/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "src/ir/transforms/utils/physical_sync_schedule.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::ir::sync {
namespace physical {
// Installs preselected descriptors. This class makes no transformation choices.
class ScheduleMaterializer : public IRMutator {
  const CutInventory& inventory_;
  SyncTokenNames& names_;
  PhysicalSyncRealization& work_;
  const BoundarySchedule* schedule_ = nullptr;
  ExprPtr block_;
  const PreparedLoops* regular_ = nullptr;
  const BoundarySchedules* boundary_ = nullptr;
  std::pair<int64_t, int64_t> blocks_{};
  StmtPtr EmitRegular(const PreparedLoopSchedule& selected) {
    const auto& plan = selected.loop;
    const auto& groups = selected.groups;
    auto emit_group = [&](size_t version) {
      const auto& phases = selected.versions[version];
      std::vector<StmtPtr> output;
      auto emit = [&](const PreparedSchedulePhase& prepared) {
        const auto first = prepared.first, end = prepared.end, period = prepared.period;
        const auto steady = prepared.steady;
        std::vector<StmtPtr> bodies;
        auto phase_iv = std::make_shared<Var>(
            names_.Next(plan.loop->loop_var_->name_hint_ + (steady ? "_sync_steady_" : "_sync_boundary_")),
            plan.loop->loop_var_->GetType(), plan.loop->span_);
        const auto copies = period;
        for (int64_t residue = 0; residue < copies; ++residue) {
          const auto& cuts = prepared.visits[residue];
          ScheduleMaterializer payload(cuts, names_, work_);
          auto body = payload.VisitStmt(plan.loop->body_);
          // Empty yield belongs to the new enclosing group, not each copied visit.
          class NoYield : public IRMutator {
            StmtPtr VisitStmt_(const YieldStmtPtr& y) override {
              return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, y->span_);
            }
          };
          body = NoYield().VisitStmt(body);
          ExprPtr coordinate = phase_iv;
          if (residue) {
            auto variable = std::make_shared<Var>(names_.Next("sync_visit_"), plan.loop->loop_var_->GetType(),
                                                  plan.loop->span_);
            auto offset = std::make_shared<ConstInt>(residue * plan.step, DataType::INDEX, plan.loop->span_);
            bodies.push_back(std::make_shared<AssignStmt>(
                variable, std::make_shared<Add>(coordinate, offset, DataType::INDEX, plan.loop->span_),
                plan.loop->span_));
            coordinate = variable;
          }
          bodies.push_back(
              DeepClone(body, {{plan.loop->loop_var_.get(), coordinate}}, true, [&](const std::string& name) {
                return names_.Next(name + "_sync_visit_");
              }).cloned_body);
          ++work_.regular_schedule_body_copies;
          work_.regular_schedule_payload_copies += plan.payloads;
        }
        bodies.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, plan.loop->span_));
        auto bound = [&](int64_t digit) {
          return std::make_shared<ConstInt>(
              static_cast<int64_t>(__int128(plan.start) + __int128(digit) * plan.step), DataType::INDEX,
              plan.loop->span_);
        };
        auto loop = std::make_shared<ForStmt>(
            phase_iv, selected.identity ? plan.loop->start_ : bound(first),
            selected.identity ? plan.loop->stop_ : bound(end),
            selected.identity
                ? plan.loop->step_
                : std::make_shared<ConstInt>(period * plan.step, DataType::INDEX, plan.loop->span_),
            std::vector<IterArgPtr>{}, SeqStmts::Flatten(std::move(bodies), plan.loop->span_),
            std::vector<VarPtr>{}, plan.loop->span_, ForKind::Sequential);
        const auto entry = MaterializeFrozenCommands(prepared.entry, names_, work_);
        output.insert(output.end(), entry.begin(), entry.end());
        output.push_back(loop);
        const auto exit = MaterializeFrozenCommands(prepared.exit, names_, work_);
        output.insert(output.end(), exit.begin(), exit.end());
      };
      for (const auto& phase : phases) emit(phase);
      return SeqStmts::Flatten(std::move(output), plan.loop->span_);
    };
    if (groups.empty()) return emit_group(0);
    StmtPtr result = emit_group(groups.size() - 1);
    const auto& block = block_;
    for (size_t i = groups.size() - 1; i-- > 0;)
      result = std::make_shared<IfStmt>(
          InvocationGroupGuard(block, blocks_, groups[i].intervals, plan.loop->span_), emit_group(i), result,
          std::vector<VarPtr>{}, plan.loop->span_);
    return result;
  }
  StmtPtr Wrap(const StmtPtr& payload, const PhysicalCuts& cuts) {
    const auto found = schedule_ ? schedule_->cuts.find(&cuts) : decltype(schedule_->cuts)::const_iterator{};
    const auto* selected = schedule_ && found != schedule_->cuts.end() ? &found->second : nullptr;
    auto statements = MaterializeFrozenCommands(selected ? selected->first : cuts.before, names_, work_);
    statements.push_back(payload);
    const auto after = MaterializeFrozenCommands(selected ? selected->second : cuts.after, names_, work_);
    statements.insert(statements.end(), after.begin(), after.end());
    return std::make_shared<SeqStmts>(statements, payload->span_);
  }
  StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
    if (regular_) {
      if (auto found = regular_->find(loop.get()); found != regular_->end()) {
        return EmitRegular(found->second);
      }
    }
    const auto found = boundary_ ? boundary_->find(loop.get()) : BoundarySchedules::const_iterator{};
    if (!boundary_ || found == boundary_->end()) return IRMutator::VisitStmt_(loop);
    const auto& plan = found->second;
    auto output = MaterializeFrozenCommands(plan.entry, names_, work_);
    const auto* previous = schedule_;
    schedule_ = &plan;
    auto body = VisitStmt(loop->body_);
    schedule_ = previous;
    output.push_back(std::make_shared<ForStmt>(
        loop->loop_var_, loop->start_, loop->stop_, loop->step_, loop->iter_args_, body, loop->return_vars_,
        loop->span_, loop->kind_, loop->attrs_, loop->leading_comments_, loop->hardware_loop_policy_));
    auto exit = MaterializeFrozenCommands(plan.exit, names_, work_);
    output.insert(output.end(), exit.begin(), exit.end());
    return SeqStmts::Flatten(std::move(output), loop->span_);
  }
  static const Call* ArmCall(StmtPtr arm) {
    if (auto seq = As<SeqStmts>(arm)) {
      if (seq->stmts_.size() != 1) return nullptr;
      arm = seq->stmts_.front();
    }
    const auto eval = As<EvalStmt>(arm);
    return eval ? As<Call>(eval->expr_).get() : nullptr;
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    const auto found = inventory_.find(call.get());
    return found == inventory_.end() ? statement : Wrap(statement, *found->second);
  }
  StmtPtr VisitStmt_(const IfStmtPtr& statement) override {
    const auto found = inventory_.find(ArmCall(statement->then_body_));
    if (found == inventory_.end()) return IRMutator::VisitStmt_(statement);
    CHECK(statement->else_body_) << "Invocation cut requires the original initialization alternatives";
    const auto other = inventory_.find(ArmCall(*statement->else_body_));
    CHECK(other != inventory_.end() && other->second == found->second)
        << "Invocation initialization alternatives have incompatible command cuts";
    return Wrap(statement, *found->second);
  }

 public:
  ScheduleMaterializer(const CutInventory& inventory, SyncTokenNames& names, PhysicalSyncRealization& work,
                       const BoundarySchedules* boundaries = nullptr)
      : inventory_(inventory), names_(names), work_(work), boundary_(boundaries) {}
  ScheduleMaterializer(const CutInventory& inventory, const ExprPtr& block, const PreparedLoops& regular,
                       const BoundarySchedules& boundaries, std::pair<int64_t, int64_t> blocks,
                       PhysicalSyncRealization& work, SyncTokenNames& names)
      : inventory_(inventory),
        names_(names),
        work_(work),
        block_(block),
        regular_(&regular),
        boundary_(&boundaries),
        blocks_(blocks) {}
};

FunctionPtr MaterializeSelectedSchedule(const FunctionPtr& owner, const ExprPtr& block,
                                        const CutInventory& cuts, const PreparedLoops& loops,
                                        const BoundarySchedules& boundaries,
                                        std::pair<int64_t, int64_t> blocks, PhysicalSyncRealization& work,
                                        SyncTokenNames& names) {
  return ScheduleMaterializer(cuts, block, loops, boundaries, blocks, work, names).VisitFunction(owner);
}

}  // namespace physical

}  // namespace pypto::ir::sync
