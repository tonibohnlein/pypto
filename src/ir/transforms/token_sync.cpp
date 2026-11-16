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

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/simplify.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/conservative_sync_dataflow.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/scoped_logical_sync.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::pass {
namespace {
// Fold only a singleton cardinality proved across every caller. This is
// scoped specialization of an immutable function, not inference from an index
// hull. On optional realization failure the caller retains the original body.
FunctionPtr BindLaunchCount(const FunctionPtr& function, std::optional<int64_t> count) {
  if (!count) return function;
  class Binder : public IRMutator {
    int64_t count_;

   public:
    bool changed = false;
    explicit Binder(int64_t count) : count_(count) {}
    ExprPtr VisitExpr_(const CallPtr& call) override {
      if (IsOp(call, "buffer.get_block_num") && call->args_.empty()) {
        changed = true;
        return std::make_shared<ConstInt>(count_, DataType::INDEX, call->span_);
      }
      return IRMutator::VisitExpr_(call);
    }
  } binder(*count);
  auto bound = binder.VisitFunction(function);
  return binder.changed ? transform_utils::SimplifyFunction(bound, true) : function;
}
std::optional<int64_t> FixedLaunchCount(const sync::SyncInvocationEvidence& evidence,
                                        const std::string& name) {
  const auto count = evidence.block_counts.find(name);
  if (count == evidence.block_counts.end() || count->second.first != count->second.second)
    return std::nullopt;
  return count->second.first;
}
bool IsOwnedLeaf(const FunctionPtr& function) {
  return (function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV) &&
         !function->requires_runtime_binding_ && !function->HasAttr(kAttrExternalSource) &&
         !function->HasAttr(kAttrBuiltinTemplateDir);
}
// Explicit whole-module requests are atomic and deliberately narrower than
// per-leaf delegation. Orchestration/group wrappers are not local PTO bodies;
// unknown, external and per-thread helper domains require their own owner.
void CheckWholeModuleScope(const ProgramPtr& program) {
  const auto* context = PassContext::Current();
  CHECK(context && context->GetMemoryPlanner() != MemoryPlanner::PtoAS)
      << "Whole-module synchronization ownership requires PyPTO-placed storage";
  size_t leaves = 0;
  for (const auto& [global, function] : program->functions_) {
    const auto type = function->func_type_;
    if (type == FunctionType::Orchestration || type == FunctionType::Graph || type == FunctionType::Group ||
        type == FunctionType::Spmd)
      continue;
    CHECK_SPAN(IsOwnedLeaf(function) && function->ir_stage_ == FunctionIRStage::Buffer, function->span_)
        << "Whole-module synchronization ownership requires qualified local Buffer AIC/AIV leaves; "
           "mixed, external or helper ownership is unavailable";
    CHECK_SPAN(!function->GetAttr<bool>(kAttrPyptoAutosync), function->span_)
        << "Whole-module synchronization ownership cannot reprocess an owned physical plan";
    ++leaves;
  }
  CHECK(leaves) << "Whole-module synchronization ownership requires a local compilation domain";
}
ProgramPtr PublishWholeModuleOutcome(const ProgramPtr& program) {
  for (const auto& [global, function] : program->functions_)
    if (IsOwnedLeaf(function))
      CHECK_SPAN(function->GetAttr<bool>(kAttrPyptoAutosync), function->span_)
          << "Whole-module synchronization ownership refused for " << function->name_ << ": "
          << function->GetAttr<std::string>(kAttrAutosyncFallbackReason);
  auto functions = program->functions_;
  for (auto& [global, function] : functions) {
    if (!IsOwnedLeaf(function)) continue;
    auto marked = std::make_shared<Function>(*function);
    marked->attrs_.emplace_back(kAttrAutosyncWholeModule, true);
    function = std::move(marked);
  }
  return std::make_shared<Program>(functions, program->name_, program->span_);
}
bool HasPhysicalCommands(const sync::LoweredLogicalSync& result) {
  if (result.set_sites || result.wait_sites) return true;
  return std::any_of(result.physical_cases.begin(), result.physical_cases.end(), [](const auto& candidate) {
    return candidate.set_sites || candidate.wait_sites || candidate.barrier_sites;
  });
}
// Consuming every flag does not complete the last asynchronous payload.
// Whole-kernel ownership suppresses PTOAS's terminal drain, so explicitly
// discharge that obligation using the existing A2/A3 ALL-pipe primitive.
// Internal child lowering never calls this wrapper: a child may return pending
// payloads. This changes the invocation boundary, not the internal demand set.
FunctionPtr CompleteOwnedInvocation(const FunctionPtr& function, const std::string& target) {
  sync::CheckPhysicalSyncResult(function);
  const auto& execution = backend::RequireLocalSyncExecutionContract(target);
  // Current leaf import admits a canonical terminal return. Reject fallthrough
  // rather than publish an ownership marker with an uncovered exit.
  auto body = As<SeqStmts>(function->body_);
  CHECK(body && !body->stmts_.empty() && As<ReturnStmt>(body->stmts_.back()))
      << "Invocation completion requires a canonical terminal return";
  class CompleteReturns : public sync::SyncStatementMutator {
    std::string core_;
    const char* completion_;

   public:
    explicit CompleteReturns(FunctionType type, const char* completion)
        : core_(type == FunctionType::AIC ? "aic" : "aiv"), completion_(completion) {}
    StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
      // Return operands are already computed Buffer/Tensor handles or scalars.
      // Do not move an effectful return expression past the completion cut.
      for (const auto& value : op->value_)
        CHECK(As<Var>(value) || As<ConstInt>(value) || As<ConstFloat>(value) || As<ConstBool>(value))
            << "Invocation completion requires evaluated return operands";
      auto completion = OpRegistry::GetInstance().Create(completion_, {}, {{"core_type", core_}}, op->span_);
      return std::make_shared<SeqStmts>(
          std::vector<StmtPtr>{std::make_shared<EvalStmt>(completion, op->span_), op}, op->span_);
    }
  } complete(function->func_type_, execution.terminal_completion_op);
  auto completed = MutableCopy(function);
  completed->body_ = complete.VisitStmt(function->body_);
  return sync::SyncOutcome(completed, true, "");
}
// Validate the requested original cuts in one statement occurrence walk before
// constructing mathematics. Pointer membership alone cannot detect a shared
// subtree appearing twice, and an accepted ancestor would hide a nested import.
class NativeChildScopeRequests : public IRVisitor {
 public:
  explicit NativeChildScopeRequests(const std::vector<ForStmtPtr>& scopes) {
    for (const auto& scope : scopes) {
      CHECK(scope) << "Native child preparation received a missing loop scope";
      CHECK_SPAN(occurrences_.emplace(scope.get(), 0).second, scope->span_)
          << "Native child preparation received duplicate loop scopes";
    }
  }
  void Validate(const FunctionPtr& function) {
    VisitFunction(function);
    for (const auto& [scope, count] : occurrences_) {
      CHECK_SPAN(count == 1, scope->span_)
          << "Native child scope must occur exactly once in the original function";
    }
  }
  void VisitStmt(const StmtPtr& statement) override {
    const auto loop = As<ForStmt>(statement);
    const auto selected = loop ? occurrences_.find(loop.get()) : occurrences_.end();
    if (selected != occurrences_.end()) {
      CHECK_SPAN(!inside_selected_, statement->span_)
          << "Native child preparation refuses overlapping requested scopes";
      ++selected->second;
      inside_selected_ = true;
    }
    IRVisitor::VisitStmt(statement);
    if (selected != occurrences_.end()) {
      inside_selected_ = false;
    }
  }
  void VisitExpr(const ExprPtr&) override {}

 private:
  std::unordered_map<const ForStmt*, size_t> occurrences_;
  bool inside_selected_ = false;
};

FunctionPtr InsertLogicalSync(const FunctionPtr& function, const std::string& target,
                              const sync::SyncParameterBounds& parameters,
                              std::optional<std::pair<int64_t, int64_t>> blocks,
                              std::optional<int64_t> launch_count) {
  if (blocks) {
    const auto invocation =
        sync::AnalyzeSyncInvocation(function, true, target, parameters, *blocks, launch_count);
    return invocation.cases.size() == 1
               ? sync::RealizeCompactSync(*invocation.cases.front().mathematics).function
               : sync::RealizeLogicalInvocation(invocation).function;
  }
  auto input = sync::ImportPlacedSync(function, true, target, parameters, blocks);
  return sync::RealizeCompactSync(sync::AnalyzeCompactSync(std::move(input))).function;
}
}  // namespace

std::vector<std::shared_ptr<const sync::NativeSyncChild>> PrepareNativeSyncChildren(
    const FunctionPtr& function, const std::vector<ForStmtPtr>& scopes) {
  auto* context = PassContext::Current();
  CHECK(context && context->GetAutoSyncDisjointGM())
      << "Native child preparation requires the explicit disjoint GM argument contract";
  CHECK(function) << "Native child preparation requires an original function";
  CHECK_SPAN(IsOwnedLeaf(function), function->span_)
      << "Native child preparation requires an owned local leaf";
  CHECK_SPAN(
      function->ir_stage_ == FunctionIRStage::Buffer && context->GetMemoryPlanner() != MemoryPlanner::PtoAS,
      function->span_)
      << "Native child preparation requires physically placed Buffer IR";
  CheckLogicalSync(function);
  CHECK_SPAN(!sync::HasLogicalSync(function), function->span_)
      << "Native child preparation refuses preexisting logical synchronization";
  if (scopes.empty()) {
    return {};  // No evidence is exported; in particular this proves no empty plan.
  }
  NativeChildScopeRequests(scopes).Validate(function);
  const auto target = context->GetBackendHandler()->GetPtoTargetArch();
  // One explicit whole-root request, not detached scope analysis or automatic
  // backend search. Every returned capability shares this immutable result.
  auto input = sync::ImportPlacedSync(function, context->GetAutoSyncDisjointGM(), target);
  const auto mathematics = sync::AnalyzeCompactSync(std::move(input));
  std::vector<std::shared_ptr<const sync::NativeSyncChild>> children;
  children.reserve(scopes.size());
  for (const auto& scope : scopes) {
    auto exported = sync::ExportNativeSyncChild(mathematics, scope, target);
    CHECK_SPAN(exported.child, scope->span_)
        << "Native child export unavailable (" << sync::NativeSyncChildStatusName(exported.status)
        << "): " << exported.reason;
    children.push_back(std::move(exported.child));
  }
  return children;
}

std::vector<std::shared_ptr<const sync::EventfulSyncChild>> PrepareEventfulSyncChildren(
    const FunctionPtr& function, const std::vector<ForStmtPtr>& scopes) {
  auto* context = PassContext::Current();
  CHECK(context && function) << "Eventful child preparation requires an original function and active context";
  CHECK_SPAN(IsOwnedLeaf(function) && function->ir_stage_ == FunctionIRStage::Buffer &&
                 context->GetMemoryPlanner() != MemoryPlanner::PtoAS,
             function->span_)
      << "Eventful child preparation requires a physically placed local Buffer owner";
  CheckLogicalSync(function);
  CHECK(!sync::HasLogicalSync(function)) << "Eventful child preparation requires original payload IR";
  if (scopes.empty()) return {};
  NativeChildScopeRequests(scopes).Validate(function);
  const auto target = context->GetBackendHandler()->GetPtoTargetArch();
  std::vector<std::shared_ptr<const sync::EventfulSyncChild>> children;
  children.reserve(scopes.size());
  const sync::SyncRegionBindings bindings(function, target);
  for (const auto& scope : scopes)
    children.push_back(sync::PrepareEventfulSyncChild(function, scope, target, bindings));
  return children;
}

Pass InsertSync(std::vector<std::shared_ptr<const sync::NativeSyncChild>> accepted_children,
                std::vector<std::shared_ptr<const sync::EventfulSyncChild>> eventful_children) {
  // Invocation evidence belongs to this program, never to a process-global
  // sidecar. Standalone failure leaves the immutable input program untouched.
  return CreateProgramPass(
      [accepted_children = std::move(accepted_children),
       eventful_children = std::move(eventful_children)](const ProgramPtr& program) {
        auto* context = PassContext::Current();
        CHECK(context &&
              (context->GetAutoSyncPolicy() == "conservative_prefix" || context->GetAutoSyncDisjointGM()))
            << "InsertSync requires the explicit disjoint GM argument contract";
        const auto target = context->GetBackendHandler()->GetPtoTargetArch();
        const bool prefix = context->GetAutoSyncPolicy() == "conservative_prefix";
        CHECK(prefix || (accepted_children.empty() && eventful_children.empty()))
            << "Accepted synchronization children require the explicit conservative_prefix policy";
        // Validate the complete request before constructing any transformed
        // function. Names or a structurally equivalent clone cannot authorize
        // reuse of retained coordinates, placement, or caller evidence.
        std::unordered_map<const Function*, size_t> owner_counts;
        if (!accepted_children.empty() || !eventful_children.empty())
          for (const auto& [global, function] : program->functions_) ++owner_counts[function.get()];
        std::unordered_map<const Function*, std::vector<std::shared_ptr<const sync::NativeSyncChild>>>
            children_by_owner;
        std::unordered_map<const Function*, std::unordered_set<const ForStmt*>> requested_scopes;
        for (const auto& child : accepted_children) {
          CHECK(child) << "InsertSync received an unavailable child capability";
          const auto& owner = child->Owner();
          const auto found = owner_counts.find(owner.get());
          CHECK(found != owner_counts.end() && found->second == 1)
              << "InsertSync accepted child has no unique original owner in this program";
          CHECK(IsOwnedLeaf(owner)) << "InsertSync accepted child requires an owned local leaf";
          child->ValidateFor(owner, target);
          CHECK(requested_scopes[owner.get()].insert(child->Scope().get()).second)
              << "InsertSync received duplicate accepted child scopes";
          children_by_owner[owner.get()].push_back(child);
        }
        std::unordered_map<const Function*, std::vector<std::shared_ptr<const sync::EventfulSyncChild>>>
            eventful_by_owner;
        for (const auto& child : eventful_children) {
          CHECK(child) << "InsertSync received an unavailable eventful child capability";
          const auto& owner = child->Owner();
          const auto found = owner_counts.find(owner.get());
          CHECK(found != owner_counts.end() && found->second == 1)
              << "InsertSync eventful child has no unique original owner";
          child->ValidateFor(owner, target);
          CHECK(requested_scopes[owner.get()].insert(child->Scope().get()).second)
              << "InsertSync received duplicate accepted child scopes";
          eventful_by_owner[owner.get()].push_back(child);
        }
        const std::vector<std::shared_ptr<const sync::NativeSyncChild>> no_children;
        const std::vector<std::shared_ptr<const sync::EventfulSyncChild>> no_eventful;
        const auto evidence =
            prefix ? sync::SyncInvocationEvidence{} : sync::CollectSyncInvocationEvidence(program);
        auto functions = program->functions_;
        for (auto& [global, function] : functions) {
          CheckLogicalSync(function);
          CHECK(!sync::HasLogicalSync(function)) << "InsertSync refuses preexisting logical synchronization";
          if (!IsOwnedLeaf(function)) continue;
          CHECK(function->ir_stage_ == FunctionIRStage::Buffer &&
                context->GetMemoryPlanner() != MemoryPlanner::PtoAS)
              << "InsertSync requires physically placed Buffer IR";
          if (prefix) {
            const auto found = children_by_owner.find(function.get());
            const auto& children = found == children_by_owner.end() ? no_children : found->second;
            const auto events = eventful_by_owner.find(function.get());
            const auto& eventful = events == eventful_by_owner.end() ? no_eventful : events->second;
            function = sync::InsertConservativePrefixes(sync::AnalyzeConservativeSync(
                function, target, children, eventful, false, children.empty() && eventful.empty()));
            sync::CheckGeneratedLogicalSync(function);
            continue;
          }
          auto bounds = evidence.parameters.find(global->name_);
          const auto parameters =
              bounds == evidence.parameters.end() ? sync::SyncParameterBounds{} : bounds->second;
          auto block = evidence.block_indices.find(global->name_);
          const auto blocks =
              block == evidence.block_indices.end() ? std::nullopt : std::make_optional(block->second);
          function = InsertLogicalSync(BindLaunchCount(function, FixedLaunchCount(evidence, global->name_)),
                                       target, parameters, blocks, FixedLaunchCount(evidence, global->name_));
          sync::CheckGeneratedLogicalSync(function);
        }
        return std::make_shared<Program>(functions, program->name_, program->span_);
      },
      "InsertSync", PassProperties{.produced = {IRProperty::LogicalSyncValid}});
}

Pass OptimizeSync() {
  return CreateProgramPass(
      [](const ProgramPtr& program) {
        auto functions = program->functions_;
        for (auto& [global, function] : functions) function = sync::OptimizeLogicalSync(function);
        return std::make_shared<Program>(functions, program->name_, program->span_);
      },
      "OptimizeSync", PassProperties{.produced = {IRProperty::LogicalSyncValid}});
}

Pass LowerSync(bool require_whole_module) {
  return CreateProgramPass(
      [require_whole_module](const ProgramPtr& program) {
        if (require_whole_module) CheckWholeModuleScope(program);
        // Re-establish caller facts from this token program. Neither insertion's
        // old analysis nor a process-global sidecar authorizes physical lowering.
        for (const auto& [global, function] : program->functions_) CheckLogicalSync(function);
        bool needs_occurrences = false;
        for (const auto& [global, function] : program->functions_)
          needs_occurrences |= IsOwnedLeaf(function) && sync::HasLogicalSync(function) &&
                               !sync::HasOnlyClosedPrefixSyntax(function);
        const auto evidence =
            needs_occurrences ? sync::CollectSyncInvocationEvidence(program) : sync::SyncInvocationEvidence{};
        auto* context = PassContext::Current();
        CHECK(context) << "LowerSync requires a target PassContext";
        auto functions = program->functions_;
        for (auto& [global, function] : functions) {
          if (!IsOwnedLeaf(function)) continue;
          auto bound = evidence.parameters.find(global->name_);
          const auto parameters =
              bound == evidence.parameters.end() ? sync::SyncParameterBounds{} : bound->second;
          auto block = evidence.block_indices.find(global->name_);
          const auto blocks =
              block == evidence.block_indices.end() ? std::nullopt : std::make_optional(block->second);
          const bool payloads =
              require_whole_module &&
              sync::QualifyLocalSyncInvocation(function, context->GetBackendHandler()->GetPtoTargetArch());
          auto result = sync::LowerLogicalSync(
              function, context->GetBackendHandler()->GetPtoTargetArch(), std::nullopt, parameters, blocks,
              {}, sync::SyncExternalPolicy::Geometry, FixedLaunchCount(evidence, global->name_));
          // Default deployment still delegates plans lacking literal flags.
          // The explicit request also qualifies resource-empty/native-only
          // plans. A genuinely empty program has no asynchronous exit debt.
          if (require_whole_module && !payloads && !HasPhysicalCommands(result))
            function = sync::SyncOutcome(result.function, true, "");
          else
            function = (require_whole_module || (result.set_sites && result.wait_sites))
                           ? CompleteOwnedInvocation(result.function,
                                                     context->GetBackendHandler()->GetPtoTargetArch())
                           : result.function;
        }
        auto result = std::make_shared<Program>(functions, program->name_, program->span_);
        return require_whole_module ? PublishWholeModuleOutcome(result) : result;
      },
      "LowerSync", PassProperties{.produced = {IRProperty::LogicalSyncValid}});
}

Pass RealizeSync(bool require_whole_module) {
  auto transform = [require_whole_module](const FunctionPtr& function,
                                          const sync::SyncParameterBounds& bounds,
                                          const std::optional<std::pair<int64_t, int64_t>>& block_bounds,
                                          std::optional<int64_t> block_count) -> FunctionPtr {
    if (!IsInCoreType(function->func_type_)) return function;
    // These functions do not enter PTO code generation; their external owner
    // remains responsible for synchronization. Do not claim PTOAS delegation.
    if (function->requires_runtime_binding_ || function->HasAttr(kAttrExternalSource) ||
        function->HasAttr(kAttrBuiltinTemplateDir))
      return function;
    auto* ctx = PassContext::Current();
    CHECK(ctx && ctx->GetEnablePyptoAutoSync()) << "RealizeSync requires enable_pypto_autosync=True";
    if (function->func_type_ != FunctionType::AIC && function->func_type_ != FunctionType::AIV)
      return sync::SyncOutcome(function, false, "PyPTO synchronization supports only AIC/AIV leaves");
    if (function->ir_stage_ != FunctionIRStage::Buffer || ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS) {
      std::string reason = "PyPTO synchronization requires physically placed Buffer IR";
      if (function->HasAttr(kAttrBufferAdmissionProperty)) {
        reason += "; first obligation: " + function->GetAttr<std::string>(kAttrBufferAdmissionStage) + "/" +
                  function->GetAttr<std::string>(kAttrBufferAdmissionProperty) + " at " +
                  function->GetAttr<std::string>(kAttrBufferAdmissionOperation);
        const auto location = function->GetAttr<std::string>(kAttrBufferAdmissionLocation);
        if (!location.empty()) {
          reason += " (" + location + ")";
        }
      }
      return sync::SyncOutcome(function, false, reason);
    }
    const bool prefix = ctx->GetAutoSyncPolicy() == "conservative_prefix";
    if (!prefix && !ctx->GetAutoSyncDisjointGM())
      return sync::SyncOutcome(function, false, "Disjoint GM argument contract was not supplied");
    auto target = ctx->GetBackendHandler()->GetPtoTargetArch();
    if (target != "a2a3")
      return sync::SyncOutcome(function, false,
                               "PyPTO physical synchronization supports only manual a2a3 macros");
    // Only publish after complete analysis AND realization. A qualification or
    // resource refusal returns the original body, never partially emitted IR.
    // Internal errors are deliberately not caught. The strict retained-result
    // APIs still throw on these user-facing input/proof refusals.
    try {
      // The same logical producer and actual-token consumer as the public
      // stages. Insertion preserves callers, payloads and scalar coordinates,
      // so this transaction may reuse its own freshly collected evidence.
      const auto logical = prefix ? sync::InsertConservativePrefixes(
                                        sync::AnalyzeConservativeSync(function, target, {}, {}, false, true))
                                  : InsertLogicalSync(BindLaunchCount(function, block_count), target, bounds,
                                                      block_bounds, block_count);
      sync::CheckGeneratedLogicalSync(logical);
      const auto optimized = sync::OptimizeLogicalSync(logical);
      sync::CheckGeneratedLogicalSync(optimized);
      const auto physical = sync::LowerLogicalSync(optimized, target, std::nullopt, bounds, block_bounds, {},
                                                   sync::SyncExternalPolicy::Geometry, block_count);
      // PTOAS recognizes literal flags, not empty plans or barriers, as manual
      // ownership. Do not fabricate a dummy flag or mix partial ownership.
      if (!require_whole_module && (!physical.set_sites || !physical.wait_sites))
        return sync::SyncOutcome(function, false,
                                 "Candidate has no SET/WAIT marker; delegate the original kernel");
      if (require_whole_module && !sync::QualifyLocalSyncInvocation(optimized, target) &&
          !HasPhysicalCommands(physical))
        return sync::SyncOutcome(physical.function, true, "");
      return CompleteOwnedInvocation(physical.function, target);
    } catch (const pypto::ValueError& error) {
      return sync::SyncOutcome(function, false, error.what());
    }
  };
  // Range evidence belongs to this closed program's call sites, not mutable
  // function metadata. Do not infer bounds when the standalone API has none.
  return CreateProgramPass(
      [transform, require_whole_module](const ProgramPtr& program) {
        if (require_whole_module) CheckWholeModuleScope(program);
        // Never delegate unresolved logical IR or reprocess an owned physical
        // plan, even when an earlier eligibility test would skip that function.
        for (const auto& [global, function] : program->functions_) {
          CHECK(!sync::HasLogicalSync(function))
              << "RealizeSync refuses preexisting logical synchronization; use LowerSync";
          CHECK(!function->GetAttr<bool>(kAttrPyptoAutosync))
              << "RealizeSync cannot reprocess an owned physical plan";
        }
        const auto* context = PassContext::Current();
        auto evidence = context && context->GetAutoSyncPolicy() == "conservative_prefix"
                            ? sync::SyncInvocationEvidence{}
                            : sync::CollectSyncInvocationEvidence(program);
        auto functions = program->functions_;
        for (auto& [global, function] : functions) {
          auto block = evidence.block_indices.find(global->name_);
          function = transform(
              function, evidence.parameters[global->name_],
              block == evidence.block_indices.end() ? std::nullopt : std::make_optional(block->second),
              FixedLaunchCount(evidence, global->name_));
        }
        auto result = std::make_shared<Program>(functions, program->name_, program->span_);
        return require_whole_module ? PublishWholeModuleOutcome(result) : result;
      },
      "RealizeSync", PassProperties{});
}
}  // namespace pypto::ir::pass
