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

#include "pypto/codegen/gm_pipe_layout_evidence.h"

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/gm_pipe_layout.h"
#include "pypto/core/logging.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto::codegen::gm_pipe {
namespace {
struct CallEvidence {
  ir::CallPtr call;
  bool submit = false;
};

struct FunctionEvidence {
  ir::FunctionPtr function;
  std::vector<std::string> callees;
  std::vector<CallEvidence> calls;
  std::optional<size_t> workspace;
  bool ambiguous_workspace = false;
  std::vector<std::pair<bool, size_t>> actions;  // setup index or local callee index
  std::unordered_map<const ir::Stmt*, size_t> cuts;
};

class Discover : public ir::IRVisitor {
 public:
  Discover(FunctionEvidence& function, std::vector<SetupEvidence>& setups, LayoutWork& work)
      : function_(function), setups_(setups), work_(work) {}
  void VisitStmt(const ir::StmtPtr& statement) override {
    if (!statement) {
      return;
    }
    ++work_.statements;
    ir::ExprPtr value;
    if (const auto assignment = ir::As<ir::AssignStmt>(statement)) {
      value = assignment->value_;
    } else if (const auto evaluation = ir::As<ir::EvalStmt>(statement)) {
      value = evaluation->expr_;
    }
    const auto call = ir::transform_utils::AsCallOrSubmitView(value);
    if (call) {
      Record(statement, call, ir::IsA<ir::Submit>(value));
    }
    ir::IRVisitor::VisitStmt(statement);
  }
  void VisitExpr(const ir::ExprPtr&) override {}

 private:
  FunctionEvidence& function_;
  std::vector<SetupEvidence>& setups_;
  LayoutWork& work_;
  std::unordered_set<std::string> callees_;
  void Record(const ir::StmtPtr& cut, const ir::CallPtr& call, bool submit) {
    if (ir::op_predicates::IsInitializePipe(call)) {
      const size_t index = setups_.size();
      setups_.push_back({function_.function, cut, call});
      const auto [entry, unique] = function_.cuts.emplace(cut.get(), index);
      if (!unique) {
        entry->second = std::numeric_limits<size_t>::max();
      }
      function_.actions.emplace_back(true, index);
      ++work_.setup_entries;
    } else if (const auto callee = ir::As<ir::GlobalVar>(call->op_)) {
      function_.calls.push_back({call, submit});
      if (callees_.insert(callee->name_).second) {
        function_.actions.emplace_back(false, function_.callees.size());
        function_.callees.push_back(callee->name_);
        ++work_.call_edges;
      }
    }
  }
};

struct SetupGeometry {
  PipeKey key;
  int slot_bytes = 0;
  int slots = 0;
};

// Resolve only requested setup records: an unrelated malformed function must
// not turn a formerly successful reachable-root workspace query into failure.
SetupGeometry Geometry(const SetupEvidence& setup) {
  const int direction = setup.call->GetKwarg<int>("dir_mask", 0);
  const int bytes = setup.call->GetKwarg<int>("slot_size", 0);
  if (direction <= 0 || bytes <= 0) {
    return {};  // Preserve existing non-GM layout behavior; no native admission.
  }
  const int slots = EffectiveSlotCount(direction, setup.call->GetKwarg<int>("slot_num", 0));
  return {{setup.call->GetKwarg<int>("id", 0), direction}, bytes, slots};
}
bool Merge(const SetupGeometry& setup, std::map<PipeKey, const SetupGeometry*>& selected,
           LayoutResult& result) {
  const auto [entry, inserted] = selected.emplace(setup.key, &setup);
  const auto prefix = "initialize_pipe for frontend pipe id " + std::to_string(setup.key.first) +
                      " and dir_mask " + std::to_string(setup.key.second);
  if (!inserted && entry->second->slots != setup.slots) {
    result.error = prefix + " uses inconsistent slot counts: " + std::to_string(entry->second->slots) +
                   " and " + std::to_string(setup.slots);
    return false;
  }
  if (!inserted && entry->second->slot_bytes != setup.slot_bytes) {
    result.error = prefix +
                   " uses inconsistent slot_size values: " + std::to_string(entry->second->slot_bytes) +
                   " and " + std::to_string(setup.slot_bytes);
    return false;
  }
  return true;
}

void Place(const std::map<PipeKey, const SetupGeometry*>& selected, LayoutResult& result) {
  for (const auto& [key, setup] : selected) {
    if (SlotCountForDirMask(key.second) <= 0) {
      result.error = "initialize_pipe has invalid dir_mask for GM slot buffer: " + std::to_string(key.second);
      return;
    }
    const auto bytes = FootprintBytes(key.second, setup->slots, setup->slot_bytes);
    if (result.bytes > std::numeric_limits<int64_t>::max() - bytes) {
      result.error =
          "GM slot buffer size overflow while sizing frontend pipe id " + std::to_string(key.first);
      return;
    }
    result.offsets.emplace(key, result.bytes);
    result.bytes += bytes;
  }
}
}  // namespace

namespace {
// Immutable once constructed. Query geometry and result caches deliberately
// live outside this object: exported contexts cannot retain their own cache.
struct LayoutSource {
  explicit LayoutSource(ir::ProgramPtr input, LayoutWork& work) : program(std::move(input)) {
    CHECK(program) << "GM pipe layout requires an original Program";
    for (const auto& [name, function] : program->functions_) {
      (void)name;
      const auto index = functions.size();
      identities.emplace(function.get(), index);
      names.emplace(function->name_, index);
      FunctionEvidence fact;
      fact.function = function;
      for (size_t i = 0; i < function->params_.size(); ++i) {
        if (function->params_[i]->name_hint_ == "__gm_pipe_buffer") {
          fact.ambiguous_workspace |= fact.workspace.has_value();
          fact.workspace = i;
        }
      }
      functions.push_back(std::move(fact));
      Discover(functions.back(), setups, work).VisitStmt(function->body_);
    }
  }
  ir::ProgramPtr program;
  std::vector<FunctionEvidence> functions;
  std::vector<SetupEvidence> setups;
  std::unordered_map<const ir::Function*, size_t> identities;
  std::unordered_map<std::string, size_t> names;
};
}  // namespace

struct WorkspaceContext::Data {
  std::shared_ptr<const LayoutSource> source;
  ir::FunctionPtr root;
  std::shared_ptr<const LayoutResult> layout;
  std::unordered_set<size_t> admitted;
};

struct LayoutEvidence::State {
  explicit State(ir::ProgramPtr input) {
    source = std::make_shared<LayoutSource>(std::move(input), work);
    geometry.resize(source->setups.size());
  }
  struct Binding {
    std::shared_ptr<const WorkspaceContext> context;
    std::string error;
  };
  LayoutWork work;
  std::shared_ptr<const LayoutSource> source;
  std::vector<std::optional<SetupGeometry>> geometry;
  std::unordered_map<const ir::Function*, std::shared_ptr<const LayoutResult>> cache;
  std::unordered_map<const ir::Function*, Binding> bindings;

  std::shared_ptr<const LayoutResult> Query(const ir::FunctionPtr& root, bool whole) {
    ++work.layout_requests;
    const auto key = whole ? nullptr : root.get();
    if (!whole && (!root || !source->identities.count(key))) {
      auto failure = std::make_shared<LayoutResult>();
      failure->error = "GM pipe layout root does not belong to the original Program";
      ++work.layouts_rejected;
      return failure;
    }
    if (const auto found = cache.find(key); found != cache.end()) {
      ++work.layout_cache_hits;
      return found->second;
    }
    std::vector<size_t> pending;
    if (whole) {
      for (size_t i = source->functions.size(); i > 0; --i) {
        pending.push_back(i - 1);
      }
    } else {
      pending.push_back(source->identities.at(root.get()));
    }
    auto result = Build(std::move(pending), !whole);
    cache.emplace(key, result);
    return result;
  }

  std::shared_ptr<const LayoutResult> Build(std::vector<size_t> pending, bool follow_calls) {
    auto result = std::make_shared<LayoutResult>();
    std::unordered_set<size_t> visited;
    std::map<PipeKey, const SetupGeometry*> selected;
    // Function expansion and setup application share one explicit stack, so
    // root queries keep the original depth-first call/setup order.
    std::vector<std::pair<bool, size_t>> actions;
    for (const auto function : pending) {
      actions.emplace_back(false, function);
    }
    while (!actions.empty() && result->error.empty()) {
      const auto [is_setup, index] = actions.back();
      actions.pop_back();
      if (is_setup) {
        ++work.setup_incidences;
        if (!geometry[index]) {
          ++work.setup_decodes;
          geometry[index] = Geometry(source->setups[index]);
        }
        if (geometry[index]->slot_bytes > 0) {
          (void)Merge(*geometry[index], selected, *result);
        }
        continue;
      }
      if (!visited.insert(index).second) {
        continue;
      }
      ++work.functions_visited;
      const auto& function = source->functions[index];
      for (auto action = function.actions.rbegin(); action != function.actions.rend(); ++action) {
        if (action->first) {
          actions.emplace_back(true, action->second);
        } else if (follow_calls) {
          const auto found = source->names.find(function.callees[action->second]);
          if (found != source->names.end()) {
            actions.emplace_back(false, found->second);
          }
        }
      }
    }
    if (result->error.empty()) {
      Place(selected, *result);
    }
    if (!result->error.empty()) {
      result->offsets.clear();
      result->bytes = 0;
      ++work.layouts_rejected;
    }
    return result;
  }
};

namespace {
struct WorkspaceProof {
  std::unordered_set<size_t> functions;
  std::string error;
};

// Recover the requested closure and its GM-dependent subset from the indexed
// original calls. No source body or expression is visited during a query.
WorkspaceProof ProveWorkspace(const LayoutSource& source, size_t root,
                              const std::vector<std::optional<SetupGeometry>>& geometry, LayoutWork& work) {
  WorkspaceProof proof;
  std::vector<size_t> pending{root};
  std::unordered_map<size_t, std::vector<size_t>> parents;
  std::unordered_set<size_t> required;
  while (!pending.empty()) {
    const auto index = pending.back();
    pending.pop_back();
    if (!proof.functions.insert(index).second) continue;
    const auto& function = source.functions[index];
    for (const auto& [is_setup, child] : function.actions) {
      if (is_setup) {
        ++work.setup_validations;
        const auto& setup = source.setups[child];
        if (function.cuts.at(setup.cut.get()) == std::numeric_limits<size_t>::max()) {
          proof.error = "GM workspace setup has no unique original statement occurrence";
          return proof;
        }
        if (geometry[child] && geometry[child]->slot_bytes > 0) required.insert(index);
      } else {
        const auto found = source.names.find(function.callees[child]);
        if (found != source.names.end()) {
          parents[found->second].push_back(index);
          pending.push_back(found->second);
        }
      }
    }
  }
  pending.assign(required.begin(), required.end());
  while (!pending.empty()) {
    const auto index = pending.back();
    pending.pop_back();
    for (const auto parent : parents[index]) {
      if (required.insert(parent).second) pending.push_back(parent);
    }
  }
  for (const auto index : required) {
    const auto& fact = source.functions[index];
    if (!fact.workspace || fact.ambiguous_workspace) {
      proof.error =
          "GM workspace needs one original injected formal in function '" + fact.function->name_ + "'";
      return proof;
    }
    const auto& formal = fact.function->params_[*fact.workspace];
    const auto type = ir::As<ir::TensorType>(formal->GetType());
    if (!type || type->shape_.size() != 1 || type->dtype_ != DataType::FP32) {
      proof.error = "GM workspace formal must be an original rank-one FP32 tensor";
      return proof;
    }
    for (const auto& edge : fact.calls) {
      ++work.forwarding_edges;
      const auto found = source.names.find(edge.call->op_->name_);
      if (found == source.names.end() || !required.count(found->second)) continue;
      const auto& callee = source.functions[found->second];
      if (!callee.workspace || callee.ambiguous_workspace) {
        proof.error = "GM workspace callee has no unique original injected formal";
        return proof;
      }
      const bool coverage = edge.submit ? edge.call->args_.size() <= callee.function->params_.size()
                                        : edge.call->args_.size() == callee.function->params_.size();
      const auto actual =
          ir::transform_utils::CallerSuppliedArg(edge.call, callee.function, *callee.workspace);
      if (!coverage || actual.get() != formal.get()) {
        proof.error = "GM workspace call must forward the exact original workspace formal from '" +
                      fact.function->name_ + "' to '" + callee.function->name_ + "'";
        return proof;
      }
    }
  }
  return proof;
}
}  // namespace

LayoutEvidence::LayoutEvidence(ir::ProgramPtr program)
    : state_(std::make_unique<State>(std::move(program))) {}
LayoutEvidence::~LayoutEvidence() = default;
std::shared_ptr<const LayoutResult> LayoutEvidence::WholeProgram() { return state_->Query({}, true); }
std::shared_ptr<const LayoutResult> LayoutEvidence::ReachableFrom(const ir::FunctionPtr& root) {
  return state_->Query(root, false);
}

std::shared_ptr<const WorkspaceContext> LayoutEvidence::BindRoot(const ir::FunctionPtr& root) {
  ++state_->work.binding_requests;
  const auto& source = *state_->source;
  const auto found = source.identities.find(root.get());
  CHECK(root && found != source.identities.end())
      << "GM workspace root does not belong to the original Program";
  if (const auto cached = state_->bindings.find(root.get()); cached != state_->bindings.end()) {
    ++state_->work.binding_cache_hits;
    CHECK(cached->second.error.empty()) << cached->second.error;
    return cached->second.context;
  }
  State::Binding binding;
  const auto layout = ReachableFrom(root);
  binding.error = layout->error;
  if (binding.error.empty()) {
    auto proof = ProveWorkspace(source, found->second, state_->geometry, state_->work);
    binding.error = std::move(proof.error);
    if (binding.error.empty()) {
      auto data = std::make_shared<WorkspaceContext::Data>();
      data->source = state_->source;
      data->root = root;
      data->layout = layout;
      data->admitted = std::move(proof.functions);
      binding.context = std::shared_ptr<const WorkspaceContext>(new WorkspaceContext(std::move(data)));
    }
  }
  if (!binding.error.empty()) ++state_->work.bindings_rejected;
  const auto& retained = state_->bindings.emplace(root.get(), std::move(binding)).first->second;
  CHECK(retained.error.empty()) << retained.error;
  return retained.context;
}

std::shared_ptr<const WorkspaceContext> LayoutEvidence::InferCodegenContext() {
  auto data = std::make_shared<WorkspaceContext::Data>();
  data->source = state_->source;
  // Most modules have no GM pipe setup. The shared source scan already proves
  // that fact; do not reconstruct every entry's call closure to rediscover it.
  if (state_->source->setups.empty()) {
    data->layout = std::make_shared<LayoutResult>();
    return std::shared_ptr<const WorkspaceContext>(new WorkspaceContext(std::move(data)));
  }
  for (const auto& fact : state_->source->functions) {
    const auto layout = ReachableFrom(fact.function);
    CHECK(layout->error.empty()) << layout->error;
    if (layout->bytes == 0) continue;
    const auto bound = BindRoot(fact.function);
    if (data->layout) {
      CHECK(data->layout->bytes == layout->bytes && data->layout->offsets == layout->offsets)
          << "PTO entries have incompatible GM workspace layouts; provide an explicit original-root context";
    } else {
      data->layout = layout;
    }
    data->admitted.insert(bound->data_->admitted.begin(), bound->data_->admitted.end());
  }
  if (!data->layout) data->layout = std::make_shared<LayoutResult>();
  return std::shared_ptr<const WorkspaceContext>(new WorkspaceContext(std::move(data)));
}

WorkspaceContext::WorkspaceContext(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
WorkspaceContext::~WorkspaceContext() = default;
const LayoutResult& WorkspaceContext::Layout() const { return *data_->layout; }
const ir::FunctionPtr& WorkspaceContext::Root() const { return data_->root; }
std::string WorkspaceContext::ValidateProgram(const ir::ProgramPtr& program) const {
  if (!program) return "GM workspace context requires an emission Program";
  for (const auto& [name, function] : program->functions_) {
    (void)name;
    const auto found = data_->source->identities.find(function.get());
    if (found == data_->source->identities.end()) {
      return "GM workspace context does not own the original emitted Function";
    }
    // Original Function identity seals its body and every SetupAt identity.
    // Helpers without setup may be included without claiming shared workspace.
    if (!data_->admitted.count(found->second) && !data_->source->functions[found->second].cuts.empty()) {
      return "GM workspace context does not cover an emitted original setup";
    }
  }
  return {};
}

const SetupEvidence* LayoutEvidence::SetupAt(const ir::FunctionPtr& function, const ir::StmtPtr& cut) const {
  ++state_->work.setup_lookups;
  const auto owner = state_->source->identities.find(function.get());
  if (owner == state_->source->identities.end()) {
    return nullptr;
  }
  const auto& cuts = state_->source->functions[owner->second].cuts;
  const auto found = cuts.find(cut.get());
  if (found == cuts.end() || found->second == std::numeric_limits<size_t>::max()) {
    return nullptr;
  }
  return &state_->source->setups[found->second];
}
bool LayoutEvidence::HasWorkspaceFormal() const {
  for (const auto& fact : state_->source->functions) {
    if (fact.workspace) return true;
  }
  return false;
}
const LayoutWork& LayoutEvidence::Work() const { return state_->work; }
const ir::ProgramPtr& LayoutEvidence::Owner() const { return state_->source->program; }
}  // namespace pypto::codegen::gm_pipe
