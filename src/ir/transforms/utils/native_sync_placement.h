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
#ifndef SRC_IR_TRANSFORMS_UTILS_NATIVE_SYNC_PLACEMENT_H_
#define SRC_IR_TRANSFORMS_UTILS_NATIVE_SYNC_PLACEMENT_H_

#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir::sync {
// Native scalar flattening needs actual inherited layout provenance. Reuse the
// same indexed window proof as Buffer code generation and exact import rather
// than treating a syntactically dense descriptor as a physical-layout proof.
// The scan caches expression nodes; logical token DAGs remain compact.
inline void RequireBufferSyncScalarLayout(const FunctionPtr& owner) {
  class ScalarUse : public IRVisitor {
   public:
    bool found = false;
    void VisitExpr(const ExprPtr& expression) override {
      if (!found && expression && seen_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
    }

   protected:
    void VisitVarLike_(const VarPtr&) override {}
    void VisitExpr_(const IterArgPtr&) override {}
    void VisitExpr_(const CallPtr& call) override {
      found = IsOp(call, "buffer.read") || IsOp(call, "buffer.write");
      if (!found) IRVisitor::VisitExpr_(call);
    }

   private:
    std::unordered_set<const Expr*> seen_;
  } scalar;
  scalar.VisitFunction(owner);
  if (!scalar.found) return;
  CHECK(owner->ir_stage_ == FunctionIRStage::Buffer)
      << "Local scalar synchronization requires Buffer IR layout evidence";
  const auto program = std::make_shared<Program>(std::vector<FunctionPtr>{owner}, owner->name_, owner->span_);
  std::vector<Diagnostic> diagnostics;
  CreateBufferIRPropertyVerifier()->Verify(program, diagnostics);
  for (const auto& diagnostic : diagnostics)
    CHECK(diagnostic.severity != DiagnosticSeverity::Error) << diagnostic.message;
}

// Indexed physical enclosures, independent of exact storage selectors, GM
// geometry and trip counts. Original SSA definitions are visited once. BufferIR
// declares runtime subviews contained in their source (02-types.md); their
// conservative parent enclosure is sufficient here. Slot selectors require
// separate range evidence. Borrowed/carry handles remain unavailable. These
// enclosures are not exact access footprints.
class NativeSyncPlacement : public IRVisitor {
 public:
  explicit NativeSyncPlacement(const FunctionPtr& owner) {
    RequireBufferSyncScalarLayout(owner);
    VisitStmt(owner->body_);
    for (const auto& target : dynamic_valid_targets_) {
      if (const auto window = windows_.find(target); window != windows_.end()) {
        const auto space = MemorySpaceOf(target);
        if (space) dynamic_valid_windows_.push_back({*space, window->second});
      }
    }
  }
  // Sound enclosing allocation bytes, never an exact access footprint.
  std::optional<backend::BufferSyncByteWindow> Window(const ExprPtr& value) const {
    const auto found = windows_.find(value.get());
    return found == windows_.end() ? std::optional<backend::BufferSyncByteWindow>{}
                                   : std::optional<backend::BufferSyncByteWindow>{found->second};
  }
  // Exact full physical allocation bytes for certified whole-buffer overwrites.
  // Excludes multi-buffer slots, subviews, partial valid shapes, and handles
  // touched by buffer.set_validshape.
  std::optional<backend::BufferSyncByteWindow> ExactFullWindow(const ExprPtr& value) const {
    if (!value || dynamic_valid_targets_.count(value.get())) return std::nullopt;
    const auto found = exact_full_windows_.find(value.get());
    if (found == exact_full_windows_.end()) return std::nullopt;
    const auto space = MemorySpaceOf(value.get());
    if (!space) return std::nullopt;
    for (const auto& entry : dynamic_valid_windows_) {
      if (entry.first == *space && entry.second.begin < found->second.end &&
          found->second.begin < entry.second.end) {
        return std::nullopt;
      }
    }
    return found->second;
  }
  void Require(const backend::BufferSyncOperation& operation) const {
    backend::RequireBufferSyncWorkspaceSeparation(operation,
                                                  [this](const ExprPtr& value) { return Window(value); });
  }

 protected:
  void VisitExpr(const ExprPtr&) override {}
  void VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    if (call && IsOp(call, "buffer.set_validshape") && !call->args_.empty()) {
      dynamic_valid_targets_.insert(call->args_[0].get());
    }
  }
  void VisitStmt_(const AssignStmtPtr& statement) override {
    const auto& value = statement->value_;
    if (const auto existing = windows_.find(value.get()); existing != windows_.end()) {
      windows_.emplace(statement->var_.get(), existing->second);
      if (const auto full = exact_full_windows_.find(value.get());
          full != exact_full_windows_.end() &&
          HasFullStaticValidShape(As<BufferType>(statement->var_->GetType()))) {
        exact_full_windows_.emplace(statement->var_.get(), full->second);
        spaces_.emplace(statement->var_.get(), As<BufferType>(statement->var_->GetType())->memory_space_);
      }
      return;
    }
    const auto call = As<Call>(value);
    if (!call) return;
    if (IsOp(call, "buffer.alloc") || IsOp(call, "buffer.alloc_multi")) {
      if (call->args_.size() != 2) return;
      const auto address = As<ConstInt>(call->args_[1]);
      const auto multi = As<MultiBufferType>(statement->var_->GetType());
      const auto type = multi ? multi->element_type_ : As<BufferType>(statement->var_->GetType());
      const auto bytes = backend::PhysicalBufferBytes(type);
      if (!address || address->value_ < 0 || !bytes) return;
      __int128 length = *bytes;
      if (multi && multi->slot_count_ > 1) {
        const auto context = PassContext::Current();
        const auto* selected = context ? context->GetBackend() : backend::GetBackend();
        if (!selected) return;
        const auto alignment = selected->GetMemAlignment(type->memory_space_);
        // This is the same native stride premise required by Buffer emission.
        if (!alignment || *bytes % alignment != 0) return;
        length *= multi->slot_count_;
      }
      const __int128 end = __int128(address->value_) + length;
      if (end > std::numeric_limits<int64_t>::max()) return;
      const backend::BufferSyncByteWindow window{address->value_, static_cast<int64_t>(end)};
      windows_.emplace(statement->var_.get(), window);
      if (!multi && HasFullStaticValidShape(type)) {
        exact_full_windows_.emplace(statement->var_.get(), window);
        spaces_.emplace(statement->var_.get(), type->memory_space_);
      }
      return;
    }
    if ((IsOp(call, "buffer.subview") || IsOp(call, "buffer.reshape") || IsOp(call, "buffer.slot")) &&
        !call->args_.empty()) {
      const auto parent = windows_.find(call->args_[0].get());
      if (parent == windows_.end()) return;
      if (IsOp(call, "buffer.slot")) {
        if (call->args_.size() != 2) return;
        const auto multi = As<MultiBufferType>(call->args_[0]->GetType());
        const auto index = As<ConstInt>(call->args_[1]);
        if (!multi || !index || index->value_ < 0 || index->value_ >= multi->slot_count_) return;
      }
      // Runtime subviews use the existing declared containment contract.
      // Slots have no corresponding automatic index proof: retain their
      // parent enclosure only after qualifying the selector above.
      windows_.emplace(statement->var_.get(), parent->second);
      if (IsOp(call, "buffer.reshape")) {
        const auto type = As<BufferType>(statement->var_->GetType());
        const auto bytes = backend::PhysicalBufferBytes(type);
        const auto full = exact_full_windows_.find(call->args_[0].get());
        if (full != exact_full_windows_.end() && HasFullStaticValidShape(type) && bytes &&
            __int128(full->second.end) - full->second.begin == *bytes) {
          exact_full_windows_.emplace(statement->var_.get(), full->second);
          spaces_.emplace(statement->var_.get(), type->memory_space_);
        }
      }
    }
  }

 private:
  static bool HasFullStaticValidShape(const BufferTypePtr& type) {
    return type && type->memory_space_ != MemorySpace::DDR && type->shape_.size() == 2 &&
           type->shape_[0] > 0 && type->shape_[1] > 0 &&
           (type->valid_shape_.empty() || type->valid_shape_ == type->shape_) &&
           type->compact_ == CompactMode::null;
  }
  std::optional<MemorySpace> MemorySpaceOf(const Expr* expr) const {
    if (const auto found = spaces_.find(expr); found != spaces_.end()) return found->second;
    if (const auto buffer = As<BufferType>(expr->GetType())) return buffer->memory_space_;
    return std::nullopt;
  }
  std::unordered_map<const Expr*, backend::BufferSyncByteWindow> windows_;
  std::unordered_map<const Expr*, backend::BufferSyncByteWindow> exact_full_windows_;
  std::unordered_map<const Expr*, MemorySpace> spaces_;
  std::unordered_set<const Expr*> dynamic_valid_targets_;
  std::vector<std::pair<MemorySpace, backend::BufferSyncByteWindow>> dynamic_valid_windows_;
};
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_NATIVE_SYNC_PLACEMENT_H_
