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

#include "src/ir/transforms/utils/sync_footprint_resolver.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/verifier/verifier.h"
#include "src/ir/transforms/utils/sync_transform_common.h"

namespace pypto::ir::sync {
namespace {

bool HasFullStaticValidShape(const BufferTypePtr& type) {
  return type && type->memory_space_ != MemorySpace::DDR && type->shape_.size() == 2 && type->shape_[0] > 0 &&
         type->shape_[1] > 0 && (type->valid_shape_.empty() || type->valid_shape_ == type->shape_) &&
         type->compact_ == CompactMode::null;
}

bool IsMatrixSpace(MemorySpace space) {
  return space == MemorySpace::Mat || space == MemorySpace::Left || space == MemorySpace::Right ||
         space == MemorySpace::Acc;
}

bool FitsAffineByteInterval(__int128 origin, __int128 extent) {
  constexpr auto kMax = std::numeric_limits<int64_t>::max();
  // The constant origin can be negative when the symbolic base adds a displacement.
  return origin >= std::numeric_limits<int64_t>::min() && origin <= kMax && extent > 0 && extent <= kMax &&
         origin + extent <= kMax;
}

IterationFiber MakeVecAffineFiber(const SymbolicIntRange& axis, int64_t stride_bytes, __int128 fixed_origin,
                                  __int128 extent) {
  const __int128 coeff = __int128(axis.affine_coeff) * stride_bytes;
  const __int128 origin = fixed_origin + __int128(axis.affine_base) * stride_bytes;
  if (coeff < std::numeric_limits<int64_t>::min() || coeff > std::numeric_limits<int64_t>::max() ||
      !FitsAffineByteInterval(origin, extent)) {
    return IterationFiber::Wildcard();
  }

  IterationFiber fiber;
  fiber.kind = IterationFiber::Kind::AffineSlice;
  fiber.frame = axis.affine_frame;
  fiber.coeff = static_cast<int64_t>(coeff);
  fiber.offset = static_cast<int64_t>(origin);
  fiber.extent = static_cast<int64_t>(extent);
  fiber.basis = {IterationFiberBasis::Kind::VecBytes, MemorySpace::Vec};
  if (axis.symbolic_base) {
    const auto& span = axis.symbolic_base->span_;
    fiber.symbolic_base = MakeMul(axis.symbolic_base, MakeIndexConst(stride_bytes, span), span);
  }
  return fiber;
}

bool IsTransposedMatPair(const SyncCanonicalRoot& root, const BufferTypePtr& type) {
  if (!root.canonical_layout || !type) return false;
  if (type->memory_space_ != MemorySpace::Mat || root.space != MemorySpace::Mat) return false;
  if (type->slayout_ == TileLayout::none_box || type->fractal_ != 16 * 16 * type->dtype_.GetByte())
    return false;
  if (root.canonical_layout->dtype_ != type->dtype_ || root.canonical_layout->fractal_ != type->fractal_)
    return false;
  if (root.rows != type->shape_[1] || root.columns != type->shape_[0]) return false;
  if (root.canonical_layout->blayout_ != type->slayout_ || root.canonical_layout->slayout_ != type->blayout_)
    return false;
  return (root.canonical_layout->blayout_ == TileLayout::col_major &&
          root.canonical_layout->slayout_ == TileLayout::row_major) ||
         (root.canonical_layout->blayout_ == TileLayout::row_major &&
          root.canonical_layout->slayout_ == TileLayout::col_major);
}

bool IsIdenticalMatLayout(const SyncCanonicalRoot& root, const BufferTypePtr& type) {
  if (!root.canonical_layout || !type) return false;
  return root.space == type->memory_space_ && root.rows == type->shape_[0] &&
         root.columns == type->shape_[1] && root.canonical_layout->dtype_ == type->dtype_ &&
         root.canonical_layout->fractal_ == type->fractal_ &&
         root.canonical_layout->blayout_ == type->blayout_ &&
         root.canonical_layout->slayout_ == type->slayout_;
}

bool IsFullOverwriteOpOperand(const CallPtr& call, BufferAccess mode, MemorySpace space,
                              const ExprPtr& operand) {
  if (space == MemorySpace::DDR) return false;
  const std::string& name = call->op_->name_;
  if (name == "buffer.load") {
    if (mode != BufferAccess::Write) return false;
    const auto buf = As<BufferType>(operand->GetType());
    return buf && buf->shape_.size() >= 2;
  }
  // A Vec insertion writes only the source prefix at the requested offset.
  // Its conservative footprint covers the destination without killing it.
  if (name == "buffer.insert" && space == MemorySpace::Vec) return false;
  if (name == "buffer.copy" || name == "buffer.extract" || name == "buffer.insert" ||
      name == "buffer.matmul") {
    return mode == BufferAccess::Write;
  }
  if (name == "buffer.ci_workspace") {
    return mode == BufferAccess::Write && !call->args_.empty() && operand == call->args_.back();
  }
  if (name == "buffer.matmul_acc") {
    return mode == BufferAccess::ReadWrite;
  }
  if (backend::FindBufferElementwiseRecipe(name)) {
    return mode == BufferAccess::Write && !call->args_.empty() && operand == call->args_.back();
  }
  if (backend::FindBufferConversionRecipe(name)) {
    return mode == BufferAccess::Write;
  }
  return false;
}

}  // namespace

void RequireBufferSyncScalarLayout(const FunctionPtr& owner) {
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

std::optional<backend::BufferSyncByteWindow> StridedByteFootprint::ByteHull() const {
  if (rows <= 0 || width <= 0 || base < 0 || (rows > 1 && pitch <= 0)) return std::nullopt;
  const __int128 end = __int128(base) + (rows > 1 ? __int128(rows - 1) * pitch : 0) + width;
  if (end <= base || end > std::numeric_limits<int64_t>::max()) return std::nullopt;
  return backend::BufferSyncByteWindow{base, static_cast<int64_t>(end)};
}

bool StridedByteFootprint::MayOverlap(const StridedByteFootprint& a, const StridedByteFootprint& b) {
  const auto ha = a.ByteHull();
  const auto hb = b.ByteHull();
  if (!ha || !hb) return true;
  if (ha->end <= hb->begin || hb->end <= ha->begin) return false;

  const int64_t p1 = a.rows > 1 ? a.pitch : 0;
  const int64_t p2 = b.rows > 1 ? b.pitch : 0;
  if (p1 == 0 && p2 == 0) return true;  // Both are contiguous intervals with overlapping hulls.

  const int64_t g = std::gcd(p1, p2);
  if (g > 0 && a.width < g && b.width < g) {
    // Project both periodic sets onto Z_g and test circular interval intersection.
    const int64_t r1 = PositiveMod64(a.base, g);
    const int64_t r2 = PositiveMod64(b.base, g);
    const int64_t d = PositiveMod64(r2 - r1, g);
    // I1 = [0, a.width), I2 = [d, d + b.width) mod g.
    return d < a.width || __int128(d) + b.width > g;
  }
  return true;
}

bool MatrixBoxFootprint::MayOverlap(const MatrixBoxFootprint& a, const MatrixBoxFootprint& b,
                                    const std::vector<SyncCanonicalRoot>& roots) {
  if (a.root_id == 0 || b.root_id == 0) return true;
  if (a.root_id == b.root_id) {
    if (a.slot_hi < b.slot_lo || b.slot_hi < a.slot_lo) return false;
    if (a.row_end <= b.row_begin || b.row_end <= a.row_begin) return false;
    if (a.col_end <= b.col_begin || b.col_end <= a.col_begin) return false;
    return true;
  }
  const auto& ra = roots.at(a.root_id - 1);
  const auto& rb = roots.at(b.root_id - 1);
  if (!ra.placed || !rb.placed) return true;
  const __int128 a_begin = __int128(ra.address) + __int128(a.slot_lo) * ra.slot_bytes;
  const __int128 a_end = __int128(ra.address) + __int128(a.slot_hi + 1) * ra.slot_bytes;
  const __int128 b_begin = __int128(rb.address) + __int128(b.slot_lo) * rb.slot_bytes;
  const __int128 b_end = __int128(rb.address) + __int128(b.slot_hi + 1) * rb.slot_bytes;
  return a_begin < b_end && b_begin < a_end;
}

bool IterationFiber::HasComparableCoordinates(const IterationFiber& other) const {
  if (!frame || frame != other.frame || kind != other.kind || coeff != other.coeff ||
      basis.kind == IterationFiberBasis::Kind::Unknown || !(basis == other.basis)) {
    return false;
  }
  if (kind == Kind::ModularSlot) {
    return modulus > 0 && modulus == other.modulus;
  }
  if (kind != Kind::AffineSlice || slice_axis != other.slice_axis || extent <= 0 || other.extent <= 0) {
    return false;
  }
  if (symbolic_base == other.symbolic_base) {
    return true;
  }
  if (!symbolic_base || !other.symbolic_base) {
    return false;
  }
  try {
    return structural_equal(symbolic_base, other.symbolic_base, false);
  } catch (const StructuralComparisonUnsupported&) {
    return false;
  }
}

bool IterationFiber::MayOverlap(const IterationFiber& other) const {
  if (!HasComparableCoordinates(other)) {
    return true;
  }
  if (kind == Kind::ModularSlot) {
    return offset == other.offset;
  }
  return offset < __int128(other.offset) + other.extent && other.offset < __int128(offset) + extent;
}

bool ResolvedOperandFootprint::MayOverlap(const ResolvedOperandFootprint& other,
                                          const std::vector<SyncCanonicalRoot>& roots) const {
  if (space != other.space) {
    return false;
  }
  if (space == MemorySpace::DDR) {
    return !gm_root || !other.gm_root || gm_root == other.gm_root;
  }
  if (!byte_hull || !other.byte_hull) {
    return true;
  }
  if (byte_hull->end <= other.byte_hull->begin || other.byte_hull->end <= byte_hull->begin) {
    return false;
  }
  if (vec_lattice && other.vec_lattice &&
      !StridedByteFootprint::MayOverlap(*vec_lattice, *other.vec_lattice)) {
    return false;
  }
  if (matrix_box && other.matrix_box &&
      !MatrixBoxFootprint::MayOverlap(*matrix_box, *other.matrix_box, roots)) {
    return false;
  }
  return slot_fiber.MayOverlap(other.slot_fiber) && fiber.MayOverlap(other.fiber);
}

bool ResolvedOperandFootprint::DefinitelyOverwritesWindow(
    const std::optional<backend::BufferSyncByteWindow>& window) const {
  return write && !read && definite_overwrite && window && window->begin < window->end &&
         definite_overwrite->begin <= window->begin && window->end <= definite_overwrite->end;
}

bool ResolvedOperandFootprint::DefinitelyOverwrites(const ResolvedOperandFootprint& other) const {
  return space == other.space && root_id != 0 && root_id == other.root_id &&
         DefinitelyOverwritesWindow(other.byte_hull);
}

std::array<ExprPtr, 2> FootprintRangeOperands(const CallPtr& call) {
  if (IsOp(call, "buffer.slot") && call->args_.size() == 2) {
    return {call->args_[1], nullptr};
  }
  if (IsOp(call, "buffer.subview") && call->args_.size() >= 2) {
    if (const auto offsets = As<MakeTuple>(call->args_[1]); offsets && offsets->elements_.size() == 2) {
      return {offsets->elements_[0], offsets->elements_[1]};
    }
  }
  if ((IsOp(call, "buffer.extract") || IsOp(call, "buffer.insert")) && call->args_.size() >= 4) {
    return {call->args_[1], call->args_[2]};
  }
  return {};
}

SyncFootprintResolver::SyncFootprintResolver(const FunctionPtr& owner, const SyncControlRangeProver* prover,
                                             bool run_scalar_layout_check, std::string target,
                                             bool resolve_peer_slots)
    : target_(std::move(target)), resolve_peer_slots_(resolve_peer_slots) {
  if (run_scalar_layout_check) RequireBufferSyncScalarLayout(owner);
  if (prover) {
    prover_ = prover;
  } else {
    owned_prover_ = std::make_unique<SyncControlRangeProver>(owner);
    prover_ = owned_prover_.get();
  }
  if (!owner) return;
  PrescanDynamicValidTargets(owner->body_);
  WalkFunction(owner);
  for (const auto* target : dynamic_valid_targets_) {
    if (const auto* v = ViewOf(target); v && v->byte_hull) {
      dynamic_valid_windows_.push_back({v->space, *v->byte_hull});
      if (v->root_id > 0 && v->root_id <= roots_.size()) {
        roots_[v->root_id - 1].touched_by_set_validshape = true;
      }
    } else if (v) {
      dynamic_valid_unbounded_spaces_.insert(v->space);
    } else if (target) {
      if (const auto bt = As<BufferType>(target->GetType())) {
        dynamic_valid_unbounded_spaces_.insert(bt->memory_space_);
      } else if (const auto mt = As<MultiBufferType>(target->GetType()); mt && mt->element_type_) {
        dynamic_valid_unbounded_spaces_.insert(mt->element_type_->memory_space_);
      }
    }
  }
  for (auto& [expr_ptr, view] : views_) {
    if (IsDynamicValidTouched(expr_ptr) || dynamic_valid_unbounded_spaces_.count(view.space) != 0 ||
        (view.byte_hull && OverlapsDynamicValid(view.space, *view.byte_hull))) {
      view.static_full_valid = false;
      if (view.vec_lattice) view.vec_lattice->exact = false;
      if (view.matrix_box) view.matrix_box->exact = false;
    }
  }
}

const SyncCanonicalRoot* SyncFootprintResolver::Root(size_t root_id) const {
  if (root_id == 0 || root_id > roots_.size()) return nullptr;
  return &roots_[root_id - 1];
}

const SyncResolvedView* SyncFootprintResolver::ViewOf(const ExprPtr& expr) const {
  return ViewOf(expr.get());
}

const SyncResolvedView* SyncFootprintResolver::ViewOf(const Expr* expr) const {
  if (!expr) return nullptr;
  const auto it = views_.find(expr);
  return it == views_.end() ? nullptr : &it->second;
}

std::optional<backend::BufferSyncByteWindow> SyncFootprintResolver::ByteHull(const ExprPtr& expr) const {
  const auto* v = ViewOf(expr);
  return v ? (v->peer_borrow ? v->peer_reserve : v->byte_hull) : std::nullopt;
}

bool SyncFootprintResolver::IsDynamicValidTouched(const Expr* expr) const {
  return dynamic_valid_targets_.count(expr) != 0;
}

bool SyncFootprintResolver::OverlapsDynamicValid(MemorySpace space,
                                                 const backend::BufferSyncByteWindow& window) const {
  if (dynamic_valid_unbounded_spaces_.count(space) != 0) return true;
  for (const auto& entry : dynamic_valid_windows_) {
    if (entry.first == space && entry.second.begin < window.end && window.begin < entry.second.end) {
      return true;
    }
  }
  return false;
}

std::optional<backend::BufferSyncByteWindow> SyncFootprintResolver::ExactFullWindow(
    const ExprPtr& expr) const {
  if (!expr || IsDynamicValidTouched(expr.get())) return std::nullopt;
  const auto* v = ViewOf(expr);
  if (!v || v->peer_borrow || !v->byte_hull || !v->static_full_valid) return std::nullopt;
  if (v->slot_selector && !v->slot_selector->is_const()) return std::nullopt;
  if (v->kind == SyncViewKind::VecSubview && v->vec_lattice) {
    // A strided subview (width < pitch && rows > 1) is NOT a contiguous full-window overwrite,
    // wait: single-slot or subview with rows == 1 or width == pitch IS contiguous, AND
    // full-overwrite check in RefineStorage compares dom.enclosure / descriptor uniformity!
    if (v->vec_lattice->rows > 1 && v->vec_lattice->width != v->vec_lattice->pitch) {
      return std::nullopt;
    }
  }
  if (OverlapsDynamicValid(v->space, *v->byte_hull)) return std::nullopt;
  return v->byte_hull;
}

void SyncFootprintResolver::RequireWorkspaceSeparation(const backend::BufferSyncOperation& operation) const {
  backend::RequireBufferSyncWorkspaceSeparation(operation,
                                                [this](const ExprPtr& value) { return ByteHull(value); });
}

void SyncFootprintResolver::RequireVecTransferPlacement(const backend::BufferSyncOperation& operation) const {
  const auto& call = operation.Operation();
  const bool insert = IsOp(call, "buffer.insert");
  if ((!insert && !IsOp(call, "buffer.copy")) || operation.Implementation().Pipe() != PipeType::V) return;
  constexpr int64_t kBlockBytes = 32;
  const char* name = insert ? "Vec insertion" : "Vec copy";
  const auto full_window = [&](const ExprPtr& operand) {
    const auto type = As<BufferType>(operand->GetType());
    const auto* view = ViewOf(operand);
    // Row-prefix TMOV uses a contiguous prefix of a placed allocation.
    // The allocation enclosure is suitable for alignment and separation,
    // but does not certify a whole-window overwrite.
    const bool row_prefix =
        backend::IsMinimumPaddedVecRowGeometry(type) || backend::IsColumnVectorBufferGeometry(type);
    auto window = row_prefix && view && view->kind == SyncViewKind::PlacedLocal && !view->peer_borrow
                      ? view->byte_hull
                      : ExactFullWindow(operand);
    if (insert) {
      // TINSERT consumes a partial-valid source but requires the same fixed
      // dense physical pitch. Native placement does not certify a full write.
      window.reset();
      const int64_t width = type->shape_[1] * type->dtype_.GetByte();
      if (view && !view->peer_borrow && view->byte_hull && view->vec_lattice &&
          view->vec_lattice->base == view->byte_hull->begin && view->vec_lattice->pitch == width &&
          view->vec_lattice->width == width && view->vec_lattice->rows == type->shape_[0]) {
        window = view->byte_hull;
      }
    }
    const auto bytes = backend::PhysicalBufferBytes(type);
    CHECK_SPAN(window && bytes && window->begin % kBlockBytes == 0 &&
                   __int128(window->end) - window->begin == *bytes,
               call->span_)
        << name << " needs an aligned, static contiguous placed byte window for each operand";
    return *window;
  };
  const auto source = full_window(call->args_[0]);
  const auto destination = full_window(call->args_[insert ? 3 : 1]);
  CHECK_SPAN(source.end <= destination.begin || destination.end <= source.begin, call->span_)
      << name << " requires physically disjoint source and destination windows";
}

void SyncFootprintResolver::PrescanDynamicValidTargets(const StmtPtr& stmt) {
  if (!stmt) return;
  if (const auto seq = As<SeqStmts>(stmt)) {
    for (const auto& c : seq->stmts_) PrescanDynamicValidTargets(c);
  } else if (const auto branch = As<IfStmt>(stmt)) {
    PrescanDynamicValidTargets(branch->then_body_);
    if (branch->else_body_) PrescanDynamicValidTargets(*branch->else_body_);
  } else if (const auto loop = As<ForStmt>(stmt)) {
    PrescanDynamicValidTargets(loop->body_);
  } else if (const auto loop = As<WhileStmt>(stmt)) {
    PrescanDynamicValidTargets(loop->body_);
  } else if (const auto eval = As<EvalStmt>(stmt)) {
    if (const auto call = As<Call>(eval->expr_);
        call && IsOp(call, "buffer.set_validshape") && !call->args_.empty()) {
      dynamic_valid_targets_.insert(call->args_[0].get());
    }
  }
}

void SyncFootprintResolver::WalkFunction(const FunctionPtr& owner) {
  for (const auto& param : owner->params_) {
    if (const auto tensor = As<TensorType>(param->GetType())) {
      const auto memory = tensor->memref_.value_or(nullptr);
      ExprPtr gm_root = memory ? ExprPtr(memory->base_) : ExprPtr(param);
      gm_roots_.emplace(param.get(), gm_root);
      SyncResolvedView view;
      view.kind = SyncViewKind::ExternalTensor;
      view.space = MemorySpace::DDR;
      views_.emplace(param.get(), std::move(view));
    }
  }
  WalkStmt(owner->body_);
}

void SyncFootprintResolver::WalkStmt(const StmtPtr& stmt) {
  if (!stmt) return;
  if (const auto seq = As<SeqStmts>(stmt)) {
    for (const auto& c : seq->stmts_) WalkStmt(c);
  } else if (const auto branch = As<IfStmt>(stmt)) {
    WalkStmt(branch->then_body_);
    if (branch->else_body_) WalkStmt(*branch->else_body_);
  } else if (const auto loop = As<ForStmt>(stmt)) {
    WalkStmt(loop->body_);
  } else if (const auto loop = As<WhileStmt>(stmt)) {
    WalkStmt(loop->body_);
  } else if (const auto assign = As<AssignStmt>(stmt)) {
    ResolveAssign(assign);
  }
}

void SyncFootprintResolver::ResolveAssign(const AssignStmtPtr& assign) {
  const auto& var = assign->var_;
  const auto& value = assign->value_;

  if (const auto gm = gm_roots_.find(value.get()); gm != gm_roots_.end()) {
    gm_roots_.emplace(var.get(), gm->second);
  }
  if (const auto existing = ViewOf(value)) {
    SyncResolvedView copy = *existing;
    if (const auto buf_type = As<BufferType>(var->GetType())) {
      copy.type = buf_type;
      copy.static_full_valid = copy.static_full_valid && HasFullStaticValidShape(buf_type);
      if (copy.vec_lattice) copy.vec_lattice->exact = copy.static_full_valid;
      if (copy.matrix_box) copy.matrix_box->exact = copy.static_full_valid;
    }
    views_.emplace(var.get(), std::move(copy));
    return;
  }

  const auto call = As<Call>(value);
  if (!call) return;

  // Handle GM tensor alias non-payload descriptors (e.g. tensor.view / tensor.alias).
  if (As<TensorType>(var->GetType())) {
    if (!target_.empty()) {
      if (const auto descriptor = backend::QualifyBufferSyncNonPayload(call, target_)) {
        const auto& ownership = descriptor->Result();
        if (ownership.behavior == BufferResultBehavior::Alias && ownership.alias_arg &&
            *ownership.alias_arg < call->args_.size()) {
          if (const auto found = gm_roots_.find(call->args_[*ownership.alias_arg].get());
              found != gm_roots_.end()) {
            gm_roots_.emplace(var.get(), found->second);
          }
        }
      }
      return;
    }
    if (!call->args_.empty()) {
      if (const auto found = gm_roots_.find(call->args_[0].get()); found != gm_roots_.end()) {
        if (IsOp(call, "tensor.view") || IsOp(call, "tensor.reshape")) {
          gm_roots_.emplace(var.get(), found->second);
        }
      }
    }
    return;
  }

  if (resolve_peer_slots_ && IsOp(call, "buffer.peer_reserve")) {
    const auto base = call->GetKwarg<int>("base", -1);
    const auto size = call->GetKwarg<int>("size", 0);
    if (base >= 0 && size > 0) {
      peer_reserves_.emplace(var.get(), backend::BufferSyncByteWindow{base, int64_t{base} + size});
    }
    return;
  }

  const bool peer_pop = resolve_peer_slots_ && IsOp(call, "buffer.peer_pop");
  if (peer_pop || IsOp(call, "buffer.alloc") || IsOp(call, "buffer.alloc_multi")) {
    if (call->args_.size() != 2) return;
    std::optional<int64_t> address;
    std::optional<backend::BufferSyncByteWindow> peer_reserve;
    if (peer_pop) {
      if (const auto found = peer_reserves_.find(call->args_[0].get()); found != peer_reserves_.end()) {
        address = found->second.begin;
        peer_reserve = found->second;
      }
    } else if (const auto placed = As<ConstInt>(call->args_[1])) {
      address = placed->value_;
    }
    const auto multi = As<MultiBufferType>(var->GetType());
    const auto elem_type = multi ? multi->element_type_ : As<BufferType>(var->GetType());
    const auto slot_bytes_opt = backend::PhysicalBufferBytes(elem_type);
    if (!address || *address < 0 || !slot_bytes_opt || !elem_type ||
        *slot_bytes_opt > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return;
    }

    const int64_t slot_count = multi ? multi->slot_count_ : 1;
    const int64_t slot_bytes = *slot_bytes_opt;
    if (multi && slot_count > 1) {
      const auto context = PassContext::Current();
      const auto* selected = context ? context->GetBackend() : backend::GetBackend();
      if (!selected) return;
      const auto alignment = selected->GetMemAlignment(elem_type->memory_space_);
      if (!alignment || slot_bytes % alignment != 0) return;
    }
    const __int128 total_bytes = __int128(slot_bytes) * slot_count;
    const __int128 end = __int128(*address) + total_bytes;
    if (end > std::numeric_limits<int64_t>::max()) return;

    const int64_t rows = elem_type->shape_.size() >= 1 ? elem_type->shape_[0] : 0;
    const int64_t cols = elem_type->shape_.size() >= 2 ? elem_type->shape_[1] : 0;

    // Find matching co-addressed canonical root (including Mat NZ <-> ZN transpose views).
    size_t root_id = 0;
    bool axis_swapped = false;
    if (!peer_pop && IsMatrixSpace(elem_type->memory_space_)) {
      for (const auto& root : roots_) {
        if (!root.placed || root.peer_borrow || root.address != *address || root.bytes != total_bytes ||
            root.slot_count != slot_count)
          continue;
        if (IsIdenticalMatLayout(root, elem_type)) {
          root_id = root.id;
          axis_swapped = false;
          break;
        }
        if (IsTransposedMatPair(root, elem_type)) {
          root_id = root.id;
          axis_swapped = true;
          break;
        }
      }
    }
    if (root_id == 0) {
      root_id = roots_.size() + 1;
      SyncCanonicalRoot root;
      root.id = root_id;
      root.space = elem_type->memory_space_;
      root.placed = true;
      root.address = *address;
      root.bytes = static_cast<int64_t>(total_bytes);
      root.slot_count = slot_count;
      root.slot_bytes = slot_bytes;
      root.rows = rows;
      root.columns = cols;
      root.canonical_layout = IsMatrixSpace(elem_type->memory_space_) ? elem_type : nullptr;
      root.peer_borrow = peer_pop ? var.get() : nullptr;
      roots_.push_back(std::move(root));
    }

    SyncResolvedView view;
    view.kind = SyncViewKind::PlacedLocal;
    view.space = elem_type->memory_space_;
    view.root_id = root_id;
    view.type = elem_type;
    view.byte_hull = backend::BufferSyncByteWindow{*address, static_cast<int64_t>(end)};
    view.base_byte_offset = *address;
    view.peer_borrow = peer_pop ? var.get() : nullptr;
    view.peer_reserve = peer_reserve;
    view.axis_swapped = axis_swapped;
    view.valid_matrix_coords = IsMatrixSpace(elem_type->memory_space_);
    view.static_full_valid =
        !multi && HasFullStaticValidShape(elem_type) && !IsDynamicValidTouched(var.get());

    if (elem_type->memory_space_ == MemorySpace::Vec && elem_type->shape_.size() == 2 && rows > 0 &&
        cols > 0 && elem_type->dtype_.GetBit() % 8 == 0 &&
        (backend::DenseBufferBytes(elem_type) || backend::IsColumnVectorBufferGeometry(elem_type) ||
         backend::IsFullPaddedVec(elem_type) || backend::IsMinimumPaddedVecRowGeometry(elem_type))) {
      const int64_t elem_bytes = elem_type->dtype_.GetByte();
      view.row_pitch_bytes = cols * elem_bytes;
      if (!multi) {
        view.vec_lattice = StridedByteFootprint{*address, view.row_pitch_bytes, rows, cols * elem_bytes,
                                                view.static_full_valid};
      }
    } else if (!multi && IsMatrixSpace(elem_type->memory_space_)) {
      const auto& root = roots_[root_id - 1];
      view.matrix_box =
          MatrixBoxFootprint{root_id, 0, 0, 0, root.rows, 0, root.columns, view.static_full_valid};
    }

    views_.emplace(var.get(), std::move(view));
    return;
  }

  if (IsOp(call, "buffer.slot") && call->args_.size() == 2) {
    const auto* parent = ViewOf(call->args_[0]);
    const auto multi = As<MultiBufferType>(call->args_[0]->GetType());
    const auto dst_type = As<BufferType>(var->GetType());
    if (!parent || !parent->byte_hull || !multi || !dst_type || parent->root_id == 0) return;
    const auto& root = roots_[parent->root_id - 1];
    const auto idx_range = prover_->LookupAtCut(assign.get(), FootprintRangeOperands(call)[0]);
    if (!idx_range || idx_range->lo < 0 || idx_range->hi >= multi->slot_count_) {
      // Unproved slot index: leave unmapped (`Unknown`, byte_hull = std::nullopt) to preserve
      // workspace-separation refusals and Wildcard Domain 0 fallback.
      return;
    }

    const __int128 begin = __int128(root.address) + __int128(idx_range->lo) * root.slot_bytes;
    const __int128 end = __int128(root.address) + __int128(idx_range->hi + 1) * root.slot_bytes;
    if (end <= begin || end > parent->byte_hull->end) return;

    SyncResolvedView view;
    view.kind = SyncViewKind::Slot;
    view.space = dst_type->memory_space_;
    view.root_id = parent->root_id;
    view.type = dst_type;
    view.byte_hull = backend::BufferSyncByteWindow{static_cast<int64_t>(begin), static_cast<int64_t>(end)};
    view.base_byte_offset = static_cast<int64_t>(begin);
    view.row_pitch_bytes = parent->row_pitch_bytes;
    view.slot_selector = *idx_range;
    view.axis_swapped = parent->axis_swapped;
    view.valid_matrix_coords = parent->valid_matrix_coords && IsMatrixSpace(dst_type->memory_space_);
    const bool is_const_slot = idx_range->is_const();
    view.static_full_valid =
        is_const_slot && HasFullStaticValidShape(dst_type) && !IsDynamicValidTouched(var.get());

    if (dst_type->memory_space_ == MemorySpace::Vec && dst_type->shape_.size() == 2 &&
        dst_type->shape_[0] > 0 && dst_type->shape_[1] > 0 && dst_type->dtype_.GetBit() % 8 == 0 &&
        view.row_pitch_bytes > 0) {
      const int64_t elem_bytes = dst_type->dtype_.GetByte();
      if (is_const_slot) {
        view.vec_lattice =
            StridedByteFootprint{static_cast<int64_t>(begin), view.row_pitch_bytes, dst_type->shape_[0],
                                 dst_type->shape_[1] * elem_bytes, view.static_full_valid};
      } else {
        view.vec_lattice = StridedByteFootprint{static_cast<int64_t>(begin), view.row_pitch_bytes,
                                                (idx_range->hi - idx_range->lo + 1) * dst_type->shape_[0],
                                                dst_type->shape_[1] * elem_bytes, false};
      }
    } else if (IsMatrixSpace(dst_type->memory_space_)) {
      view.matrix_box =
          MatrixBoxFootprint{parent->root_id, idx_range->lo,         idx_range->hi, 0, root.rows, 0,
                             root.columns,    view.static_full_valid};
    }

    if (!is_const_slot && idx_range->affine_frame != nullptr && idx_range->modulus > 0) {
      view.slot_fiber = {
          IterationFiber::Kind::ModularSlot,
          idx_range->affine_frame,
          idx_range->modulus,
          idx_range->affine_coeff,
          idx_range->affine_base,
          1,
          0,
          nullptr,
          {IterationFiberBasis::Kind::SlotPartition, root.space, 0, root.address, root.slot_bytes}};
    }

    views_.emplace(var.get(), std::move(view));
    return;
  }

  if (IsOp(call, "buffer.subview") && call->args_.size() >= 2) {
    const auto* parent = ViewOf(call->args_[0]);
    const auto parent_type = As<BufferType>(call->args_[0]->GetType());
    const auto dst_type = As<BufferType>(var->GetType());
    const auto offsets = As<MakeTuple>(call->args_[1]);
    if (!parent || !parent->byte_hull) return;

    SyncResolvedView view;
    view.kind = SyncViewKind::VecSubview;
    view.space = dst_type ? dst_type->memory_space_ : parent->space;
    view.root_id = parent->root_id;
    view.type = dst_type;
    view.byte_hull = parent->byte_hull;
    view.base_byte_offset = parent->base_byte_offset;
    view.row_pitch_bytes = parent->row_pitch_bytes;
    view.slot_selector = parent->slot_selector;
    view.slot_fiber = parent->slot_fiber;
    view.peer_borrow = parent->peer_borrow;
    view.peer_reserve = parent->peer_reserve;
    view.axis_swapped = parent->axis_swapped;
    view.static_full_valid = false;

    if (parent_type && dst_type && offsets && offsets->elements_.size() == 2 &&
        parent_type->memory_space_ == MemorySpace::Vec && dst_type->memory_space_ == MemorySpace::Vec &&
        parent_type->shape_.size() == 2 && dst_type->shape_.size() == 2 && dst_type->shape_[0] > 0 &&
        dst_type->shape_[1] > 0 && dst_type->dtype_.GetBit() % 8 == 0 && parent->row_pitch_bytes > 0) {
      const int64_t elem_bytes = dst_type->dtype_.GetByte();
      const int64_t pitch = parent->row_pitch_bytes;
      const int64_t max_r = std::max<int64_t>(0, parent_type->shape_[0] - dst_type->shape_[0]);
      const int64_t max_c = std::max<int64_t>(0, parent_type->shape_[1] - dst_type->shape_[1]);

      const auto [row, col] = FootprintRangeOperands(call);
      const auto r_opt = prover_->LookupAtCut(assign.get(), row);
      const auto c_opt = prover_->LookupAtCut(assign.get(), col);
      int64_t r_lo = 0, r_hi = max_r;
      int64_t c_lo = 0, c_hi = max_c;
      if (r_opt) {
        r_lo = std::max<int64_t>(0, r_opt->lo);
        r_hi = std::min<int64_t>(max_r, r_opt->hi);
        if (r_lo > r_hi) {
          r_lo = 0;
          r_hi = max_r;
        }
      }
      if (c_opt) {
        c_lo = std::max<int64_t>(0, c_opt->lo);
        c_hi = std::min<int64_t>(max_c, c_opt->hi);
        if (c_lo > c_hi) {
          c_lo = 0;
          c_hi = max_c;
        }
      }
      // A moving parent's base_byte_offset is only a lower bound. For a chained
      // subview, require a lattice of exactly the parent's static shape: any
      // movement in the lattice adds rows or widens its byte intervals.
      const bool parent_static_origin =
          (parent->kind == SyncViewKind::PlacedLocal) ||
          (parent->kind == SyncViewKind::Slot && parent->slot_selector &&
           parent->slot_selector->is_const()) ||
          (parent->kind == SyncViewKind::VecSubview && parent->fiber.kind == IterationFiber::Kind::Wildcard &&
           parent->byte_hull->begin == parent->base_byte_offset && parent->vec_lattice.has_value() &&
           parent->vec_lattice->base == parent->base_byte_offset &&
           parent->vec_lattice->rows == parent_type->shape_[0] &&
           parent->vec_lattice->width == __int128(parent_type->shape_[1]) * elem_bytes);

      if (parent_static_origin) {
        const __int128 begin =
            __int128(parent->base_byte_offset) + __int128(r_lo) * pitch + __int128(c_lo) * elem_bytes;
        const __int128 end = __int128(parent->base_byte_offset) +
                             (__int128(r_hi) + dst_type->shape_[0] - 1) * pitch +
                             (__int128(c_hi) + dst_type->shape_[1]) * elem_bytes;
        if (begin >= parent->byte_hull->begin && end > begin && end <= parent->byte_hull->end) {
          view.byte_hull =
              backend::BufferSyncByteWindow{static_cast<int64_t>(begin), static_cast<int64_t>(end)};
          view.base_byte_offset = static_cast<int64_t>(begin);
          const bool const_offsets = (r_lo == r_hi && c_lo == c_hi);
          const auto valid = call->args_.size() == 3 ? As<MakeTuple>(call->args_[2]) : nullptr;
          const auto valid_matches_shape = [&](size_t axis) {
            if (!valid || valid->elements_.size() != 2) return false;
            const auto extent = As<ConstInt>(valid->elements_[axis]);
            return extent && extent->value_ == dst_type->shape_[axis];
          };
          const bool full_valid_operand =
              call->args_.size() == 2 || (valid_matches_shape(0) && valid_matches_shape(1));
          const bool proved_offsets = r_opt && c_opt && r_opt->is_const() && c_opt->is_const() &&
                                      r_opt->lo >= 0 && r_opt->hi <= max_r && c_opt->lo >= 0 &&
                                      c_opt->hi <= max_c;
          view.static_full_valid = parent->static_full_valid && proved_offsets && full_valid_operand &&
                                   HasFullStaticValidShape(dst_type) && !IsDynamicValidTouched(var.get());
          if (const_offsets) {
            view.vec_lattice = StridedByteFootprint{static_cast<int64_t>(begin), pitch, dst_type->shape_[0],
                                                    dst_type->shape_[1] * elem_bytes, view.static_full_valid};
          } else if (c_lo == c_hi) {
            if (dst_type->shape_[0] == 1 && r_opt && r_opt->affine_frame && r_opt->modulus == 0 &&
                r_opt->affine_coeff > 1 && !r_opt->symbolic_base && r_opt->lo >= 0 && r_opt->hi <= max_r &&
                r_lo == r_opt->lo && (r_hi - r_lo) % r_opt->affine_coeff == 0) {
              const __int128 step_bytes = __int128(r_opt->affine_coeff) * pitch;
              const int64_t count = (r_hi - r_lo) / r_opt->affine_coeff + 1;
              if (step_bytes <= std::numeric_limits<int64_t>::max()) {
                view.vec_lattice =
                    StridedByteFootprint{static_cast<int64_t>(begin), static_cast<int64_t>(step_bytes), count,
                                         dst_type->shape_[1] * elem_bytes, false};
              }
            }
            if (!view.vec_lattice) {
              view.vec_lattice =
                  StridedByteFootprint{static_cast<int64_t>(begin), pitch, r_hi - r_lo + dst_type->shape_[0],
                                       dst_type->shape_[1] * elem_bytes, false};
            }
          } else if (r_lo == r_hi && dst_type->shape_[0] == 1) {
            if (c_opt && c_opt->affine_frame && c_opt->modulus == 0 && c_opt->affine_coeff > 1 &&
                !c_opt->symbolic_base && c_opt->lo >= 0 && c_opt->hi <= max_c && c_lo == c_opt->lo &&
                (c_hi - c_lo) % c_opt->affine_coeff == 0) {
              const __int128 step_bytes = __int128(c_opt->affine_coeff) * elem_bytes;
              const int64_t count = (c_hi - c_lo) / c_opt->affine_coeff + 1;
              if (step_bytes <= std::numeric_limits<int64_t>::max()) {
                view.vec_lattice =
                    StridedByteFootprint{static_cast<int64_t>(begin), static_cast<int64_t>(step_bytes), count,
                                         dst_type->shape_[1] * elem_bytes, false};
              }
            }
            if (!view.vec_lattice) {
              view.vec_lattice =
                  StridedByteFootprint{static_cast<int64_t>(begin), pitch, 1,
                                       (c_hi - c_lo + dst_type->shape_[1]) * elem_bytes, false};
            }
          } else {
            // Both row and column vary: use bounding strided rows with combined column width.
            view.vec_lattice =
                StridedByteFootprint{static_cast<int64_t>(begin), pitch, r_hi - r_lo + dst_type->shape_[0],
                                     (c_hi - c_lo + dst_type->shape_[1]) * elem_bytes, false};
          }

          // Fiber consumers can compare distinct allocations whose byte hulls
          // overlap. Use the parent's physical origin and the fixed orthogonal
          // offset to put both fibers in the same byte coordinate system.
          const __int128 slice_extent =
              __int128(dst_type->shape_[0] - 1) * pitch + __int128(dst_type->shape_[1]) * elem_bytes;
          const __int128 parent_origin = parent->base_byte_offset;
          if (r_opt && r_opt->affine_frame && r_opt->modulus == 0 && r_opt->affine_coeff != 0 &&
              ((c_opt && !c_opt->affine_frame && c_opt->is_const()) || max_c == 0)) {
            view.fiber =
                MakeVecAffineFiber(*r_opt, pitch, parent_origin + __int128(c_lo) * elem_bytes, slice_extent);
          } else if (c_opt && c_opt->affine_frame && c_opt->modulus == 0 && c_opt->affine_coeff != 0 &&
                     ((r_opt && !r_opt->affine_frame && r_opt->is_const()) || max_r == 0) &&
                     dst_type->shape_[0] == 1) {
            view.fiber =
                MakeVecAffineFiber(*c_opt, elem_bytes, parent_origin + __int128(r_lo) * pitch, slice_extent);
          }
        }
      } else if (parent->vec_lattice && c_lo == c_hi && r_lo == r_hi && r_lo == 0 &&
                 dst_type->shape_[0] == parent_type->shape_[0]) {
        // Chained column subview inside a moving parent: preserve parent column movement!
        const __int128 parent_static_width = __int128(parent_type->shape_[1]) * elem_bytes;
        const __int128 col_spread = __int128(parent->vec_lattice->width) - parent_static_width;
        if (col_spread >= 0) {
          const __int128 new_base = __int128(parent->vec_lattice->base) + __int128(c_lo) * elem_bytes;
          const __int128 new_width = __int128(dst_type->shape_[1]) * elem_bytes + col_spread;
          const __int128 row_span = parent->vec_lattice->rows > 1
                                        ? __int128(parent->vec_lattice->rows - 1) * parent->vec_lattice->pitch
                                        : 0;
          if (new_base >= parent->byte_hull->begin &&
              new_base + row_span + new_width <= parent->byte_hull->end &&
              new_width <= std::numeric_limits<int64_t>::max() &&
              (parent->vec_lattice->rows == 1 || new_width <= parent->vec_lattice->pitch)) {
            view.vec_lattice =
                StridedByteFootprint{static_cast<int64_t>(new_base), parent->vec_lattice->pitch,
                                     parent->vec_lattice->rows, static_cast<int64_t>(new_width), false};
          }
          if (parent->fiber.kind == IterationFiber::Kind::AffineSlice && dst_type->shape_[0] == 1) {
            const __int128 slice_extent = __int128(dst_type->shape_[1]) * elem_bytes;
            // The parent fiber already carries the physical origin and movement.
            const __int128 new_off = __int128(parent->fiber.offset) + __int128(c_lo) * elem_bytes;
            if (FitsAffineByteInterval(new_off, slice_extent)) {
              view.fiber = parent->fiber;
              view.fiber.offset = static_cast<int64_t>(new_off);
              view.fiber.extent = static_cast<int64_t>(slice_extent);
            }
          }
        }
      }
    }

    views_.emplace(var.get(), std::move(view));
    return;
  }

  if (IsOp(call, "buffer.reshape") && call->args_.size() == 1) {
    const auto* parent = ViewOf(call->args_[0]);
    const auto src_type = As<BufferType>(call->args_[0]->GetType());
    const auto dst_type = As<BufferType>(var->GetType());
    if (!parent || !parent->byte_hull || !dst_type) return;

    SyncResolvedView view = *parent;
    view.kind = SyncViewKind::Reshape;
    view.type = dst_type;
    view.static_full_valid = false;

    if (dst_type->memory_space_ == MemorySpace::Vec) {
      const bool src_contiguous =
          src_type && src_type->shape_.size() == 2 &&
          (src_type->shape_[0] == 1 ||
           (src_type->dtype_.GetBit() % 8 == 0 &&
            parent->row_pitch_bytes == src_type->shape_[1] * src_type->dtype_.GetByte()));
      const auto bytes = backend::PhysicalBufferBytes(dst_type);
      const __int128 hull_bytes = __int128(parent->byte_hull->end) - parent->byte_hull->begin;
      if (src_contiguous && bytes && *bytes == hull_bytes && dst_type->shape_.size() == 2 &&
          dst_type->shape_[0] > 0 && dst_type->shape_[1] > 0 && dst_type->dtype_.GetBit() % 8 == 0) {
        view.row_pitch_bytes = dst_type->shape_[1] * dst_type->dtype_.GetByte();
        view.static_full_valid = parent->static_full_valid && HasFullStaticValidShape(dst_type) &&
                                 !IsDynamicValidTouched(var.get());
        view.vec_lattice =
            StridedByteFootprint{parent->byte_hull->begin, view.row_pitch_bytes, dst_type->shape_[0],
                                 view.row_pitch_bytes, view.static_full_valid};
      } else {
        view.row_pitch_bytes = 0;
        view.vec_lattice.reset();
      }
    } else if (IsMatrixSpace(dst_type->memory_space_) && parent->root_id > 0) {
      const auto& root = roots_[parent->root_id - 1];
      if (parent->valid_matrix_coords && IsIdenticalMatLayout(root, dst_type)) {
        view.axis_swapped = false;
        view.valid_matrix_coords = true;
      } else if (parent->valid_matrix_coords && IsTransposedMatPair(root, dst_type)) {
        view.axis_swapped = true;
        view.valid_matrix_coords = true;
      } else {
        view.axis_swapped = false;
        view.valid_matrix_coords = false;
      }
      if (view.valid_matrix_coords) {
        view.static_full_valid = parent->static_full_valid && HasFullStaticValidShape(dst_type) &&
                                 !IsDynamicValidTouched(var.get());
        if (parent->matrix_box) {
          view.matrix_box = parent->matrix_box;
          view.matrix_box->exact = view.static_full_valid;
        }
      } else {
        view.static_full_valid = false;
        view.matrix_box.reset();
      }
    }

    views_.emplace(var.get(), std::move(view));
    return;
  }
}

ResolvedOperandFootprint SyncFootprintResolver::ResolveOperand(const Stmt* cut, const CallPtr& call,
                                                               size_t arg_idx, BufferAccess mode,
                                                               MemorySpace space, bool disjoint_gm,
                                                               bool peer_slot_relative) const {
  ResolvedOperandFootprint out;
  out.space = space;
  out.read = (mode == BufferAccess::Read || mode == BufferAccess::ReadWrite);
  out.write = (mode == BufferAccess::Write || mode == BufferAccess::ReadWrite);
  if (!call || arg_idx >= call->args_.size()) return out;

  const auto& operand = call->args_[arg_idx];
  if (space == MemorySpace::DDR) {
    if (disjoint_gm) {
      if (const auto found = gm_roots_.find(operand.get()); found != gm_roots_.end()) {
        out.gm_root = found->second;
      }
    }
    return out;
  }

  const auto* view = ViewOf(operand);
  if (!view) return out;
  if (view->peer_borrow && !peer_slot_relative) {
    out.byte_hull = view->peer_reserve;
    return out;
  }
  out.root_id = view->root_id;
  out.byte_hull = view->byte_hull;
  out.slot_fiber = view->slot_fiber;
  out.fiber = view->fiber;

  if (space == MemorySpace::Vec) {
    out.vec_lattice = view->vec_lattice;
    if (!out.vec_lattice && out.byte_hull) {
      out.vec_lattice = StridedByteFootprint{out.byte_hull->begin, out.byte_hull->end - out.byte_hull->begin,
                                             1, out.byte_hull->end - out.byte_hull->begin, false};
    }
    if (IsFullOverwriteOpOperand(call, mode, space, operand) && ExactFullWindow(operand).has_value()) {
      out.definite_overwrite = out.byte_hull;
      if (out.vec_lattice) out.vec_lattice->exact = true;
    } else if (out.vec_lattice && !IsFullOverwriteOpOperand(call, mode, space, operand)) {
      // Partial scalar writes (e.g. buffer.write) or indexed reads are not exact full overwrites.
      if (IsOp(call, "buffer.write") || IsOp(call, "buffer.read")) {
        out.vec_lattice->exact = false;
      }
    }
    return out;
  }

  // Matrix space: Mat, Left, Right, Acc.
  if (view->root_id == 0) return out;
  const auto& root = roots_[view->root_id - 1];
  out.matrix_box = view->matrix_box;

  // Subtile coordinate resolution for buffer.extract (Mat -> Left/Right) and buffer.insert (Acc -> Mat).
  const bool is_mat_extract =
      IsOp(call, "buffer.extract") && arg_idx == 0 && space == MemorySpace::Mat && call->args_.size() >= 4;
  const bool is_mat_insert =
      IsOp(call, "buffer.insert") && arg_idx == 3 && space == MemorySpace::Mat && call->args_.size() >= 4;
  if (view->valid_matrix_coords && (is_mat_extract || is_mat_insert)) {
    const auto sub_type = As<BufferType>(call->args_[is_mat_extract ? 3 : 0]->GetType());
    if (sub_type && sub_type->shape_.size() == 2 && sub_type->shape_[0] > 0 && sub_type->shape_[1] > 0) {
      int64_t sub_rows = sub_type->shape_[0];
      int64_t sub_cols = sub_type->shape_[1];
      const auto [row, col] = FootprintRangeOperands(call);
      auto r_opt = prover_->LookupAtCut(cut, row);
      auto c_opt = prover_->LookupAtCut(cut, col);
      if (view->axis_swapped) {
        std::swap(sub_rows, sub_cols);
        std::swap(r_opt, c_opt);
      }
      int64_t r_begin = 0, r_end = root.rows;
      int64_t c_begin = 0, c_end = root.columns;
      if (r_opt) {
        const int64_t lo = std::max<int64_t>(0, r_opt->lo);
        const int64_t hi = std::min<int64_t>(root.rows - sub_rows, r_opt->hi);
        if (lo <= hi) {
          r_begin = lo;
          r_end = hi + sub_rows;
        }
      }
      if (c_opt) {
        const int64_t lo = std::max<int64_t>(0, c_opt->lo);
        const int64_t hi = std::min<int64_t>(root.columns - sub_cols, c_opt->hi);
        if (lo <= hi) {
          c_begin = lo;
          c_end = hi + sub_cols;
        }
      }
      const bool const_coords = r_opt && c_opt && r_opt->is_const() && c_opt->is_const();
      const int64_t slot_lo = view->slot_selector ? view->slot_selector->lo : 0;
      const int64_t slot_hi = view->slot_selector ? view->slot_selector->hi : 0;
      const bool single_slot = slot_lo == slot_hi;
      const bool box_exact = const_coords && single_slot && view->static_full_valid &&
                             (!out.byte_hull || !OverlapsDynamicValid(space, *out.byte_hull));
      out.matrix_box =
          MatrixBoxFootprint{view->root_id, slot_lo, slot_hi, r_begin, r_end, c_begin, c_end, box_exact};
      if (r_opt && c_opt && r_opt->lo >= 0 && c_opt->lo >= 0 && r_opt->hi <= root.rows - sub_rows &&
          c_opt->hi <= root.columns - sub_cols) {
        out.matrix_subtile = MatrixSubtileFootprint{*r_opt, *c_opt, sub_rows, sub_cols};
      }

      // A comparable within-slot row or column fiber can coexist with the
      // modular slot fiber. The slot partition origin defines the matrix axes.
      if (out.matrix_subtile && out.fiber.kind == IterationFiber::Kind::Wildcard) {
        if (r_opt && r_opt->affine_frame && r_opt->modulus == 0 && r_opt->affine_coeff != 0 &&
            ((c_opt && !c_opt->affine_frame && c_opt->is_const()) || root.columns == sub_cols) &&
            !r_opt->symbolic_base && c_begin == 0 && c_end == root.columns &&
            __int128(r_opt->affine_base) + sub_rows <= std::numeric_limits<int64_t>::max()) {
          out.fiber.kind = IterationFiber::Kind::AffineSlice;
          out.fiber.frame = r_opt->affine_frame;
          out.fiber.coeff = r_opt->affine_coeff;
          out.fiber.offset = r_opt->affine_base;
          out.fiber.extent = sub_rows;
          out.fiber.slice_axis = 0;
        } else if (c_opt && c_opt->affine_frame && c_opt->modulus == 0 && c_opt->affine_coeff != 0 &&
                   ((r_opt && !r_opt->affine_frame && r_opt->is_const()) || root.rows == sub_rows) &&
                   !c_opt->symbolic_base && r_begin == 0 && r_end == root.rows &&
                   __int128(c_opt->affine_base) + sub_cols <= std::numeric_limits<int64_t>::max()) {
          out.fiber.kind = IterationFiber::Kind::AffineSlice;
          out.fiber.frame = c_opt->affine_frame;
          out.fiber.coeff = c_opt->affine_coeff;
          out.fiber.offset = c_opt->affine_base;
          out.fiber.extent = sub_cols;
          out.fiber.slice_axis = 1;
        }
        if (out.fiber.kind == IterationFiber::Kind::AffineSlice) {
          out.fiber.basis = {IterationFiberBasis::Kind::MatrixSlot, root.space, root.id,
                             single_slot ? view->base_byte_offset : root.address};
        }
      }
    }
  }

  if (IsFullOverwriteOpOperand(call, mode, space, operand) && out.matrix_box && out.matrix_box->exact &&
      out.matrix_box->slot_lo == out.matrix_box->slot_hi && out.matrix_box->row_begin == 0 &&
      out.matrix_box->row_end == root.rows && out.matrix_box->col_begin == 0 &&
      out.matrix_box->col_end == root.columns &&
      (!out.byte_hull || !OverlapsDynamicValid(space, *out.byte_hull))) {
    out.definite_overwrite = out.byte_hull;
  }

  return out;
}

}  // namespace pypto::ir::sync
