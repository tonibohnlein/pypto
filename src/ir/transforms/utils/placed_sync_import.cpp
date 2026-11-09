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
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_indexed_recipes.h"
#include "pypto/backend/common/buffer_row_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir::sync {
namespace {
int64_t Checked(__int128 value) {
  CHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max())
      << "Synchronization import requires nonnegative nonoverflowing index arithmetic";
  return static_cast<int64_t>(value);
}
int64_t Constant(const ExprPtr& e) {
  auto c = As<ConstInt>(e);
  CHECK(c) << "Synchronization import requires a constant extent/address";
  return Checked(c->value_);
}
void RequireIndex(const ExprPtr& e) {
  auto type = As<ScalarType>(e->GetType());
  CHECK(type && type->dtype_ == DataType::INDEX)
      << "Synchronization affine coordinates require an entirely INDEX expression chain";
}
std::pair<int64_t, int64_t> Affine(const ExprPtr& e, const SyncSegment& segment);
std::pair<int64_t, int64_t> AffineImpl(const ExprPtr& e, const SyncSegment& segment) {
  if (auto c = As<ConstInt>(e)) return {Constant(c), 0};
  if (segment.original_loop && e == segment.original_loop->loop_var_) return {segment.start, segment.step};
  if (auto a = As<Add>(e)) {
    auto x = Affine(a->left_, segment), y = Affine(a->right_, segment);
    return {Checked(__int128(x.first) + y.first), Checked(__int128(x.second) + y.second)};
  }
  if (auto m = As<Mul>(e)) {
    auto x = Affine(m->left_, segment), y = Affine(m->right_, segment);
    CHECK(x.second == 0 || y.second == 0) << "Nonaffine synchronization access";
    return {Checked(__int128(x.first) * y.first),
            Checked(__int128(x.second) * y.first + __int128(y.second) * x.first)};
  }
  CHECK(false) << "Unavailable affine synchronization access";
  return {};
}
std::pair<int64_t, int64_t> Affine(const ExprPtr& e, const SyncSegment& segment) {
  RequireIndex(e);
  auto result = AffineImpl(e, segment);
  // Check every intermediate, including one hidden by a later multiplication by zero.
  Checked(__int128(result.first) + __int128(result.second) * std::max<int64_t>(0, segment.count - 1));
  return result;
}
std::vector<StmtPtr> Statements(const StmtPtr& s) {
  if (auto seq = As<SeqStmts>(s)) return seq->stmts_;
  return {s};
}
CallPtr SingleCall(const StmtPtr& s) {
  auto statements = Statements(s);
  CHECK(statements.size() == 1) << "Conditional synchronization body must contain one payload";
  auto eval = As<EvalStmt>(statements.front());
  CHECK(eval) << "Conditional synchronization body must be a payload evaluation";
  auto call = As<Call>(eval->expr_);
  CHECK(call) << "Submit/indirect execution is outside the synchronization prototype";
  return call;
}
class Importer {
 public:
  CompactSyncInput result;
  std::unordered_map<const Expr*, size_t> objects;
  // Vec descriptors retain their own geometry; all placed Vec bytes share one
  // physical domain. SSA roots, typed aliases and gaps do not create disjointness.
  struct VecView {
    int64_t address, pitch;
    BufferTypePtr type;
  };
  std::unordered_map<const Expr*, VecView> vec_views;
  std::optional<size_t> vec_storage;
  void RegisterVec(const VarPtr& var, VecView view) {
    const auto type = view.type;
    // A full-valid column vector [R,1] occupies consecutive elements in either
    // major order. This does not admit general column-major matrices or views.
    CHECK(type->shape_ == type->valid_shape_ &&
          (backend::DenseBufferBytes(type) || backend::IsColumnVectorBuffer(type)))
        << "Synchronization Vec views require static full-valid dense descriptors";
    const auto end = Checked(__int128(view.address) + __int128(type->shape_[0] - 1) * view.pitch +
                             __int128(type->shape_[1]) * type->dtype_.GetByte());
    if (!vec_storage) {
      vec_storage = result.storage.size();
      result.storage.push_back({"Vec physical bytes", MemorySpace::Vec, 0, 1, end, end, nullptr, true});
    } else {
      auto& domain = result.storage[*vec_storage];
      domain.bytes = domain.columns = std::max(domain.bytes, end);
    }
    objects[var.get()] = *vec_storage;
    vec_views.emplace(var.get(), std::move(view));
  }
  void View(const AssignStmtPtr& assign, const CallPtr& call) {
    OpRegistry::GetInstance().ValidateBufferCall(call);
    auto source = vec_views.find(call->args_[0].get());
    CHECK(source != vec_views.end()) << "Synchronization view needs a qualified Vec source";
    auto view = source->second;
    auto type = As<BufferType>(assign->var_->GetType());
    CHECK(type && type->shape_.size() == 2) << "Synchronization view requires rank two";
    if (IsOp(call, "buffer.subview")) {
      const auto offsets = As<MakeTuple>(call->args_[1]);
      const auto row = Constant(offsets->elements_[0]), column = Constant(offsets->elements_[1]);
      view.address = Checked(__int128(view.address) + __int128(row) * view.pitch +
                             __int128(column) * view.type->dtype_.GetByte());
      // Inherit the parent's pitch, including through nested subviews.
    } else {
      CHECK(view.pitch == Checked(__int128(view.type->shape_[1]) * view.type->dtype_.GetByte()))
          << "Synchronization cannot reshape a noncontiguous window";
      view.pitch = Checked(__int128(type->shape_[1]) * type->dtype_.GetByte());
    }
    view.type = type;
    RegisterVec(assign->var_, view);
  }
  size_t Storage(const ExprPtr& e) const {
    auto found = objects.find(e.get());
    CHECK(found != objects.end()) << "Synchronization access has no qualified storage identity";
    return found->second;
  }
  void Allocate(const AssignStmtPtr& a) {
    auto c = As<Call>(a->value_);
    if (c && (IsOp(c, "buffer.subview") || IsOp(c, "buffer.reshape"))) {
      View(a, c);
      return;
    }
    CHECK(c && IsOp(c, "buffer.alloc"))
        << "Unavailable storage/view or scalar definition in synchronization import";
    OpRegistry::GetInstance().ValidateBufferCall(c);
    CHECK(c->args_.size() == 2) << "Synchronization import requires placed buffer allocation addresses";
    auto t = As<BufferType>(a->var_->GetType());
    CHECK(t && t->shape_.size() == 2 && t->shape_ == t->valid_shape_ && t->dtype_.GetBit() % 8 == 0)
        << "Synchronization import needs fully valid byte-addressable rank-two buffers";
    CHECK(t->compact_ == CompactMode::null && t->pad_ == PadValue::null)
        << "Packed/padded buffer interpretation is not qualified";
    auto valid = As<MakeTuple>(c->args_[0]);
    CHECK(valid && valid->elements_.empty()) << "Dynamic valid extents are not qualified";
    const auto address = Constant(c->args_[1]);
    auto bytes = Checked(__int128(t->shape_[0]) * t->shape_[1] * t->dtype_.GetByte());
    Checked(__int128(address) + bytes);
    if (t->memory_space_ == MemorySpace::Vec) {
      RegisterVec(a->var_, {address, Checked(__int128(t->shape_[1]) * t->dtype_.GetByte()), t});
      return;
    }
    // A whole boxed buffer is exact only when no partial box is present.
    if (t->slayout_ != TileLayout::none_box) {
      CHECK(t->fractal_ && t->fractal_ % (16 * t->dtype_.GetByte()) == 0) << "Unavailable fractal geometry";
      auto columns = t->fractal_ / (16 * t->dtype_.GetByte());
      CHECK(t->shape_[0] % 16 == 0 && t->shape_[1] % columns == 0)
          << "Partial fractal-box footprints require separate qualification";
    }
    for (size_t i = 0; i < result.storage.size(); ++i) {
      const auto& old = result.storage[i];
      if (old.space != t->memory_space_ || address >= old.address + old.bytes ||
          old.address >= address + bytes)
        continue;
      CHECK(old.address == address && old.bytes == bytes && old.layout && old.rows == t->shape_[0] &&
            old.columns == t->shape_[1] && old.layout->dtype_ == t->dtype_ &&
            old.layout->blayout_ == t->blayout_ && old.layout->slayout_ == t->slayout_ &&
            old.layout->fractal_ == t->fractal_)
          << "Overlapping placed buffers need a common physical-layout partition";
      objects[a->var_.get()] = i;
      return;
    }
    objects[a->var_.get()] = result.storage.size();
    result.storage.push_back(
        {a->var_->name_hint_, t->memory_space_, address, t->shape_[0], t->shape_[1], bytes, t});
  }
  SyncAccess Access(size_t storage, bool write, std::pair<int64_t, int64_t> row,
                    std::pair<int64_t, int64_t> column, int64_t rows, int64_t columns,
                    const SyncSegment& segment) {
    const auto& s = result.storage[storage];
    const auto last = std::max<int64_t>(0, segment.count - 1);
    CHECK(rows > 0 && columns > 0 &&
          Checked(__int128(row.first) + __int128(last) * row.second + rows) <= s.rows &&
          Checked(__int128(column.first) + __int128(last) * column.second + columns) <= s.columns)
        << "Synchronization access exceeds its proved allocation";
    return {storage, write, row.first, column.first, row.second, column.second, rows, columns, false};
  }
  SyncSite Site(const CallPtr& call, const SyncSegment& segment) {
    OpRegistry::GetInstance().ValidateBufferCall(call);
    SyncSite site{call, nullptr, PipeType::ALL, {}};
    // Track modes by operand, not storage: in-place operands can share bytes
    // while retaining different registered read/write modes.
    std::vector<std::pair<bool, bool>> modes(call->args_.size());
    auto whole = [&](size_t arg, bool write) {
      (write ? modes[arg].second : modes[arg].first) = true;
      auto id = Storage(call->args_[arg]);
      if (auto found = vec_views.find(call->args_[arg].get()); found != vec_views.end()) {
        const auto& view = found->second;
        const auto width = Checked(__int128(view.type->shape_[1]) * view.type->dtype_.GetByte());
        // Dense footprints use one interval. Strided views retain one interval
        // per physical row, never an enclosing rectangle that fills the gaps.
        if (view.pitch == width) {
          site.accesses.push_back(Access(id, write, {0, 0}, {view.address, 0}, 1,
                                         Checked(__int128(width) * view.type->shape_[0]), segment));
        } else {
          for (int64_t row = 0; row < view.type->shape_[0]; ++row)
            site.accesses.push_back(Access(id, write, {0, 0},
                                           {Checked(__int128(view.address) + __int128(row) * view.pitch), 0},
                                           1, width, segment));
        }
        return;
      }
      const auto& s = result.storage[id];
      site.accesses.push_back(Access(id, write, {0, 0}, {0, 0}, s.rows, s.columns, segment));
    };
    auto enclosure = [&](size_t arg, bool write) {
      const auto first = site.accesses.size();
      whole(arg, write);
      for (size_t i = first; i < site.accesses.size(); ++i) site.accesses[i].exact_footprint = false;
    };
    if (IsOp(call, "buffer.load") || IsOp(call, "buffer.store")) {
      const bool load = IsOp(call, "buffer.load");
      auto offsets = As<MakeTuple>(call->args_[1]), shape = As<MakeTuple>(call->args_[2]);
      CHECK(offsets && shape && offsets->elements_.size() == 2 && shape->elements_.size() == 2)
          << "Synchronization DMA requires a rank-two rectangle";
      auto id = Storage(call->args_[load ? 0 : 3]);
      site.accesses.push_back(Access(id, !load, Affine(offsets->elements_[0], segment),
                                     Affine(offsets->elements_[1], segment), Constant(shape->elements_[0]),
                                     Constant(shape->elements_[1]), segment));
      const auto gm_arg = load ? 0 : 3;
      (load ? modes[gm_arg].first : modes[gm_arg].second) = true;
      const auto local = As<BufferType>(call->args_[load ? 3 : 0]->GetType());
      CHECK(local && local->shape_[0] == Constant(shape->elements_[0]) &&
            local->shape_[1] == Constant(shape->elements_[1]))
          << "Partial DMA local footprint is not qualified";
      CHECK(local->memory_space_ == MemorySpace::Vec ||
            (load ? local->memory_space_ == MemorySpace::Mat : local->memory_space_ == MemorySpace::Acc))
          << "Only GM/Mat, Acc/GM and GM/Vec DMA contracts are qualified";
      whole(load ? 3 : 0, load);
      site.pipe =
          load ? PipeType::MTE2 : (local->memory_space_ == MemorySpace::Vec ? PipeType::MTE3 : PipeType::FIX);
    } else if (IsOp(call, "buffer.copy") || IsOp(call, "buffer.extract")) {
      const bool extract = IsOp(call, "buffer.extract");
      const auto target = extract ? 3 : 1;
      auto sid = Storage(call->args_[0]), tid = Storage(call->args_[target]);
      const auto& s = result.storage[sid];
      const auto& t = result.storage[tid];
      CHECK(s.space == MemorySpace::Mat && (t.space == MemorySpace::Left || t.space == MemorySpace::Right))
          << "Only Mat to operand-buffer movement is qualified";
      auto row = extract ? Affine(call->args_[1], segment) : std::pair<int64_t, int64_t>{0, 0};
      auto column = extract ? Affine(call->args_[2], segment) : std::pair<int64_t, int64_t>{0, 0};
      if (!extract) CHECK(s.rows == t.rows && s.columns == t.columns) << "Partial copy is not qualified";
      if (s.layout->slayout_ != TileLayout::none_box) {
        const auto box = s.layout->fractal_ / (16 * s.layout->dtype_.GetByte());
        CHECK(row.first % 16 == 0 && row.second % 16 == 0 && t.rows % 16 == 0 && column.first % box == 0 &&
              column.second % box == 0 && t.columns % box == 0)
            << "Unaligned boxed extraction requires a separate footprint contract";
      }
      site.accesses.push_back(Access(sid, false, row, column, t.rows, t.columns, segment));
      modes[0].first = true;
      whole(target, true);
      site.pipe = PipeType::MTE1;
    } else if (IsOp(call, "buffer.matmul") || IsOp(call, "buffer.matmul_acc")) {
      whole(0, false);
      whole(1, false);
      if (IsOp(call, "buffer.matmul_acc")) whole(2, false);
      whole(2, true);
      site.pipe = PipeType::M;
      const auto lhs = As<BufferType>(call->args_[0]->GetType());
      const auto rhs = As<BufferType>(call->args_[1]->GetType());
      const auto dst = As<BufferType>(call->args_[2]->GetType());
      // Ascend C Mmad synchronization optimization: consecutive K accumulation
      // on the same L0C footprint needs no M barrier when floor(M/16)*floor(N/16)
      // >= 10. Admission already validates full-valid shapes and native layouts.
      // https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html
      if (result.hardware_target == "a2a3" && lhs && rhs && dst && lhs->memory_space_ == MemorySpace::Left &&
          rhs->memory_space_ == MemorySpace::Right && dst->memory_space_ == MemorySpace::Acc &&
          lhs->shape_[0] > 0 && lhs->shape_[0] <= 4095 && lhs->shape_[1] > 0 && lhs->shape_[1] <= 4095 &&
          rhs->shape_[1] > 0 && rhs->shape_[1] <= 4095 && lhs->shape_[1] == rhs->shape_[0] &&
          dst->shape_[0] == lhs->shape_[0] && dst->shape_[1] == rhs->shape_[1] &&
          (lhs->shape_[0] / 16) * (rhs->shape_[1] / 16) >= 10) {
        site.protected_accumulator = Storage(call->args_[2]);
        site.initializes_accumulator = IsOp(call, "buffer.matmul");
      }
    } else if (auto recipe = backend::FindBufferIndexedRecipe(call->op_->name_)) {
      CHECK(result.owner->func_type_ == FunctionType::AIV) << "Indexed synchronization requires an AIV leaf";
      // The shared contract uses in-range physical element offsets into a
      // known local buffer, not arbitrary pointers. Data-dependent gather
      // reads and native scratch use are enclosed without inventing selectors.
      if (recipe->scatter) {
        whole(0, false);
      } else {
        enclosure(0, false);
        enclosure(2, true);
      }
      whole(1, false);
      // TSCATTER clears the entire destination before its indexed writes.
      whole(recipe->input_count, true);
      site.pipe = PipeType::V;
    } else if (auto recipe = backend::FindBufferRowRecipe(call->op_->name_)) {
      CHECK(result.owner->func_type_ == FunctionType::AIV) << "Row synchronization requires an AIV leaf";
      whole(0, false);
      // The shared contract distinguishes writable reduction scratch from a
      // read-only broadcast column. Model the full scratch window as a write
      // enclosure; do not claim it is the exact native scratch footprint.
      if (recipe->reduction) {
        enclosure(1, true);
      } else {
        whole(1, false);
      }
      whole(2, true);
      site.pipe = PipeType::V;
    } else if (auto recipe = backend::FindBufferElementwiseRecipe(call->op_->name_)) {
      CHECK(result.owner->func_type_ == FunctionType::AIV) << "Vec synchronization requires an AIV leaf";
      for (size_t arg = 0; arg < recipe->inputs.size(); ++arg) {
        if (recipe->inputs[arg].kind == backend::BufferElementwiseOperandKind::Buffer) {
          whole(arg, false);
        } else {
          CHECK(As<ConstFloat>(call->args_[arg]) || As<ConstInt>(call->args_[arg]))
              << "Synchronization scalar operand readiness is unavailable";
        }
      }
      whole(recipe->inputs.size(), true);
      site.pipe = PipeType::V;
    } else {
      CHECK(false) << "Unavailable instruction contract in synchronization import: " << call->op_->name_;
    }
    const bool vector = result.owner->func_type_ == FunctionType::AIV;
    CHECK(vector ? (site.pipe == PipeType::MTE2 || site.pipe == PipeType::MTE3 || site.pipe == PipeType::V)
                 : (site.pipe == PipeType::MTE2 || site.pipe == PipeType::MTE1 || site.pipe == PipeType::M ||
                    site.pipe == PipeType::FIX))
        << "Payload pipe is unavailable on the owning synchronization core";
    // Geometry above qualifies one explicit payload contract. Validate its
    // data modes against the registry rather than guessing other op effects.
    for (size_t arg = 0; arg < call->args_.size(); ++arg) {
      if (!As<BufferType>(call->args_[arg]->GetType())) continue;
      auto effect = OpRegistry::GetInstance().GetEntry(call->op_->name_).GetBufferArgEffect(arg).data;
      const auto [reads, writes] = modes[arg];
      CHECK(reads == (effect == BufferAccess::Read || effect == BufferAccess::ReadWrite) &&
            writes == (effect == BufferAccess::Write || effect == BufferAccess::ReadWrite))
          << "Imported data modes disagree with registered operation contract";
    }
    return site;
  }
  void Body(const StmtPtr& stmt, SyncSegment& segment) {
    bool preceding_matrix = false;
    for (const auto& s : Statements(stmt)) {
      if (auto eval = As<EvalStmt>(s)) {
        auto call = As<Call>(eval->expr_);
        CHECK(call) << "Submit or non-call payload is not qualified";
        segment.sites.push_back(Site(call, segment));
        preceding_matrix |= segment.sites.back().pipe == PipeType::M;
      } else if (auto branch = As<IfStmt>(s)) {
        auto eq = As<Eq>(branch->condition_);
        CHECK(segment.original_loop && eq && eq->left_ == segment.original_loop->loop_var_ &&
              Constant(eq->right_) == segment.start && branch->else_body_ && branch->return_vars_.empty())
            << "Only first-visit initialization/continuation selection is qualified";
        auto first = SingleCall(branch->then_body_), later = SingleCall(*branch->else_body_);
        CHECK(IsOp(first, "buffer.matmul") && IsOp(later, "buffer.matmul_acc") &&
              first->args_ == later->args_)
            << "Conditional payloads must preserve operands and destination";
        auto site = Site(first, segment);
        auto continuation = Site(later, segment);
        site.continuation = later;
        // A nonleading initialization has startup-only internal demands. Keep
        // the existing uniform family contract conservative for that site.
        if (preceding_matrix) site.protected_accumulator.reset();
        preceding_matrix = true;
        for (auto a : continuation.accesses)
          if (!a.write && result.storage[a.storage].space == MemorySpace::Acc) {
            a.after_first = true;
            site.accesses.push_back(a);
          }
        segment.sites.push_back(std::move(site));
      } else {
        CHECK(As<YieldStmt>(s)) << "Nested or data-dependent control is outside this compact provider";
        CHECK(As<YieldStmt>(s)->value_.empty()) << "Loop-carried scalar readiness is not qualified";
      }
    }
  }
};
}  // namespace
CompactSyncInput ImportPlacedSync(const FunctionPtr& function, bool disjoint_gm, const std::string& target) {
  CHECK(function && (function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV))
      << "Synchronization prototype requires one AIC or AIV leaf";
  if (function->func_type_ == FunctionType::AIV) {
    // Reuse the production view/recipe alias verifier. Synchronization cannot
    // repair an illegal partial in-place instruction or a malformed descriptor.
    CHECK(function->ir_stage_ == FunctionIRStage::Buffer) << "Vec synchronization requires Buffer IR";
    auto program =
        std::make_shared<Program>(std::vector<FunctionPtr>{function}, function->name_, function->span_);
    std::vector<Diagnostic> diagnostics;
    CreateBufferIRPropertyVerifier()->Verify(program, diagnostics);
    for (const auto& diagnostic : diagnostics)
      CHECK(diagnostic.severity != DiagnosticSeverity::Error) << diagnostic.message;
  }
  CHECK(disjoint_gm)
      << "Disjoint GM argument contract must be explicitly supplied; may-alias import unavailable";
  Importer importer;
  importer.result.owner = function;
  importer.result.hardware_target = target;
  for (const auto& param : function->params_) {
    auto t = As<TensorType>(param->GetType());
    CHECK(t && t->shape_.size() == 2 && t->dtype_.GetBit() % 8 == 0 && !t->tensor_view_)
        << "Synchronization prototype requires unviewed ND tensor arguments";
    auto rows = Constant(t->shape_[0]), cols = Constant(t->shape_[1]);
    importer.objects[param.get()] = importer.result.storage.size();
    importer.result.storage.push_back({param->name_hint_, MemorySpace::DDR, 0, rows, cols,
                                       Checked(__int128(rows) * cols * t->dtype_.GetByte()), nullptr});
  }
  for (const auto& stmt : Statements(function->body_)) {
    if (auto a = As<AssignStmt>(stmt)) {
      importer.Allocate(a);
      continue;
    }
    if (As<ReturnStmt>(stmt)) continue;
    if (auto loop = As<ForStmt>(stmt)) {
      RequireIndex(loop->loop_var_);
      RequireIndex(loop->start_);
      RequireIndex(loop->stop_);
      RequireIndex(loop->step_);
      CHECK(loop->kind_ == ForKind::Sequential && loop->iter_args_.empty() && loop->return_vars_.empty())
          << "Only sequential coordinate-only loops are qualified";
      SyncSegment s;
      s.original_loop = loop;
      s.start = Constant(loop->start_);
      s.step = Constant(loop->step_);
      auto stop = Constant(loop->stop_);
      CHECK(s.step > 0) << "Nonpositive loop step is not qualified";
      s.count = stop <= s.start ? 0 : 1 + (stop - s.start - 1) / s.step;
      Checked(__int128(s.start) + __int128(s.count) * s.step);
      importer.Body(loop->body_, s);
      importer.result.segments.push_back(std::move(s));
    } else {
      // Keep maximal adjacent finite statements in one numerical region.
      if (importer.result.segments.empty() || importer.result.segments.back().original_loop)
        importer.result.segments.emplace_back();
      importer.Body(stmt, importer.result.segments.back());
    }
  }
  return std::move(importer.result);
}
}  // namespace pypto::ir::sync
