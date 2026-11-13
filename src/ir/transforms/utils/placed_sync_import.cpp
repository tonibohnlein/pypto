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
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_indexed_recipes.h"
#include "pypto/backend/common/buffer_row_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/verifier/verifier.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::sync {
namespace {
int64_t Checked(__int128 value) {
  CHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max())
      << "Synchronization import requires nonnegative nonoverflowing index arithmetic";
  return static_cast<int64_t>(value);
}
// Exact contiguous byte evidence for both admission and retained exports.
// No enclosure or partial boxed layout is flattened into an exact interval.
std::optional<std::pair<int64_t, int64_t>> PhysicalInterval(const SyncStorage& storage,
                                                            const SyncAccess& access) {
  if (!access.exact_footprint || access.row_step || access.column_step || access.after_first ||
      !storage.row_parameters.empty() || !storage.column_parameters.empty() || storage.row_block ||
      storage.column_block || storage.row_extent_parameter)
    return std::nullopt;
  const auto row_end = __int128(access.row) + access.rows;
  const auto column_end = __int128(access.column) + access.columns;
  if (access.row < 0 || access.column < 0 || access.rows <= 0 || access.columns <= 0 ||
      row_end > storage.rows || column_end > storage.columns)
    return std::nullopt;
  __int128 begin = 0, end = 0;
  if (storage.byte_coordinates) {
    if (access.row != 0 || access.rows != 1) return std::nullopt;
    begin = access.column;
    end = column_end;
  } else {
    const auto elements = __int128(storage.rows) * storage.columns;
    const auto slot_bytes = storage.slot_count == 1 ? storage.bytes : storage.slot_bytes;
    if (elements <= 0 || slot_bytes <= 0 || slot_bytes % elements) return std::nullopt;
    const auto bytes = slot_bytes / elements;
    if (storage.layout && storage.layout->slayout_ != TileLayout::none_box) {
      // Whole qualified fractal tiles are contiguous; partial logical
      // rectangles need a layout-specific interval proof, not flattening.
      if (access.row || access.column || access.rows != storage.rows || access.columns != storage.columns)
        return std::nullopt;
      end = slot_bytes;
    } else if (!storage.layout || storage.layout->blayout_ == TileLayout::row_major) {
      if (access.rows != 1 && (access.column || access.columns != storage.columns)) return std::nullopt;
      begin = (__int128(access.row) * storage.columns + access.column) * bytes;
      end = begin + __int128(access.rows) * access.columns * bytes;
    } else if (storage.layout->blayout_ == TileLayout::col_major) {
      if (access.columns != 1 && (access.row || access.rows != storage.rows)) return std::nullopt;
      begin = (__int128(access.column) * storage.rows + access.row) * bytes;
      end = begin + __int128(access.rows) * access.columns * bytes;
    } else {
      return std::nullopt;
    }
  }
  return std::make_pair(Checked(begin), Checked(end));
}
int64_t Constant(const ExprPtr& e) {
  auto c = As<ConstInt>(e);
  CHECK(c) << "Synchronization import requires a constant extent/address";
  return Checked(c->value_);
}
void RequireIndex(const ExprPtr& e) {
  auto type = As<ScalarType>(e->GetType());
  CHECK(type && type->dtype_ == DataType::INDEX)
      << "Synchronization address roots and loop coordinates require INDEX type";
}
// Prove value-preserving machine arithmetic, not merely unbounded integer
// algebra. INDEX is signed 64-bit in the qualified backend contract.

bool SameCoordinateType(const ExprPtr& a, const ExprPtr& b) {
  const auto x = As<ScalarType>(a->GetType()), y = As<ScalarType>(b->GetType());
  return x && y && x->dtype_ == y->dtype_;
}
struct AffineCoordinate {
  int64_t first = 0, second = 0;
  SyncTranslation parameters;
  int64_t block = 0;
  std::map<const ForStmt*, int64_t> frames;
};
using ParameterIndices = std::unordered_map<const Expr*, size_t>;
// Retain checked scalar definitions as a DAG. Coefficients are only materialized
// for requested access roots, never for every SSA prefix definition.
class CoordinateGraph {
  enum class Kind { Constant, Parameter, Block, Loop, Add, Scale, ModularShift };
  struct Node {
    Kind kind;
    int64_t upper = 0, factor = 0, start = 0, step = 0, lower = 0;
    size_t left = 0, right = 0, parameter = 0;
    const ForStmt* frame = nullptr;
    bool constant = false;
  };
  std::vector<Node> nodes_;
  std::map<std::pair<const Expr*, const ForStmt*>, size_t> memo_;
  std::unordered_map<const Expr*, size_t> definitions_;

 public:
  bool Defined(const ExprPtr& e) const { return definitions_.count(e.get()); }
  size_t Build(const ExprPtr& e, const SyncSegment& segment, const ParameterIndices& parameters,
               CompactSyncInput& input) {
    const auto maximum = SyncCoordinateMaximum(e);
    CHECK(maximum) << "Synchronization coordinates require integer arithmetic";
    const auto frame = segment.original_loop.get();
    if (auto found = definitions_.find(e.get()); found != definitions_.end()) {
      return found->second;
    }
    const auto key = std::make_pair(e.get(), frame);
    if (auto found = memo_.find(key); found != memo_.end()) {
      return found->second;
    }
    if (auto cast = As<Cast>(e)) {
      const auto child = Build(cast->operand_, segment, parameters, input);
      CHECK(nodes_[child].upper <= *maximum) << "Synchronization coordinate cast may change its value";
      memo_.emplace(key, child);
      return child;
    }
    Node n{Kind::Constant};
    if (auto c = As<ConstInt>(e)) {
      n.lower = n.upper = n.factor = Constant(c);
      n.constant = true;
    } else if (auto call = As<Call>(e); call && IsOp(call, "buffer.get_block_idx")) {
      CHECK(input.block_bounds && call->args_.empty()) << "Launch coordinate has no established caller range";
      n.kind = Kind::Block;
      n.lower = input.block_bounds->first;
      n.upper = input.block_bounds->second;
    } else if (auto loop = std::find_if(segment.nested_frames.begin(), segment.nested_frames.end(),
                                        [&](const auto& f) { return e == f.loop->loop_var_; });
               loop != segment.nested_frames.end()) {
      if (loop->start_expression && loop->count <= 1) {
        // Uniform singleton, not a sampled IV. Preserve the source frame in
        // the region tree while substituting its proved invariant start.
        const auto id = Build(loop->start_expression, segment, parameters, input);
        memo_.emplace(key, id);
        return id;
      }
      n.kind = Kind::Loop;
      n.frame = loop->loop.get();
      n.start = loop->start;
      n.lower = n.start;
      n.step = loop->step;
      n.upper = Checked(__int128(n.start) + __int128(n.step) * std::max<int64_t>(0, loop->count - 1));
    } else if (segment.original_loop && e == segment.original_loop->loop_var_) {
      n.kind = Kind::Loop;
      n.frame = frame;
      n.start = segment.start;
      n.lower = n.start;
      n.step = segment.step;
      n.upper = Checked(__int128(n.start) + __int128(n.step) * std::max<int64_t>(0, segment.count - 1));
    } else if (auto it = parameters.find(e.get()); it != parameters.end()) {
      CHECK(input.parameter_bounds.count(it->second))
          << "Invocation coordinate has no established caller range";
      n.kind = Kind::Parameter;
      n.parameter = it->second;
      n.lower = input.parameter_bounds.at(it->second).first;
      n.upper = input.parameter_bounds.at(it->second).second;
    } else if (auto mod = As<FloorMod>(e)) {
      CHECK(SameCoordinateType(e, mod->left_) && SameCoordinateType(e, mod->right_))
          << "Coordinate modulo requires matching original types";
      n.left = Build(mod->left_, segment, parameters, input);
      Build(mod->right_, segment, parameters, input);
      const auto modulus = Constant(mod->right_);
      CHECK(modulus > 0) << "Coordinate modulo requires a positive constant modulus";
      const auto child = nodes_[n.left];
      CHECK(child.lower / modulus == child.upper / modulus)
          << "Invocation modulo crosses a wrap; exact activation partition is required";
      n.kind = Kind::ModularShift;
      n.factor = (child.lower / modulus) * modulus;
      n.lower = child.lower - n.factor;
      n.upper = child.upper - n.factor;
      n.constant = child.constant;
    } else {
      ExprPtr left, right;
      const bool multiply = static_cast<bool>(As<Mul>(e));
      if (auto add = As<Add>(e)) {
        left = add->left_;
        right = add->right_;
      } else if (auto mul = As<Mul>(e)) {
        left = mul->left_;
        right = mul->right_;
      } else {
        CHECK(false) << "Unavailable affine synchronization access";
      }
      CHECK(SameCoordinateType(e, left) && SameCoordinateType(e, right))
          << "Coordinate arithmetic requires matching operand types and explicit casts";
      n.left = Build(left, segment, parameters, input);
      n.right = Build(right, segment, parameters, input);
      // Copy scalar node metadata: recursive construction can reallocate nodes_.
      auto x = nodes_[n.left], y = nodes_[n.right];
      if (multiply) {
        CHECK(x.constant || y.constant) << "Nonaffine synchronization access";
        if (!y.constant) {
          std::swap(n.left, n.right);
          std::swap(x, y);
        }
        n.kind = Kind::Scale;
        n.factor = y.upper;
        n.upper = Checked(__int128(x.upper) * n.factor);
        n.lower = Checked(__int128(x.lower) * n.factor);
        n.constant = x.constant || !n.factor;
      } else {
        n.kind = Kind::Add;
        n.upper = Checked(__int128(x.upper) + y.upper);
        n.lower = Checked(__int128(x.lower) + y.lower);
        n.constant = x.constant && y.constant;
      }
    }
    // Each intermediate has been range-checked in its original machine type,
    // including operands of zero and values before a widening INDEX cast.
    CHECK(n.upper <= *maximum) << "Synchronization coordinate arithmetic may overflow its source type";
    const auto id = nodes_.size();
    nodes_.push_back(n);
    memo_.emplace(key, id);
    ++input.coordinate_nodes;
    return id;
  }
  void Define(const ExprPtr& var, const ExprPtr& value, const SyncSegment& segment,
              const ParameterIndices& parameters, CompactSyncInput& input) {
    CHECK(SameCoordinateType(var, value)) << "Coordinate definition requires an explicit type conversion";
    definitions_.emplace(var.get(), Build(value, segment, parameters, input));
  }
  AffineCoordinate Flatten(size_t root, CompactSyncInput& input) const {
    // Construction IDs are topological. Propagate aggregate weights once per
    // reachable node, avoiding path enumeration through shared subexpressions.
    std::map<size_t, int64_t, std::greater<size_t>> pending{{root, 1}};
    AffineCoordinate result;
    __int128 origin = 0;
    auto shift_origin = [&](__int128 delta) {
      CHECK(!__builtin_add_overflow(origin, delta, &origin))
          << "Coordinate affine proof exceeds supported precision";
    };
    while (!pending.empty()) {
      const auto [id, weight] = *pending.begin();
      pending.erase(pending.begin());
      ++input.coordinate_flatten_visits;
      const auto& n = nodes_[id];
      auto add = [&](size_t child, int64_t w) {
        if (w) pending[child] = Checked(__int128(pending[child]) + w);
      };
      if (n.constant) {
        shift_origin(__int128(weight) * n.upper);
      } else if (n.kind == Kind::Parameter) {
        auto& coefficient = result.parameters[n.parameter];
        coefficient = Checked(__int128(coefficient) + weight);
      } else if (n.kind == Kind::Block) {
        // Anchor each proved case at its lower launch coordinate. Modular
        // rewrites may subtract a constant, but the final origin stays >=0.
        shift_origin(__int128(weight) * input.block_bounds->first);
        result.block = Checked(__int128(result.block) + weight);
      } else if (n.kind == Kind::Loop) {
        shift_origin(__int128(weight) * n.start);
        auto& coefficient = result.frames[n.frame];
        coefficient = Checked(__int128(coefficient) + __int128(weight) * n.step);
      } else if (n.kind == Kind::Scale) {
        add(n.left, Checked(__int128(weight) * n.factor));
      } else if (n.kind == Kind::ModularShift) {
        shift_origin(-__int128(weight) * n.factor);
        add(n.left, weight);
      } else {
        add(n.left, weight);
        add(n.right, weight);
      }
    }
    result.first = Checked(origin);
    input.coordinate_coefficient_entries +=
        result.parameters.size() + result.frames.size() + (result.block != 0);
    return result;
  }
};
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
  // Native instruction, descriptor and control qualification is shared. Only
  // dependence clients construct storage effects and their selector evidence.
  explicit Importer(bool recover_accesses) : recover_accesses(recover_accesses) {}
  const bool recover_accesses;
  CompactSyncInput result;
  std::unordered_map<const Expr*, size_t> objects;
  // Vec descriptors retain their own geometry; all placed Vec bytes share one
  // physical domain. SSA roots, typed aliases and gaps do not create disjointness.
  struct VecView {
    int64_t address, pitch;
    BufferTypePtr type;
    // Byte translation in one original counted-loop frame. The maximum
    // displacement is proved against the parent view, not sampled at visit 0.
    int64_t step = 0, last_displacement = 0;
    ForStmtPtr frame;
  };
  std::unordered_map<const Expr*, VecView> vec_views;
  std::optional<size_t> vec_storage;
  struct SlotSelection {
    int64_t first, step;
    const ForStmt* frame;
  };
  std::unordered_map<const Expr*, SlotSelection> slots;
  ParameterIndices parameters;
  CoordinateGraph coordinates;
  std::set<size_t> translated_storage;
  AffineCoordinate Coordinate(const ExprPtr& e, const SyncSegment& segment) {
    RequireIndex(e);
    auto out = coordinates.Flatten(coordinates.Build(e, segment, parameters, result), result);
    for (auto [frame, coefficient] : out.frames)
      CHECK(frame == segment.original_loop.get() ||
            std::any_of(segment.nested_frames.begin(), segment.nested_frames.end(),
                        [&](const auto& f) { return f.loop.get() == frame; }))
          << "Coordinate definition belongs to another occurrence frame";
    if (auto found = out.frames.find(segment.original_loop.get()); found != out.frames.end()) {
      out.second = found->second;
      out.frames.erase(found);
    }
    return out;
  }
  void DefineCoordinate(const AssignStmtPtr& assign, const SyncSegment& segment) {
    CHECK(As<ScalarType>(assign->var_->GetType()))
        << "Nested or data-dependent storage/view definitions are outside this compact provider";
    CHECK(SyncCoordinateMaximum(assign->var_)) << "Unavailable scalar coordinate type";
    CHECK(!parameters.count(assign->var_.get()) && !coordinates.Defined(assign->var_) &&
          (!segment.original_loop || assign->var_ != segment.original_loop->loop_var_))
        << "Invocation and loop coordinates must retain their original single-assignment identities";
    coordinates.Define(assign->var_, assign->value_, segment, parameters, result);
  }
  void SelectSlot(const AssignStmtPtr& assign, const SyncSegment& segment) {
    auto call = As<Call>(assign->value_);
    OpRegistry::GetInstance().ValidateBufferCall(call);
    const auto id = Storage(call->args_[0]);
    const auto count = result.storage[id].slot_count;
    RequireIndex(call->args_[1]);
    ExprPtr index = call->args_[1];
    bool modulo = false;
    if (auto mod = As<FloorMod>(index)) {
      RequireIndex(mod->left_);
      RequireIndex(mod->right_);
      CHECK(Constant(mod->right_) == count) << "Slot modulus must equal the allocation slot count";
      index = mod->left_;
      modulo = true;
    }
    const auto coordinate = Coordinate(index, segment);
    CHECK(coordinate.parameters.empty() && coordinate.frames.empty() && !coordinate.block)
        << "Parameterized slot phase requires an explicit phase certificate";
    CHECK(modulo || (!coordinate.second && coordinate.first < count))
        << "Slot selection must be a constant in range or a proved nonnegative affine rotation";
    objects[assign->var_.get()] = id;
    slots[assign->var_.get()] = {coordinate.first % count, coordinate.second % count,
                                 segment.original_loop.get()};
  }
  void QualifySlot(const ExprPtr& handle, const SyncSegment& segment) const {
    if (auto slot = slots.find(handle.get()); slot != slots.end()) {
      CHECK(!slot->second.frame || slot->second.frame == segment.original_loop.get())
          << "Slot selection belongs to another occurrence frame";
    }
  }
  void ApplySlot(SyncAccess& access, const ExprPtr& handle, const SyncSegment& segment) const {
    QualifySlot(handle, segment);
    if (auto slot = slots.find(handle.get()); slot != slots.end()) {
      access.slot_first = slot->second.first;
      access.slot_step = slot->second.step;
    }
  }
  void RegisterVec(const VarPtr& var, VecView view) {
    const auto type = view.type;
    // A full-valid column vector [R,1] occupies consecutive elements in either
    // major order. This does not admit general column-major matrices or views.
    CHECK(type->shape_ == type->valid_shape_ &&
          (backend::DenseBufferBytes(type) || backend::IsColumnVectorBuffer(type)))
        << "Synchronization Vec views require static full-valid dense descriptors";
    const auto end =
        Checked(__int128(view.address) + view.last_displacement + __int128(type->shape_[0] - 1) * view.pitch +
                __int128(type->shape_[1]) * type->dtype_.GetByte());
    for (const auto& old : result.storage) {
      CHECK(old.space != MemorySpace::Vec || old.byte_coordinates ||
            view.address >= old.address + old.bytes || old.address >= end)
          << "Overlapping slot and ordinary Vec views require a common physical-layout partition";
    }
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
  void View(const AssignStmtPtr& assign, const CallPtr& call, const SyncSegment& segment = {}) {
    OpRegistry::GetInstance().ValidateBufferCall(call);
    auto source = vec_views.find(call->args_[0].get());
    CHECK(source != vec_views.end()) << "Synchronization storage/view needs a qualified Vec source";
    auto view = source->second;
    auto type = As<BufferType>(assign->var_->GetType());
    CHECK(type && type->shape_.size() == 2) << "Synchronization view requires rank two";
    if (IsOp(call, "buffer.subview")) {
      const auto offsets = As<MakeTuple>(call->args_[1]);
      const auto row = As<ConstInt>(offsets->elements_[0])
                           ? AffineCoordinate{Constant(offsets->elements_[0]), 0}
                           : Coordinate(offsets->elements_[0], segment);
      const auto column = As<ConstInt>(offsets->elements_[1])
                              ? AffineCoordinate{Constant(offsets->elements_[1]), 0}
                              : Coordinate(offsets->elements_[1], segment);
      CHECK(row.parameters.empty() && column.parameters.empty() && row.frames.empty() &&
            column.frames.empty() && !row.block && !column.block)
          << "Vec subview translation needs one original loop frame";
      CHECK(!view.frame || view.frame == segment.original_loop)
          << "Vec subview belongs to another occurrence frame";
      const auto last = std::max<int64_t>(0, segment.count - 1);
      CHECK(__int128(row.first) + __int128(last) * row.second + type->shape_[0] <= view.type->shape_[0] &&
            __int128(column.first) + __int128(last) * column.second + type->shape_[1] <= view.type->shape_[1])
          << "Vec subview exceeds its proved parent window";
      view.address = Checked(__int128(view.address) + __int128(row.first) * view.pitch +
                             __int128(column.first) * view.type->dtype_.GetByte());
      view.step = Checked(__int128(view.step) + __int128(row.second) * view.pitch +
                          __int128(column.second) * view.type->dtype_.GetByte());
      if (view.step) {
        CHECK(segment.original_loop) << "Moving Vec view has no original counted frame";
        view.frame = segment.original_loop;
        view.last_displacement = Checked(__int128(last) * view.step);
      }
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
    CHECK(c && (IsOp(c, "buffer.alloc") || IsOp(c, "buffer.alloc_multi")))
        << "Unavailable storage/view or scalar definition in synchronization import";
    OpRegistry::GetInstance().ValidateBufferCall(c);
    CHECK(c->args_.size() == 2) << "Synchronization import requires placed buffer allocation addresses";
    auto multi = As<MultiBufferType>(a->var_->GetType());
    auto t = multi ? multi->element_type_ : As<BufferType>(a->var_->GetType());
    const auto slot_count = multi ? multi->slot_count_ : 1;
    CHECK(t && t->shape_.size() == 2 && t->shape_ == t->valid_shape_ && t->dtype_.GetBit() % 8 == 0)
        << "Synchronization import needs fully valid byte-addressable rank-two buffers";
    CHECK(t->compact_ == CompactMode::null && t->pad_ == PadValue::null)
        << "Packed/padded buffer interpretation is not qualified";
    auto valid = As<MakeTuple>(c->args_[0]);
    CHECK(valid && valid->elements_.empty()) << "Dynamic valid extents are not qualified";
    const auto address = Constant(c->args_[1]);
    const auto slot_bytes = Checked(__int128(t->shape_[0]) * t->shape_[1] * t->dtype_.GetByte());
    if (multi && slot_count > 1) {
      // Native multi-tile lowering rounds each slot to the target alignment.
      // Until padded slot geometry is represented, accept only equal strides.
      const auto context = PassContext::Current();
      const auto* backend = context ? context->GetBackend() : backend::GetBackend();
      const auto alignment = backend->GetMemAlignment(t->memory_space_);
      CHECK(alignment && slot_bytes % alignment == 0 && address % alignment == 0)
          << "Multi-buffer placement requires aligned whole-slot geometry";
    }
    auto bytes = Checked(__int128(slot_bytes) * slot_count);
    Checked(__int128(address) + bytes);
    if (t->memory_space_ == MemorySpace::Vec && !multi) {
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
      CHECK(old.address == address && old.bytes == bytes && old.slot_count == slot_count && old.layout &&
            old.rows == t->shape_[0] && old.columns == t->shape_[1] && old.layout->dtype_ == t->dtype_ &&
            old.layout->blayout_ == t->blayout_ && old.layout->slayout_ == t->slayout_ &&
            old.layout->fractal_ == t->fractal_)
          << "Overlapping placed buffers need a common physical-layout partition";
      objects[a->var_.get()] = i;
      return;
    }
    objects[a->var_.get()] = result.storage.size();
    result.storage.push_back(
        {a->var_->name_hint_, t->memory_space_, address, t->shape_[0], t->shape_[1], bytes, t});
    result.storage.back().slot_count = slot_count;
    result.storage.back().slot_bytes = slot_bytes;
  }
  SyncAccess Access(size_t storage, bool write, AffineCoordinate row, AffineCoordinate column, int64_t rows,
                    int64_t columns, const SyncSegment& segment) {
    const auto& s = result.storage[storage];
    std::set<const ForStmt*> moving;
    for (const auto& [frame, coefficient] : row.frames)
      if (coefficient) moving.insert(frame);
    for (const auto& [frame, coefficient] : column.frames)
      if (coefficient) moving.insert(frame);
    if (row.second || column.second) moving.insert(segment.original_loop.get());
    CHECK(moving.size() <= 1) << "Coupled moving coordinates need a multidimensional selector certificate";
    ForStmtPtr coordinate_frame;
    int64_t count = segment.count;
    if (!moving.empty()) {
      const auto* selected = *moving.begin();
      if (selected == segment.original_loop.get()) coordinate_frame = segment.original_loop;
      for (const auto& frame : segment.nested_frames)
        if (frame.loop.get() == selected) {
          coordinate_frame = frame.loop;
          count = frame.count;
        }
      CHECK(coordinate_frame) << "Moving access has no original occurrence frame";
      if (auto found = row.frames.find(selected); found != row.frames.end()) {
        row.second = found->second;
      }
      if (auto found = column.frames.find(selected); found != column.frames.end()) {
        column.second = found->second;
      }
    }
    const auto last = std::max<int64_t>(0, count - 1);
    auto translation_bound = [&](const SyncTranslation& translation) {
      int64_t upper = 0;
      for (auto [p, coefficient] : translation)
        upper = Checked(__int128(upper) + __int128(coefficient) * result.parameter_bounds.at(p).second);
      return upper;
    };
    CHECK(rows > 0 && columns > 0 &&
          Checked(__int128(row.first) + __int128(last) * row.second + rows +
                  translation_bound(row.parameters) +
                  __int128(row.block) *
                      (result.block_bounds ? result.block_bounds->second - result.block_bounds->first : 0)) <=
              (s.row_extent_parameter ? result.parameter_bounds.at(*s.row_extent_parameter).first : s.rows) &&
          Checked(__int128(column.first) + __int128(last) * column.second + columns +
                  translation_bound(column.parameters) +
                  __int128(column.block) *
                      (result.block_bounds ? result.block_bounds->second - result.block_bounds->first : 0)) <=
              s.columns)
        << "Synchronization access exceeds its proved allocation";
    SyncAccess access{storage,       write, row.first, column.first, row.second,
                      column.second, rows,  columns,   false};
    access.coordinate_frame = coordinate_frame;
    return access;
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
        CHECK(!view.frame || view.frame == segment.original_loop)
            << "Vec view use belongs to another occurrence frame";
        if (!recover_accesses) return;
        const auto width = Checked(__int128(view.type->shape_[1]) * view.type->dtype_.GetByte());
        // Dense footprints use one interval. Strided views retain one interval
        // per physical row, never an enclosing rectangle that fills the gaps.
        if (view.pitch == width) {
          site.accesses.push_back(Access(id, write, {0, 0}, {view.address, view.step}, 1,
                                         Checked(__int128(width) * view.type->shape_[0]), segment));
        } else {
          for (int64_t row = 0; row < view.type->shape_[0]; ++row)
            site.accesses.push_back(Access(
                id, write, {0, 0}, {Checked(__int128(view.address) + __int128(row) * view.pitch), view.step},
                1, width, segment));
        }
        return;
      }
      if (!recover_accesses) {
        QualifySlot(call->args_[arg], segment);
        return;
      }
      const auto& s = result.storage[id];
      site.accesses.push_back(Access(id, write, {0, 0}, {0, 0}, s.rows, s.columns, segment));
      ApplySlot(site.accesses.back(), call->args_[arg], segment);
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
      // Coordinate readiness and checked arithmetic remain native admission
      // evidence even when no dependence footprint is requested.
      const auto row = Coordinate(offsets->elements_[0], segment);
      const auto column = Coordinate(offsets->elements_[1], segment);
      if (recover_accesses) {
        auto id = Storage(call->args_[load ? 0 : 3]);
        auto& storage = result.storage[id];
        if (translated_storage.insert(id).second) {
          storage.row_parameters = row.parameters;
          storage.column_parameters = column.parameters;
          storage.row_block = row.block;
          storage.column_block = column.block;
        } else {
          CHECK(storage.row_parameters == row.parameters && storage.column_parameters == column.parameters &&
                storage.row_block == row.block && storage.column_block == column.block)
              << "GM accesses need a common invocation translation; parameter-dependent overlap is unproved";
        }
        site.accesses.push_back(Access(id, !load, row, column, Constant(shape->elements_[0]),
                                       Constant(shape->elements_[1]), segment));
      }
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
      auto row = extract ? Coordinate(call->args_[1], segment) : AffineCoordinate{};
      auto column = extract ? Coordinate(call->args_[2], segment) : AffineCoordinate{};
      CHECK(row.parameters.empty() && column.parameters.empty() && !row.block && !column.block)
          << "Parameterized local extraction needs a separate membership and alignment certificate";
      if (!extract) CHECK(s.rows == t.rows && s.columns == t.columns) << "Partial copy is not qualified";
      if (s.layout->slayout_ != TileLayout::none_box) {
        const auto box = s.layout->fractal_ / (16 * s.layout->dtype_.GetByte());
        CHECK(row.first % 16 == 0 && row.second % 16 == 0 && t.rows % 16 == 0 && column.first % box == 0 &&
              column.second % box == 0 && t.columns % box == 0)
            << "Unaligned boxed extraction requires a separate footprint contract";
      }
      auto source_access = Access(sid, false, row, column, t.rows, t.columns, segment);
      CHECK(!extract || s.slot_count == 1 || PhysicalInterval(s, source_access))
          << "Rotating-slot extraction requires an exact contiguous within-slot interval";
      ApplySlot(source_access, call->args_[0], segment);
      if (recover_accesses) {
        site.accesses.push_back(std::move(source_access));
      }
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
        CHECK(result.storage[Storage(call->args_[2])].slot_count == 1)
            << "Rotating accumulator protection requires same-physical-slot qualification";
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
      if (auto assign = As<AssignStmt>(s)) {
        auto call = As<Call>(assign->value_);
        if (call && IsOp(call, "buffer.slot"))
          SelectSlot(assign, segment);
        else if (call && (IsOp(call, "buffer.subview") || IsOp(call, "buffer.reshape")))
          View(assign, call, segment);
        else
          DefineCoordinate(assign, segment);
      } else if (auto eval = As<EvalStmt>(s)) {
        auto call = As<Call>(eval->expr_);
        CHECK(call) << "Submit or non-call payload is not qualified";
        segment.sites.push_back(Site(call, segment));
        preceding_matrix |= segment.sites.back().pipe == PipeType::M;
      } else if (auto branch = As<IfStmt>(s)) {
        // Admit exactly a suffix of first-visit frame tests. Testing only an
        // outer coordinate would reinitialize on every untested inner visit.
        std::map<const Expr*, int64_t> first_visits;
        std::function<bool(const ExprPtr&)> collect = [&](const ExprPtr& condition) {
          if (auto both = As<And>(condition)) return collect(both->left_) && collect(both->right_);
          auto eq = As<Eq>(condition);
          auto value = eq ? As<ConstInt>(eq->right_) : nullptr;
          return eq && value && first_visits.emplace(eq->left_.get(), value->value_).second;
        };
        CHECK(segment.original_loop && collect(branch->condition_) && branch->else_body_ &&
              branch->return_vars_.empty())
            << "Only first-visit initialization/continuation selection is qualified";
        auto frames = segment.nested_frames;
        if (frames.empty())
          frames.push_back({segment.original_loop, segment.count, segment.start, segment.step});
        CHECK(!first_visits.empty() && first_visits.size() <= frames.size())
            << "Initialization guard has an unavailable occurrence frame";
        // The current shared boundary view changes one enclosing repeat seam.
        // More frames need continuation views at every intermediate seam, not
        // just at the outermost K reset; retain explicit unavailability.
        CHECK(first_visits.size() <= 2)
            << "Initialization across intermediate repetitions lacks continuation seam evidence";
        const size_t first_frame = frames.size() - first_visits.size();
        for (size_t f = first_frame; f < frames.size(); ++f) {
          CHECK(!frames[f].start_expression)
              << "Symbolic first-visit initialization needs its exact activation predicate";
          auto found = first_visits.find(frames[f].loop->loop_var_.get());
          CHECK(found != first_visits.end() && found->second == frames[f].start)
              << "Initialization guard must test consecutive first-visit frames";
        }
        auto first = SingleCall(branch->then_body_), later = SingleCall(*branch->else_body_);
        CHECK(IsOp(first, "buffer.matmul") && IsOp(later, "buffer.matmul_acc") &&
              first->args_ == later->args_)
            << "Conditional payloads must preserve operands and destination";
        auto site = Site(first, segment);
        auto continuation = Site(later, segment);
        site.continuation = later;
        if (frames[first_frame].loop != segment.original_loop)
          site.initialization_scope = frames[first_frame].loop;
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
// Discover source shape in one postorder traversal. Nested loops retain their
// original sequence/repetition tree; flat constant loops use the leaf path.
std::pair<bool, bool> NestedShape(const StmtPtr& statement, std::map<const ForStmt*, bool>& nested) {
  if (!statement) return {false, false};
  if (auto loop = As<ForStmt>(statement)) {
    const auto child = NestedShape(loop->body_, nested);
    nested.emplace(loop.get(), child.first);
    // Invocation-varying starts need the source-frame-aware numerical path,
    // including a nonnested zero/one loop after analytic case qualification.
    return {true, !As<ConstInt>(loop->start_) || child.second || child.first};
  }
  std::pair<bool, bool> result{};
  auto join = [&](const StmtPtr& child) {
    const auto shape = NestedShape(child, nested);
    result.first |= shape.first;
    result.second |= shape.second;
  };
  if (auto sequence = As<SeqStmts>(statement)) {
    for (const auto& child : sequence->stmts_) join(child);
  } else if (auto branch = As<IfStmt>(statement)) {
    join(branch->then_body_);
    if (branch->else_body_) join(*branch->else_body_);
  }
  return result;
}
SyncLoopFrame NumericalFrame(Importer& importer, const ForStmtPtr& loop,
                             const std::vector<SyncLoopFrame>& parents) {
  RequireIndex(loop->loop_var_);
  RequireIndex(loop->start_);
  RequireIndex(loop->stop_);
  RequireIndex(loop->step_);
  CHECK(loop->kind_ == ForKind::Sequential && loop->iter_args_.empty() && loop->return_vars_.empty())
      << "Structured synchronization requires coordinate-only sequential loops";
  const auto stop = Constant(loop->stop_), step = Constant(loop->step_);
  CHECK(step > 0) << "Nonpositive structured loop step";
  if (!As<ConstInt>(loop->start_)) {
    SyncSegment context;
    context.nested_frames = parents;
    const auto start = importer.Coordinate(loop->start_, context);
    CHECK(start.frames.empty() && !start.second) << "Structured loop start is not invocation invariant";
    __int128 lo = start.first, hi = start.first;
    for (const auto& [p, coefficient] : start.parameters) {
      lo += __int128(coefficient) * importer.result.parameter_bounds.at(p).first;
      hi += __int128(coefficient) * importer.result.parameter_bounds.at(p).second;
    }
    if (start.block) {
      INTERNAL_CHECK(importer.result.block_bounds) << "Internal error: missing launch proof";
      hi += __int128(start.block) *
            (importer.result.block_bounds->second - importer.result.block_bounds->first);
    }
    // No sampling: every admitted start executes exactly once. Optional visits
    // require an absence-aware parent and are not converted to count one.
    if (lo >= stop) return {loop, 0, 0, step, loop->start_};
    CHECK(lo >= 0 && hi < stop && __int128(stop) - lo <= step)
        << "Optional or varying outer visits need a guarded regional activation interface";
    Checked(hi + step);
    return {loop, 1, 0, step, loop->start_};
  }
  const auto start = Constant(loop->start_);
  const auto count = stop <= start ? 0 : 1 + (stop - start - 1) / step;
  Checked(__int128(start) + __int128(count) * step);
  return {loop, count, start, step};
}
SyncRegion ImportStructuredBody(Importer& importer, const StmtPtr& statement,
                                std::vector<SyncLoopFrame>& parents,
                                const std::map<const ForStmt*, bool>& nested) {
  SyncRegion sequence;
  SyncSegment finite;
  finite.nested_frames = parents;
  auto add = [&](SyncSegment segment) {
    const auto local_count = segment.count;
    segment.count = 1;
    for (const auto& frame : parents) segment.count = Checked(__int128(segment.count) * frame.count);
    segment.nested_frames = parents;
    if (segment.original_loop) segment.reset_period = local_count;
    SyncRegion leaf;
    leaf.leaf = importer.result.segments.size();
    importer.result.segments.push_back(std::move(segment));
    sequence.children.push_back(std::move(leaf));
  };
  auto flush = [&]() {
    if (!finite.sites.empty()) {
      add(std::move(finite));
      finite = SyncSegment{};
      finite.nested_frames = parents;
    }
  };
  for (const auto& child : Statements(statement)) {
    if (auto loop = As<ForStmt>(child)) {
      flush();
      const auto frame = NumericalFrame(importer, loop, parents);
      if (!frame.count && frame.start_expression) {
        // This complete case proves the original loop absent. Keep its frame,
        // but do not fabricate payload effects for unexecuted operations.
        SyncRegion absent;
        absent.repetition = frame;
        sequence.children.push_back(std::move(absent));
        continue;
      }
      parents.push_back(frame);
      if (nested.at(loop.get())) {
        auto region = ImportStructuredBody(importer, loop->body_, parents, nested);
        region.repetition = frame;
        sequence.children.push_back(std::move(region));
      } else {
        SyncSegment segment;
        segment.original_loop = loop;
        segment.start = frame.start;
        segment.step = frame.step;
        segment.count = frame.count;
        segment.nested_frames = parents;
        importer.Body(loop->body_, segment);
        add(std::move(segment));
      }
      parents.pop_back();
    } else if (auto assign = As<AssignStmt>(child)) {
      if (As<ScalarType>(assign->var_->GetType()))
        importer.DefineCoordinate(assign, finite);
      else
        importer.Allocate(assign);
    } else if (!As<YieldStmt>(child) && !As<ReturnStmt>(child)) {
      importer.Body(child, finite);
    }
  }
  flush();
  return sequence;
}
// Qualify startup at its actual source scope, not at a sampled leaf visit.
// The only changed effect between startup and continuation is the first
// accumulator read. No earlier M or access through a physical alias is allowed.
// Summary maps are proportional to supplied source interfaces, not trip counts.
struct StartupSummary {
  using Site = std::pair<size_t, size_t>;
  std::optional<Site> first_matrix;
  std::map<size_t, Site> first_access;
  std::map<const ForStmt*, Site> pending;
};
StartupSummary QualifyStartupScopes(SyncRegion& region, const CompactSyncInput& input) {
  StartupSummary out;
  if (region.leaf) {
    const auto& segment = input.segments[*region.leaf];
    const bool live = !segment.original_loop || segment.nested_frames.back().count != 0;
    for (size_t s = 0; s < segment.sites.size(); ++s) {
      const auto& site = segment.sites[s];
      const auto position = std::make_pair(*region.leaf, s);
      if (live && site.pipe == PipeType::M && !out.first_matrix) out.first_matrix = position;
      if (live)
        for (const auto& access : site.accesses) out.first_access.emplace(access.storage, position);
      if (site.initialization_scope)
        CHECK(out.pending.emplace(site.initialization_scope.get(), position).second)
            << "Multiple initializers in one named scope are unavailable";
    }
  } else {
    for (auto& child : region.children) {
      auto next = QualifyStartupScopes(child, input);
      if (!out.first_matrix) out.first_matrix = next.first_matrix;
      out.first_access.insert(next.first_access.begin(), next.first_access.end());
      for (const auto& [scope, site] : next.pending)
        CHECK(out.pending.emplace(scope, site).second)
            << "Multiple initializers in one named scope are unavailable";
    }
  }
  if (region.repetition) {
    auto found = out.pending.find(region.repetition->loop.get());
    if (found != out.pending.end()) {
      const auto [leaf, index] = found->second;
      const auto& site = input.segments[leaf].sites[index];
      if (out.first_matrix) {
        CHECK(site.protected_accumulator && out.first_matrix == found->second &&
              out.first_access.at(*site.protected_accumulator) == found->second)
            << "Scoped initialization requires first-touch qualified accumulator evidence";
        region.startup = found->second;
      }
      out.pending.erase(found);
    }
    // An empty descendant cannot supply an ancestor's entry evidence. Keep
    // pending scope names for validation, but discard its unexecuted effects.
    if (!region.repetition->count) {
      out.first_matrix.reset();
      out.first_access.clear();
    }
  }
  return out;
}
// Reuse PyPTO's bound environment, but qualify integer operations before
// accepting mathematical bounds as machine-index evidence. Saturation, unknown
// parameters and data-loaded values remain unproved. No branch is sampled.
class InvocationBounds : public IRVisitor {
  using Bound = std::pair<int64_t, int64_t>;
  arith::ConstIntBoundAnalyzer analyzer_;
  const std::map<std::string, FunctionPtr>& functions_;
  std::map<std::string, std::map<size_t, std::optional<Bound>>>& evidence_;
  std::map<std::string, std::optional<Bound>>& launches_;
  FunctionPtr caller_;
  std::optional<Bound> block_index_;
  SyncParameterWork& work_;
  std::unordered_map<const Expr*, std::optional<Bound>> ranges_;
  std::set<const Expr*> visited_;
  // Expressions may be shared across scopes; cached unknowns and successes
  // both belong to the current bound environment, never to the entire program.
  void Invalidate() {
    ranges_.clear();
    visited_.clear();
  }
  std::optional<Bound> Range(const ExprPtr& expression) {
    if (auto found = ranges_.find(expression.get()); found != ranges_.end()) {
      return found->second;
    }
    ++work_.range_visits;
    auto result = ComputeRange(expression);
    ranges_.emplace(expression.get(), result);
    return result;
  }
  std::optional<Bound> ComputeRange(const ExprPtr& expression) {
    const auto maximum = SyncCoordinateMaximum(expression);
    if (!maximum) return std::nullopt;
    if (auto call = As<Call>(expression);
        call && (IsOp(call, "buffer.get_block_idx") || IsOp(call, "tile.get_block_idx"))) {
      return block_index_;
    }
    if (auto cast = As<Cast>(expression)) {
      auto source = Range(cast->operand_);
      return source && source->second <= *maximum ? source : std::nullopt;
    }
    if (As<Var>(expression) || As<ConstInt>(expression)) {
      const auto b = analyzer_(expression);
      if (b.min_value < 0 || b.max_value >= arith::ConstIntBound::kPosInf || b.max_value > *maximum)
        return std::nullopt;
      return Bound{b.min_value, b.max_value};
    }
    ExprPtr left, right;
    bool multiply = false, modulo = false;
    if (auto add = As<Add>(expression)) {
      left = add->left_;
      right = add->right_;
    } else if (auto mul = As<Mul>(expression)) {
      left = mul->left_;
      right = mul->right_;
      multiply = true;
    } else if (auto mod = As<FloorMod>(expression)) {
      left = mod->left_;
      right = mod->right_;
      modulo = true;
    } else {
      return std::nullopt;
    }
    if (!SameCoordinateType(expression, left) || !SameCoordinateType(expression, right)) return std::nullopt;
    auto a = Range(left), b = Range(right);
    if (!a || !b) return std::nullopt;
    // Only a proved positive constant divisor is admitted. Both operands were
    // checked in their original machine types before reducing the range.
    if (modulo) {
      if (b->first <= 0 || b->first != b->second) return std::nullopt;
      return a->second < b->first ? a : std::optional<Bound>{{0, b->first - 1}};
    }
    const __int128 lo = multiply ? __int128(a->first) * b->first : __int128(a->first) + b->first;
    const __int128 hi = multiply ? __int128(a->second) * b->second : __int128(a->second) + b->second;
    if (hi >= arith::ConstIntBound::kPosInf || hi > *maximum) return std::nullopt;
    return Bound{static_cast<int64_t>(lo), static_cast<int64_t>(hi)};
  }
  void RecordCall(const CallPtr& call) {
    if (auto global = As<GlobalVar>(call->op_)) {
      auto found = functions_.find(global->name_);
      if (found != functions_.end()) {
        std::optional<Bound> launch;
        if (found->second->func_type_ == FunctionType::Spmd) {
          auto count = call->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
          if (count) {
            auto bound = Range(count);
            if (bound && bound->first > 0) launch = Bound{0, bound->second - 1};
          } else if (auto constant = As<ConstInt>(found->second->GetAttr<ExprPtr>(kAttrCoreNum, nullptr));
                     constant && constant->value_ > 0) {
            launch = Bound{0, constant->value_ - 1};
          }
        } else if (caller_->func_type_ == FunctionType::Spmd &&
                   found->second->func_type_ == FunctionType::AIC) {
          // An outlined AIC worker inherits its wrapper's logical block index.
          // Other call/core kinds require their own dispatch interpretation.
          launch = block_index_;
        }
        auto [launch_it, first_launch] = launches_.emplace(global->name_, launch);
        if (!first_launch) {
          if (!launch_it->second || !launch)
            launch_it->second = std::nullopt;
          else
            launch_it->second->second = std::max(launch_it->second->second, launch->second);
        }
        const auto& params = found->second->params_;
        auto& callee = evidence_[global->name_];
        for (size_t i = 0; i < params.size(); ++i) {
          if (!As<ScalarType>(params[i]->GetType())) continue;
          auto argument = transform_utils::CallerSuppliedArg(call, found->second, i);
          auto bound = argument ? Range(argument) : std::nullopt;
          auto [it, inserted] = callee.emplace(i, bound);
          if (!inserted) {
            if (!it->second || !bound) {
              it->second = std::nullopt;
            } else {
              it->second->first = std::min(it->second->first, bound->first);
              it->second->second = std::max(it->second->second, bound->second);
            }
          }
        }
      }
    }
  }

 public:
  InvocationBounds(const std::map<std::string, FunctionPtr>& functions,
                   std::map<std::string, std::map<size_t, std::optional<Bound>>>& evidence,
                   std::map<std::string, std::optional<Bound>>& launches, FunctionPtr caller,
                   SyncParameterWork& work)
      : functions_(functions),
        evidence_(evidence),
        launches_(launches),
        caller_(std::move(caller)),
        work_(work) {
    // Every predecessor has been processed before this environment is built.
    // Missing/unknown formal bounds are not inferred from a sample invocation.
    if (auto incoming = evidence.find(caller_->name_); incoming != evidence.end())
      for (const auto& [index, bound] : incoming->second)
        if (bound) analyzer_.Update(caller_->params_.at(index), {bound->first, bound->second});
    if (auto launch = launches.find(caller_->name_); launch != launches.end()) {
      block_index_ = launch->second;
    }
  }
  void VisitExpr(const ExprPtr& expression) override {
    if (expression && visited_.insert(expression.get()).second) {
      ++work_.expression_visits;
      IRVisitor::VisitExpr(expression);
    }
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const CallPtr& call) override {
    RecordCall(call);
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& call) override {
    RecordCall(SubmitToCallView(call));
    IRVisitor::VisitExpr_(call);
  }
  void VisitStmt_(const AssignStmtPtr& statement) override {
    VisitExpr(statement->value_);
    auto b = Range(statement->value_);
    if (b && SameCoordinateType(statement->var_, statement->value_))
      analyzer_.Update(statement->var_, {b->first, b->second});
    else
      analyzer_.Unbind(statement->var_);
    Invalidate();
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    // Calls in bounds and initial values execute outside the iteration domain.
    // Missing one such site could incorrectly retain a narrower callee range.
    VisitExpr(loop->start_);
    VisitExpr(loop->stop_);
    VisitExpr(loop->step_);
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    auto start = Range(loop->start_), stop = Range(loop->stop_), step = Range(loop->step_);
    const auto maximum = SyncCoordinateMaximum(loop->loop_var_);
    if (maximum && start && stop && step && step->first > 0 && stop->second > start->first &&
        __int128(stop->second) - 1 + step->second < arith::ConstIntBound::kPosInf &&
        __int128(stop->second) - 1 + step->second <= *maximum) {
      int64_t last = stop->second - 1;
      if (start->first == start->second && stop->first == stop->second && step->first == step->second)
        last = start->first + (stop->first - start->first - 1) / step->first * step->first;
      analyzer_.Update(loop->loop_var_, {start->first, last});
    } else {
      analyzer_.Unbind(loop->loop_var_);
    }
    // Inclusive hull is deliberately conservative for nonunit strides. Calls
    // in empty/unknown loops do not manufacture vacuous range certificates.
    Invalidate();
    VisitStmt(loop->body_);
    analyzer_.Unbind(loop->loop_var_);
    Invalidate();
  }
};
// Index edges once, including Submit's original call ABI. Kahn traversal joins
// all predecessors before inspecting a function; no whole-program fixpoint or
// visit expansion is needed. Cycles and their descendants are left unproved.
class InvocationCallGraph : public IRVisitor {
  std::set<const Expr*> visited_;
  void Record(const OpPtr& op) {
    if (auto global = As<GlobalVar>(op)) callees.insert(global->name_);
  }

 public:
  std::set<std::string> callees;
  size_t expression_visits = 0;
  void VisitExpr(const ExprPtr& expression) override {
    if (expression && visited_.insert(expression.get()).second) {
      ++expression_visits;
      IRVisitor::VisitExpr(expression);
    }
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const CallPtr& call) override {
    Record(call->op_);
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& call) override {
    Record(call->op_);
    IRVisitor::VisitExpr_(call);
  }
};
}  // namespace
std::optional<int64_t> SyncCoordinateMaximum(const ExprPtr& e) {
  auto type = As<ScalarType>(e->GetType());
  if (!type || (type->dtype_ != DataType::INDEX && !type->dtype_.IsInt())) return std::nullopt;
  const auto bits = type->dtype_ == DataType::INDEX ? 64 : type->dtype_.GetBit();
  if (!bits || bits > 64) return std::nullopt;
  if (bits == 64) return std::numeric_limits<int64_t>::max();
  return static_cast<int64_t>((uint64_t{1} << (bits - type->dtype_.IsSignedInt())) - 1);
}
SyncInvocationEvidence CollectSyncInvocationEvidence(const ProgramPtr& program, SyncParameterWork* work) {
  SyncParameterWork local_work;
  if (!work) work = &local_work;
  std::map<std::string, FunctionPtr> functions;
  for (const auto& [global, function] : program->functions_) functions.emplace(global->name_, function);
  std::map<std::string, std::map<size_t, std::optional<std::pair<int64_t, int64_t>>>> evidence;
  std::map<std::string, std::optional<std::pair<int64_t, int64_t>>> launches;
  std::map<std::string, std::set<std::string>> edges;
  std::map<std::string, size_t> incoming;
  for (const auto& [name, function] : functions) incoming.emplace(name, 0);
  for (const auto& [name, function] : functions) {
    InvocationCallGraph graph;
    graph.VisitFunction(function);
    work->call_graph_expression_visits += graph.expression_visits;
    ++work->call_graph_nodes;
    for (const auto& callee : graph.callees)
      if (functions.count(callee)) {
        edges[name].insert(callee);
        ++incoming.at(callee);
        ++work->call_graph_edges;
      }
  }
  std::set<std::string> ready, processed;
  for (const auto& [name, count] : incoming)
    if (!count) ready.insert(name);
  while (!ready.empty()) {
    const auto name = *ready.begin();
    ready.erase(ready.begin());
    InvocationBounds collector(functions, evidence, launches, functions.at(name), *work);
    collector.VisitFunction(functions.at(name));
    processed.insert(name);
    ++work->processed_functions;
    for (const auto& callee : edges[name])
      if (!--incoming.at(callee)) ready.insert(callee);
  }
  SyncInvocationEvidence result;
  for (const auto& [name, params] : evidence)
    for (const auto& [index, bound] : params)
      if (bound && processed.count(name)) result.parameters[name].emplace(index, *bound);
  for (const auto& [name, bound] : launches)
    if (bound && processed.count(name)) result.block_indices.emplace(name, *bound);
  return result;
}
// Qualify only already recovered exact geometry. This is one pass over static
// access descriptions: a noncontiguous rectangle is not expanded into rows and
// no enclosing interval is substituted for its footprint. Coordinates are
// relative to the canonical storage identity (including the Vec byte domain).
static void QualifyPhysicalIntervals(CompactSyncInput& input) {
  for (auto& segment : input.segments) {
    for (auto& site : segment.sites) {
      for (auto& access : site.accesses) {
        ++input.interval_access_visits;
        const auto& storage = input.storage.at(access.storage);
        access.physical_interval = PhysicalInterval(storage, access);
        if (access.physical_interval) ++input.qualified_intervals;
      }
    }
  }
}

static CompactSyncInput ImportQualifiedSync(const FunctionPtr& function, const std::string& target,
                                            const SyncParameterBounds& parameter_bounds,
                                            const std::optional<std::pair<int64_t, int64_t>>& block_bounds,
                                            bool recover_accesses) {
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
  Importer importer(recover_accesses);
  importer.result.owner = function;
  importer.result.hardware_target = target;
  importer.result.parameter_bounds = parameter_bounds;
  if (block_bounds)
    CHECK(block_bounds->first >= 0 && block_bounds->second >= block_bounds->first &&
          block_bounds->second < std::numeric_limits<int64_t>::max())
        << "Invalid logical launch range";
  importer.result.block_bounds = block_bounds;
  for (const auto& [index, range] : parameter_bounds) {
    CHECK(index < function->params_.size() && As<ScalarType>(function->params_[index]->GetType()))
        << "Invocation range must identify an original scalar parameter";
    const auto maximum = SyncCoordinateMaximum(function->params_[index]);
    CHECK(maximum && range.first >= 0 && range.second >= range.first &&
          range.second < std::numeric_limits<int64_t>::max() && range.second <= *maximum)
        << "Invalid invocation coordinate range";
  }
  // Discover scalar identities before tensor shapes, regardless of ABI order.
  for (size_t index = 0; index < function->params_.size(); ++index) {
    const auto& param = function->params_[index];
    if (As<ScalarType>(param->GetType())) {
      if (recover_accesses) {
        CHECK(SyncCoordinateMaximum(param)) << "Unavailable invocation coordinate type";
        CHECK(parameter_bounds.count(index)) << "Invocation coordinate has no established caller range";
      }
      importer.parameters.emplace(param.get(), index);
    }
  }
  for (const auto& param : function->params_) {
    if (!recover_accesses) break;  // GM allocation geometry is not token-order evidence.
    if (As<ScalarType>(param->GetType())) continue;
    auto t = As<TensorType>(param->GetType());
    CHECK(t && t->shape_.size() == 2 && t->dtype_.GetBit() % 8 == 0 && !t->tensor_view_)
        << "Synchronization prototype requires unviewed ND tensor arguments";
    std::optional<size_t> row_parameter;
    int64_t rows;
    if (As<ConstInt>(t->shape_[0])) {
      rows = Constant(t->shape_[0]);
    } else {
      RequireIndex(t->shape_[0]);
      auto bound = importer.parameters.find(t->shape_[0].get());
      CHECK(bound != importer.parameters.end())
          << "Dynamic allocation extent has no established caller range";
      row_parameter = bound->second;
      rows = parameter_bounds.at(*row_parameter).second;
    }
    auto cols = Constant(t->shape_[1]);
    importer.objects[param.get()] = importer.result.storage.size();
    importer.result.storage.push_back({param->name_hint_, MemorySpace::DDR, 0, rows, cols,
                                       Checked(__int128(rows) * cols * t->dtype_.GetByte()), nullptr});
    importer.result.storage.back().row_extent_parameter = row_parameter;
  }
  std::map<const ForStmt*, bool> nested;
  const auto shape = NestedShape(function->body_, nested);
  if (shape.second) {
    std::vector<SyncLoopFrame> frames;
    importer.result.region_tree = ImportStructuredBody(importer, function->body_, frames, nested);
    if (recover_accesses) {
      CHECK(QualifyStartupScopes(*importer.result.region_tree, importer.result).pending.empty())
          << "Initialization scope is outside the retained source tree";
      QualifyPhysicalIntervals(importer.result);
    }
    return std::move(importer.result);
  }
  for (const auto& stmt : Statements(function->body_)) {
    if (auto a = As<AssignStmt>(stmt)) {
      if (auto call = As<Call>(a->value_); call && IsOp(call, "buffer.slot")) {
        importer.SelectSlot(a, SyncSegment{});
      } else if (As<ScalarType>(a->var_->GetType())) {
        importer.DefineCoordinate(a, SyncSegment{});
      } else {
        importer.Allocate(a);
      }
      continue;
    }
    if (As<ReturnStmt>(stmt)) continue;
    if (auto loop = As<ForStmt>(stmt)) {
      SyncSegment s;
      RequireIndex(loop->loop_var_);
      RequireIndex(loop->start_);
      RequireIndex(loop->stop_);
      RequireIndex(loop->step_);
      CHECK(loop->kind_ == ForKind::Sequential && loop->iter_args_.empty() && loop->return_vars_.empty())
          << "Only sequential coordinate-only loops are qualified";
      const auto start = Constant(loop->start_), step = Constant(loop->step_), stop = Constant(loop->stop_);
      CHECK(step > 0) << "Nonpositive loop step is not qualified";
      const auto count = stop <= start ? 0 : 1 + (stop - start - 1) / step;
      Checked(__int128(start) + __int128(count) * step);
      s.original_loop = loop;
      s.start = start;
      s.step = step;
      s.count = count;
      importer.Body(loop->body_, s);
      importer.result.segments.push_back(std::move(s));
    } else {
      // Keep maximal adjacent finite statements in one numerical region.
      if (importer.result.segments.empty() || importer.result.segments.back().original_loop)
        importer.result.segments.emplace_back();
      importer.Body(stmt, importer.result.segments.back());
    }
  }
  // Each leaf has one permutation of every slot family. Different offsets
  // are permitted, but different strides do not justify a period-one graph.
  for (const auto& segment : importer.result.segments) {
    std::map<size_t, int64_t> strides;
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses) {
        if (importer.result.storage[access.storage].slot_count == 1) continue;
        auto [found, inserted] = strides.emplace(access.storage, access.slot_step);
        CHECK(inserted || found->second == access.slot_step)
            << "Rotating family requires one common stride within a region";
      }
  }
  if (recover_accesses) QualifyPhysicalIntervals(importer.result);
  return std::move(importer.result);
}
CompactSyncInput ImportPlacedSync(const FunctionPtr& function, bool disjoint_gm, const std::string& target,
                                  const SyncParameterBounds& parameters,
                                  const std::optional<std::pair<int64_t, int64_t>>& blocks) {
  CHECK(disjoint_gm)
      << "Disjoint GM argument contract must be explicitly supplied; may-alias import unavailable";
  return ImportQualifiedSync(function, target, parameters, blocks, true);
}
CompactSyncInput ImportSyncPayloads(const FunctionPtr& function, const std::string& target,
                                    const SyncParameterBounds& parameters,
                                    const std::optional<std::pair<int64_t, int64_t>>& blocks) {
  return ImportQualifiedSync(function, target, parameters, blocks, false);
}
}  // namespace pypto::ir::sync
