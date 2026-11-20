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

#include "src/ir/transforms/utils/peer_local_effects.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/buffer_runtime_valid_analysis.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "src/ir/transforms/utils/sync_footprint_resolver.h"
#include "src/ir/transforms/utils/sync_loop_trip_proof.h"
#include "src/ir/transforms/utils/sync_scalar_readiness.h"

namespace pypto::ir::sync {
namespace {
using cross_core_pipe::PeerTransportEffects;
using cross_core_pipe::PeerTransportKind;
using cross_core_pipe::PeerTransportSite;
using transform_utils::GetCallFromStmt;

// Project the original Tile operations onto their native payload contracts.
// Runtime validity needs a scalar proof at the original cut, while each data
// effect conservatively encloses the whole placed allocation.

bool HasRuntimeExtents(const std::vector<ExprPtr>& extents) {
  return std::any_of(extents.begin(), extents.end(),
                     [](const auto& extent) { return !As<ConstInt>(extent); });
}

struct TileWindow {
  MemorySpace space;
  int64_t begin;
  int64_t end;
  // A view inherits its original borrow and its lifetime.
  const PeerTransportSite* pop = nullptr;
  // Ordinary alloc-backed handles may be updated in their defining segment.
  const Var* mutable_handle = nullptr;
  size_t definition_segment = 0;
};

std::vector<int64_t> StaticExtents(const std::vector<ExprPtr>& expressions, const Span& span) {
  std::vector<int64_t> result;
  for (const auto& expression : expressions) {
    const auto constant = As<ConstInt>(expression);
    CHECK_SPAN(constant && constant->value_ >= 0, span)
        << "Peer local planning requires bounded static tile geometry";
    result.push_back(constant->value_);
  }
  return result;
}

BufferTypePtr DescribeTile(const TileTypePtr& tile, const Span& span) {
  CHECK_SPAN(tile && tile->memory_space_ && tile->shape_.size() == 2 && tile->dtype_.GetBit() % 8 == 0, span)
      << "Peer local planning requires a byte-addressable rank-two Tile descriptor";
  const auto space = *tile->memory_space_;
  const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
  const auto shape = StaticExtents(tile->shape_, span);
  CHECK_SPAN(shape[0] > 0 && shape[1] > 0 && view.valid_shape.size() == 2, span)
      << "Peer local planning requires a two-dimensional tile valid shape and positive capacity";
  const bool runtime = HasRuntimeExtents(view.valid_shape);
  CHECK_SPAN(!runtime || space == MemorySpace::Vec, span)
      << "Peer runtime valid extents require ordinary Vec storage";
  const auto valid =
      runtime ? backend::MutableBufferValidShape(view.valid_shape) : StaticExtents(view.valid_shape, span);
  CHECK_SPAN(valid[0] <= shape[0] && valid[1] <= shape[1], span)
      << "Peer local tile valid extent exceeds its capacity";
  if (space == MemorySpace::Vec) {
    CHECK_SPAN(backend::PhysicalLogicalVecDescriptor(tile, /*allow_full_zero_pad=*/true), span)
        << "Peer local planning requires an ordinary contiguous Vec descriptor";
  } else {
    const auto box = tile_view_semantics::GetBoxedTileAlignment(view, tile->dtype_);
    const auto layout = tile_view_semantics::GetImplicitTileLayout(tile->shape_, space);
    const auto start = As<ConstInt>(view.start_offset);
    CHECK_SPAN((space == MemorySpace::Mat || space == MemorySpace::Left || space == MemorySpace::Right ||
                space == MemorySpace::Acc) &&
                   box && shape[0] % box->rows == 0 && shape[1] % box->cols == 0 &&
                   view.blayout == layout.blayout && view.slayout == layout.slayout &&
                   view.fractal == layout.fractal && view.stride.empty() &&
                   (!view.start_offset || (start && start->value_ == 0)) && view.compact == CompactMode::null,
               span)
        << "Peer local planning requires a complete native matrix allocation";
  }
  const auto descriptor = std::make_shared<BufferType>(shape, tile->dtype_, space, valid, view.blayout,
                                                       view.slayout, view.fractal, view.pad, view.compact);
  CHECK_SPAN(!runtime || backend::IsAlignedDenseVecBufferGeometry(descriptor), span)
      << "Peer runtime valid extents require aligned dense Vec storage";
  return descriptor;
}

std::vector<ExprPtr> ValidExtents(const TileTypePtr& type) {
  return tile_view_semantics::GetEffectiveTileView(*type).valid_shape;
}

std::vector<ExprPtr> RuntimeValidOperands(const StmtPtr& stmt, const CallPtr& call) {
  if (!IsOp(call, "tile.set_validshape") && !IsOp(call, "tile.load") && !IsOp(call, "tile.store") &&
      !backend::FindLogicalBufferElementwiseRecipe(call)) {
    return {};
  }
  std::vector<ExprPtr> expressions;
  const auto append = [&expressions](const ExprPtr& value) {
    if (const auto tile = As<TileType>(value->GetType())) {
      for (const auto& extent : ValidExtents(tile)) {
        if (!As<ConstInt>(extent)) {
          expressions.push_back(extent);
        }
      }
    }
  };
  if (const auto assign = As<AssignStmt>(stmt)) {
    append(assign->var_);
  }
  if (!IsOp(call, "tile.set_validshape")) {
    for (const auto& arg : call->args_) {
      append(arg);
    }
  }
  return expressions;
}

bool SameExtents(const std::vector<ExprPtr>& a, const std::vector<ExprPtr>& b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](const auto& lhs, const auto& rhs) {
           const auto left = As<ConstInt>(lhs);
           const auto right = As<ConstInt>(rhs);
           return left && right ? left->value_ == right->value_ : structural_equal(lhs, rhs);
         });
}

bool IsSubblockQuery(const CallPtr& call) {
  return (IsOp(call, "tile.get_subblock_idx") || IsOp(call, "buffer.get_subblock_idx")) &&
         call->args_.empty() && call->kwargs_.empty();
}

void Flatten(const StmtPtr& statement, std::vector<StmtPtr>& into) {
  if (const auto seq = As<SeqStmts>(statement)) {
    for (const auto& child : seq->stmts_) Flatten(child, into);
  } else {
    into.push_back(statement);
  }
}

// A terminal complementary lane guard preserves original issue order in each
// lane. Both lists include the common prefix so lowering can check commands
// at shared cuts and both terminal drains before changing the original body.
std::vector<std::vector<StmtPtr>> LaneStatements(const FunctionPtr& function) {
  std::vector<StmtPtr> statements;
  Flatten(function->body_, statements);
  if (statements.empty()) return {std::move(statements)};
  const auto branch = As<IfStmt>(statements.back());
  if (!branch) return {std::move(statements)};
  const auto condition = As<Eq>(branch->condition_);
  CHECK_SPAN(function->func_type_ == FunctionType::AIV && condition && branch->else_body_ &&
                 branch->return_vars_.empty(),
             branch->span_)
      << "Peer local planning requires a terminal AIV subblock guard with two independent arms";
  std::unordered_set<const Var*> subblock;
  statements.pop_back();
  for (const auto& statement : statements) {
    if (const auto assignment = As<AssignStmt>(statement)) {
      if (IsSubblockQuery(As<Call>(assignment->value_)) ||
          subblock.count(AsVarLike(assignment->value_).get())) {
        subblock.insert(assignment->var_.get());
      }
    }
  }
  const bool left_is_subblock =
      subblock.count(AsVarLike(condition->left_).get()) || IsSubblockQuery(As<Call>(condition->left_));
  const auto literal = As<ConstInt>(left_is_subblock ? condition->right_ : condition->left_);
  CHECK_SPAN(literal && (literal->value_ == 0 || literal->value_ == 1) &&
                 (left_is_subblock || subblock.count(AsVarLike(condition->right_).get()) ||
                  IsSubblockQuery(As<Call>(condition->right_))),
             branch->span_)
      << "Peer local planning cannot prove the original AIV subblock guard";
  auto other = statements;
  Flatten(branch->then_body_, statements);
  Flatten(*branch->else_body_, other);
  return {std::move(statements), std::move(other)};
}

class LaneImporter {
 public:
  LaneImporter(const FunctionPtr& function, const PeerTransportEffects& transport, const std::string& target,
               const SyncParameterBounds& parameters, PeerGmEffects& gm, const SyncControlRangeProver& ranges,
               const SyncFootprintResolver* footprint)
      : transport_(transport),
        backend_(*backend::BackendConfig::GetBackend()),
        gm_(gm),
        ranges_(ranges),
        footprint_(footprint) {
    result_.input.owner = function;
    result_.input.hardware_target = target;
    result_.input.segments.emplace_back();
    for (const auto& site : transport.sites) {
      CHECK(sites_.emplace(site.statement.get(), &site).second) << "Duplicate peer transport cut";
    }
    placement_ = backend_.CreateMemoryAllocatorPolicy();
    INTERNAL_CHECK(placement_) << "A2/A3 peer planning requires the existing placement policy";
    for (const auto& [id, binding] : transport.pipes) {
      initializers_.insert(binding.initializer.get());
      if (!binding.receive) continue;
      // The invocation starts with a distinct cursor for each receive pipe.
      // A phase without a pop must still preserve those symbolic origins.
      ordinals_.emplace(id, 0);
      const auto& ring = *binding.receive;
      const auto capacity = backend_.GetHandler()->GetCapacityBytes(ring.space);
      CHECK(ring.placed_base && *ring.placed_base >= 0 && ring.reserve_bytes > 0 &&
            static_cast<uint64_t>(*ring.placed_base) <= capacity &&
            static_cast<uint64_t>(ring.reserve_bytes) <= capacity - *ring.placed_base &&
            ring.reserve_bytes <= std::numeric_limits<int64_t>::max() - *ring.placed_base)
          << "Peer receive reserve has no bounded physical placement";
      const auto base = *ring.placed_base;
      CHECK(ring.local_slots > 0 && ring.slot_bytes > 0 &&
            ring.slot_bytes <= ring.reserve_bytes / ring.local_slots &&
            placement_->AlignAddress(base, ring.space) == static_cast<uint64_t>(base) &&
            placement_->AlignAddress(ring.slot_bytes, ring.space) == static_cast<uint64_t>(ring.slot_bytes))
          << "Peer receive slots do not have aligned, bounded native geometry";
      CHECK(reserves_[ring.space].emplace(base, base + ring.reserve_bytes).second)
          << "Peer receive reserves overlap";
    }
    for (const auto& [space, intervals] : reserves_) {
      int64_t end = 0;
      for (const auto& [begin, next_end] : intervals) {
        CHECK(end <= begin) << "Peer receive reserves overlap";
        end = next_end;
      }
    }
    for (size_t formal = 0; formal < function->params_.size(); ++formal) {
      const auto& parameter = function->params_[formal];
      if (As<ScalarType>(parameter->GetType())) {
        ready_.insert(parameter.get());
        if (const auto bound = parameters.find(formal); bound != parameters.end()) {
          coordinate_bounds_.emplace(parameter.get(), bound->second);
        }
      } else if (As<TensorType>(parameter->GetType())) {
        gm_roots_.emplace(parameter.get(), formal);
        ready_.insert(parameter.get());
      } else {
        CHECK_SPAN(false, parameter->span_) << "Unqualified peer endpoint parameter type";
      }
    }
  }

  PeerLocalLane Run(const std::vector<StmtPtr>& statements) {
    const auto& owner = result_.input.owner;
    CHECK_SPAN(owner->return_types_.empty() || (!statements.empty() && As<ReturnStmt>(statements.back())),
               owner->span_)
        << "Peer local path with results requires a terminal return";
    for (size_t i = 0; i < statements.size(); ++i) {
      const auto& statement = statements[i];
      if (IsOp(GetCallFromStmt(statement), "system.syncall")) {
        ImportSyncAll(statement);
      } else if (const auto loop = As<ForStmt>(statement)) {
        ImportLoop(loop);
      } else {
        Import(statement, i + 1 == statements.size());
      }
    }
    CHECK(closed_.size() == births_.size()) << "Peer endpoint exits with an outstanding receive borrow";
    result_.regions.push_back({region_start_, result_.input.segments.size(), std::move(work_loop_)});
    return std::move(result_);
  }

 private:
  const PeerTransportEffects& transport_;
  const backend::Backend& backend_;
  MemoryAllocatorPolicyPtr placement_;
  PeerGmEffects& gm_;
  const SyncControlRangeProver& ranges_;
  const SyncFootprintResolver* footprint_;
  PeerLocalLane result_;
  size_t region_start_ = 0;
  std::optional<PeerLocalWorkLoop> work_loop_;
  std::unordered_map<const Stmt*, const PeerTransportSite*> sites_;
  std::unordered_set<const Call*> initializers_, payload_calls_;
  std::map<MemorySpace, std::map<int64_t, int64_t>> reserves_;
  std::map<int, int64_t> ordinals_;
  std::map<int, size_t> ring_domains_;
  std::map<int, int64_t> pops_per_visit_;

  std::unordered_map<const PeerTransportSite*, int64_t> births_;
  std::unordered_set<const PeerTransportSite*> closed_;
  std::unordered_set<const Var*> ready_;
  using CoordinateBounds = std::pair<int64_t, int64_t>;
  std::unordered_map<const Var*, CoordinateBounds> coordinate_bounds_;
  mutable std::unordered_set<const Expr*> ready_expressions_;
  std::unordered_map<const Var*, TileWindow> local_;
  std::unordered_map<const Var*, std::vector<ExprPtr>> current_valid_extents_;
  std::unordered_set<const Var*> ambiguous_metadata_handles_;
  std::unordered_map<const Var*, size_t> gm_roots_;
  std::map<MemorySpace, size_t> storage_;

  void ImportSyncAll(const StmtPtr& statement) {
    const auto call = GetCallFromStmt(statement);
    CHECK_SPAN(IsA<EvalStmt>(statement) && cross_core_pipe::IsHardMixedSyncAll(call), statement->span_)
        << "Peer local region requires an authored hard MIX SYNCALL";
    CHECK_SPAN(births_.size() == closed_.size(), statement->span_)
        << "A live FIFO handle crosses a hard MIX SYNCALL";
    // The pinned A2/A3 hard MIX implementation drains all local pipes before
    // reaching the cross-core rendezvous. Keep the call in place and preserve
    // the FIFO state: completion does not rewind any transport counter.
    const auto end = result_.input.segments.size();
    result_.regions.push_back({region_start_, end, std::exchange(work_loop_, std::nullopt)});
    region_start_ = end;
    result_.input.segments.emplace_back();
  }

  // The lane's immutable SSA definitions only add ready scalar values.
  // Cache positive results so a shared expression DAG is walked once. A
  // native payload or embedded GM load has no ready-scalar contract here.
  bool Ready(const ExprPtr& value) const {
    if (const auto var = AsVarLike(value)) return ready_.count(var.get());
    if (ready_expressions_.count(value.get())) return true;
    bool ready;
    if (const auto binary = std::dynamic_pointer_cast<const BinaryExpr>(value)) {
      ready = Ready(binary->left_) && Ready(binary->right_);
    } else if (const auto unary = std::dynamic_pointer_cast<const UnaryExpr>(value)) {
      ready = Ready(unary->operand_);
    } else {
      ready = As<ConstInt>(value) || As<ConstFloat>(value) || As<ConstBool>(value);
    }
    if (ready) ready_expressions_.insert(value.get());
    return ready;
  }

  std::optional<CoordinateBounds> Bounds(const ExprPtr& value) const {
    const auto maximum = SyncCoordinateMaximum(value);
    if (!maximum) {
      return std::nullopt;
    }
    std::optional<CoordinateBounds> bound;
    if (const auto known = As<ConstInt>(value)) {
      bound = {{known->value_, known->value_}};
    } else if (const auto variable = AsVarLike(value)) {
      if (const auto found = coordinate_bounds_.find(variable.get()); found != coordinate_bounds_.end()) {
        bound = found->second;
      }
    } else if (const auto cast = As<Cast>(value)) {
      bound = Bounds(cast->operand_);
    }
    if (!bound || bound->first < 0 || bound->second > *maximum) {
      return std::nullopt;
    }
    return bound;
  }

  void RequireReadyTuple(const ExprPtr& value, const Span& span) const {
    const auto tuple = As<MakeTuple>(value);
    CHECK_SPAN(tuple && std::all_of(tuple->elements_.begin(), tuple->elements_.end(),
                                    [this](const ExprPtr& expr) { return Ready(expr); }),
               span)
        << "Peer local planning requires a ready scalar coordinate tuple";
  }

  size_t Storage(MemorySpace space, int64_t end) {
    auto& storage = result_.input.storage;
    const auto [entry, inserted] = storage_.try_emplace(space, storage.size());
    if (inserted) {
      // DDR has no established physical pointer ordering. A shared singleton
      // covers every possible intra-endpoint alias; Group validation checks
      // separation from the separately allocated FIFO workspace.
      storage.push_back(
          {MemorySpaceToString(space), space, 0, 1, end, end, nullptr, space != MemorySpace::DDR});
    }
    auto& domain = storage[entry->second];
    domain.columns = std::max(domain.columns, end);
    domain.bytes = domain.columns;
    return entry->second;
  }

  void Access(SyncSite& site, const TileWindow& window, bool write, bool exact = false) {
    if (window.begin == window.end) return;
    if (window.pop) {
      const auto id = window.pop->pipe_id;
      const auto& ring = *transport_.pipes.at(id).receive;
      auto& storage = result_.input.storage;
      const auto [it, fresh] = ring_domains_.try_emplace(id, storage.size());
      if (fresh) {
        SyncStorage domain{"receive_" + std::to_string(id),
                           ring.space,
                           *ring.placed_base,
                           1,
                           ring.slot_bytes,
                           ring.local_slots * ring.slot_bytes,
                           nullptr,
                           true};
        domain.slot_count = ring.local_slots;
        domain.slot_bytes = ring.slot_bytes;
        storage.push_back(std::move(domain));
      }
      const auto slot = (births_.at(window.pop) - 1) % ring.local_slots;
      const auto origin = *ring.placed_base + slot * ring.slot_bytes;
      CHECK(window.begin >= origin && window.end <= origin + ring.slot_bytes)
          << "Peer local footprint escapes its captured slot";
      SyncAccess access{it->second, write, 0, window.begin - origin, 0, 0, 1, window.end - window.begin,
                        false,      exact};
      access.slot_first = slot;
      const auto repeated = pops_per_visit_.find(id);
      access.slot_step = repeated == pops_per_visit_.end() ? 0 : repeated->second % ring.local_slots;
      if (access.slot_step) access.slot_coordinate_frame = result_.input.segments.back().original_loop;
      site.accesses.push_back(std::move(access));
      return;
    }
    site.accesses.push_back({Storage(window.space, window.end), write, 0, window.begin, 0, 0, 1,
                             window.end - window.begin, false, exact});
  }

  void GmAccess(SyncSite& site, const StmtPtr& statement, const ExprPtr& value, bool write) {
    const auto it = gm_roots_.find(AsVarLike(value).get());
    CHECK_SPAN(it != gm_roots_.end(), statement->span_) << "Peer GM transfer has no original formal root";
    gm_.formals.insert(it->second);
    if (write) {
      gm_.writes.insert(it->second);
    }
    Access(site, {MemorySpace::DDR, 0, 1}, write);
  }

  int64_t ScalarGmIndexMaximum(const ExprPtr& index, const ExprPtr& extent, const Span& span) const {
    const auto maximum = SyncCoordinateMaximum(index);
    const auto type = As<ScalarType>(index->GetType());
    const auto literal = As<ConstInt>(index);
    CHECK_SPAN(maximum && type && (type->dtype_ == DataType::INDEX || type->dtype_.IsSignedInt()) &&
                   (!literal || (literal->value_ >= 0 && literal->value_ <= *maximum)),
               span)
        << "Peer scalar GM address requires a signed index with a representable value";
    const auto bound = Bounds(index);
    const auto length = As<ConstInt>(extent);
    CHECK_SPAN(!length || !bound || bound->first < length->value_, span)
        << "Peer scalar GM address is outside the source extent";
    const auto upper = bound ? bound->second : *maximum;
    // As for the original tensor.read, only in-bounds executions are defined.
    return length ? std::min(upper, length->value_ - 1) : upper;
  }

  void QualifyScalarGmAddress(const TensorType& tensor, const MakeTuple& indices, const Span& span) const {
    const auto rank = tensor.shape_.size();
    const auto columns = rank == 2 ? As<ConstInt>(tensor.shape_.back()) : nullptr;
    CHECK_SPAN(indices.elements_.size() == rank && (rank == 1 || columns), span)
        << "Peer scalar GM address requires a ready coordinate tuple and a static inner extent";
    __int128 maximum = ScalarGmIndexMaximum(indices.elements_.front(), tensor.shape_.front(), span);
    if (columns) {
      maximum = maximum * columns->value_ +
                ScalarGmIndexMaximum(indices.elements_.back(), tensor.shape_.back(), span);
    }
    // PTOAS emits a signed int64 INDEX for the flattened typed pointer
    // subscript. Check its enclosing byte extent before relying on address
    // arithmetic. The original caller still supplies valid backing storage.
    CHECK_SPAN(maximum < std::numeric_limits<int64_t>::max() / tensor.dtype_.GetByte(), span)
        << "Peer scalar GM address may exceed the signed native byte offset";
  }

  void ImportScalarGmRead(const CallPtr& call, const VarPtr& result) {
    const auto& owner = result_.input.owner;
    CHECK_SPAN(owner->func_type_ == FunctionType::AIC && result_.input.hardware_target == "a2a3" && result &&
                   call->args_.size() == 2 && call->kwargs_.empty(),
               call->span_)
        << "Peer scalar GM read requires a direct A2/A3 AIC scalar definition";
    const auto tensor = As<TensorType>(call->args_[0]->GetType());
    const auto root = gm_roots_.find(AsVarLike(call->args_[0]).get());
    CHECK_SPAN(root != gm_roots_.end() && tensor && !tensor->tensor_view_ &&
                   backend::IsBufferTensorShape(tensor->shape_),
               call->span_)
        << "Peer scalar GM read requires a plain rank-one or rank-two formal or alias";
    const auto formal = As<TensorType>(owner->params_[root->second]->GetType());
    const auto scalar = As<ScalarType>(result->GetType());
    const auto expression = As<ScalarType>(call->GetType());
    CHECK_SPAN(formal && !formal->tensor_view_ && formal->dtype_ == tensor->dtype_ &&
                   tile_view_semantics::ShapeExprListsEquivalent(formal->shape_, tensor->shape_) &&
                   (tensor->dtype_ == DataType::INT32 || tensor->dtype_ == DataType::FP32) && scalar &&
                   expression && scalar->dtype_ == tensor->dtype_ && scalar->dtype_ == expression->dtype_,
               call->span_)
        << "Peer scalar GM read must preserve its original packed pointer and scalar dtype";
    RequireReadyTuple(call->args_[1], call->span_);
    QualifyScalarGmAddress(*tensor, *As<MakeTuple>(call->args_[1]), call->span_);
    gm_.formals.insert(root->second);
    gm_.scalar_reads.insert(root->second);
    CHECK_SPAN(ready_.insert(result.get()).second, call->span_) << "Peer scalar GM result is defined twice";
    // The retained native read is a cacheable, blocking pointer dereference.
    // There is no asynchronous issue or cache-maintenance event to model.
    // Publication at invocation entry remains an original caller obligation.
    // The Group certificate must exclude all aliased writes during execution.
  }

  TileWindow Window(const ExprPtr& value, const Span& span) const {
    const auto found = local_.find(AsVarLike(value).get());
    CHECK_SPAN(found != local_.end(), span) << "Peer local operand has no proven physical storage";
    const auto& window = found->second;
    if (window.pop) {
      const auto id = window.pop->pipe_id;
      CHECK_SPAN(!closed_.count(window.pop) &&
                     ordinals_.at(id) - births_.at(window.pop) < transport_.pipes.at(id).receive->local_slots,
                 span)
          << "Peer local operation uses a released or overwritten receive tile";
    }
    const auto current = current_valid_extents_.find(window.mutable_handle);
    CHECK_SPAN(current == current_valid_extents_.end() ||
                   SameExtents(current->second, ValidExtents(As<TileType>(value->GetType()))),
               span)
        << "Peer local operand has stale native tile valid metadata";
    return window;
  }

  TileWindow Place(const VarPtr& var) const {
    const auto tile = As<TileType>(var->GetType());
    const auto descriptor = DescribeTile(tile, var->span_);
    CHECK_SPAN(tile->memref_ && *tile->memref_, var->span_)
        << "Peer local result requires a placed allocation";
    const auto& memref = *tile->memref_;
    const auto address = As<ConstInt>(memref->byte_offset_);
    const auto space = descriptor->memory_space_;
    const auto bytes = utils::StaticPhysicalAllocationBytes(tile, space, backend_.GetHandler());
    const auto capacity = backend_.GetHandler()->GetCapacityBytes(space);
    CHECK_SPAN(
        address && address->value_ >= 0 && bytes && *bytes <= memref->size_ &&
            static_cast<uint64_t>(address->value_) <= capacity &&
            memref->size_ <= capacity - address->value_ &&
            memref->size_ <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - address->value_) &&
            placement_->AlignAddress(address->value_, space) == static_cast<uint64_t>(address->value_),
        var->span_)
        << "Peer local result has no aligned, bounded physical allocation";
    const bool ordinary = memref->slot_count_ == 1 && !memref->slot_index_;
    const TileWindow window{space,
                            address->value_,
                            address->value_ + static_cast<int64_t>(memref->size_),
                            nullptr,
                            ordinary ? var.get() : nullptr,
                            result_.input.segments.size()};
    if (const auto found = reserves_.find(space); found != reserves_.end()) {
      const auto next = found->second.lower_bound(window.end);
      CHECK_SPAN(next == found->second.begin() || std::prev(next)->second <= window.begin, var->span_)
          << "Ordinary Tile placement intersects a live receive reserve";
    }
    return window;
  }

  static ExprPtr BufferOperand(const ExprPtr& value) {
    return std::make_shared<Var>("", DescribeTile(As<TileType>(value->GetType()), value->span_),
                                 value->span_);
  }

  void RequireRuntimeValid(const StmtPtr& statement, const ExprPtr& value) const {
    const auto type = As<TileType>(value->GetType());
    if (!type) {
      return;
    }
    const auto valid = ValidExtents(type);
    if (!HasRuntimeExtents(valid)) {
      return;
    }
    CHECK_SPAN(result_.input.owner->func_type_ == FunctionType::AIV, statement->span_)
        << "Peer runtime tile valid extents require an AIV endpoint";
    for (size_t axis = 0; axis < valid.size(); ++axis) {
      const auto& extent = valid[axis];
      const auto capacity = As<ConstInt>(type->shape_[axis]);
      const auto literal = As<ConstInt>(extent);
      const auto bounds = literal ? std::optional{std::pair{literal->value_, literal->value_}}
                                  : ranges_.LookupScalarBounds(statement.get(), extent.get());
      CHECK_SPAN(
          Ready(extent) && capacity && bounds && bounds->first > 0 && bounds->second <= capacity->value_,
          statement->span_)
          << "Peer runtime tile valid extent has no positive in-capacity bound";
    }
  }

  void AddSite(SyncSite site) {
    CHECK_SPAN(payload_calls_.insert(site.original.get()).second, site.original->span_)
        << "A shared call cannot identify two independent physical payload cuts";
    result_.input.segments.back().sites.push_back(std::move(site));
  }

  void CheckReceiveReserveSeparation(const TileWindow& window, const Span& span) const {
    if (const auto found = reserves_.find(window.space); found != reserves_.end()) {
      const auto next = found->second.lower_bound(window.end);
      CHECK_SPAN(next == found->second.begin() || std::prev(next)->second <= window.begin, span)
          << "Buffer footprint intersects a live receive reserve";
    }
  }

  void CheckOrdinaryPlacement(const TileWindow& window, const Span& span) const {
    const auto capacity = backend_.GetHandler()->GetCapacityBytes(window.space);
    CHECK_SPAN(
        window.begin >= 0 && window.begin < window.end && static_cast<uint64_t>(window.end) <= capacity, span)
        << "Ordinary Buffer footprint has no bounded physical placement";
    CheckReceiveReserveSeparation(window, span);
  }

  TileWindow ResolveBufferWindow(const StmtPtr& statement, const ExprPtr& operand,
                                 const ResolvedOperandFootprint& resolved) const {
    CHECK_SPAN(resolved.byte_hull, statement->span_) << "Peer payload has no placed Buffer footprint";
    const auto& bytes = *resolved.byte_hull;
    TileWindow window{resolved.space, bytes.begin, bytes.end};
    const auto view = footprint_->ViewOf(operand);
    if (view && view->peer_borrow) {
      const auto found = local_.find(view->peer_borrow);
      CHECK_SPAN(found != local_.end() && found->second.pop, statement->span_)
          << "Peer Buffer footprint has no executed pop binding";
      const auto& captured = found->second;
      window.pop = captured.pop;
      const auto id = window.pop->pipe_id;
      const auto& ring = *transport_.pipes.at(id).receive;
      CHECK_SPAN(!closed_.count(window.pop) && ordinals_.at(id) - births_.at(window.pop) < ring.local_slots &&
                     !window.pop->empty_payload && bytes.begin >= *ring.placed_base &&
                     bytes.end - *ring.placed_base <= window.pop->capacity_bytes,
                 statement->span_)
          << "Peer Buffer footprint uses an empty, released or overwritten receive slot";
      const auto offset = captured.begin - *ring.placed_base;
      window.begin += offset;
      window.end += offset;
    } else {
      CheckOrdinaryPlacement(window, statement->span_);
    }
    return window;
  }

  void ImportBufferPayload(const StmtPtr& statement, const CallPtr& call, const VarPtr& result) {
    const auto operation = backend::QualifyBufferSyncOperation(call, result_.input.owner->func_type_,
                                                               result_.input.hardware_target, true);
    const auto& implementation = operation.Implementation();
    if (implementation.RequiresReadOnlyGM()) {
      // The unit resolves the original Group roots and proves that the
      // operand remains read-only before publishing either local plan.
      ImportScalarGmRead(call, result);
      return;
    }
    footprint_->RequireVecCopyPlacement(operation);
    footprint_->RequireWorkspaceSeparation(operation);
    for (const auto& claim : implementation.Claims()) {
      auto& reserved = result_.reservations[{claim.pool.src, claim.pool.dst}];
      for (const auto id : claim.ids) {
        reserved.insert(static_cast<int>(id));
      }
    }
    SyncSite site{call, nullptr, implementation.Pipe(), {}};
    for (size_t i = 0; i < operation.ArgumentEffects().size(); ++i) {
      const auto mode = operation.ArgumentEffects()[i].data;
      if (mode == BufferAccess::None) {
        continue;
      }
      const auto& operand = call->args_[i];
      const auto type = As<BufferType>(operand->GetType());
      const auto space = type ? type->memory_space_ : MemorySpace::DDR;
      const auto resolved = footprint_->ResolveOperand(statement.get(), call, i, mode, space, true, false,
                                                       /*peer_slot_relative=*/true);
      if (space == MemorySpace::DDR) {
        if (resolved.read) {
          GmAccess(site, statement, operand, false);
        }
        if (resolved.write) {
          GmAccess(site, statement, operand, true);
        }
      } else {
        const auto window = ResolveBufferWindow(statement, operand, resolved);
        if (resolved.read) {
          Access(site, window, false);
        }
        if (resolved.write) {
          Access(site, window, true, resolved.candidate_full_overwrite);
        }
      }
    }
    for (const auto& implicit : implementation.ImplicitEffects()) {
      const TileWindow window{implicit.space, implicit.begin, implicit.end};
      // The selected native contract supplies physical bounds for its own
      // scratch. It can use ISA-reserved bytes outside the allocation limit.
      INTERNAL_CHECK_SPAN(window.space != MemorySpace::DDR && window.begin >= 0 && window.begin < window.end,
                          statement->span_)
          << "Invalid physical bounds for qualified native scratch";
      CheckReceiveReserveSeparation(window, statement->span_);
      if (implicit.data == BufferAccess::Read || implicit.data == BufferAccess::ReadWrite) {
        Access(site, window, false, implicit.exact_footprint);
      }
      if (implicit.data == BufferAccess::Write || implicit.data == BufferAccess::ReadWrite) {
        Access(site, window, true, implicit.exact_footprint);
      }
    }
    CHECK_SPAN(static_cast<bool>(result) == implementation.ProducesReadyScalar(), call->span_)
        << "Peer Buffer payload result has no selected scalar completion contract";
    if (result) {
      ready_.insert(result.get());
    }
    AddSite(std::move(site));
  }

  void ImportBufferDefinition(const StmtPtr& statement, const CallPtr& call, const VarPtr& result) {
    for (const auto& operand : call->args_) {
      CheckSyncScalarReadiness(operand, [this](const VarPtr& var) { return ready_.count(var.get()) != 0; });
    }
    if (auto descriptor = backend::QualifyBufferSyncNonPayload(call, result_.input.hardware_target)) {
      CHECK_SPAN(static_cast<bool>(result) == (descriptor->Result().behavior != BufferResultBehavior::None),
                 call->span_)
          << "Peer Buffer descriptor requires its declared result binding";
      if (result) {
        ready_.insert(result.get());
        if (IsOp(call, "buffer.get_block_idx") || IsOp(call, "buffer.get_block_num")) {
          coordinate_bounds_.emplace(result.get(),
                                     CoordinateBounds{IsOp(call, "buffer.get_block_num") ? 1 : 0, INT32_MAX});
        }
      }
    } else {
      ImportBufferPayload(statement, call, result);
    }
  }

  void AddOperation(const StmtPtr& statement, const CallPtr& original, const CallPtr& qualified,
                    const std::vector<ExprPtr>& operands) {
    const auto operation = backend::QualifyBufferSyncOperation(qualified, result_.input.owner->func_type_,
                                                               result_.input.hardware_target);
    const auto& implementation = operation.Implementation();
    const auto selected = backend_.TryInferPipe(original);
    CHECK_SPAN(selected == implementation.Pipe() && implementation.ImplicitEffects().empty(), original->span_)
        << "Peer local operation has a different native pipe or unbound implicit storage";
    for (const auto& claim : implementation.Claims()) {
      auto& reserved = result_.reservations[{claim.pool.src, claim.pool.dst}];
      for (const auto id : claim.ids) reserved.insert(static_cast<int>(id));
    }
    SyncSite site{original, nullptr, implementation.Pipe(), {}};
    for (size_t index = 0; index < operands.size(); ++index) {
      const auto access = operation.ArgumentEffects()[index].data;
      if (access == BufferAccess::None) continue;
      const bool read = access == BufferAccess::Read || access == BufferAccess::ReadWrite;
      const bool write = access == BufferAccess::Write || access == BufferAccess::ReadWrite;
      if (As<TensorType>(operands[index]->GetType())) {
        if (read) GmAccess(site, statement, operands[index], false);
        if (write) GmAccess(site, statement, operands[index], true);
      } else {
        RequireRuntimeValid(statement, operands[index]);
        const auto window = Window(operands[index], original->span_);
        if (read) Access(site, window, false);
        if (write) Access(site, window, true);
      }
    }
    AddSite(std::move(site));
  }

  void ImportTransfer(const StmtPtr& statement, const CallPtr& call, const VarPtr& result, bool load) {
    CHECK_SPAN(
        (load && (call->args_.size() == 3 || call->args_.size() == 4)) || (!load && call->args_.size() == 3),
        call->span_)
        << "Peer local transfer requires ordinary rank-two GM operands";
    const auto tile = load ? ExprPtr(result) : call->args_[0];
    CHECK_SPAN(tile, call->span_) << "Peer local load requires a Tile result";
    const auto type = As<TileType>(tile->GetType());
    CHECK_SPAN(type, call->span_) << "Peer local transfer requires a Tile operand";
    const auto& tensor = call->args_[load ? 0 : 2];
    const auto tensor_type = As<TensorType>(tensor->GetType());
    const auto root = gm_roots_.find(AsVarLike(tensor).get());
    CHECK_SPAN(tensor_type && tensor_type->shape_.size() == 2 && !tensor_type->tensor_view_ &&
                   root != gm_roots_.end(),
               call->span_)
        << "Peer local transfer requires a plain rank-two GM formal or alias";
    CHECK_SPAN(std::all_of(tensor_type->shape_.begin(), tensor_type->shape_.end(),
                           [this](const ExprPtr& dim) { return Ready(dim); }),
               call->span_)
        << "Peer GM tensor shape is unavailable at the transfer cut";
    const auto& offsets = call->args_[1];
    const auto valid = load ? call->args_.back()
                            : std::make_shared<MakeTuple>(
                                  tile_view_semantics::GetEffectiveTileView(*type).valid_shape, call->span_);
    RequireReadyTuple(offsets, call->span_);
    RequireReadyTuple(valid, call->span_);
    const auto tile_extents = ValidExtents(type);
    CHECK_SPAN(!HasRuntimeExtents(tile_extents) || SameExtents(As<MakeTuple>(valid)->elements_, tile_extents),
               call->span_)
        << "Peer local transfer does not match its tile valid metadata";
    for (const auto& [name, value] : call->kwargs_) {
      CHECK_SPAN(load ? name == "target_memory" : name == "atomic" || name == "cache" || name == "st_phase",
                 call->span_)
          << "Unqualified native transfer attribute " << name;
    }
    CHECK_SPAN(call->GetKwarg<int>("atomic", 0) == 0 && call->GetKwarg<int>("cache", 0) == 0 &&
                   call->GetKwarg<int>("st_phase", 0) == 0,
               call->span_)
        << "Peer local planning requires a plain GM transfer";
    const std::vector<ExprPtr> operands = load ? std::vector<ExprPtr>{tensor, offsets, valid, tile}
                                               : std::vector<ExprPtr>{tile, offsets, valid, tensor};
    auto projected = operands;
    projected[load ? 3 : 0] = BufferOperand(tile);
    const auto qualified = OpRegistry::GetInstance().CreateInternal(load ? "buffer.load" : "buffer.store",
                                                                    projected, call->span_);
    AddOperation(statement, call, qualified, operands);
    if (!load && result) gm_roots_.emplace(result.get(), root->second);
  }

  void RequirePayloadAliasing(const ExprPtr& source, const VarPtr& destination, bool allows_exact_alias,
                              const Span& span) const {
    const auto from = Window(source, span);
    const auto to = Window(destination, span);
    const bool overlaps = from.space == to.space && from.begin < to.end && to.begin < from.end;
    CHECK_SPAN(!overlaps || (allows_exact_alias && from.begin == to.begin && from.end == to.end), span)
        << "Native peer payload has an unsupported source and destination alias";
  }

  void ImportCompute(const StmtPtr& statement, const CallPtr& call, const VarPtr& result) {
    CHECK_SPAN(result && As<TileType>(result->GetType()), call->span_)
        << "Unqualified peer local operation " << call->op_->name_;
    CHECK_SPAN(local_.emplace(result.get(), Place(result)).second, call->span_)
        << "Peer local result is assigned more than once";
    if (IsOp(call, "tile.load")) {
      ImportTransfer(statement, call, result, true);
      return;
    }
    std::vector<ExprPtr> operands;
    std::vector<ExprPtr> projected;
    std::string op;
    std::vector<std::pair<std::string, std::any>> kwargs;
    if (IsOp(call, "tile.move") || IsOp(call, "tile.matmul")) {
      const bool move = IsOp(call, "tile.move");
      CHECK_SPAN(call->args_.size() == (move ? 1u : 2u), call->span_)
          << "Unqualified native peer move or matmul operands";
      const auto destination = DescribeTile(As<TileType>(result->GetType()), call->span_);
      for (const auto& [name, value] : call->kwargs_) {
        const bool describes_result =
            (name == "blayout" && call->GetKwarg<TileLayout>(name) == destination->blayout_) ||
            (name == "slayout" && call->GetKwarg<TileLayout>(name) == destination->slayout_);
        CHECK_SPAN(move && (name == "target_memory" || describes_result), call->span_)
            << "Unqualified native move or matmul attribute " << name;
      }
      op = move ? "buffer.copy" : "buffer.matmul";
      operands = call->args_;
      for (const auto& operand : operands) {
        RequirePayloadAliasing(operand, result, /*allows_exact_alias=*/false, call->span_);
        projected.push_back(BufferOperand(operand));
      }
    } else {
      const auto* recipe = backend::FindLogicalBufferElementwiseRecipe(call);
      CHECK_SPAN(recipe && backend::SupportsLogicalBufferElementwise(*recipe, call), call->span_)
          << "Unqualified native peer elementwise recipe " << call->op_->name_;
      op = recipe->buffer_op;
      kwargs = call->kwargs_;
      const auto output_extents = ValidExtents(As<TileType>(result->GetType()));
      for (const auto& input : recipe->inputs) {
        const auto& value = call->args_[input.logical_index];
        operands.push_back(value);
        if (input.kind == backend::BufferElementwiseOperandKind::ElementScalar) {
          CHECK_SPAN(Ready(value), call->span_) << "Peer elementwise scalar operand is unavailable";
          projected.push_back(std::make_shared<Var>(
              "", std::make_shared<ScalarType>(As<TileType>(result->GetType())->dtype_), value->span_));
        } else {
          CHECK_SPAN(input.kind == backend::BufferElementwiseOperandKind::Buffer, call->span_)
              << "Peer local elementwise requires an audited workspace-free variant";
          const auto input_extents = ValidExtents(As<TileType>(value->GetType()));
          CHECK_SPAN((!HasRuntimeExtents(input_extents) && !HasRuntimeExtents(output_extents)) ||
                         SameExtents(input_extents, output_extents),
                     call->span_)
              << "Peer local elementwise requires matching native tile valid metadata";
          const bool may_alias =
              recipe->destination_alias == backend::BufferDestinationAliasPolicy::ExactOrDisjoint;
          RequirePayloadAliasing(value, result, may_alias, call->span_);
          projected.push_back(BufferOperand(value));
        }
      }
      for (size_t i = 0; i < call->args_.size(); ++i) {
        const auto found = std::find_if(recipe->inputs.begin(), recipe->inputs.end(),
                                        [i](const auto& input) { return input.logical_index == i; });
        if (found == recipe->inputs.end()) RequireReadyTuple(call->args_[i], call->span_);
      }
    }
    operands.push_back(result);
    projected.push_back(BufferOperand(result));
    const auto qualified = OpRegistry::GetInstance().CreateInternal(op, projected, kwargs, call->span_);
    AddOperation(statement, call, qualified, operands);
  }

  void ImportPeer(const CallPtr& call, const PeerTransportSite& transfer) {
    const auto& binding = transport_.pipes.at(transfer.pipe_id);
    CHECK_SPAN(footprint_ || std::all_of(transfer.descriptor->valid_shape_.begin(),
                                         transfer.descriptor->valid_shape_.end(),
                                         [](int64_t extent) { return extent >= 0; }),
               call->span_)
        << "Peer local planning needs a statically bounded native valid extent";
    if (transfer.kind == PeerTransportKind::Free) {
      const auto window = Window(transfer.tile, call->span_);
      CHECK_SPAN(window.pop && window.pop->pipe_id == transfer.pipe_id && closed_.insert(window.pop).second,
                 call->span_)
          << "Peer TFREE cannot be bound to its live local borrow";
      return;
    }
    SyncSite site{call, nullptr, transfer.data_pipe, {}};
    if (transfer.kind == PeerTransportKind::Pop) {
      CHECK_SPAN(binding.receive && binding.receive->placed_base, call->span_)
          << "Peer local receive reserve has no placed base";
      const auto& ring = *binding.receive;
      auto& ordinal = ordinals_[transfer.pipe_id];
      const auto begin = *ring.placed_base + (ordinal % ring.local_slots) * ring.slot_bytes;
      CHECK_SPAN(ordinal < INT64_MAX, call->span_) << "Peer FIFO ordinal exceeds index range";
      ++ordinal;
      // No-split C2V may widen a nonempty logical payload to the physical box.
      // With zero rows the pinned Vec receive issues no local bytes, but the
      // call still waits, returns credit and advances the local slot cursor.
      const TileWindow window{ring.space, begin, begin + transfer.capacity_bytes, &transfer};
      CHECK_SPAN(local_.emplace(transfer.tile.get(), window).second, call->span_)
          << "Peer TPOP result is assigned more than once";
      births_.emplace(&transfer, ordinal);
      ready_.insert(transfer.tile.get());
      if (!transfer.empty_payload) {
        Access(site, window, true, transfer.descriptor->valid_shape_ == transfer.descriptor->shape_);
      }
    } else {
      if (!transfer.empty_payload) {
        const auto window =
            footprint_ ? ResolveBufferWindow(
                             transfer.statement, transfer.tile,
                             footprint_->ResolveOperand(transfer.statement.get(), call, 0, BufferAccess::Read,
                                                        transfer.memory_space, false, false, true))
                       : Window(transfer.tile, call->span_);
        CHECK_SPAN(
            window.space == transfer.memory_space && window.end - window.begin >= transfer.capacity_bytes,
            call->span_)
            << "Peer TPUSH source exceeds its placed tile";
        Access(site, window, false);
      }
    }
    AddSite(std::move(site));
  }

  void ImportValidShape(const StmtPtr& statement, const CallPtr& call, const VarPtr& result) {
    CHECK_SPAN(result_.input.owner->func_type_ == FunctionType::AIV && result && call->args_.size() == 3 &&
                   call->kwargs_.empty(),
               call->span_)
        << "Peer tile.set_validshape requires an ordinary AIV Vec handle";
    const auto& source = call->args_[0];
    const auto window = Window(source, call->span_);
    const auto from = backend::PhysicalLogicalVecDescriptor(source->GetType());
    const auto to = backend::PhysicalLogicalVecDescriptor(result->GetType());
    CHECK_SPAN(window.mutable_handle && !ambiguous_metadata_handles_.count(window.mutable_handle) &&
                   window.definition_segment == result_.input.segments.size() &&
                   backend::IsAlignedDenseVecBufferGeometry(from) && to && structural_equal(from, to),
               call->span_)
        << "Peer tile.set_validshape requires an aligned dense Vec allocation in its defining region";
    const auto valid = ValidExtents(As<TileType>(result->GetType()));
    CHECK_SPAN(SameExtents(valid, {call->args_[1], call->args_[2]}) && Ready(call->args_[1]) &&
                   Ready(call->args_[2]),
               call->span_)
        << "Peer tile.set_validshape result does not match ready native metadata";
    const auto placed = Place(result);
    CHECK_SPAN(placed.begin == window.begin && placed.end == window.end, call->span_)
        << "Peer tile.set_validshape must retain the original placed allocation";
    RequireRuntimeValid(statement, result);
    current_valid_extents_.insert_or_assign(window.mutable_handle, valid);
    CHECK_SPAN(local_.emplace(result.get(), window).second, call->span_)
        << "Peer tile.set_validshape result is assigned more than once";
  }

  void BindAlias(const VarPtr& destination, const ExprPtr& source, bool reshape = false) {
    const auto var = AsVarLike(source);
    if (var && local_.count(var.get())) {
      auto window = Window(var, destination->span_);
      if (window.mutable_handle) {
        // Addressed and unaddressed lowering can choose different descriptor
        // identity for plain assignments and reshapes with shared storage.
        CHECK_SPAN(!current_valid_extents_.count(window.mutable_handle), destination->span_)
            << "Peer Tile alias cannot preserve updated native tile valid metadata";
        ambiguous_metadata_handles_.insert(window.mutable_handle);
        window.mutable_handle = nullptr;
      }
      const auto source_type = As<TileType>(var->GetType());
      const auto from = DescribeTile(source_type, destination->span_);
      const auto dest_type = As<TileType>(destination->GetType());
      const auto to = DescribeTile(dest_type, destination->span_);
      const auto from_extents = ValidExtents(source_type);
      const auto to_extents = ValidExtents(dest_type);
      CHECK_SPAN(!reshape || (!HasRuntimeExtents(from_extents) && !HasRuntimeExtents(to_extents)),
                 destination->span_)
          << "Peer Tile reshape requires static valid metadata";
      CHECK_SPAN((structural_equal(from, to) && SameExtents(from_extents, to_extents)) ||
                     (reshape && from->memory_space_ == MemorySpace::Vec &&
                      to->memory_space_ == MemorySpace::Vec && backend::IsContiguousVectorReshape(from, to)),
                 destination->span_)
          << "Peer Tile alias has an unproved physical view";
      if (dest_type->memref_) {
        const auto placed = Place(destination);
        CHECK_SPAN(placed.space == window.space && placed.begin == window.begin && placed.end <= window.end,
                   destination->span_)
            << "Peer Tile alias does not preserve its native storage";
      }
      CHECK_SPAN(local_.emplace(destination.get(), window).second, destination->span_)
          << "Peer Tile alias is assigned more than once";
    } else if (var && gm_roots_.count(var.get())) {
      CHECK_SPAN(!reshape && As<TensorType>(destination->GetType()), destination->span_)
          << "Peer GM alias does not preserve its native tensor handle";
      gm_roots_.emplace(destination.get(), gm_roots_.at(var.get()));
    } else {
      CHECK_SPAN(!reshape && As<ScalarType>(destination->GetType()) && Ready(source), destination->span_)
          << "Unqualified peer local expression";
      ready_.insert(destination.get());
      if (const auto bound = Bounds(source)) {
        coordinate_bounds_.emplace(destination.get(), *bound);
      }
    }
  }

  void QualifyWorkLoop(const ForStmtPtr& loop) const {
    const auto start = Bounds(loop->start_);
    const auto stop = Bounds(loop->stop_);
    const auto step = Bounds(loop->step_);
    const auto maximum = SyncCoordinateMaximum(loop->loop_var_);
    CHECK_SPAN(start && stop && step && step->first > 0 && maximum && start->second <= *maximum &&
                   stop->second <= *maximum && step->second <= *maximum &&
                   (__int128(stop->second) - 1 + step->second <= *maximum),
               loop->span_)
        << "Peer runtime work loop requires ready nonnegative bounds and a proved positive, bounded stride";
  }

  void BindLoopResults(const ForStmtPtr& loop, const YieldStmtPtr& yield) {
    CHECK_SPAN(loop->iter_args_.empty() ? !yield || yield->value_.empty()
                                        : yield && yield->value_.size() == loop->iter_args_.size(),
               loop->span_)
        << "Unsupported loop yield";
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& arg = loop->iter_args_[i];
      const auto value = AsVarLike(yield->value_[i]);
      CHECK_SPAN(value && gm_roots_.count(arg.get()) && gm_roots_.count(value.get()) &&
                     gm_roots_.at(arg.get()) == gm_roots_.at(value.get()),
                 loop->span_)
          << "Loop must carry an invariant tensor handle";
      BindAlias(loop->return_vars_[i], arg);
    }
    CHECK_SPAN(births_.size() == closed_.size(), loop->span_) << "A live FIFO handle crosses a loop backedge";
  }

  void ImportWorkLoop(const ForStmtPtr& loop, bool allow_runtime, const std::vector<StmtPtr>& body,
                      const YieldStmtPtr& yield) {
    CHECK_SPAN(allow_runtime && !work_loop_, loop->span_)
        << "Peer local planning requires at most one outer runtime work loop per region";
    QualifyWorkLoop(loop);
    const auto entry = ordinals_;
    auto& segments = result_.input.segments;
    const auto first = segments.size();
    segments.emplace_back();
    for (const auto& stmt : body) {
      if (const auto child = As<ForStmt>(stmt)) {
        ImportLoop(child, /*allow_runtime=*/false);
      } else {
        Import(stmt, false);
      }
    }
    BindLoopResults(loop, yield);
    std::map<size_t, int64_t> advance;
    for (const auto& [id, domain] : ring_domains_) {
      advance.emplace(domain, (ordinals_.at(id) - entry.at(id)) % result_.input.storage[domain].slot_count);
    }
    // Rebase the suffix to the same ring origin as the item. Its actual origin
    // can differ by any number of item advances; only seam analysis uses that
    // uncertainty. A later region can keep this representative origin because
    // only closed borrows reach the cut. Each region's physical aliases are
    // invariant under a common rotation.
    ordinals_ = entry;
    work_loop_ = PeerLocalWorkLoop{loop, first, segments.size(), std::move(advance)};
    segments.emplace_back();
  }

  void ImportLoop(const ForStmtPtr& loop, bool allow_runtime = true) {
    const auto start = transform_utils::EvalConstInt(loop->start_);
    const auto stop = transform_utils::EvalConstInt(loop->stop_);
    const auto step = transform_utils::EvalConstInt(loop->step_);
    CHECK_SPAN(loop->kind_ == ForKind::Sequential, loop->span_) << "Peer endpoint requires sequential loops";
    CHECK_SPAN(births_.size() == closed_.size(), loop->span_) << "A live FIFO handle crosses loop entry";
    CHECK_SPAN(loop->iter_args_.size() == loop->return_vars_.size(), loop->span_) << "Invalid loop carry";
    std::vector<StmtPtr> body;
    Flatten(loop->body_, body);
    const auto yield = body.empty() ? nullptr : As<YieldStmt>(body.back());
    if (yield) body.pop_back();
    for (const auto& arg : loop->iter_args_) BindAlias(arg, arg->initValue_);
    ready_.insert(loop->loop_var_.get());
    if (!start || !stop || !step) {
      ImportWorkLoop(loop, allow_runtime, body, yield);
      return;
    }
    CHECK_SPAN(*step > 0, loop->span_) << "Peer endpoint requires a positive sequential loop stride";
    const __int128 trips = std::max<__int128>(0, (__int128(*stop) - *start + *step - 1) / *step);
    CHECK_SPAN(trips <= INT64_MAX && __int128(*start) + trips * *step <= INT64_MAX, loop->span_)
        << "Peer loop coordinate exceeds the index range";
    auto& seg = result_.input.segments.emplace_back();
    seg.original_loop = loop;
    seg.count = static_cast<int64_t>(trips);
    seg.start = *start;
    seg.step = *step;
    for (const auto& stmt : body) {
      const auto it = sites_.find(stmt.get());
      if (it != sites_.end() && it->second->kind == PeerTransportKind::Pop)
        ++pops_per_visit_[it->second->pipe_id];
    }
    const auto entry = ordinals_;
    for (const auto& stmt : body) {
      CHECK_SPAN(!IsA<ReturnStmt>(stmt) && !IsA<ForStmt>(stmt), stmt->span_)
          << "Peer loop body requires a single sequential region";
      Import(stmt, false);
    }
    BindLoopResults(loop, yield);
    for (const auto& [id, n] : pops_per_visit_) {
      const auto first = entry.find(id);
      const auto end = (first == entry.end() ? 0 : first->second) + trips * n;
      CHECK_SPAN(end < INT64_MAX, loop->span_) << "Peer FIFO ordinal exceeds index range";
      ordinals_[id] = static_cast<int64_t>(end);
    }
    pops_per_visit_.clear();
    result_.input.segments.emplace_back();
  }

  void Import(const StmtPtr& statement, bool terminal) {
    if (const auto returned = As<ReturnStmt>(statement)) {
      CHECK_SPAN(terminal, statement->span_) << "Unqualified peer local early return";
      for (const auto& value : returned->value_) {
        CHECK_SPAN(gm_roots_.count(AsVarLike(value).get()) || Ready(value), statement->span_)
            << "Unqualified peer local return operand";
      }
      return;
    }
    const auto assignment = As<AssignStmt>(statement);
    const auto result = assignment ? assignment->var_ : nullptr;
    const auto call = GetCallFromStmt(statement);
    if (!call) {
      CHECK_SPAN(assignment, statement->span_) << "Peer local effects require bounded straight-line paths";
      BindAlias(result, assignment->value_);
      return;
    }
    const auto peer = sites_.find(statement.get());
    if (peer != sites_.end()) {
      ImportPeer(call, *peer->second);
      return;
    }
    CHECK_SPAN(!As<GlobalVar>(call->op_), statement->span_) << "Unqualified outlined peer payload";
    CHECK_SPAN(!IsOp(call, "system.syncall"), statement->span_)
        << "Peer local hard MIX SYNCALL must be outside loops";
    CHECK_SPAN(call->attrs_.empty(), statement->span_)
        << "Peer local planning requires qualified native call attributes";
    if (initializers_.count(call.get())) {
      CHECK_SPAN(std::all_of(call->args_.begin(), call->args_.end(),
                             [this](const ExprPtr& arg) { return Ready(arg); }),
                 call->span_)
          << "Peer pipe initializer has an unqualified descriptor operand";
      return;
    }
    if (footprint_) {
      if (cross_core_pipe::IsPeerPipeSetup(call)) {
        CHECK_SPAN(result, call->span_) << "Peer reserve or imported address requires its SSA result";
        for (const auto& arg : call->args_) {
          CheckSyncScalarReadiness(arg, [this](const VarPtr& var) { return ready_.count(var.get()) != 0; });
        }
        ready_.insert(result.get());
      } else {
        ImportBufferDefinition(statement, call, result);
      }
      return;
    }
    if (IsOp(call, "tile.alloc") || IsOp(call, "system.reserve_buffer") ||
        IsOp(call, "system.import_peer_buffer") || IsSubblockQuery(call) ||
        IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num")) {
      CHECK_SPAN(result && std::all_of(call->args_.begin(), call->args_.end(),
                                       [this](const ExprPtr& arg) { return Ready(arg); }),
                 call->span_)
          << "Unqualified peer descriptor or ready scalar recipe";
      ready_.insert(result.get());
      if (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num")) {
        CHECK_SPAN(call->args_.empty() && call->kwargs_.empty(), call->span_)
            << "Invalid original block query";
        // The emitted Group ABI forwards positive int32 dispatch cardinality
        // and zero-based int32 block index to the Tile subfunction.
        coordinate_bounds_.emplace(result.get(),
                                   CoordinateBounds{IsOp(call, "tile.get_block_num") ? 1 : 0, INT32_MAX});
      }
    } else if (IsOp(call, "tile.create")) {
      // Tile codegen emits only the alloc_tile descriptor. The replay lane
      // uses this uninitialized zero-row carrier to signal a no-data push.
      CHECK_SPAN(result && call->args_.size() == 1, call->span_)
          << "Unqualified peer Tile allocation descriptor";
      RequireReadyTuple(call->args_[0], call->span_);
      const auto descriptor = DescribeTile(As<TileType>(result->GetType()), call->span_);
      CHECK_SPAN(descriptor->memory_space_ == MemorySpace::Vec && descriptor->valid_shape_[0] == 0 &&
                     local_.emplace(result.get(), Place(result)).second,
                 call->span_)
          << "Peer tile.create requires a zero-row replay allocation";
    } else if (IsOp(call, "tile.set_validshape")) {
      ImportValidShape(statement, call, result);
    } else if (IsOp(call, "tile.reshape")) {
      CHECK_SPAN(result && call->args_.size() == 2, call->span_) << "Unqualified peer Tile reshape";
      RequireReadyTuple(call->args_[1], call->span_);
      BindAlias(result, call->args_[0], /*reshape=*/true);
    } else if (IsOp(call, "tensor.read")) {
      ImportScalarGmRead(call, result);
    } else if (IsOp(call, "tile.store")) {
      ImportTransfer(statement, call, result, false);
    } else {
      ImportCompute(statement, call, result);
    }
  }
};
}  // namespace

PeerLocalEffects ImportPeerLocalEffects(const FunctionPtr& function, const PeerTransportEffects& transport,
                                        const std::string& target, const SyncParameterBounds& parameters) {
  backend::RequireLocalSyncExecutionContract(target);
  CHECK(function &&
        (function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV) &&
        transport.unavailable.empty() && !transport.sites.empty())
      << "Peer local planning requires a device endpoint and its qualified transport effects";
  auto paths = LaneStatements(function);
  const bool buffer_ir = function->ir_stage_ == FunctionIRStage::Buffer;
  if (buffer_ir) {
    const BufferRuntimeValidAnalysis valid(function, /*collect_guards=*/false,
                                           /*check_payload_capacity=*/true);
    if (const auto& issue = valid.issue()) {
      CHECK_SPAN(false, issue->call->span_) << issue->message;
    }
  }
  const SyncControlRangeProver ranges(
      function, parameters, std::nullopt,
      !buffer_ir && function->func_type_ == FunctionType::AIV ? RuntimeValidOperands : nullptr);
  const auto footprint = buffer_ir ? std::make_unique<SyncFootprintResolver>(function, &ranges, true, target,
                                                                             /*resolve_peer_slots=*/true)
                                   : nullptr;
  PeerLocalEffects result;
  for (const auto& path : paths) {
    LaneImporter importer(function, transport, target, parameters, result.gm, ranges, footprint.get());
    result.lanes.push_back(importer.Run(path));
  }
  return result;
}

}  // namespace pypto::ir::sync
