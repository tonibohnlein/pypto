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
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto::ir::sync {
namespace {
using cross_core_pipe::PeerTransportEffects;
using cross_core_pipe::PeerTransportKind;
using cross_core_pipe::PeerTransportSite;
using transform_utils::GetCallFromStmt;

struct TileWindow {
  MemorySpace space;
  int64_t begin;
  int64_t end;
  // A view inherits its original borrow and its lifetime.
  const PeerTransportSite* pop = nullptr;
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
  const auto valid = StaticExtents(view.valid_shape, span);
  CHECK_SPAN(
      shape[0] > 0 && shape[1] > 0 && valid.size() == 2 && valid[0] <= shape[0] && valid[1] <= shape[1], span)
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
  return std::make_shared<BufferType>(shape, tile->dtype_, space, valid, view.blayout, view.slayout,
                                      view.fractal, view.pad, view.compact);
}

bool IsSubblockQuery(const CallPtr& call) {
  return IsOp(call, "tile.get_subblock_idx") && call->args_.empty() && call->kwargs_.empty();
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
               std::set<size_t>& gm_formals)
      : transport_(transport), backend_(*backend::BackendConfig::GetBackend()), gm_formals_(gm_formals) {
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
      } else if (As<TensorType>(parameter->GetType())) {
        gm_roots_.emplace(parameter.get(), formal);
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
      if (const auto loop = As<ForStmt>(statements[i]))
        ImportLoop(loop);
      else
        Import(statements[i], i + 1 == statements.size());
    }
    CHECK(closed_.size() == births_.size()) << "Peer endpoint exits with an outstanding receive borrow";
    return std::move(result_);
  }

 private:
  const PeerTransportEffects& transport_;
  const backend::Backend& backend_;
  MemoryAllocatorPolicyPtr placement_;
  std::set<size_t>& gm_formals_;
  PeerLocalLane result_;
  std::unordered_map<const Stmt*, const PeerTransportSite*> sites_;
  std::unordered_set<const Call*> initializers_, payload_calls_;
  std::map<MemorySpace, std::map<int64_t, int64_t>> reserves_;
  std::map<int, int64_t> ordinals_;
  std::map<int, size_t> ring_domains_;
  std::map<int, int64_t> pops_per_visit_;

  std::unordered_map<const PeerTransportSite*, int64_t> births_;
  std::unordered_set<const PeerTransportSite*> closed_;
  std::unordered_set<const Var*> ready_;
  mutable std::unordered_set<const Expr*> ready_expressions_;
  std::unordered_map<const Var*, TileWindow> local_;
  std::unordered_map<const Var*, size_t> gm_roots_;
  std::map<MemorySpace, size_t> storage_;

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

  void Access(SyncSite& site, const TileWindow& window, bool write) {
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
      SyncAccess access{it->second, write, 0, 0, 0, 0, 1, window.end - window.begin, false, false};
      access.slot_first = (births_.at(window.pop) - 1) % ring.local_slots;
      const auto repeated = pops_per_visit_.find(id);
      access.slot_step = repeated == pops_per_visit_.end() ? 0 : repeated->second % ring.local_slots;
      if (access.slot_step) access.slot_coordinate_frame = result_.input.segments.back().original_loop;
      site.accesses.push_back(std::move(access));
      return;
    }
    site.accesses.push_back({Storage(window.space, window.end), write, 0, window.begin, 0, 0, 1,
                             window.end - window.begin, false, false});
  }

  void GmAccess(SyncSite& site, const StmtPtr& statement, const ExprPtr& value, bool write) {
    const auto it = gm_roots_.find(AsVarLike(value).get());
    CHECK_SPAN(it != gm_roots_.end(), statement->span_) << "Peer GM transfer has no original formal root";
    gm_formals_.insert(it->second);
    Access(site, {MemorySpace::DDR, 0, 1}, write);
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
    const TileWindow window{space, address->value_, address->value_ + static_cast<int64_t>(memref->size_)};
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

  void AddSite(SyncSite site) {
    CHECK_SPAN(payload_calls_.insert(site.original.get()).second, site.original->span_)
        << "A shared call cannot identify two independent physical payload cuts";
    result_.input.segments.back().sites.push_back(std::move(site));
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
    CHECK_SPAN(std::all_of(transfer.descriptor->valid_shape_.begin(), transfer.descriptor->valid_shape_.end(),
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
      const bool empty =
          transfer.memory_space == MemorySpace::Vec && transfer.descriptor->valid_shape_[0] == 0;
      const TileWindow window{ring.space, begin, begin + transfer.capacity_bytes, &transfer};
      CHECK_SPAN(local_.emplace(transfer.tile.get(), window).second, call->span_)
          << "Peer TPOP result is assigned more than once";
      births_.emplace(&transfer, ordinal);
      if (!empty) Access(site, window, true);
    } else {
      const auto window = Window(transfer.tile, call->span_);
      CHECK_SPAN(
          window.space == transfer.memory_space && window.end - window.begin >= transfer.capacity_bytes,
          call->span_)
          << "Peer TPUSH source exceeds its placed tile";
      const bool empty =
          transfer.memory_space == MemorySpace::Vec && transfer.descriptor->valid_shape_[0] == 0;
      if (!empty) Access(site, window, false);
    }
    AddSite(std::move(site));
  }

  void BindAlias(const VarPtr& destination, const ExprPtr& source, bool reshape = false) {
    const auto var = AsVarLike(source);
    if (var && local_.count(var.get())) {
      const auto window = Window(var, destination->span_);
      const auto from = DescribeTile(As<TileType>(var->GetType()), destination->span_);
      const auto dest_type = As<TileType>(destination->GetType());
      const auto to = DescribeTile(dest_type, destination->span_);
      CHECK_SPAN(structural_equal(from, to) ||
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
    }
  }

  void ImportLoop(const ForStmtPtr& loop) {
    const auto start = transform_utils::EvalConstInt(loop->start_);
    const auto stop = transform_utils::EvalConstInt(loop->stop_);
    const auto step = transform_utils::EvalConstInt(loop->step_);
    CHECK_SPAN(loop->kind_ == ForKind::Sequential && start && stop && step && *step > 0, loop->span_)
        << "Peer endpoint requires a statically bounded sequential loop";
    const __int128 trips = std::max<__int128>(0, (__int128(*stop) - *start + *step - 1) / *step);
    CHECK_SPAN(trips <= INT64_MAX && __int128(*start) + trips * *step <= INT64_MAX, loop->span_)
        << "Peer loop coordinate exceeds the index range";
    CHECK_SPAN(births_.size() == closed_.size(), loop->span_) << "A live FIFO handle crosses loop entry";
    CHECK_SPAN(loop->iter_args_.size() == loop->return_vars_.size(), loop->span_) << "Invalid loop carry";
    for (const auto& arg : loop->iter_args_) BindAlias(arg, arg->initValue_);
    ready_.insert(loop->loop_var_.get());
    auto& seg = result_.input.segments.emplace_back();
    seg.original_loop = loop;
    seg.count = static_cast<int64_t>(trips);
    seg.start = *start;
    seg.step = *step;
    std::vector<StmtPtr> body;
    Flatten(loop->body_, body);
    for (const auto& stmt : body) {
      const auto it = sites_.find(stmt.get());
      if (it != sites_.end() && it->second->kind == PeerTransportKind::Pop)
        ++pops_per_visit_[it->second->pipe_id];
    }
    const auto entry = ordinals_;
    const auto yield = body.empty() ? nullptr : As<YieldStmt>(body.back());
    if (yield) body.pop_back();
    CHECK_SPAN(loop->iter_args_.empty() ? !yield || yield->value_.empty()
                                        : yield && yield->value_.size() == loop->iter_args_.size(),
               loop->span_)
        << "Unsupported loop yield";
    for (const auto& stmt : body) {
      CHECK_SPAN(!IsA<ReturnStmt>(stmt) && !IsA<ForStmt>(stmt), stmt->span_)
          << "Peer loop body requires a single sequential region";
      Import(stmt, false);
    }
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      auto arg = loop->iter_args_[i];
      const auto value = AsVarLike(yield->value_[i]);
      CHECK_SPAN(value && gm_roots_.count(arg.get()) && gm_roots_.count(value.get()) &&
                     gm_roots_.at(arg.get()) == gm_roots_.at(value.get()),
                 loop->span_)
          << "Loop must carry an invariant tensor handle";
      BindAlias(loop->return_vars_[i], arg);
    }
    CHECK_SPAN(births_.size() == closed_.size(), loop->span_) << "A live FIFO handle crosses a loop backedge";
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
    CHECK_SPAN(call->attrs_.empty(), statement->span_)
        << "Peer local planning requires qualified native call attributes";
    if (initializers_.count(call.get())) {
      CHECK_SPAN(std::all_of(call->args_.begin(), call->args_.end(),
                             [this](const ExprPtr& arg) { return Ready(arg); }),
                 call->span_)
          << "Peer pipe initializer has an unqualified descriptor operand";
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
    } else if (IsOp(call, "tile.reshape")) {
      CHECK_SPAN(result && call->args_.size() == 2, call->span_) << "Unqualified peer Tile reshape";
      RequireReadyTuple(call->args_[1], call->span_);
      BindAlias(result, call->args_[0], /*reshape=*/true);
    } else if (IsOp(call, "tile.store")) {
      ImportTransfer(statement, call, result, false);
    } else {
      ImportCompute(statement, call, result);
    }
  }
};
}  // namespace

PeerLocalEffects ImportPeerLocalEffects(const FunctionPtr& function, const PeerTransportEffects& transport,
                                        const std::string& target) {
  backend::RequireLocalSyncExecutionContract(target);
  CHECK(function && function->ir_stage_ == FunctionIRStage::Functional &&
        (function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV) &&
        transport.unavailable.empty() && !transport.sites.empty())
      << "Peer local planning requires a native Tile endpoint and its qualified transport effects";
  auto paths = LaneStatements(function);
  PeerLocalEffects result;
  for (const auto& path : paths) {
    LaneImporter importer(function, transport, target, result.gm_formals);
    result.lanes.push_back(importer.Run(path));
  }
  return result;
}

}  // namespace pypto::ir::sync
