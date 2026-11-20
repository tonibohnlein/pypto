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

#include "pypto/ir/transforms/utils/cross_core_pipe.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/core_side_ops.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace cross_core_pipe {

namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;

struct ConsumerFifoResult {
  std::map<int, size_t> peak_depth;
  size_t first_transport;
  bool unsupported_control = false;
};

bool IsPeerTransport(const CallPtr& call) {
  return op_predicates::IsTPush(call) || op_predicates::IsTPop(call) || op_predicates::IsTFree(call);
}

int IncomingDirection(core_affinity::CoreSide side) {
  return side == core_affinity::CoreSide::AIC ? core_affinity::kDirMaskV2C : core_affinity::kDirMaskC2V;
}

using ReserveDeclarations = std::unordered_map<const Var*, CallPtr>;

// Tile TPOP on A2/A3 issues TLOAD on MTE2, with the local slot offset based
// on SLOT_SIZE even when its payload is shorter (TPush.hpp: Consumer::pop).
// The incoming operand names a local reserve. A scalar address or imported
// outbound address does not establish the identity and extent of that reserve.
std::optional<ConsumerReceiveStorage> BindConsumerStorage(const CallPtr& setup, core_affinity::CoreSide side,
                                                          const ReserveDeclarations& reserves) {
  const int mask = setup->GetKwarg<int>("dir_mask", 0);
  const int both = core_affinity::kDirMaskV2C | core_affinity::kDirMaskC2V;
  const int slots = setup->GetKwarg<int>("slot_num", GetPtoasImplicitSlotNum(mask));
  const int local = setup->GetKwarg<int>("local_slot_num", slots);
  const int64_t stride = setup->GetKwarg<int>("slot_size", 0);
  if (!(mask & IncomingDirection(side)) || (mask & ~both) || local <= 0 || local > slots || stride <= 0 ||
      setup->args_.size() != 2) {
    return std::nullopt;
  }
  const bool cube = side == core_affinity::CoreSide::AIC;
  const auto reserve = AsVarLike(setup->args_[cube ? 1 : 0]);
  const auto declaration = reserves.find(reserve.get());
  if (declaration == reserves.end()) {
    return std::nullopt;
  }
  const int64_t size = declaration->second->GetKwarg<int>("size", 0);
  // All operands are widened from i32. The depth-stride product and the
  // reserve's placed end therefore fit i64 before performing the arithmetic.
  if (size < local * stride) {
    return std::nullopt;
  }
  const int64_t base = declaration->second->GetKwarg<int>("base", kAutoBufferBase);
  return ConsumerReceiveStorage{
      reserve, cube ? MemorySpace::Mat : MemorySpace::Vec,    PipeType::MTE2, local, stride,
      size,    base >= 0 ? std::optional{base} : std::nullopt};
}

// Each nested body closes its own borrows without popping an inherited queue
// entry. Thus it restores the incoming state even for skipped or repeated visits.
// Independent pipe IDs have independent slots and can release in either order.
class ConsumerFifo {
 public:
  using TransportCuts = std::vector<std::pair<StmtPtr, int>>;

  ConsumerFifo(core_affinity::CoreSide side, bool established_pipe_ids, TransportCuts* cuts = nullptr)
      : pop_(core_side_ops::TPopOp(side)),
        free_(core_side_ops::TFreeOp(side)),
        established_pipe_ids_(established_pipe_ids),
        cuts_(cuts) {}

  ConsumerFifoResult Run(const std::vector<StmtPtr>& stmts) {
    result_.first_transport = stmts.size();
    for (root_index_ = 0; root_index_ < stmts.size(); ++root_index_) Read(stmts[root_index_], 0, false);
    CHECK(active_.empty()) << "Cross-core FIFO requires a logical tfree for every tpop";
    return std::move(result_);
  }

 private:
  struct Borrow {
    int id;
    size_t scope;
  };

  void Scope(const StmtPtr& body, size_t scope, bool parallel) {
    const auto entry = active_.size();
    Read(body, scope + 1, parallel);
    CHECK_SPAN(active_.size() == entry, body->span_)
        << "Cross-core FIFO cannot prove ownership escaping a control-flow body; "
           "place/consume and release each gathered tile within its scope.";
  }

  void Read(const StmtPtr& stmt, size_t scope, bool parallel) {
    if (const auto call = transform_utils::GetCallFromStmt(stmt)) {
      if (IsPeerTransport(call)) {
        result_.first_transport = std::min(result_.first_transport, root_index_);
        result_.unsupported_control |= parallel;
        if (cuts_ && !op_predicates::IsTFree(call)) cuts_->emplace_back(stmt, call->GetKwarg<int>("id", 0));
      }
      if (op_predicates::IsTPop(call)) {
        const auto assign = As<AssignStmt>(stmt);
        CHECK_SPAN(assign && IsOp(call, pop_), stmt->span_)
            << "Cross-core FIFO requires a tpop result on its consumer core";
        const int id = established_pipe_ids_ ? call->GetKwarg<int>("id", 0) : 0;
        CHECK_SPAN(active_.emplace(assign->var_.get(), Borrow{id, scope}).second, stmt->span_)
            << "Cross-core FIFO has a repeated outstanding tpop result";
        auto& queue = queues_[id];
        queue.push_back(assign->var_.get());
        auto& peak = result_.peak_depth[id];
        peak = std::max(peak, queue.size());
      } else if (op_predicates::IsTFree(call)) {
        const auto tile = call->args_.empty() ? nullptr : AsVarLike(call->args_[0]);
        const auto borrow = active_.find(tile.get());
        CHECK_SPAN(IsOp(call, free_) && borrow != active_.end() && borrow->second.scope == scope, stmt->span_)
            << "Cross-core FIFO cannot match tfree with an outstanding tpop in the same scope";
        const int id = borrow->second.id;
        auto& queue = queues_.at(id);
        CHECK_SPAN(
            (!established_pipe_ids_ || call->GetKwarg<int>("id", id) == id) && queue.front() == tile.get(),
            stmt->span_)
            << "Cross-core pipe requires FIFO release order within each pipe; a retained earlier tile "
               "must be consumed before a later gathered tile is released.";
        queue.pop_front();
        if (cuts_) cuts_->emplace_back(stmt, id);
        active_.erase(borrow);
      }
    }
    if (const auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) Read(child, scope, parallel);
    } else if (const auto loop = As<ForStmt>(stmt)) {
      Scope(loop->body_, scope,
            parallel || loop->kind_ == ForKind::Parallel || loop->kind_ == ForKind::Pipeline);
    } else if (const auto loop = As<WhileStmt>(stmt)) {
      Scope(loop->body_, scope, parallel);
    } else if (const auto branch = As<IfStmt>(stmt)) {
      Scope(branch->then_body_, scope, parallel);
      if (branch->else_body_) Scope(*branch->else_body_, scope, parallel);
    } else if (const auto region = As<ScopeStmt>(stmt)) {
      Scope(region->body_, scope, true);
    } else if (const auto region = As<SimtScopeStmt>(stmt)) {
      Scope(region->body_, scope, true);
    } else if (As<BreakStmt>(stmt) || As<ContinueStmt>(stmt) || As<InlineStmt>(stmt) ||
               As<SimtLaunchStmt>(stmt) || (As<ReturnStmt>(stmt) && !active_.empty())) {
      result_.unsupported_control = true;
    }
  }

  std::string pop_;
  std::string free_;
  bool established_pipe_ids_;
  TransportCuts* cuts_;
  std::unordered_map<const Var*, Borrow> active_;
  std::unordered_map<int, std::deque<const Var*>> queues_;
  ConsumerFifoResult result_;
  size_t root_index_ = 0;
};

size_t AutomaticConsumerDepth(const std::vector<StmtPtr>& stmts, core_affinity::CoreSide side) {
  // Automatic setup constructs just one pipe. Size the original aggregate
  // queue without interpreting handwritten operation IDs as independent pools.
  const auto fifo = ConsumerFifo(side, false).Run(stmts);
  return fifo.peak_depth.empty() ? 0 : fifo.peak_depth.at(0);
}

class PeerCutVerifier : public IRVisitor {
 public:
  explicit PeerCutVerifier(const std::unordered_set<const Call*>& initializers)
      : initializers_(initializers) {}

  void VisitStmt(const StmtPtr& stmt) override {
    const auto call = transform_utils::GetCallFromStmt(stmt);
    expected_ = call.get();
    IRVisitor::VisitStmt(stmt);
  }

  void VisitExpr(const ExprPtr& expr) override {
    const auto call = As<Call>(expr);
    const bool at_cut = expected_ == expr.get();
    if (at_cut) expected_ = nullptr;
    CHECK(!call || !As<GlobalVar>(call->op_)) << "Outlined calls do not establish local peer FIFO order";
    const bool initializes = op_predicates::IsInitializePipe(call);
    if (IsPeerTransport(call) || initializes) {
      // Unrolling can share an immutable TPOP expression across assignments.
      // ConsumerFifo counts every statement occurrence and matches each borrow
      // using that assignment's result, even when the right-hand side is shared.
      CHECK_SPAN(at_cut, call->span_) << "Peer FIFO requires a peer operation at a statement cut";
      if (initializes) {
        const bool first = seen_initializers_.insert(call.get()).second;
        CHECK_SPAN(first && initializers_.count(call.get()), call->span_)
            << "Peer FIFO requires initializers that dominate all transport cuts";
      }
    }
    if (seen_values_.insert(expr.get()).second) IRVisitor::VisitExpr(expr);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}

 private:
  const std::unordered_set<const Call*>& initializers_;
  const Call* expected_ = nullptr;
  std::unordered_set<const Call*> seen_initializers_;
  std::unordered_set<const Expr*> seen_values_;
};

std::optional<Span> FindOverwrittenConsumer(const std::vector<StmtPtr>& stmts, size_t begin, int id,
                                            const ConsumerReceiveStorage& storage) {
  std::vector<std::pair<const Var*, int64_t>> prior_slots;
  std::unordered_set<const Var*> overwritten;
  std::unordered_map<const Var*, const Var*> aliases;
  size_t slot = 0;
  for (size_t i = begin; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];
    const auto call = transform_utils::GetCallFromStmt(stmt);
    const auto assign = As<AssignStmt>(stmt);
    // Only an unconditional read with a proven overwritten source is an
    // error here. Branch, loop and unknown-footprint liveness remain outside
    // this witness. TFREE does not change the physical receive cursor.
    if (!assign && !As<EvalStmt>(stmt)) {
      break;
    }
    if (op_predicates::IsTPop(call) && call->GetKwarg<int>("id", 0) == id) {
      if (!assign) {
        return std::nullopt;
      }
      const auto tile = backend::DescribePeerTransferBuffer(As<TileType>(assign->var_->GetType()));
      const auto type = tile ? tile->descriptor : nullptr;
      const bool full_vec = type && type->memory_space_ == MemorySpace::Vec &&
                            type->dtype_ == DataType::FP32 && type->pad_ == PadValue::null &&
                            type->valid_shape_ == type->shape_;
      const auto elements = full_vec && call->GetKwarg<int>("split", 0) == kSplitNone &&
                                    tile->capacity_bytes <= storage.slot_bytes
                                ? tile->capacity_bytes / 4
                                : 0;
      const auto current = std::pair{assign->var_.get(), elements};
      if (slot < prior_slots.size()) {
        const auto& [earlier, earlier_elements] = prior_slots[slot];
        if (earlier_elements > 0 && earlier_elements <= elements) {
          overwritten.insert(earlier);
        }
        prior_slots[slot] = current;
      } else {
        prior_slots.push_back(current);
      }
      slot = (slot + 1) % storage.local_slots;
      continue;
    }
    if (assign) {
      if (const auto source = AsVarLike(assign->value_)) {
        const auto original = aliases.find(source.get());
        aliases.emplace(assign->var_.get(), original == aliases.end() ? source.get() : original->second);
      }
    }
    if (overwritten.empty()) {
      continue;
    }
    const auto* op = call ? LookupOpEntry(call->op_) : nullptr;
    const bool reads_data =
        op && !op_predicates::IsTFree(call) &&
        (op->GetExecutionMemoryAccessEvidence() == ExecutionMemoryAccessEvidence::Functional ||
         IsOp(call, "tile.store"));
    if (!reads_data) {
      continue;
    }
    for (size_t arg = 0; arg < call->args_.size(); ++arg) {
      const auto value = AsVarLike(call->args_[arg]);
      const auto original = aliases.find(value.get());
      const auto* root = original == aliases.end() ? value.get() : original->second;
      const bool reads_arg = ArgEffectReads(op->GetArgEffect(arg, call->kwargs_));
      const bool expired = overwritten.count(root) != 0;
      if (expired && reads_arg) {
        return stmt->span_;
      }
    }
  }
  return std::nullopt;
}

struct C2VReceiveSite {
  const Stmt* stmt;
  int pipe_id;
  const Stmt* loop;
  bool unconditional;
  bool may_write;
  std::optional<uint64_t> tile_bytes;
};

// A no-split Vec TPOP with zero valid rows passes nBurst=0 to TLOAD.
// Restrict the proof to this identified no-data case; a zero column alone
// is a separate native width/padding case. Both still advance the FIFO.
bool EmptyReceive(const BufferTypePtr& type) {
  return type->memory_space_ == MemorySpace::Vec && type->valid_shape_.front() == 0;
}

class C2VReceiveSites : public IRVisitor {
 public:
  void VisitStmt(const StmtPtr& stmt) override {
    const auto call = transform_utils::GetCallFromStmt(stmt);
    if (IsOp(call, "system.aiv_initialize_pipe")) {
      RecordRing(call);
    } else if (IsOp(call, "tile.tpop_from_aic")) {
      RecordPop(stmt, call);
    } else if (IsOp(call, "system.reserve_buffer")) {
      if (const auto assign = As<AssignStmt>(stmt)) {
        reserve_declarations.emplace(assign->var_.get(), call);
      }
    } else if (IsOp(call, "system.wait_flag")) {
      has_explicit_wait = true;
    } else if (loop_ && (As<BreakStmt>(stmt) || As<ContinueStmt>(stmt) || As<ReturnStmt>(stmt))) {
      early_exit_loops.insert(loop_);
    }
    IRVisitor::VisitStmt(stmt);
  }

  // Only statement cuts affect transport. Skip the scalar expression DAG.
  void VisitExpr(const ExprPtr&) override {}

  std::vector<C2VReceiveSite> sites;
  std::map<int, std::optional<ConsumerReceiveStorage>> rings;
  std::unordered_set<const Stmt*> early_exit_loops;
  bool has_explicit_wait = false;

 protected:
  void VisitStmt_(const ForStmtPtr& loop) override { VisitLoop(loop, loop->body_); }
  void VisitStmt_(const WhileStmtPtr& loop) override { VisitLoop(loop, loop->body_); }
  void VisitStmt_(const IfStmtPtr& branch) override {
    const bool unconditional = unconditional_;
    unconditional_ = false;
    IRVisitor::VisitStmt_(branch);
    unconditional_ = unconditional;
  }

 private:
  void VisitLoop(const StmtPtr& loop, const StmtPtr& body) {
    const auto enclosing = loop_;
    const bool unconditional = unconditional_;
    if (!enclosing) {
      loop_ = loop.get();
    }
    unconditional_ = !enclosing;
    VisitStmt(body);
    loop_ = enclosing;
    unconditional_ = unconditional;
  }

  void RecordRing(const CallPtr& call) {
    constexpr auto side = core_affinity::CoreSide::AIV;
    if (!(call->GetKwarg<int>("dir_mask", 0) & IncomingDirection(side))) {
      return;
    }
    const int id = call->GetKwarg<int>("id", 0);
    const auto [entry, unique] = rings.emplace(id, BindConsumerStorage(call, side, reserve_declarations));
    if (!unique) {
      entry->second.reset();
    }
  }

  void RecordPop(const StmtPtr& stmt, const CallPtr& call) {
    const auto assign = As<AssignStmt>(stmt);
    const auto tile = assign ? As<TileType>(assign->var_->GetType()) : nullptr;
    if (!tile || tile->memory_space_ != MemorySpace::Vec) {
      return;
    }
    if (const auto buffer = backend::DescribePeerTransferBuffer(tile);
        buffer && call->GetKwarg<int>("split", kSplitNone) == kSplitNone) {
      const bool empty = EmptyReceive(buffer->descriptor);
      sites.push_back({stmt.get(), call->GetKwarg<int>("id", 0), loop_, unconditional_, !empty,
                       static_cast<uint64_t>(buffer->capacity_bytes)});
      return;
    }
    // Delegated variants outside the native transport contract retain their
    // conservative fence when placement or payload geometry is unknown.
    const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
    const bool empty = std::any_of(view.valid_shape.begin(), view.valid_shape.end(),
                                   [](const ExprPtr& dim) { return TryGetNonNegativeConstInt(dim) == 0; });
    sites.push_back({stmt.get(), call->GetKwarg<int>("id", 0), loop_, unconditional_, !empty,
                     utils::StaticPhysicalAllocationBytes(tile, MemorySpace::Vec, nullptr)});
  }

  ReserveDeclarations reserve_declarations;
  const Stmt* loop_ = nullptr;
  bool unconditional_ = false;
};

bool MayAliasOtherRings(const C2VReceiveSites& receives) {
  std::map<int64_t, int64_t> address_ranges;
  std::unordered_set<const Var*> seen_reserves;
  for (const auto& site : receives.sites) {
    const auto entry = receives.rings.find(site.pipe_id);
    if (entry == receives.rings.end() || !entry->second || !entry->second->placed_base) {
      return true;
    }
    const auto& ring = *entry->second;
    const auto base = *ring.placed_base;
    if (site.may_write && (!site.tile_bytes || *site.tile_bytes > static_cast<uint64_t>(ring.slot_bytes))) {
      return true;
    }
    if (seen_reserves.insert(ring.reserve.get()).second &&
        !address_ranges.emplace(base, base + ring.reserve_bytes).second) {
      return true;
    }
  }
  int64_t end = 0;
  for (const auto& [base, next_end] : address_ranges) {
    if (base < end) {
      return true;
    }
    end = next_end;
  }
  return false;
}

struct LocalSlotHistory {
  int distinct_sites = 0;
  int first_pipe = 0;
  int safe_depth = 1;
  int preceding = 0;
  const Stmt* uniform_loop = nullptr;
  size_t period = 0;

  void Include(const C2VReceiveSite& site, int local_slots) {
    if (!site.may_write) {
      return;
    }
    if (distinct_sites == 0) {
      first_pipe = site.pipe_id;
      safe_depth = local_slots;
      uniform_loop = site.loop;
    }
    if (!site.unconditional || site.loop != uniform_loop) {
      uniform_loop = nullptr;
    }
    // Different pipe counters sharing a reserve can start at the same byte.
    if (site.pipe_id != first_pipe) {
      safe_depth = 1;
    }
    distinct_sites = std::min(2, distinct_sites + 1);
  }

  void ObservePeriod(const C2VReceiveSite& site) {
    if (uniform_loop && site.loop == uniform_loop) {
      if (site.unconditional) {
        ++period;
      } else {
        uniform_loop = nullptr;
      }
    }
  }

  bool NeedsFence(const C2VReceiveSite& site, const std::unordered_set<const Stmt*>& early_exits) const {
    // In an exact repeated sequence, a slot revisited after local_slots
    // receives belongs to the original static site iff that distance is an
    // integer multiple of the sequence's period.
    const bool stock_covers_period = uniform_loop && period != 0 && !early_exits.count(uniform_loop) &&
                                     static_cast<size_t>(safe_depth) % period == 0;
    return site.may_write && distinct_sites > 1 && !stock_covers_period &&
           (site.loop || preceding >= safe_depth);
  }

  void Advance(const C2VReceiveSite& site) {
    if (site.loop) {
      preceding = safe_depth;
    } else if (preceding < safe_depth) {
      ++preceding;
    }
  }
};

}  // namespace

std::optional<int64_t> TryGetNonNegativeConstInt(const ExprPtr& expr) {
  auto value = transform_utils::EvalConstInt(expr);
  if (!value.has_value() || *value < 0) return std::nullopt;
  return value;
}

std::optional<int64_t> TryGetTileSlotSizeBytes(const TypePtr& type) {
  auto tile_type = std::dynamic_pointer_cast<const TileType>(type);
  if (!tile_type) return std::nullopt;

  int64_t element_count = 1;
  for (const auto& dim : tile_type->shape_) {
    auto dim_value = TryGetNonNegativeConstInt(dim);
    if (!dim_value.has_value()) return std::nullopt;
    INTERNAL_CHECK(*dim_value == 0 || element_count <= std::numeric_limits<int64_t>::max() / *dim_value)
        << "Tile element count overflow while inferring cross-core slot size";
    element_count *= *dim_value;
  }

  const int64_t bit_width = static_cast<int64_t>(tile_type->dtype_.GetBit());
  INTERNAL_CHECK(bit_width > 0) << "Unsupported dtype for cross-core slot size inference: "
                                << tile_type->dtype_.ToString();
  INTERNAL_CHECK(element_count <= (std::numeric_limits<int64_t>::max() - 7) / bit_width)
      << "Tile byte size overflow while inferring cross-core slot size";
  return (element_count * bit_width + 7) / 8;
}

void RecordObservedSlotSize(PipeDirectionMetadata& metadata, int64_t slot_size) {
  metadata.has_ops = true;
  if (std::find(metadata.observed_slot_sizes.begin(), metadata.observed_slot_sizes.end(), slot_size) ==
      metadata.observed_slot_sizes.end()) {
    metadata.observed_slot_sizes.push_back(slot_size);
  }
  if (!metadata.slot_size_bytes.has_value()) {
    metadata.slot_size_bytes = slot_size;
    return;
  }
  if (metadata.slot_size_bytes.value() != slot_size) {
    metadata.has_inconsistent_slot_size = true;
    metadata.slot_size_bytes = std::max(metadata.slot_size_bytes.value(), slot_size);
  }
}

void RecordTileSlotSize(PipeDirectionMetadata& metadata, const TypePtr& type) {
  metadata.has_ops = true;
  auto slot_size = TryGetTileSlotSizeBytes(type);
  if (slot_size.has_value()) {
    RecordObservedSlotSize(metadata, slot_size.value());
  }
}

void MergeDirectionMetadata(PipeDirectionMetadata& dst, const PipeDirectionMetadata& src) {
  dst.has_ops = dst.has_ops || src.has_ops;
  dst.has_inconsistent_slot_size = dst.has_inconsistent_slot_size || src.has_inconsistent_slot_size;
  for (int64_t slot_size : src.observed_slot_sizes) {
    RecordObservedSlotSize(dst, slot_size);
  }
}

CrossCorePipeMetadata MergeCrossCorePipeMetadata(const CrossCorePipeMetadata& lhs,
                                                 const CrossCorePipeMetadata& rhs) {
  CrossCorePipeMetadata merged;
  MergeDirectionMetadata(merged.c2v, lhs.c2v);
  MergeDirectionMetadata(merged.c2v, rhs.c2v);
  MergeDirectionMetadata(merged.v2c, lhs.v2c);
  MergeDirectionMetadata(merged.v2c, rhs.v2c);
  merged.has_reserve_buffer = lhs.has_reserve_buffer || rhs.has_reserve_buffer;
  merged.has_import_peer_buffer = lhs.has_import_peer_buffer || rhs.has_import_peer_buffer;
  merged.has_aic_initialize_pipe = lhs.has_aic_initialize_pipe || rhs.has_aic_initialize_pipe;
  merged.has_aiv_initialize_pipe = lhs.has_aiv_initialize_pipe || rhs.has_aiv_initialize_pipe;
  return merged;
}

int BuildDirMask(const CrossCorePipeMetadata& metadata) {
  int dir_mask = 0;
  if (metadata.c2v.has_ops) dir_mask |= core_affinity::kDirMaskC2V;
  if (metadata.v2c.has_ops) dir_mask |= core_affinity::kDirMaskV2C;
  return dir_mask;
}

int GetPtoasImplicitSlotNum(int dir_mask) {
  return dir_mask == (core_affinity::kDirMaskC2V | core_affinity::kDirMaskV2C) ? 4 : 8;
}

std::optional<int64_t> GetCommonSlotSizeBytes(const CrossCorePipeMetadata& metadata) {
  std::optional<int64_t> common_slot_size;
  for (const auto* direction : {&metadata.c2v, &metadata.v2c}) {
    if (!direction->has_ops) continue;
    if (!direction->slot_size_bytes.has_value()) {
      return std::nullopt;
    }
    if (!common_slot_size.has_value()) {
      common_slot_size = direction->slot_size_bytes;
      continue;
    }
    common_slot_size = std::max(common_slot_size.value(), direction->slot_size_bytes.value());
  }
  return common_slot_size;
}

std::string BuildPipeBufferName(const std::string& func_name, core_affinity::PipeDirection direction) {
  return func_name +
         ((direction == core_affinity::PipeDirection::C2V) ? "_c2v_slot_buffer" : "_v2c_slot_buffer");
}

CallPtr CreateSystemOpCall(const std::string& op_name,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span) {
  return CreateSystemOpCall(op_name, {}, kwargs, span);
}

CallPtr CreateSystemOpCall(const std::string& op_name, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span) {
  return OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
}

CallPtr CreateReserveBuffer(const std::string& buffer_name, int64_t size_bytes, const Span& span) {
  INTERNAL_CHECK_SPAN(size_bytes >= 0 && size_bytes <= std::numeric_limits<int>::max(), span)
      << "Cross-core reserve_buffer size out of range: " << size_bytes;
  return CreateSystemOpCall("system.reserve_buffer",
                            {{"name", std::any(buffer_name)},
                             {"size", std::any(static_cast<int>(size_bytes))},
                             {"base", std::any(kAutoBufferBase)}},
                            span);
}

CallPtr CreateImportPeerBuffer(const std::string& buffer_name, const std::string& peer_func,
                               const Span& span) {
  return CreateSystemOpCall("system.import_peer_buffer",
                            {{"name", std::any(buffer_name)}, {"peer_func", std::any(peer_func)}}, span);
}

CallPtr CreateInitializePipe(core_affinity::CoreSide side, int dir_mask, int slot_size_bytes,
                             const ExprPtr& c2v_consumer_buf, const ExprPtr& v2c_consumer_buf, int slot_num,
                             std::optional<int> local_slot_num, const Span& span) {
  INTERNAL_CHECK_SPAN(slot_size_bytes >= 0 && slot_size_bytes <= std::numeric_limits<int>::max(), span)
      << "Cross-core slot_size out of range: " << slot_size_bytes;
  INTERNAL_CHECK_SPAN(slot_num > 0, span) << "Cross-core slot_num must be positive: " << slot_num;
  // Always emit the transport depth so GM sizing and peer ring indexing agree.
  // The optional local depth separately describes the on-chip reserve.
  std::vector<std::pair<std::string, std::any>> kwargs = {{"dir_mask", std::any(dir_mask)},
                                                          {"slot_size", std::any(slot_size_bytes)},
                                                          {"slot_num", std::any(slot_num)}};
  if (local_slot_num.has_value()) {
    kwargs.emplace_back("local_slot_num", local_slot_num.value());
  }
  const std::string op_name = core_side_ops::InitializePipeOp(side);
  return CreateSystemOpCall(op_name, {c2v_consumer_buf, v2c_consumer_buf}, kwargs, span);
}

void CollectCrossCorePipeMetadata(const std::vector<StmtPtr>& stmts, CrossCorePipeMetadata& metadata) {
  for (const auto& stmt : stmts) {
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
    CallPtr call;
    if (assign) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (eval) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    if (op) {
      if (IsOp(op, "system.reserve_buffer")) {
        metadata.has_reserve_buffer = true;
      } else if (IsOp(op, "system.import_peer_buffer")) {
        metadata.has_import_peer_buffer = true;
      } else if (IsOp(op, "system.aic_initialize_pipe")) {
        metadata.has_aic_initialize_pipe = true;
      } else if (IsOp(op, "system.aiv_initialize_pipe")) {
        metadata.has_aiv_initialize_pipe = true;
      } else if (IsOp(op, "tile.tpush_to_aiv") && call->args_.size() == 1) {
        RecordTileSlotSize(metadata.c2v, call->args_[0]->GetType());
      } else if (IsOp(op, "tile.tpush_to_aic") && call->args_.size() == 1) {
        RecordTileSlotSize(metadata.v2c, call->args_[0]->GetType());
      } else if (IsOp(op, "tile.tpop_from_aiv") && assign) {
        RecordTileSlotSize(metadata.v2c, assign->var_->GetType());
      } else if (IsOp(op, "tile.tpop_from_aic") && assign) {
        RecordTileSlotSize(metadata.c2v, assign->var_->GetType());
      }
    }

    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(for_stmt->body_), metadata);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(if_stmt->then_body_), metadata);
      const auto& else_body = if_stmt->else_body_;
      if (else_body) {
        CollectCrossCorePipeMetadata(FlattenBody(*else_body), metadata);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(while_stmt->body_), metadata);
    }
  }
}

CrossCorePipeMetadata CollectDominatingPipeSetupMetadata(const std::vector<StmtPtr>& stmts) {
  CrossCorePipeMetadata metadata;
  for (const auto& stmt : stmts) {
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
    CallPtr call;
    if (assign) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (eval) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    CrossCorePipeMetadata stmt_metadata;
    CollectCrossCorePipeMetadata({stmt}, stmt_metadata);
    if (stmt_metadata.HasCrossCoreOps()) {
      break;
    }
    if (op) {
      if (IsOp(op, "system.reserve_buffer")) {
        metadata.has_reserve_buffer = true;
      } else if (IsOp(op, "system.import_peer_buffer")) {
        metadata.has_import_peer_buffer = true;
      } else if (IsOp(op, "system.aic_initialize_pipe")) {
        metadata.has_aic_initialize_pipe = true;
      } else if (IsOp(op, "system.aiv_initialize_pipe")) {
        metadata.has_aiv_initialize_pipe = true;
      }
    }
  }
  return metadata;
}

namespace {

PeerPipeBinding BindNativePipe(const CallPtr& call, core_affinity::CoreSide side,
                               const std::optional<ConsumerReceiveStorage>& receive) {
  const int mask = call->GetKwarg<int>("dir_mask", 0);
  const int depth = call->GetKwarg<int>("slot_num", GetPtoasImplicitSlotNum(mask));
  const int64_t stride = call->GetKwarg<int>("slot_size", 0);
  CHECK(mask >= 1 && mask <= 3 && call->args_.size() == 2 && depth > 0 && stride > 0)
      << "Unqualified tile FIFO direction or ring geometry";
  CHECK(!call->HasKwarg("c2v_dtype") && !call->GetKwarg<bool>("pre_relu", false))
      << "Peer FIFO requires ordinary A2/A3 tile transport";
  // The pinned TPipe computes its bidirectional GM partition in int and
  // computes its tile-offset products before widening to a GM address.
  CHECK(stride * depth <= std::numeric_limits<int32_t>::max())
      << "Peer FIFO ring exceeds native offset range";
  CHECK(!(mask & IncomingDirection(side)) || receive)
      << "Peer FIFO receive ring has no local storage binding";
  return {call, mask, depth, stride, receive};
}

PeerTransportSite BindNativeSite(const StmtPtr& stmt, int id, core_affinity::CoreSide side,
                                 const PeerPipeBinding& pipe) {
  const auto call = transform_utils::GetCallFromStmt(stmt);
  const bool push = op_predicates::IsTPush(call);
  const bool pop = op_predicates::IsTPop(call);
  CHECK(!push || IsOp(call, core_side_ops::TPushOp(side))) << "Peer push belongs to the other core";
  CHECK(call->GetKwarg<int>("split", push ? -1 : kSplitNone) == kSplitNone)
      << "Peer FIFO requires no-split tile transfer";
  const bool cube = side == core_affinity::CoreSide::AIC;
  const int direction = (push == cube) ? core_affinity::kDirMaskC2V : core_affinity::kDirMaskV2C;
  CHECK(pipe.direction_mask & direction) << "Peer FIFO transport direction is disabled on its initializer";
  CHECK(call->args_.size() == (pop ? 0 : 1)) << "Peer FIFO requires unary tile transport";
  const auto tile = pop ? As<AssignStmt>(stmt)->var_ : AsVarLike(call->args_[0]);
  const auto type = tile ? As<TileType>(tile->GetType()) : nullptr;
  const auto space = cube ? (push ? MemorySpace::Acc : MemorySpace::Mat) : MemorySpace::Vec;
  CHECK(type && type->memory_space_ == space) << "Peer FIFO tile has no qualified native memory space";
  const auto buffer = backend::DescribePeerTransferBuffer(type);
  CHECK(buffer && buffer->capacity_bytes <= pipe.slot_bytes) << "Peer FIFO cannot enclose tile in one slot";
  const int ready = pipe.direction_mask == 3 && direction == core_affinity::kDirMaskV2C ? 2 : 0;
  const auto data = push ? (cube ? PipeType::FIX : PipeType::MTE3) : PipeType::MTE2;
  const auto kind = push ? PeerTransportKind::Push : pop ? PeerTransportKind::Pop : PeerTransportKind::Free;
  return {stmt, kind, id, tile, space, data, buffer->capacity_bytes, buffer->descriptor, ready, ready + 1};
}

ConsumerFifoEvidence InspectConsumerFifo(const FunctionPtr& function, PeerTransportEffects* effects) {
  if (function->func_type_ != FunctionType::AIC && function->func_type_ != FunctionType::AIV) return {};
  const auto side =
      function->func_type_ == FunctionType::AIC ? core_affinity::CoreSide::AIC : core_affinity::CoreSide::AIV;
  const auto stmts = FlattenBody(function->body_);
  ConsumerFifoEvidence evidence;
  try {
    ConsumerFifo::TransportCuts cuts;
    const auto fifo = ConsumerFifo(side, true, effects ? &cuts : nullptr).Run(stmts);
    if (fifo.first_transport == stmts.size()) return {};
    evidence.has_transport = true;
    CHECK(!fifo.unsupported_control)
        << "Concurrent, outlined or early-exit control does not prove local FIFO order";
    const auto initialize = core_side_ops::InitializePipeOp(side);
    ReserveDeclarations reserves;
    std::map<int, std::optional<ConsumerReceiveStorage>> local_storage;
    std::unordered_set<const Call*> prologue_initializers;
    for (size_t i = 0; i < fifo.first_transport; ++i) {
      const auto& stmt = stmts[i];
      const auto call = transform_utils::GetCallFromStmt(stmt);
      if (!call) continue;
      if (IsOp(call, "system.reserve_buffer")) {
        if (const auto assign = As<AssignStmt>(stmt)) reserves.emplace(assign->var_.get(), call);
      } else if (IsOp(call, initialize)) {
        prologue_initializers.insert(call.get());
        const auto id = call->GetKwarg<int>("id", 0);
        const auto receive = BindConsumerStorage(call, side, reserves);
        if (call->GetKwarg<int>("dir_mask", 0) & IncomingDirection(side)) {
          const bool unique = local_storage.emplace(id, receive).second;
          CHECK(unique) << "More than one dominating consumer initializer uses the same peer pipe id";
        }
        if (effects) {
          const bool unique = effects->pipes.emplace(id, BindNativePipe(call, side, receive)).second;
          CHECK(unique && id >= 0) << "Peer FIFO requires a unique nonnegative logical pipe id";
        }
      }
    }
    PeerCutVerifier(prologue_initializers).VisitStmt(function->body_);
    for (const auto& [id, required] : fifo.peak_depth) {
      const auto it = local_storage.find(id);
      CHECK(it != local_storage.end()) << "No dominating consumer initializer for peer pipe id " << id;
      const auto& storage = it->second;
      CHECK(storage) << "Invalid consumer slot geometry or missing local reserve for peer pipe id " << id;
      evidence.capacities.emplace(id, ConsumerPipeCapacity{required, *storage});
      const bool can_prove_slot_reuse =
          side == core_affinity::CoreSide::AIV && prologue_initializers.size() == 1;
      if (effects) {
        CHECK(required <= static_cast<size_t>(storage->local_slots))
            << "Peer FIFO outstanding borrows exceed its local slot count";
      }
      if (can_prove_slot_reuse) {
        if (const auto overwritten = FindOverwrittenConsumer(stmts, fifo.first_transport, id, *storage)) {
          evidence.overwritten_input.emplace(*overwritten);
        }
      }
    }
    if (effects) {
      CHECK(!evidence.overwritten_input) << "Peer FIFO has a native read of an overwritten receive slot";
      for (const auto& [stmt, id] : cuts) {
        const auto binding = effects->pipes.find(id);
        CHECK(binding != effects->pipes.end()) << "No dominating initializer for peer pipe id " << id;
        effects->sites.push_back(BindNativeSite(stmt, id, side, binding->second));
      }
    }
  } catch (const pypto::ValueError& error) {
    return {true, {}, error.what(), std::nullopt};
  }
  return evidence;
}

}  // namespace

ConsumerFifoEvidence AnalyzeConsumerFifo(const FunctionPtr& function) {
  return InspectConsumerFifo(function, nullptr);
}

PeerTransportEffects AnalyzePeerTransportEffects(const FunctionPtr& function) {
  if (!backend::BackendConfig::IsConfigured() ||
      backend::BackendConfig::GetBackend()->GetHandler()->GetPtoTargetArch() != "a2a3") {
    return {"Native tile FIFO effects require the A2/A3 transport", {}, {}};
  }
  PeerTransportEffects effects;
  const auto fifo = InspectConsumerFifo(function, &effects);
  if (!fifo.unavailable.empty()) return {fifo.unavailable, {}, {}};
  if (!fifo.has_transport) return {"No native tile FIFO operations", {}, {}};
  return effects;
}

namespace {

// Each borrow lives entirely in a linear statement sequence. The enclosing
// sequence may execute conditionally or repeatedly. Ordinal differences
// inside a borrow then remain exact, irrespective of its entry slot phase.
class BorrowReaders : public IRVisitor {
 public:
  explicit BorrowReaders(const PeerTransportEffects& effects) : pipes_(effects.pipes) {
    for (const auto& site : effects.sites) {
      const bool unique = sites_.emplace(site.statement.get(), &site).second;
      CHECK(unique) << "Shared peer statement has no unique borrow cut";
    }
    for (const auto& [id, pipe] : pipes_) {
      if (!pipe.receive) continue;
      const auto& ring = *pipe.receive;
      CHECK(ring.placed_base) << "Local borrow cuts require a placed receive reserve";
      const auto base = *ring.placed_base;
      CHECK(reserves_[ring.space].emplace(base, base + ring.reserve_bytes).second)
          << "Overlapping peer reserves have no independent local cursor";
    }
    for (const auto& [space, intervals] : reserves_) {
      int64_t prior_end = 0;
      for (const auto& [base, end] : intervals) {
        CHECK(prior_end <= base) << "Overlapping peer reserves have no independent local cursor";
        prior_end = end;
      }
    }
  }

  std::vector<PeerBorrowUse> borrows;

  void VisitStmt(const StmtPtr& stmt) override {
    const auto assign = As<AssignStmt>(stmt);
    if (const auto it = sites_.find(stmt.get()); it != sites_.end()) {
      const auto& site = *it->second;
      if (site.kind == PeerTransportKind::Pop) {
        auto& ordinal = ordinals_[site.pipe_id];
        CHECK(!aliases_.count(site.tile.get())) << "Repeated SSA binding for receive borrow";
        aliases_[site.tile.get()] = borrows.size();
        birth_.push_back(++ordinal);
        ++active_;
        const bool empty = EmptyReceive(site.descriptor);
        borrows.push_back({stmt, site.pipe_id, pipes_.at(site.pipe_id).receive->local_slots, empty, {}});
        return;
      }
      if (site.kind == PeerTransportKind::Free) {
        closed_.insert(aliases_.at(site.tile.get()));
        --active_;
        return;
      }
      if (auto use = Lookup(site.tile))
        Record(*use, stmt, site.data_pipe);
      else
        CheckPlacement(site.tile);
      return;
    }
    if (auto call = transform_utils::GetCallFromStmt(stmt)) {
      for (size_t index = 0; index < call->args_.size(); ++index) {
        const auto use = Lookup(AsVarLike(call->args_[index]));
        if (!use) {
          VisitExpr(call->args_[index]);
          continue;
        }
        const auto entry = LookupOpEntry(call->op_);
        CHECK(entry) << "Borrow consumed by an unregistered or outlined operation";
        if (assign && entry->GetExecutionMemoryAccessEvidence() == ExecutionMemoryAccessEvidence::NoAccess &&
            op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size()) == index) {
          BindAlias(assign->var_, *use);
        } else {
          Record(*use, stmt, ReaderPipe(call, index));
        }
      }
      for (const auto& [key, attribute] : call->attrs_) {
        ForEachAttrExpr(attribute, [this](const ExprPtr& value) { VisitExpr(value); });
      }
      if (assign && !Lookup(assign->var_)) CheckPlacement(assign->var_);
      return;
    }
    if (assign) {
      if (auto use = Lookup(AsVarLike(assign->value_))) {
        BindAlias(assign->var_, *use);
        return;
      }
      CheckPlacement(assign->var_);
      VisitExpr(assign->value_);
      return;
    }
    if (!As<SeqStmts>(stmt)) CHECK(!active_) << "Receive borrow crosses a control-flow boundary";
    IRVisitor::VisitStmt(stmt);
  }

  void VisitExpr(const ExprPtr& expr) override {
    if (const auto var = AsVarLike(expr)) {
      CHECK(!Lookup(var)) << "Receive borrow escapes through a nested expression or control operand";
      CheckPlacement(var);
    }
    // Scalar expression DAGs cannot contain an accepted borrow directly.
    // Still inspect their leaves (e.g. an embedded tile.read) once.
    if (seen_.insert(expr.get()).second) IRVisitor::VisitExpr(expr);
  }

 private:
  std::optional<size_t> Lookup(const VarPtr& var) const {
    auto it = aliases_.find(var.get());
    return it == aliases_.end() ? std::nullopt : std::optional{it->second};
  }

  void BindAlias(const VarPtr& alias, size_t use) {
    const auto type = As<TileType>(alias->GetType());
    CHECK(!closed_.count(use) && type && !type->memref_)
        << "Receive view must preserve its live native slot handle";
    CHECK(aliases_.emplace(alias.get(), use).second) << "Repeated receive view binding";
  }

  void CheckPlacement(const VarPtr& value) {
    const auto type = As<TileType>(value->GetType());
    if (!type || !type->memory_space_ || !reserves_.count(*type->memory_space_)) return;
    if (!checked_.insert(type.get()).second) return;
    CHECK(type->memref_) << "Unbound local tile could alias a native receive reserve";
    const auto& memref = *type->memref_;
    const auto base = TryGetNonNegativeConstInt(memref->byte_offset_);
    CHECK(base && memref->size_ > 0 && memref->size_ <= static_cast<uint64_t>(INT64_MAX - *base))
        << "Local tile has no bounded placed interval for peer alias analysis";
    const auto& intervals = reserves_.at(*type->memory_space_);
    const auto next = intervals.lower_bound(*base + static_cast<int64_t>(memref->size_));
    CHECK(next == intervals.begin() || std::prev(next)->second <= *base)
        << "Placed local tile overlaps a native receive reserve";
  }

  PipeType ReaderPipe(const CallPtr& call, size_t index) const {
    // Only single-pipe implementations whose selected operand is ordinary
    // read-only data can close a borrow. A workspace or a multi-pipe macro
    // needs its own completion contract.
    const auto pipe = backend::BackendConfig::GetBackend()->TryInferPipe(call);
    if (IsOp(call, "tile.move") && pipe == PipeType::MTE1 && index == 0) return *pipe;
    const auto* recipe = backend::FindLogicalBufferElementwiseRecipe(call);
    if (recipe && pipe == PipeType::V && backend::SupportsLogicalBufferElementwise(*recipe, call)) {
      const auto read = std::find_if(recipe->inputs.begin(), recipe->inputs.end(), [index](const auto& arg) {
        return arg.logical_index == index && arg.kind == backend::BufferElementwiseOperandKind::Buffer;
      });
      if (read != recipe->inputs.end()) return *pipe;
    }
    CHECK(false) << "Receive reader lacks a selected single-pipe read contract: " << call->op_->name_;
    return PipeType::ALL;
  }

  void Record(size_t use, const StmtPtr& stmt, PipeType pipe) {
    auto& borrow = borrows[use];
    CHECK(!closed_.count(use)) << "Native read after logical release of a receive borrow";
    CHECK(ordinals_.at(borrow.pipe_id) - birth_[use] < static_cast<size_t>(borrow.reuse_distance))
        << "Receive slot is overwritten before a named reader";
    auto& cuts = borrow.readers.try_emplace(pipe, PeerReaderCuts{stmt, stmt}).first->second;
    cuts.last = stmt;
  }

  const std::map<int, PeerPipeBinding>& pipes_;
  std::unordered_map<const Stmt*, const PeerTransportSite*> sites_;
  std::unordered_map<const Var*, size_t> aliases_;
  std::map<MemorySpace, std::map<int64_t, int64_t>> reserves_;
  std::unordered_set<const TileType*> checked_;
  std::unordered_set<const Expr*> seen_;
  std::unordered_set<size_t> closed_;
  std::unordered_map<int, size_t> ordinals_;
  std::vector<size_t> birth_;
  size_t active_ = 0;
};

}  // namespace

PeerBorrowUsePlan AnalyzePeerBorrowUses(const FunctionPtr& function, const PeerTransportEffects& effects) {
  if (!effects.unavailable.empty()) return {effects.unavailable, {}};
  try {
    BorrowReaders readers(effects);
    readers.VisitFunction(function);
    return {{}, std::move(readers.borrows)};
  } catch (const pypto::ValueError& error) {
    return {error.what(), {}};
  }
}

std::unordered_set<const Stmt*> FindC2VSlotReuseFences(const FunctionPtr& function) {
  C2VReceiveSites ir;
  ir.VisitStmt(function->body_);
  // Authored physical waits own their pipeline schedule. An all-pipe drain
  // could block a wait whose signal is issued after the drain.
  if (ir.has_explicit_wait) {
    return {};
  }
  const bool may_alias_all = MayAliasOtherRings(ir);
  const auto reserve_for = [&](const C2VReceiveSite& site) {
    return may_alias_all ? nullptr : ir.rings.at(site.pipe_id)->reserve.get();
  };
  std::map<const Var*, LocalSlotHistory> histories;
  for (const auto& site : ir.sites) {
    const auto local_slots = may_alias_all ? 1 : ir.rings.at(site.pipe_id)->local_slots;
    histories[reserve_for(site)].Include(site, local_slots);
  }
  for (const auto& site : ir.sites) {
    histories.at(reserve_for(site)).ObservePeriod(site);
  }

  // TFREE does not complete a tile read on A2/A3. The local index is the
  // number of pops modulo depth regardless of the number of outstanding
  // borrow markers. Count all branches as a conservative upper bound, and
  // assume a loop can wrap. Stock PTOAS sees the dependence at a recurrent
  // single static site because its tile descriptor is the same every time.
  std::unordered_set<const Stmt*> fences;
  for (const auto& site : ir.sites) {
    auto& history = histories.at(reserve_for(site));
    if (history.NeedsFence(site, ir.early_exit_loops)) {
      fences.insert(site.stmt);
    }
    history.Advance(site);
  }
  return fences;
}

AutomaticPipeSetup BuildAutomaticPipeSetup(const std::string& func_name, const std::string& aic_name,
                                           const std::string& aiv_name, const std::vector<StmtPtr>& aic_stmts,
                                           const std::vector<StmtPtr>& aiv_stmts,
                                           std::optional<int> slot_num_override,
                                           std::optional<int> local_slot_num_override, const Span& span) {
  CrossCorePipeMetadata aic_metadata;
  CollectCrossCorePipeMetadata(aic_stmts, aic_metadata);
  CrossCorePipeMetadata aiv_metadata;
  CollectCrossCorePipeMetadata(aiv_stmts, aiv_metadata);
  CrossCorePipeMetadata combined = MergeCrossCorePipeMetadata(aic_metadata, aiv_metadata);

  if (!combined.HasCrossCoreOps() || aic_metadata.HasAnySetup() || aiv_metadata.HasAnySetup()) {
    return {};
  }

  const int dir_mask = BuildDirMask(combined);
  auto common_slot_size = GetCommonSlotSizeBytes(combined);
  if (dir_mask == 0 || !common_slot_size.has_value()) {
    return {};
  }

  // Always emit the transport depth, avoiding PTOAS's different implicit default.
  // Local storage follows it unless the supporting backend receives an explicit
  // local depth. GM allocation continues to consume only the transport signature.
  if (slot_num_override.has_value()) {
    INTERNAL_CHECK_SPAN(slot_num_override.value() > 0, span)
        << "Cross-core slot_num override must be positive: " << slot_num_override.value();
  }
  const int effective_slot_num = slot_num_override.value_or(kDefaultAutoPipeSlotNum);
  if (local_slot_num_override.has_value()) {
    // Keep allocation and both peer initializers in agreement. Merely shrinking
    // reserve_buffer would leave PTOAS indexing at the full transport depth.
    CHECK_SPAN(PassContext::Current()->GetBackendHandler()->SupportsLocalPipeDepth(), span)
        << "pl.cross_core_slot(local_slot_num=...) is only supported on A2/A3; "
           "omit local_slot_num on A5";
    CHECK_SPAN(*local_slot_num_override > 0 && *local_slot_num_override <= effective_slot_num, span)
        << "Cross-core local_slot_num (" << *local_slot_num_override << ") must be positive and <= slot_num ("
        << effective_slot_num << ")";
  }
  const int local_depth = local_slot_num_override.value_or(effective_slot_num);
  const size_t required_slot_num = std::max(AutomaticConsumerDepth(aic_stmts, core_affinity::CoreSide::AIC),
                                            AutomaticConsumerDepth(aiv_stmts, core_affinity::CoreSide::AIV));
  CHECK_SPAN(static_cast<size_t>(effective_slot_num) >= required_slot_num, span)
      << "Automatic cross-core pipe slot_num=" << effective_slot_num << " is insufficient for "
      << required_slot_num
      << " simultaneously retained popped tiles; set cross_core_slot(slot_num=" << required_slot_num
      << ") or place/consume each gathered tile before gathering the next.";
  CHECK_SPAN(static_cast<size_t>(local_depth) >= required_slot_num, span)
      << "Automatic cross-core pipe local_slot_num=" << local_depth << " is insufficient for "
      << required_slot_num
      << " simultaneously retained popped tiles; increase local_slot_num or place/consume "
         "each gathered tile before gathering the next.";
  // Bound-check the slot size before multiplying so an oversized inferred size
  // can't overflow the int64 buffer_size computation.
  const int64_t slot_size_i64 = common_slot_size.value();
  INTERNAL_CHECK_SPAN(slot_size_i64 >= 0 && slot_size_i64 <= std::numeric_limits<int>::max(), span)
      << "Cross-core slot_size out of range: " << slot_size_i64;
  const int slot_size_bytes = static_cast<int>(slot_size_i64);
  const int64_t buffer_size = slot_size_i64 * local_depth;
  AutomaticPipeSetup setup;

  std::shared_ptr<Var> aic_v2c_reserve_var;
  std::shared_ptr<Var> aic_c2v_import_var;
  std::shared_ptr<Var> aiv_c2v_reserve_var;
  std::shared_ptr<Var> aiv_v2c_import_var;

  auto zero_i32 = [&]() { return std::make_shared<ConstInt>(0, DataType::INT32, span); };
  auto var_as_expr = [](const std::shared_ptr<Var>& v) -> ExprPtr {
    return std::static_pointer_cast<const Expr>(v);
  };

  if (dir_mask & core_affinity::kDirMaskV2C) {
    const auto v2c_name = BuildPipeBufferName(func_name, core_affinity::PipeDirection::V2C);
    auto v2c_reserve = CreateReserveBuffer(v2c_name, buffer_size, span);
    aic_v2c_reserve_var = std::make_shared<Var>(v2c_name, v2c_reserve->GetType(), span);
    setup.aic_stmts.push_back(std::make_shared<AssignStmt>(aic_v2c_reserve_var, v2c_reserve, span));
    auto v2c_import = CreateImportPeerBuffer(v2c_name, aic_name, span);
    aiv_v2c_import_var = std::make_shared<Var>(v2c_name + "_import", v2c_import->GetType(), span);
    setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(aiv_v2c_import_var, v2c_import, span));
  }

  if (dir_mask & core_affinity::kDirMaskC2V) {
    const auto c2v_name = BuildPipeBufferName(func_name, core_affinity::PipeDirection::C2V);
    auto c2v_reserve = CreateReserveBuffer(c2v_name, buffer_size, span);
    aiv_c2v_reserve_var = std::make_shared<Var>(c2v_name, c2v_reserve->GetType(), span);
    setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(aiv_c2v_reserve_var, c2v_reserve, span));
    auto c2v_import = CreateImportPeerBuffer(c2v_name, aiv_name, span);
    aic_c2v_import_var = std::make_shared<Var>(c2v_name + "_import", c2v_import->GetType(), span);
    setup.aic_stmts.push_back(std::make_shared<AssignStmt>(aic_c2v_import_var, c2v_import, span));
  }

  // AIC: c2v operand = import on Cube; v2c operand = reserve on Cube (matches PTO codegen order).
  const ExprPtr aic_c2v_arg = aic_c2v_import_var ? var_as_expr(aic_c2v_import_var) : ExprPtr(zero_i32());
  const ExprPtr aic_v2c_arg = aic_v2c_reserve_var ? var_as_expr(aic_v2c_reserve_var) : ExprPtr(zero_i32());
  // AIV: c2v operand = reserve on Vector; v2c operand = import on Vector.
  const ExprPtr aiv_c2v_arg = aiv_c2v_reserve_var ? var_as_expr(aiv_c2v_reserve_var) : ExprPtr(zero_i32());
  const ExprPtr aiv_v2c_arg = aiv_v2c_import_var ? var_as_expr(aiv_v2c_import_var) : ExprPtr(zero_i32());

  setup.aic_stmts.push_back(std::make_shared<EvalStmt>(
      CreateInitializePipe(core_affinity::CoreSide::AIC, dir_mask, slot_size_bytes, aic_c2v_arg, aic_v2c_arg,
                           effective_slot_num, local_slot_num_override, span),
      span));
  setup.aiv_stmts.push_back(std::make_shared<EvalStmt>(
      CreateInitializePipe(core_affinity::CoreSide::AIV, dir_mask, slot_size_bytes, aiv_c2v_arg, aiv_v2c_arg,
                           effective_slot_num, local_slot_num_override, span),
      span));

  return setup;
}

std::vector<StmtPtr> PrependPipeSetup(const std::vector<StmtPtr>& prologue,
                                      const std::vector<StmtPtr>& body) {
  if (prologue.empty()) return body;
  std::vector<StmtPtr> result;
  result.reserve(prologue.size() + body.size());
  result.insert(result.end(), prologue.begin(), prologue.end());
  result.insert(result.end(), body.begin(), body.end());
  return result;
}

std::string FormatObservedSlotSizes(const std::vector<int64_t>& slot_sizes) {
  std::string result;
  for (size_t i = 0; i < slot_sizes.size(); ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(slot_sizes[i]);
  }
  return result;
}

}  // namespace cross_core_pipe
}  // namespace ir
}  // namespace pypto
