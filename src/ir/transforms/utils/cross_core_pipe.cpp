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
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/core_side_ops.h"
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

// Each nested body closes its own borrows without popping an inherited queue
// entry. Thus it restores the incoming state even for skipped or repeated visits.
// Independent pipe IDs have independent slots and can release in either order.
class ConsumerFifo {
 public:
  ConsumerFifo(core_affinity::CoreSide side, bool established_pipe_ids)
      : pop_(core_side_ops::TPopOp(side)),
        free_(core_side_ops::TFreeOp(side)),
        established_pipe_ids_(established_pipe_ids) {}

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
    if (IsPeerTransport(call) || op_predicates::IsInitializePipe(call)) {
      CHECK_SPAN(at_cut && seen_cuts_.insert(call.get()).second, call->span_)
          << "Peer FIFO requires an original peer operation at a unique statement cut";
      CHECK_SPAN(!op_predicates::IsInitializePipe(call) || initializers_.count(call.get()), call->span_)
          << "Peer FIFO requires initializers that dominate all transport cuts";
    }
    if (seen_values_.insert(expr.get()).second) IRVisitor::VisitExpr(expr);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}

 private:
  const std::unordered_set<const Call*>& initializers_;
  const Call* expected_ = nullptr;
  std::unordered_set<const Call*> seen_cuts_;
  std::unordered_set<const Expr*> seen_values_;
};

std::optional<std::pair<int64_t, int64_t>> FullVecShape(const TypePtr& value) {
  const auto type = As<TileType>(value);
  if (!type || type->memory_space_ != MemorySpace::Vec || type->dtype_ != DataType::FP32 ||
      type->shape_.size() != 2)
    return std::nullopt;
  const auto view = tile_view_semantics::GetEffectiveTileView(*type);
  if (view.blayout != TileLayout::row_major || view.slayout != TileLayout::none_box ||
      view.pad != PadValue::null || view.compact != CompactMode::null || view.valid_shape.size() != 2)
    return std::nullopt;
  const auto rows = TryGetNonNegativeConstInt(type->shape_[0]);
  const auto cols = TryGetNonNegativeConstInt(type->shape_[1]);
  if ((view.start_offset && TryGetNonNegativeConstInt(view.start_offset) != 0) || !rows || !cols ||
      *rows == 0 || *cols == 0 || *cols % 8 != 0 || rows != TryGetNonNegativeConstInt(view.valid_shape[0]) ||
      cols != TryGetNonNegativeConstInt(view.valid_shape[1]))
    return std::nullopt;
  if (!view.stride.empty() && (view.stride.size() != 2 || TryGetNonNegativeConstInt(view.stride[0]) != cols ||
                               TryGetNonNegativeConstInt(view.stride[1]) != 1))
    return std::nullopt;
  return std::pair{*rows, *cols};
}

std::optional<Span> FindOverwrittenConsumer(const std::vector<StmtPtr>& stmts, size_t begin, int id,
                                            size_t slots, int slot_size) {
  const auto& registry = OpRegistry::GetInstance();
  std::deque<VarPtr> active;
  VarPtr overwritten;
  for (size_t i = begin; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];
    const auto call = transform_utils::GetCallFromStmt(stmt);
    // The concrete refusal only uses a flat unconditional prefix. Effects of
    // arbitrary branch/carry/exit paths remain outside this witness.
    if (!As<AssignStmt>(stmt) && !As<EvalStmt>(stmt)) break;
    if (op_predicates::IsTPop(call) && call->GetKwarg<int>("id", 0) == id) {
      const auto tile = As<AssignStmt>(stmt)->var_;
      const auto shape = FullVecShape(tile->GetType());
      overwritten.reset();
      if (active.size() >= slots && shape && call->GetKwarg<int>("split", 0) == 0 &&
          shape->first <= (slot_size / 4) / shape->second) {
        const auto& earlier = active[active.size() - slots];
        if (earlier && shape == FullVecShape(earlier->GetType())) overwritten = earlier;
      }
      active.push_back(call->GetKwarg<int>("split", 0) == 0 ? tile : nullptr);
      continue;
    }
    if (op_predicates::IsTFree(call)) {
      if (!active.empty() && !call->args_.empty() && call->args_[0].get() == active.front().get())
        active.pop_front();
    } else if (overwritten && call && !As<GlobalVar>(call->op_) && registry.IsRegistered(call->op_->name_)) {
      const auto& op = registry.GetEntry(call->op_->name_);
      const auto shape = FullVecShape(call->GetType());
      bool active_read = false, matches_shape = shape == FullVecShape(overwritten->GetType());
      if (matches_shape &&
          op.GetExecutionMemoryAccessEvidence() == ExecutionMemoryAccessEvidence::Functional) {
        for (size_t arg = 0; arg < call->args_.size(); ++arg) {
          const auto& value = call->args_[arg];
          if (As<TileType>(value->GetType())) matches_shape &= shape == FullVecShape(value->GetType());
          active_read |=
              value.get() == overwritten.get() && ArgEffectReads(op.GetArgEffect(arg, call->kwargs_));
        }
        if (matches_shape && active_read) return stmt->span_;
      }
    }
    overwritten.reset();
  }
  return std::nullopt;
}

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

ConsumerFifoEvidence AnalyzeConsumerFifo(const FunctionPtr& function) {
  if (function->func_type_ != FunctionType::AIC && function->func_type_ != FunctionType::AIV) return {};
  const auto side =
      function->func_type_ == FunctionType::AIC ? core_affinity::CoreSide::AIC : core_affinity::CoreSide::AIV;
  const int incoming =
      side == core_affinity::CoreSide::AIC ? core_affinity::kDirMaskV2C : core_affinity::kDirMaskC2V;
  const size_t operand = side == core_affinity::CoreSide::AIC ? 1 : 0;
  const auto stmts = FlattenBody(function->body_);
  ConsumerFifoEvidence evidence;
  try {
    const auto fifo = ConsumerFifo(side, true).Run(stmts);
    if (fifo.first_transport == stmts.size()) return {};
    evidence.has_transport = true;
    CHECK(!fifo.unsupported_control)
        << "Concurrent, outlined or early-exit control does not prove local FIFO order";
    const auto initialize = core_side_ops::InitializePipeOp(side);
    std::unordered_map<const Var*, CallPtr> reserves;
    std::map<int, CallPtr> initializers;
    std::unordered_set<const Call*> prologue_initializers;
    for (size_t i = 0; i < fifo.first_transport; ++i) {
      const auto& stmt = stmts[i];
      const auto call = transform_utils::GetCallFromStmt(stmt);
      if (!call) continue;
      if (IsOp(call, "system.reserve_buffer")) {
        if (const auto assign = As<AssignStmt>(stmt)) reserves.emplace(assign->var_.get(), call);
      } else if (IsOp(call, initialize)) {
        prologue_initializers.insert(call.get());
        if (call->GetKwarg<int>("dir_mask", 0) & incoming) {
          CHECK(initializers.emplace(call->GetKwarg<int>("id", 0), call).second)
              << "More than one dominating consumer initializer uses the same peer pipe id";
        }
      }
    }
    PeerCutVerifier(prologue_initializers).VisitStmt(function->body_);
    for (const auto& [id, required] : fifo.peak_depth) {
      const auto it = initializers.find(id);
      CHECK(it != initializers.end()) << "No dominating consumer initializer for peer pipe id " << id;
      const auto& setup = it->second;
      const auto mask = setup->GetKwarg<int>("dir_mask", 0);
      const auto slots = setup->GetKwarg<int>("slot_num", GetPtoasImplicitSlotNum(mask));
      const auto local = setup->GetKwarg<int>("local_slot_num", slots);
      const auto slot_size = setup->GetKwarg<int>("slot_size", 0);
      CHECK((mask & ~(core_affinity::kDirMaskV2C | core_affinity::kDirMaskC2V)) == 0 && local > 0 &&
            slots >= local && slot_size > 0 && setup->args_.size() == 2)
          << "Invalid consumer slot geometry for peer pipe id " << id;
      const auto reserve = reserves.find(AsVarLike(setup->args_[operand]).get());
      CHECK(reserve != reserves.end() &&
            reserve->second->GetKwarg<int>("size", 0) >= int64_t{local} * slot_size)
          << "No reserve declaration covers the consumer slot window for peer pipe id " << id;
      const auto depth = static_cast<size_t>(local);
      evidence.capacities.emplace(id, ConsumerPipeCapacity{required, depth});
      if (side == core_affinity::CoreSide::AIV && prologue_initializers.size() == 1 && required > depth) {
        if (const auto overwritten =
                FindOverwrittenConsumer(stmts, fifo.first_transport, id, depth, slot_size))
          evidence.overwritten_input.emplace(*overwritten);
      }
    }
  } catch (const pypto::ValueError& error) {
    return {true, {}, error.what(), std::nullopt};
  }
  return evidence;
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
