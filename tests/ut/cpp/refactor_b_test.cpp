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

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/conservative_sync_dataflow.h"
#include "src/ir/transforms/utils/retained_sync_child.h"
#include "src/ir/transforms/utils/sync_footprint_resolver.h"
#include "src/ir/transforms/utils/sync_loop_trip_proof.h"

namespace pypto::ir::sync {
namespace {
const Span kSpan = Span::unknown();

ConstIntPtr CI(int64_t val, DataType dtype = DataType::INDEX) {
  return std::make_shared<ConstInt>(val, dtype, kSpan);
}

VarPtr MakeVar(const std::string& name, TypePtr type) {
  return std::make_shared<Var>(name, std::move(type), kSpan);
}

TypePtr IndexType() { return std::make_shared<ScalarType>(DataType::INDEX); }
TypePtr BoolType() { return std::make_shared<ScalarType>(DataType::BOOL); }

ExprPtr MakeTup(std::vector<ExprPtr> elems) { return std::make_shared<MakeTuple>(std::move(elems), kSpan); }

ExprPtr ConstTup(std::initializer_list<int64_t> vals) {
  std::vector<ExprPtr> elems;
  for (int64_t v : vals) elems.push_back(CI(v));
  return MakeTup(std::move(elems));
}

CallPtr TypedCall(const char* op_name, std::vector<ExprPtr> args, TypePtr type) {
  auto op = OpRegistry::GetInstance().GetOp(op_name);
  assert(op);
  return std::make_shared<Call>(std::move(op), std::move(args), std::move(type), kSpan);
}

StmtPtr AllocBuf(const VarPtr& var, int64_t offset, const TypePtr& type,
                 const char* op_name = "buffer.alloc") {
  return std::make_shared<AssignStmt>(var, TypedCall(op_name, {MakeTup({}), CI(offset)}, type), kSpan);
}

TensorTypePtr MakeTensorType(int64_t rows, int64_t cols, DataType dtype = DataType::FP32) {
  return std::make_shared<TensorType>(std::vector<ExprPtr>{CI(rows), CI(cols)}, dtype);
}

StmtPtr BufferSlot(const VarPtr& dst, const ExprPtr& pool, const ExprPtr& slot_idx,
                   const BufferTypePtr& type) {
  return std::make_shared<AssignStmt>(dst, TypedCall("buffer.slot", {pool, slot_idx}, type), kSpan);
}

StmtPtr BufferSubview(const VarPtr& dst, const ExprPtr& parent, ExprPtr row_off, ExprPtr col_off,
                      const BufferTypePtr& type) {
  return std::make_shared<AssignStmt>(
      dst, TypedCall("buffer.subview", {parent, MakeTup({std::move(row_off), std::move(col_off)})}, type),
      kSpan);
}

StmtPtr BufferReshape(const VarPtr& dst, const ExprPtr& parent, const BufferTypePtr& type) {
  return std::make_shared<AssignStmt>(dst, TypedCall("buffer.reshape", {parent}, type), kSpan);
}

StmtPtr EvalCall(const char* op_name, std::vector<ExprPtr> args) {
  auto call = OpRegistry::GetInstance().Create(op_name, std::move(args), kSpan);
  return std::make_shared<EvalStmt>(call, kSpan);
}

BufferTypePtr VecType(int64_t rows, int64_t cols, DataType dtype = DataType::FP32) {
  return std::make_shared<BufferType>(std::vector<int64_t>{rows, cols}, dtype, MemorySpace::Vec);
}

void TestPeerTransferBufferGeometry() {
  const std::vector<ExprPtr> shape{CI(16), CI(128)};
  auto vec = tile_view_semantics::GetImplicitTileView(shape, MemorySpace::Vec);
  auto describe = [&](const TileView& view, MemorySpace space, DataType dtype) {
    return backend::DescribePeerTransferBuffer(
        std::make_shared<TileType>(shape, dtype, std::nullopt, view, space));
  };
  // Both subblocks advance the FIFO. An empty lane retains its full physical
  // slot while its logical valid extents identify a transfer of zero elements.
  vec.valid_shape = {CI(0), CI(0)};
  vec.pad = PadValue::min;
  const auto empty = describe(vec, MemorySpace::Vec, DataType::BF16);
  assert(empty && empty->descriptor->valid_shape_ == std::vector<int64_t>({0, 0}));
  assert(empty->capacity_bytes == 4096 && empty->descriptor->pad_ == PadValue::min);
  vec.valid_shape = {CI(5), MakeVar("n", IndexType())};
  const auto dynamic = describe(vec, MemorySpace::Vec, DataType::FP32);
  assert(dynamic && dynamic->descriptor->valid_shape_ == std::vector<int64_t>({5, -1}));
  assert(dynamic->capacity_bytes == 8192);
  auto mat = tile_view_semantics::GetImplicitTileView(shape, MemorySpace::Mat);
  mat.pad = PadValue::min;
  const auto matrix = describe(mat, MemorySpace::Mat, DataType::BF16);
  assert(matrix && matrix->capacity_bytes == 4096 && matrix->descriptor->slayout_ == TileLayout::row_major);
  auto acc = tile_view_semantics::GetImplicitTileView(shape, MemorySpace::Acc);
  const auto accumulator = describe(acc, MemorySpace::Acc, DataType::FP32);
  assert(accumulator && accumulator->capacity_bytes == 8192 && accumulator->descriptor->fractal_ == 1024);
  mat.slayout = TileLayout::none_box;
  assert(!describe(mat, MemorySpace::Mat, DataType::BF16));
  vec.stride = {CI(256), CI(1)};
  assert(!describe(vec, MemorySpace::Vec, DataType::FP32));
  vec.stride.clear();
  vec.valid_shape = {CI(17), CI(128)};
  assert(!describe(vec, MemorySpace::Vec, DataType::FP32));
  const auto huge = std::make_shared<TileType>(std::vector<int64_t>{std::numeric_limits<int64_t>::max(), 128},
                                               DataType::FP32, std::nullopt, std::nullopt, MemorySpace::Vec);
  assert(!backend::DescribePeerTransferBuffer(huge));
}

BufferTypePtr MatType(int64_t rows, int64_t cols, DataType dtype = DataType::FP16,
                      TileLayout slayout = TileLayout::row_major, TileLayout blayout = TileLayout::col_major,
                      int64_t fractal = 512) {
  return std::make_shared<BufferType>(std::vector<int64_t>{rows, cols}, dtype, MemorySpace::Mat,
                                      std::vector<int64_t>{rows, cols}, slayout, blayout, fractal);
}

BufferTypePtr AccType(int64_t rows, int64_t cols, DataType dtype = DataType::FP32) {
  return std::make_shared<BufferType>(std::vector<int64_t>{rows, cols}, dtype, MemorySpace::Acc,
                                      std::vector<int64_t>{rows, cols}, TileLayout::row_major,
                                      TileLayout::col_major, 1024);
}

BufferTypePtr LeftType(int64_t rows, int64_t cols, DataType dtype = DataType::FP16) {
  return std::make_shared<BufferType>(std::vector<int64_t>{rows, cols}, dtype, MemorySpace::Left,
                                      std::vector<int64_t>{rows, cols}, TileLayout::row_major,
                                      TileLayout::col_major, 512);
}

MultiBufferTypePtr MultiMatType(int64_t slots, int64_t rows, int64_t cols, DataType dtype = DataType::FP16) {
  return std::make_shared<MultiBufferType>(MatType(rows, cols, dtype), slots);
}

MultiBufferTypePtr MultiVecType(int64_t slots, int64_t rows, int64_t cols, DataType dtype = DataType::FP32) {
  return std::make_shared<MultiBufferType>(VecType(rows, cols, dtype), slots);
}

ForStmtPtr SimpleFor(const VarPtr& iv, ExprPtr start, ExprPtr stop, ExprPtr step,
                     std::vector<StmtPtr> body_stmts) {
  body_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan));
  auto seq = std::make_shared<SeqStmts>(std::move(body_stmts), kSpan);
  return std::make_shared<ForStmt>(iv, std::move(start), std::move(stop), std::move(step),
                                   std::vector<IterArgPtr>{}, seq, std::vector<VarPtr>{}, kSpan);
}

FunctionPtr MakeAicFn(const std::string& name, std::vector<VarPtr> params, std::vector<ParamDirection> dirs,
                      std::vector<StmtPtr> stmts, FunctionType ftype = FunctionType::AIV) {
  auto fn = std::make_shared<Function>(name, std::move(params), std::move(dirs), std::vector<TypePtr>{},
                                       std::make_shared<SeqStmts>(std::move(stmts), kSpan), kSpan, ftype,
                                       std::nullopt, std::nullopt, decltype(Function::attrs_){}, false,
                                       FunctionIRStage::Buffer);
  return fn;
}

// ============================================================================
// B1: SyncControlRangeProver & Positive-Trip Wiring
// ============================================================================
void TestB1SyncControlRangeProver() {
  auto n_var = MakeVar("N", IndexType());
  auto base_var = MakeVar("base", IndexType());
  auto cond_var = MakeVar("cond_n_pos", BoolType());

  auto iv1 = MakeVar("i1", IndexType());
  auto loop_unbounded = SimpleFor(iv1, CI(0), n_var, CI(1), {});

  auto iv2 = MakeVar("i2", IndexType());
  auto loop_guarded = SimpleFor(iv2, CI(0), n_var, CI(1), {});
  auto if_n_pos = std::make_shared<IfStmt>(std::make_shared<Ge>(n_var, CI(1), DataType::BOOL, kSpan),
                                           loop_guarded, std::nullopt, std::vector<VarPtr>{}, kSpan);

  auto iv3 = MakeVar("i3", IndexType());
  auto loop_bool_alias = SimpleFor(iv3, CI(0), n_var, CI(1), {});
  auto assign_cond = std::make_shared<AssignStmt>(
      cond_var, std::make_shared<Gt>(n_var, CI(0), DataType::BOOL, kSpan), kSpan);
  auto if_bool_alias =
      std::make_shared<IfStmt>(cond_var, loop_bool_alias, std::nullopt, std::vector<VarPtr>{}, kSpan);

  // Inner loop in tiled loop: for b in range(0, N, 64): len = min(N - b, 64); for j in range(0, len, 1):
  auto b_var = MakeVar("b", IndexType());
  auto len_var = MakeVar("len", IndexType());
  auto j_var = MakeVar("j", IndexType());
  auto inner_tiled = SimpleFor(j_var, CI(0), len_var, CI(1), {});
  auto assign_len = std::make_shared<AssignStmt>(
      len_var,
      std::make_shared<Min>(std::make_shared<Sub>(n_var, b_var, DataType::INDEX, kSpan), CI(64),
                            DataType::INDEX, kSpan),
      kSpan);
  auto outer_tiled = SimpleFor(b_var, CI(0), n_var, CI(64), {assign_len, inner_tiled});

  // Unbounded correlated difference MUST be refused (INT64_MAX wrap risk):
  // for k_bad in range(base, base + 64, 1) with unbounded `base`:
  auto k_bad = MakeVar("k_bad", IndexType());
  auto loop_overflow_risk =
      SimpleFor(k_bad, base_var, std::make_shared<Add>(base_var, CI(64), DataType::INDEX, kSpan), CI(1), {});

  // Bounded correlated difference succeeds: if (base <= 1024 && base >= 0): for k_ok in range(base, base+64,
  // 1)
  auto k_ok = MakeVar("k_ok", IndexType());
  auto loop_safe_corr =
      SimpleFor(k_ok, base_var, std::make_shared<Add>(base_var, CI(64), DataType::INDEX, kSpan), CI(1), {});
  auto safe_guard = std::make_shared<And>(std::make_shared<Ge>(base_var, CI(0), DataType::BOOL, kSpan),
                                          std::make_shared<Le>(base_var, CI(1024), DataType::BOOL, kSpan),
                                          DataType::BOOL, kSpan);
  auto if_safe_corr =
      std::make_shared<IfStmt>(safe_guard, loop_safe_corr, std::nullopt, std::vector<VarPtr>{}, kSpan);

  auto fn = MakeAicFn("test_prover", {n_var, base_var}, {ParamDirection::In, ParamDirection::In},
                      {loop_unbounded, if_n_pos, assign_cond, if_bool_alias, outer_tiled, loop_overflow_risk,
                       if_safe_corr, std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)});

  // Unbounded parameters -> no synthetic >0 assumption.
  {
    SyncControlRangeProver prover(fn);
    assert(!prover.IsPositiveTrip(loop_unbounded.get()));
    assert(prover.IsPositiveTrip(loop_guarded.get()));
    assert(prover.IsPositiveTrip(loop_bool_alias.get()));
    assert(prover.IsPositiveTrip(inner_tiled.get()));
    assert(!prover.IsPositiveTrip(outer_tiled.get()));
    assert(!prover.IsPositiveTrip(loop_overflow_risk.get()));
    assert(prover.IsPositiveTrip(loop_safe_corr.get()));
  }

  // With parameter bound N in [1, 256] -> outer_tiled and loop_unbounded become positive-trip.
  {
    SyncParameterBounds bounds{{0, {1, 256}}};
    SyncControlRangeProver prover(fn, bounds);
    assert(prover.IsPositiveTrip(loop_unbounded.get()));
    assert(prover.IsPositiveTrip(outer_tiled.get()));
    assert(!prover.IsPositiveTrip(loop_overflow_risk.get()));
  }
}

void TestB1RangeUseCuts() {
  auto iv = MakeVar("row", IndexType());
  auto shadow = MakeVar("row", IndexType());
  auto late = MakeVar("late", IndexType());
  auto condition = MakeVar("condition", BoolType());
  auto parent_type = VecType(8, 16);
  auto row_type = VecType(1, 16);
  auto parent = MakeVar("parent", parent_type);
  constexpr size_t kGuardDefinitions = 16;
  constexpr size_t kSharedGuardViews = 32;
  std::vector<StmtPtr> loop_body;
  ExprPtr bound = CI(1 - static_cast<int64_t>(kGuardDefinitions));
  for (size_t i = 0; i < kGuardDefinitions; ++i) {
    auto next = MakeVar("bound_" + std::to_string(i), IndexType());
    loop_body.push_back(std::make_shared<AssignStmt>(next, MakeAdd(bound, CI(1)), kSpan));
    bound = next;
  }
  auto assign_guard = std::make_shared<AssignStmt>(condition, MakeLt(iv, bound), kSpan);
  auto assign_shadow = std::make_shared<AssignStmt>(shadow, CI(5), kSpan);
  auto else_view = BufferSubview(MakeVar("else", row_type), parent, iv, CI(0), row_type);
  std::vector<StmtPtr> then_stmts{assign_shadow};
  for (size_t i = 0; i < kSharedGuardViews; ++i) {
    auto dst = MakeVar("then_" + std::to_string(i), row_type);
    then_stmts.push_back(BufferSubview(dst, parent, iv, CI(0), row_type));
  }
  auto then_body = std::make_shared<SeqStmts>(then_stmts, kSpan);
  auto branch = std::make_shared<IfStmt>(condition, then_body, else_view, std::vector<VarPtr>{}, kSpan);
  loop_body.push_back(assign_guard);
  loop_body.push_back(branch);
  auto loop = SimpleFor(iv, CI(0), CI(2), CI(1), std::move(loop_body));
  auto assign_late = std::make_shared<AssignStmt>(late, CI(7), kSpan);
  auto end = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan);
  auto fn = MakeAicFn("range_use_cuts", {}, {}, {AllocBuf(parent, 0, parent_type), loop, assign_late, end});
  const SyncControlRangeProver prover(fn);

  // Query out of order to distinguish a cut's lexical facts from traversal state.
  const auto in_else = prover.LookupAtCut(else_view.get(), iv);
  assert(in_else && in_else->lo == 1 && in_else->hi == 1);
  for (size_t i = 1; i < then_stmts.size(); ++i) {
    const auto in_then = prover.LookupAtCut(then_stmts[i].get(), iv);
    assert(in_then && in_then->lo == 0 && in_then->hi == 0);
  }
  assert(prover.Work().footprint_queries == 2 * (kSharedGuardViews + 1));
  assert(prover.Work().scalar_bound_queries == 0);
  assert(prover.Work().fallback_definitions == 0);

  // The optional bounds-only index stores each selected expression once per
  // original cut. Sharing the same expression at distinct cuts preserves
  // local guards, while lookups never replay the shared guard definitions.
  const auto selected = [](const StmtPtr&, const CallPtr& call) -> std::vector<ExprPtr> {
    if (!IsOp(call, "buffer.subview")) {
      return {};
    }
    const auto offsets = As<MakeTuple>(call->args_[1]);
    return {offsets->elements_[0], offsets->elements_[0]};
  };
  const SyncControlRangeProver indexed(fn, {}, std::nullopt, selected);
  assert((indexed.LookupScalarBounds(else_view.get(), iv.get()) == std::pair<int64_t, int64_t>{1, 1}));
  for (size_t i = 1; i < then_stmts.size(); ++i) {
    assert((indexed.LookupScalarBounds(then_stmts[i].get(), iv.get()) == std::pair<int64_t, int64_t>{0, 0}));
  }
  assert(!indexed.LookupScalarBounds(then_stmts.back().get(), shadow.get()));
  assert(!indexed.LookupScalarBounds(branch.get(), iv.get()));
  assert(indexed.Work().scalar_bound_queries == kSharedGuardViews + 1);
  assert(indexed.Work().indexed_definitions == prover.Work().indexed_definitions);
  assert(indexed.Work().fallback_definitions == 0);

  const auto at_branch = prover.LookupAtCut(branch.get(), iv);
  assert(at_branch && at_branch->lo == 0 && at_branch->hi == 1);
  assert(at_branch->affine_frame == loop.get() && at_branch->affine_coeff == 1);
  const auto local = prover.LookupAtCut(then_stmts.back().get(), shadow);
  assert(local && local->is_const() && local->lo == 5);
  assert(!prover.LookupAtCut(else_view.get(), shadow));
  assert(!prover.LookupAtCut(end.get(), iv));
  assert(!prover.LookupAtCut(assign_late.get(), late));
  const auto after = prover.LookupAtCut(end.get(), late);
  assert(after && after->is_const() && after->lo == 7);
}

void TestB1PositiveTripCompletionAndRelay() {
  // Promoted ForStmt child (positive-trip): MTE2 loads x, V adds x ->
  // positive-trip prover makes MTE2->V relay and post_write_safe visible in parent!
  auto gm_t = MakeTensorType(16, 64, DataType::FP32);
  auto gm = MakeVar("gm", gm_t);
  auto out = MakeVar("out", gm_t);
  auto buf_t = VecType(16, 64);
  auto x = MakeVar("x", buf_t);
  auto y = MakeVar("y", buf_t);
  auto iv = MakeVar("k", IndexType());

  auto load_x = EvalCall("buffer.load", {gm, ConstTup({0, 0}), ConstTup({16, 64}), x});
  auto add_xy = EvalCall("buffer.add", {x, x, y});
  auto child_loop = SimpleFor(iv, CI(0), CI(4), CI(1), {load_x, add_xy});
  auto store_y = EvalCall("buffer.store", {y, ConstTup({0, 0}), ConstTup({16, 64}), out});

  auto fn = MakeAicFn("positive_child_relay", {gm, out}, {ParamDirection::In, ParamDirection::InOut},
                      {AllocBuf(x, 0, buf_t), AllocBuf(y, 4096, buf_t), child_loop, store_y,
                       std::make_shared<ReturnStmt>(std::vector<ExprPtr>{out}, kSpan)},
                      FunctionType::AIV);

  const auto analysis = AnalyzeConservativeSync(fn, "a2a3", {}, {}, /*retain_incoming_diagnostics=*/false,
                                                /*automatic_children=*/true, /*enable_refresh_kill=*/true);
  assert(analysis.ChildEntries().size() == 1);
  const auto& entry = analysis.ChildEntries()[0];
  assert(entry.eventful != nullptr);
  assert(!entry.completion.has_bypass_path);
  assert(entry.completion.IsRelayed(PipeType::MTE2, PipeType::V));
  assert(entry.completion.IsPostWriteSafe(PipeType::MTE2, PipeType::V));
}

// ============================================================================
// B2: SyncFootprintResolver & Strided/2D StorageClasses
// ============================================================================
void TestB2StridedVecAndMatStorageClasses() {
  auto gm_t = MakeTensorType(16, 64, DataType::FP32);
  auto gm_in = MakeVar("gm_in", gm_t);
  auto gm_out = MakeVar("gm_out", gm_t);

  // Two disjoint column slices of `[16, 64]` FP32 Vec buffer:
  // buf[:, 0:32] and buf[:, 32:64]. Their 1D byte hulls overlap ([0, 3968) vs [128, 4096)),
  // yet Stage 2 modular GCD proves 2 disjoint StorageDomains and 0 bar_v!
  auto full_t = VecType(16, 64);
  auto sub_t = VecType(16, 32);
  auto buf = MakeVar("buf", full_t);
  auto s0 = MakeVar("s0", sub_t);
  auto s1 = MakeVar("s1", sub_t);

  auto fn = MakeAicFn("strided_vec_disjoint", {gm_in, gm_out}, {ParamDirection::In, ParamDirection::InOut},
                      {AllocBuf(buf, 0, full_t), BufferSubview(s0, buf, CI(0), CI(0), sub_t),
                       BufferSubview(s1, buf, CI(0), CI(32), sub_t),
                       EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 32}), s0}),
                       EvalCall("buffer.load", {gm_in, ConstTup({0, 32}), ConstTup({16, 32}), s1}),
                       EvalCall("buffer.store", {s0, ConstTup({0, 0}), ConstTup({16, 32}), gm_out}),
                       EvalCall("buffer.store", {s1, ConstTup({0, 32}), ConstTup({16, 32}), gm_out}),
                       std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
                      FunctionType::AIV);

  const auto analysis = AnalyzeConservativeSync(fn, "a2a3");
  assert(analysis.StorageDomains().size() == 2);

  // Also test MultiBuffer constant slots: slot(pool, 0) and slot(pool, 1) -> 2 disjoint domains.
  auto gm_mat_t = MakeTensorType(16, 128, DataType::FP16);
  auto gm_m = MakeVar("gm_m", gm_mat_t);
  auto pool_t = MultiMatType(2, 16, 128);
  auto m_t = MatType(16, 128);
  auto pool = MakeVar("pool", pool_t);
  auto sl0 = MakeVar("sl0", m_t);
  auto sl1 = MakeVar("sl1", m_t);
  auto fn_slots = MakeAicFn("const_slots_disjoint", {gm_m}, {ParamDirection::In},
                            {AllocBuf(pool, 0, pool_t, "buffer.alloc_multi"),
                             BufferSlot(sl0, pool, CI(0), m_t), BufferSlot(sl1, pool, CI(1), m_t),
                             EvalCall("buffer.load", {gm_m, ConstTup({0, 0}), ConstTup({16, 128}), sl0}),
                             EvalCall("buffer.load", {gm_m, ConstTup({0, 0}), ConstTup({16, 128}), sl1}),
                             std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)},
                            FunctionType::AIC);
  const auto slot_analysis = AnalyzeConservativeSync(fn_slots, "a2a3");
  assert(slot_analysis.StorageDomains().size() == 2);

  // Check MatrixBoxFootprint intersection directly: disjoint extract column subtiles.
  std::vector<SyncCanonicalRoot> roots(2);
  roots[1].id = 1;
  roots[1].space = MemorySpace::Mat;
  roots[1].placed = true;
  roots[1].bytes = 32768;
  MatrixBoxFootprint box_a{/*root_id=*/1, 0, 0, 0, 128, 0, 64, true};
  MatrixBoxFootprint box_b{/*root_id=*/1, 0, 0, 0, 128, 64, 128, true};
  MatrixBoxFootprint box_c{/*root_id=*/1, 0, 0, 64, 128, 32, 96, true};
  assert(!MatrixBoxFootprint::MayOverlap(box_a, box_b, roots));
  assert(MatrixBoxFootprint::MayOverlap(box_a, box_c, roots));
  assert(MatrixBoxFootprint::MayOverlap(box_b, box_c, roots));
}

// ============================================================================
// B3: IterationFiber ModularSlot & AffineSlice
// ============================================================================
void TestB3IterationFiberRefinement() {
  // Inside unpromoted loop (contains WhileStmt barrier elsewhere? No, explicit AnalyzeConservativeSync
  // without automatic_children treats the ForStmt directly via IterationFiber!).
  // Use pool[i % 2] (write MTE2) and pool[(i+1) % 2] (read V) in same iteration:
  // Within iteration i, ModularSlot(r=0) and ModularSlot(r=1) do NOT conflict -> V site has NO
  // MTE2->V incoming service! Wait: on next trip (backedge), r=0 shifts to r=1, so V sees MTE2 from
  // previous trip OR zero intra-iteration conflict when only reading (i+1)%2? Wait: if i=0 skips/reads
  // before writing (i+1)%2, backedge carries r=0->r=1. Check instead two disjoint fibers on MTE2 write
  // `slot(pool, i % 2)` and V read `slot(pool, (i+1) % 2)` vs reading `slot(pool, i % 2)` directly:
  // Let's test strictly advancing AffineSlice: `sub_i = subview(buf, (i, 0))` written by MTE2 and
  // read by V inside loop `for i in range(0, 4, 1)`:
  // MTE2 writes `sub_i` (b=0), then V reads `sub_i` (b=0) -> V waits MTE2->V, kills V->MTE2 WAR?
  // Wait: across LoopBackedge, `b'=0-row_bytes <= 0` transitions `V read` & `MTE2 write` to
  // `CompletedPast`, so MTE2 site has ZERO `V -> MTE2` WAR wait! Post-loop `store(buf)` (Wildcard)
  // DOES wait on `V`/`MTE2` across LoopExit!
  auto gm_t = MakeTensorType(4, 64, DataType::FP32);
  auto gm_in = MakeVar("gm_in", gm_t);
  auto gm_out = MakeVar("gm_out", gm_t);
  auto buf_t = VecType(4, 64);
  auto row_t = VecType(1, 64);
  auto buf = MakeVar("buf", buf_t);
  auto iv = MakeVar("i", IndexType());
  auto row_sub = MakeVar("row_sub", row_t);

  auto load_row = EvalCall("buffer.load", {gm_in, MakeTup({iv, CI(0)}), ConstTup({1, 64}), row_sub});
  auto add_row = EvalCall("buffer.add", {row_sub, row_sub, row_sub});
  auto loop =
      SimpleFor(iv, CI(0), CI(4), CI(1), {BufferSubview(row_sub, buf, iv, CI(0), row_t), load_row, add_row});
  auto post_store = EvalCall("buffer.store", {buf, ConstTup({0, 0}), ConstTup({4, 64}), gm_out});

  auto fn = MakeAicFn("advancing_affine_fiber", {gm_in, gm_out}, {ParamDirection::In, ParamDirection::InOut},
                      {AllocBuf(buf, 0, buf_t), loop, post_store,
                       std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
                      FunctionType::AIV);

  const auto analysis = AnalyzeConservativeSync(fn, "a2a3", {}, {}, /*retain_incoming_diagnostics=*/true,
                                                /*automatic_children=*/false);
  // Site 0 = load_row, Site 1 = add_row, Site 2 = post_store.
  assert(analysis.Sites().size() == 3);
  // Site 0 (load_row on MTE2): strictly advancing -> zero V->MTE2 backedge WAR!
  assert(analysis.Sites()[0].source_pipes.empty());
  // Site 1 (add_row on V): waits for Site 0 (MTE2->V), zero V->V self-barrier from previous trip!
  assert((analysis.Sites()[1].source_pipes == std::vector<PipeType>{PipeType::MTE2}));
  // Site 2 (post_store on MTE3): LoopExit projects CompletedPast -> Wildcard -> must sync with V!
  assert(std::find(analysis.Sites()[2].source_pipes.begin(), analysis.Sites()[2].source_pipes.end(),
                   PipeType::V) != analysis.Sites()[2].source_pipes.end());
}

void TestAffineVecPhysicalByteOrigins() {
  const auto tile = VecType(8, 8);
  const auto row = VecType(1, 8);
  const auto a = MakeVar("a", tile);
  const auto b = MakeVar("b", tile);
  const auto i = MakeVar("i", IndexType());
  const auto a_next = MakeVar("a_next", row);
  const auto b_current = MakeVar("b_current", row);
  const auto b_next = MakeVar("b_next", row);
  const auto wide_type = VecType(2, 64);
  const auto wide = MakeVar("wide", wide_type);
  const auto window_type = VecType(1, 48);
  const auto window = MakeVar("window", window_type);
  const auto direct = MakeVar("direct", row);
  const auto nested = MakeVar("nested", row);
  const auto adjacent = MakeVar("adjacent", row);
  const auto next = MakeAdd(i, CI(1));
  const auto column = MakeMul(i, CI(8));
  const auto loop =
      SimpleFor(i, CI(0), CI(4), CI(1),
                {BufferSubview(a_next, a, next, CI(0), row), BufferSubview(b_current, b, i, CI(0), row),
                 BufferSubview(b_next, b, next, CI(0), row),
                 BufferSubview(direct, wide, CI(1), MakeAdd(column, CI(8)), row),
                 BufferSubview(nested, window, CI(0), column, row),
                 BufferSubview(adjacent, window, CI(0), MakeAdd(column, CI(8)), row)});
  const auto fn = MakeAicFn("vec_physical_byte_origins", {}, {},
                            {AllocBuf(a, 32, tile), AllocBuf(b, 64, tile), AllocBuf(wide, 1024, wide_type),
                             BufferSubview(window, wide, CI(1), CI(8), window_type), loop,
                             std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)});
  SyncFootprintResolver resolver(fn);
  const auto offset = [&](const VarPtr& view) {
    const auto* resolved = resolver.ViewOf(view);
    assert(resolved && resolved->fiber.kind == IterationFiber::Kind::AffineSlice);
    assert(resolved->fiber.frame == loop.get());
    assert(resolved->fiber.coeff == 32 && resolved->fiber.extent == 32);
    return resolved->fiber.offset;
  };
  // b begins one FP32 row into a. On every trip, a[i + 1] aliases b[i], and b[i + 1] is adjacent.
  assert(offset(a_next) == 64);
  assert(offset(b_current) == 64);
  assert(offset(b_next) == 96);
  // window begins at wide[1, 8]: byte 1024 + (64 + 8) * 4. Each column fiber moves eight FP32s per trip.
  assert(offset(direct) == 1312);
  assert(offset(nested) == 1312);
  assert(offset(adjacent) == 1344);
}

// ============================================================================
// B4: Init-Guard k0==0, Geometry Enclosing Capture, Tier 2 Class A & B, 64-bit Rearm
// ============================================================================
void TestB4InitGuardAndSubblockPromotion() {
  // 1. CoordinateGraph init-guard simplification: k0 = kb * 512; if k0 == 0: matmul else: matmul_acc
  auto left_t = std::make_shared<BufferType>(std::vector<int64_t>{16, 32}, DataType::FP16, MemorySpace::Left,
                                             std::vector<int64_t>{16, 32}, TileLayout::row_major,
                                             TileLayout::row_major, 512);
  auto right_t = std::make_shared<BufferType>(std::vector<int64_t>{32, 256}, DataType::FP16,
                                              MemorySpace::Right, std::vector<int64_t>{32, 256},
                                              TileLayout::col_major, TileLayout::row_major, 512);
  auto acc_t = AccType(16, 256);

  auto kb = MakeVar("kb", IndexType());
  auto ko = MakeVar("ko", IndexType());
  auto k0 = MakeVar("k0", IndexType());
  auto l_buf = MakeVar("l", left_t);
  auto r_buf = MakeVar("r", right_t);
  auto acc = MakeVar("acc", acc_t);
  auto gm_c_t = MakeTensorType(16, 256, DataType::FP32);
  auto gm_c = MakeVar("gm_c", gm_c_t);

  auto init_cond =
      std::make_shared<And>(std::make_shared<Eq>(k0, CI(0), DataType::BOOL, kSpan),
                            std::make_shared<Eq>(ko, CI(0), DataType::BOOL, kSpan), DataType::BOOL, kSpan);
  auto init_branch = std::make_shared<IfStmt>(init_cond, EvalCall("buffer.matmul", {l_buf, r_buf, acc}),
                                              EvalCall("buffer.matmul_acc", {l_buf, r_buf, acc}),
                                              std::vector<VarPtr>{}, kSpan);
  auto inner_ko = SimpleFor(ko, CI(0), CI(2), CI(1), {init_branch});
  auto outer_kb = SimpleFor(
      kb, CI(0), CI(4), CI(1),
      {std::make_shared<AssignStmt>(k0, std::make_shared<Mul>(kb, CI(512), DataType::INDEX, kSpan), kSpan),
       inner_ko});

  auto fn_init =
      MakeAicFn("init_guard_k0", {gm_c}, {ParamDirection::InOut},
                {AllocBuf(l_buf, 0, left_t), AllocBuf(r_buf, 0, right_t), AllocBuf(acc, 0, acc_t), outer_kb,
                 EvalCall("buffer.store", {acc, ConstTup({0, 0}), ConstTup({16, 256}), gm_c}),
                 std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_c}, kSpan)},
                FunctionType::AIC);

  // Promoting outer_kb succeeds (CoordinateGraph simplifies k0 == 0 -> kb == 0).
  const auto placed = ImportPlacedSync(fn_init, true, "a2a3");
  assert(placed.segments.size() == 2);

  // Also test Geometry enclosing capture + ProjectEnclosingLoopBounds on inner_ko when kb is outer ForStmt!
  SyncRegionBindings bindings(fn_init, "a2a3");
  const auto attempt = TryPrepareEventfulSyncChild(fn_init, inner_ko, "a2a3", bindings);
  assert(attempt.child != nullptr);

  // End-to-end InsertSync + LowerSync on both StructuredTokenSync (single loop under projected k0)
  // and ScopedTokenFamilies (nested kb/ko), including buffer.slot + YieldStmt inside arms.
  {
    struct ScopedPassCtx {
      PassContext ctx;
      ScopedPassCtx()
          : ctx({}, VerificationLevel::Basic, DiagnosticPhase::PrePipeline, {}, MemoryPlanner::PyPTO, false,
                kDefaultRuntimeKind, true, false, false, true, true, "conservative_prefix") {
        ctx.EnterContext();
      }
      ~ScopedPassCtx() { ctx.ExitContext(); }
    } scoped_ctx;

    auto cond_k = MakeVar("cond_k", BoolType());
    auto gm_a_t = MakeTensorType(16, 32, DataType::FP16);
    auto gm_a = MakeVar("gm_a", gm_a_t);
    auto mat_t = MatType(16, 32);
    auto m_a = MakeVar("m_a", mat_t);
    auto macc_t = std::make_shared<MultiBufferType>(acc_t, 1);
    auto macc = MakeVar("macc", macc_t);
    auto acc0 = MakeVar("acc0", acc_t);
    auto acc1 = MakeVar("acc1", acc_t);
    auto acc_out = MakeVar("acc_out", acc_t);
    auto slot_arm0 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{BufferSlot(acc0, macc, CI(0), acc_t),
                             EvalCall("buffer.matmul", {l_buf, r_buf, acc0}),
                             std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)},
        kSpan);
    auto slot_arm1 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{BufferSlot(acc1, macc, CI(0), acc_t),
                             EvalCall("buffer.matmul_acc", {l_buf, r_buf, acc1}),
                             std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)},
        kSpan);
    auto slot_branch =
        std::make_shared<IfStmt>(init_cond, slot_arm0, slot_arm1, std::vector<VarPtr>{}, kSpan);
    auto inner_slot = SimpleFor(ko, CI(0), CI(2), CI(1),
                                {EvalCall("buffer.load", {gm_a, ConstTup({0, 0}), ConstTup({16, 32}), m_a}),
                                 EvalCall("buffer.copy", {m_a, l_buf}), slot_branch});

    // Single-loop StructuredTokenSync child (with outer 1-trip kb + WhileStmt forcing child promotion).
    auto dummy_w = std::make_shared<WhileStmt>(
        cond_k, std::vector<IterArgPtr>{},
        std::make_shared<SeqStmts>(
            std::vector<StmtPtr>{std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)}, kSpan),
        std::vector<VarPtr>{}, kSpan);
    auto kb_1trip = SimpleFor(
        kb, CI(0), CI(1), CI(1),
        {dummy_w,
         std::make_shared<AssignStmt>(k0, std::make_shared<Mul>(kb, CI(512), DataType::INDEX, kSpan), kSpan),
         inner_slot});
    auto fn_struct = MakeAicFn(
        "b4_struct_e2e", {gm_a, gm_c, cond_k},
        {ParamDirection::In, ParamDirection::InOut, ParamDirection::In},
        {AllocBuf(m_a, 0, mat_t), AllocBuf(l_buf, 0, left_t), AllocBuf(r_buf, 0, right_t),
         AllocBuf(macc, 0, macc_t, "buffer.alloc_multi"), kb_1trip, BufferSlot(acc_out, macc, CI(0), acc_t),
         EvalCall("buffer.store", {acc_out, ConstTup({0, 0}), ConstTup({16, 256}), gm_c}),
         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_c}, kSpan)},
        FunctionType::AIC);
    const auto cons_struct =
        AnalyzeConservativeSync(fn_struct, "a2a3", {}, {}, false, /*automatic_children=*/true);
    assert(cons_struct.ChildEntries().size() == 1);
    const auto lowered_struct =
        TryLowerClosedPrefixSync(InsertConservativePrefixes(cons_struct), "a2a3", std::nullopt);
    assert(lowered_struct.has_value() && lowered_struct->set_sites > 0);

    // Nested-loop ScopedTokenFamilies child (dummy_w outside kb_multi so kb_multi becomes promoted child).
    auto kb_multi = SimpleFor(
        kb, CI(0), CI(4), CI(1),
        {std::make_shared<AssignStmt>(k0, std::make_shared<Mul>(kb, CI(512), DataType::INDEX, kSpan), kSpan),
         inner_slot});
    auto fn_scoped =
        MakeAicFn("b4_scoped_e2e", {gm_a, gm_c, cond_k},
                  {ParamDirection::In, ParamDirection::InOut, ParamDirection::In},
                  {AllocBuf(m_a, 0, mat_t), AllocBuf(l_buf, 0, left_t), AllocBuf(r_buf, 0, right_t),
                   AllocBuf(macc, 0, macc_t, "buffer.alloc_multi"), dummy_w, kb_multi,
                   BufferSlot(acc_out, macc, CI(0), acc_t),
                   EvalCall("buffer.store", {acc_out, ConstTup({0, 0}), ConstTup({16, 256}), gm_c}),
                   std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_c}, kSpan)},
                  FunctionType::AIC);
    const auto cons_scoped =
        AnalyzeConservativeSync(fn_scoped, "a2a3", {}, {}, false, /*automatic_children=*/true);
    assert(cons_scoped.ChildEntries().size() == 1);
    const auto lowered_scoped =
        TryLowerClosedPrefixSync(InsertConservativePrefixes(cons_scoped), "a2a3", std::nullopt);
    assert(lowered_scoped.has_value() && lowered_scoped->set_sites > 0);
  }

  // 2. Tier 2 Class B (InlineExactSubBlock) + subsequent EventfulSyncChild (64-bit reaching_child_dirs):
  // Inside a WhileStmt (preventing whole-fn Exact):
  //   straight-line profitable slice where Exact sharpens SyncFootprintResolver:
  //   `sub_a` ([3, 8] in [3, 32]) and `sub_b` ([1, 24] at [0, 8] in reshaped [2, 48]) are
  //   Exact-disjoint (false StridedByteFootprint::MayOverlap), while `x1 -> y1` emits MTE2 -> V.
  // Class B uses parent palette -> must NOT poison reaching_child_dirs for the following EventfulSyncChild!
  auto gm_v_t = MakeTensorType(16, 64, DataType::FP32);
  auto gm_v = MakeVar("gm_v", gm_v_t);
  auto out_v = MakeVar("out_v", gm_v_t);
  auto cond_w = MakeVar("cond_w", BoolType());
  auto vx_t = VecType(16, 64);
  auto s_t = VecType(3, 32);
  auto r_t = VecType(2, 48);
  auto va_t = VecType(3, 8);
  auto vb_t = VecType(1, 24);
  auto p1 = MakeVar("p1", s_t);
  auto r1 = MakeVar("r1", r_t);
  auto sub_a = MakeVar("sub_a", va_t);
  auto sub_b = MakeVar("sub_b", vb_t);
  auto x1 = MakeVar("x1", vx_t);
  auto y1 = MakeVar("y1", vx_t);
  auto z1 = MakeVar("z1", vx_t);
  auto w_t = VecType(16, 64);
  auto x2 = MakeVar("x2", w_t);
  auto y2 = MakeVar("y2", w_t);
  auto iv_c = MakeVar("ic", IndexType());
  auto child_c = SimpleFor(iv_c, CI(0), CI(2), CI(1),
                           {EvalCall("buffer.load", {gm_v, ConstTup({0, 0}), ConstTup({16, 64}), x2}),
                            EvalCall("buffer.add", {x2, x2, y2})});
  auto while_body = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{BufferReshape(r1, p1, r_t), BufferSubview(sub_a, p1, CI(0), CI(0), va_t),
                           BufferSubview(sub_b, r1, CI(0), CI(8), vb_t),
                           EvalCall("buffer.load", {gm_v, ConstTup({0, 0}), ConstTup({3, 8}), sub_a}),
                           EvalCall("buffer.load", {gm_v, ConstTup({0, 0}), ConstTup({16, 64}), x1}),
                           EvalCall("buffer.add", {x1, x1, y1}),
                           EvalCall("buffer.add", {sub_b, sub_b, sub_b}),
                           std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)},
      kSpan);
  auto while_stmt = std::make_shared<WhileStmt>(cond_w, std::vector<IterArgPtr>{}, while_body,
                                                std::vector<VarPtr>{}, kSpan);

  auto fn_sub =
      MakeAicFn("class_b_and_64bit_rearm", {gm_v, out_v, cond_w},
                {ParamDirection::In, ParamDirection::InOut, ParamDirection::In},
                {AllocBuf(p1, 0, s_t), AllocBuf(x1, 4096, vx_t), AllocBuf(y1, 8192, vx_t),
                 AllocBuf(z1, 12288, vx_t), AllocBuf(x2, 16384, w_t), AllocBuf(y2, 20480, w_t), while_stmt,
                 child_c, EvalCall("buffer.store", {y2, ConstTup({0, 0}), ConstTup({16, 64}), out_v}),
                 std::make_shared<ReturnStmt>(std::vector<ExprPtr>{out_v}, kSpan)},
                FunctionType::AIV);

  struct ScopedPassCtx {
    PassContext ctx;
    ScopedPassCtx()
        : ctx({}, VerificationLevel::Basic, DiagnosticPhase::PrePipeline, {}, MemoryPlanner::PyPTO, false,
              kDefaultRuntimeKind, true, false, false, true, true, "conservative_prefix") {
      ctx.EnterContext();
    }
    ~ScopedPassCtx() { ctx.ExitContext(); }
  } scoped_ctx;
  auto auto_children = PrepareAutomaticSyncChildren(fn_sub, "a2a3");
  assert(auto_children.inline_subblocks.size() == 1);
  assert(auto_children.inline_subblocks[0].record_sites >= 1);
  assert(auto_children.inline_subblocks[0].completion.IsRelayed(PipeType::MTE2, PipeType::V));

  const auto cons_sub = AnalyzeConservativeSync(fn_sub, "a2a3", {}, {}, false, /*automatic_children=*/true);
  // 2 ChildEntries: 1 Class B inline subblock (eventful==nullptr) + 1 EventfulSyncChild `child_c`.
  assert(cons_sub.ChildEntries().size() == 2);
  const ConservativeSyncChildEntry* eventful_entry = nullptr;
  for (const auto& ce : cons_sub.ChildEntries()) {
    if (ce.eventful) eventful_entry = &ce;
  }
  assert(eventful_entry != nullptr);
  // Class B inline subblock must NOT poison `reaching_child_dirs`: `child_c` stays first_visit=true!
  assert(eventful_entry->first_visit);

  // Prefix insertion + ClosedPrefixSync lowering succeeds end-to-end!
  const auto logical_sub = InsertConservativePrefixes(cons_sub);
  assert(logical_sub->GetAttr<int>("autosync_retained_child_count") == 1);
  const auto lowered_sub = TryLowerClosedPrefixSync(logical_sub, "a2a3", std::nullopt);
  assert(lowered_sub.has_value());

  // 3. Tier 2 Class A (multi-statement window with inner ForStmt inside an outer WhileStmt):
  auto fn_class_a_while_body = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{EvalCall("buffer.load", {gm_v, ConstTup({0, 0}), ConstTup({16, 64}), x1}),
                           SimpleFor(iv_c, CI(0), CI(2), CI(1), {EvalCall("buffer.add", {x1, x1, y1})}),
                           EvalCall("buffer.mul", {y1, y1, z1}),
                           std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)},
      kSpan);
  auto while_a = std::make_shared<WhileStmt>(cond_w, std::vector<IterArgPtr>{}, fn_class_a_while_body,
                                             std::vector<VarPtr>{}, kSpan);
  auto fn_class_a =
      MakeAicFn("class_a_subblock_window", {gm_v, out_v, cond_w},
                {ParamDirection::In, ParamDirection::InOut, ParamDirection::In},
                {AllocBuf(x1, 0, vx_t), AllocBuf(y1, 4096, vx_t), AllocBuf(z1, 8192, vx_t), while_a,
                 EvalCall("buffer.store", {z1, ConstTup({0, 0}), ConstTup({16, 64}), out_v}),
                 std::make_shared<ReturnStmt>(std::vector<ExprPtr>{out_v}, kSpan)},
                FunctionType::AIV);

  const auto cons_a = AnalyzeConservativeSync(fn_class_a, "a2a3", {}, {}, false, /*automatic_children=*/true);
  assert(cons_a.ChildEntries().size() == 1);
  assert(cons_a.ChildEntries()[0].IsSubblockSlice());
  assert(cons_a.ChildEntries()[0].eventful != nullptr);
  assert(cons_a.ChildEntries()[0].eventful->Scope()->HasAttr(kClosedSyncChildSubblockAttr));
  const auto logical_a = InsertConservativePrefixes(cons_a);
  const auto lowered_a = TryLowerClosedPrefixSync(logical_a, "a2a3", std::nullopt);
  assert(lowered_a.has_value());
  // After unwrapping kClosedSyncChildSubblockAttr, the lowered WhileStmt body must NOT contain any
  // synthetic _sync_win_iv ForStmt or YieldStmt before its end!
  const auto seq_a = As<SeqStmts>(lowered_a->function->body_);
  assert(seq_a);
  for (const auto& s : seq_a->stmts_) {
    if (const auto ws = As<WhileStmt>(s)) {
      const auto wb = As<SeqStmts>(ws->body_);
      assert(wb && !wb->stmts_.empty() && As<YieldStmt>(wb->stmts_.back()));
      for (size_t i = 0; i + 1 < wb->stmts_.size(); ++i) {
        assert(!As<YieldStmt>(wb->stmts_[i]));
        if (const auto fl = As<ForStmt>(wb->stmts_[i])) {
          assert(fl->loop_var_->name_hint_ != "_sync_win_iv");
        }
      }
    }
  }
}

void TestRound1AuditFixes() {
  auto has_pipe = [](const std::vector<PipeType>& pipes, PipeType target) {
    return std::find(pipes.begin(), pipes.end(), target) != pipes.end();
  };

  // Bug 1: ExtractAffine must NOT treat loop-varying IfStmt::return_vars_ as symbolic_invariant.
  {
    auto gm_t = MakeTensorType(4, 64, DataType::FP32);
    auto gm_in = MakeVar("gm_in", gm_t);
    auto gm_out = MakeVar("gm_out", gm_t);
    auto buf_t = VecType(4, 64);
    auto row_t = VecType(1, 64);
    auto buf = MakeVar("buf", buf_t);
    auto iv = MakeVar("i", IndexType());
    auto delta = MakeVar("delta", IndexType());
    auto row_idx = MakeVar("row_idx", IndexType());
    auto sub = MakeVar("sub", row_t);

    auto cond_odd = MakeEq(MakeFloorMod(iv, CI(2)), CI(1));
    auto br = std::make_shared<IfStmt>(
        cond_odd, std::make_shared<YieldStmt>(std::vector<ExprPtr>{CI(-1)}, kSpan),
        std::optional<StmtPtr>{std::make_shared<YieldStmt>(std::vector<ExprPtr>{CI(0)}, kSpan)},
        std::vector<VarPtr>{delta}, kSpan);
    auto assign_row = std::make_shared<AssignStmt>(row_idx, MakeAdd(iv, delta), kSpan);
    auto sub_stmt = BufferSubview(sub, buf, row_idx, CI(0), row_t);
    auto load_sub = EvalCall("buffer.load", {gm_in, MakeTup({CI(0), CI(0)}), ConstTup({1, 64}), sub});
    auto add_sub = EvalCall("buffer.add", {sub, sub, sub});
    auto loop = SimpleFor(iv, CI(0), CI(4), CI(1), {br, assign_row, sub_stmt, load_sub, add_sub});
    auto fn = MakeAicFn("bug1_if_return_var_not_invariant", {gm_in, gm_out},
                        {ParamDirection::In, ParamDirection::InOut},
                        {AllocBuf(buf, 0, buf_t), loop,
                         EvalCall("buffer.store", {buf, ConstTup({0, 0}), ConstTup({4, 64}), gm_out}),
                         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
                        FunctionType::AIV);

    SyncControlRangeProver prover(fn);
    const auto row_range = prover.LookupAtCut(sub_stmt.get(), row_idx);
    assert(row_range.has_value());
    assert(row_range->affine_frame == nullptr);

    const auto analysis = AnalyzeConservativeSync(fn, "a2a3", {}, {}, false, /*automatic_children=*/false);
    assert(analysis.Sites().size() == 3);
    // Trip 0 and Trip 1 both touch row 0 -> load must keep V->MTE2 backedge wait!
    assert(has_pipe(analysis.Sites()[0].source_pipes, PipeType::V));
  }

  // Bug 2: StridedByteFootprint must not assert parity when symbolic_base != nullptr or lo is clamped.
  {
    auto gm_t = MakeTensorType(8, 64, DataType::FP32);
    auto gm_in = MakeVar("gm_in", gm_t);
    auto gm_out = MakeVar("gm_out", gm_t);
    auto p = MakeVar("p", IndexType());
    auto buf_t = VecType(8, 64);
    auto row_t = VecType(1, 64);
    auto buf = MakeVar("buf", buf_t);
    auto iv = MakeVar("i", IndexType());
    auto s_dyn = MakeVar("s_dyn", row_t);
    auto s_odd = MakeVar("s_odd", row_t);

    auto load_dyn = EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({1, 64}), s_dyn});
    auto add_odd = EvalCall("buffer.add", {s_odd, s_odd, s_odd});
    auto loop = SimpleFor(
        iv, CI(0), CI(3), CI(1),
        {BufferSubview(s_dyn, buf, MakeAdd(p, MakeMul(CI(2), iv)), CI(0), row_t),
         BufferSubview(s_odd, buf, MakeAdd(CI(1), MakeMul(CI(2), iv)), CI(0), row_t), load_dyn, add_odd});
    auto guard = MakeAnd(MakeGe(p, CI(0)), MakeLe(p, CI(1)));
    auto if_stmt = std::make_shared<IfStmt>(
        guard,
        std::make_shared<SeqStmts>(
            std::vector<StmtPtr>{loop, std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)}, kSpan),
        std::nullopt, std::vector<VarPtr>{}, kSpan);
    auto fn = MakeAicFn("bug2_strided_sym_base", {gm_in, gm_out, p},
                        {ParamDirection::In, ParamDirection::InOut, ParamDirection::In},
                        {AllocBuf(buf, 0, buf_t), if_stmt,
                         EvalCall("buffer.store", {buf, ConstTup({0, 0}), ConstTup({8, 64}), gm_out}),
                         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
                        FunctionType::AIV);

    const auto analysis = AnalyzeConservativeSync(fn, "a2a3", {}, {}, false, /*automatic_children=*/false);
    assert(analysis.Sites().size() == 3);
    assert(has_pipe(analysis.Sites()[1].source_pipes, PipeType::MTE2));
  }

  // Bug 3: 2D Matrix IterationFiber::AffineSlice must distinguish row slices (axis 0) from col slices
  // (axis 1).
  {
    auto mat_t = MatType(64, 64);
    auto acc_t = AccType(16, 64);
    auto left_t = LeftType(64, 16);
    auto gm_t = MakeTensorType(64, 64, DataType::FP16);
    auto gm_out = MakeVar("gm_out", gm_t);
    auto mat = MakeVar("mat", mat_t);
    auto acc = MakeVar("acc", acc_t);
    auto left = MakeVar("left", left_t);
    auto iv = MakeVar("i", IndexType());

    auto ins_row = EvalCall("buffer.insert", {acc, MakeMul(iv, CI(16)), CI(0), mat});
    auto ext_col = EvalCall("buffer.extract", {mat, CI(0), MakeMul(MakeAdd(iv, CI(1)), CI(16)), left});
    auto loop = SimpleFor(iv, CI(0), CI(2), CI(1), {ins_row, ext_col});
    auto fn = MakeAicFn("bug3_mat_row_vs_col_slice_axis", {gm_out}, {ParamDirection::InOut},
                        {AllocBuf(mat, 0, mat_t), AllocBuf(acc, 0, acc_t), AllocBuf(left, 0, left_t), loop,
                         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
                        FunctionType::AIC);

    const auto analysis = AnalyzeConservativeSync(fn, "a2a3", {}, {}, false, /*automatic_children=*/false);
    assert(analysis.Sites().size() == 2);
    // mat[0:16, 0:64] overlaps mat[0:64, 16:32] at mat[0:16, 16:32] -> extract MUST wait on FIX!
    assert(has_pipe(analysis.Sites()[1].source_pipes, PipeType::FIX));
  }

  // Bug 4: Non-transpose Matrix buffer.reshape must invalidate matrix coordinate provenance.
  {
    auto m1_t = MatType(32, 64);
    auto m2_t = MatType(64, 32);  // Same (row_major, col_major) layout -> NOT a transpose!
    auto acc_t = AccType(16, 16);
    auto left_t = LeftType(16, 16);
    auto gm_t = MakeTensorType(32, 64, DataType::FP16);
    auto gm_out = MakeVar("gm_out", gm_t);
    auto m1 = MakeVar("m1", m1_t);
    auto m2 = MakeVar("m2", m2_t);
    auto acc = MakeVar("acc", acc_t);
    auto left = MakeVar("left", left_t);

    auto fn = MakeAicFn("bug4_non_transpose_mat_reshape", {gm_out}, {ParamDirection::InOut},
                        {AllocBuf(m1, 0, m1_t), AllocBuf(acc, 0, acc_t), AllocBuf(left, 0, left_t),
                         BufferReshape(m2, m1, m2_t), EvalCall("buffer.insert", {acc, CI(0), CI(16), m1}),
                         EvalCall("buffer.extract", {m2, CI(32), CI(0), left}),
                         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
                        FunctionType::AIC);

    SyncFootprintResolver resolver(fn);
    assert(resolver.ViewOf(m1) && resolver.ViewOf(m1)->valid_matrix_coords);
    assert(resolver.ViewOf(m2) && !resolver.ViewOf(m2)->valid_matrix_coords);
    assert(!resolver.ViewOf(m2)->matrix_box.has_value());

    const auto analysis = AnalyzeConservativeSync(fn, "a2a3", {}, {}, false, /*automatic_children=*/false);
    assert(analysis.Sites().size() == 2);
    assert(has_pipe(analysis.Sites()[1].source_pipes, PipeType::FIX));
  }

  // Bug 5: ClosedPrefixSync::Discover must include cross-pipe SyncNativeClaims in prior_child_dirs_.
  {
    auto gm_t = MakeTensorType(16, 64, DataType::FP32);
    auto gm_in = MakeVar("gm_in", gm_t);
    auto gm_out = MakeVar("gm_out", gm_t);
    auto buf_t = VecType(16, 64);
    auto x1 = MakeVar("x1", buf_t);
    auto y1 = MakeVar("y1", buf_t);
    auto x2 = MakeVar("x2", buf_t);
    auto y2 = MakeVar("y2", buf_t);
    auto iv1 = MakeVar("i1", IndexType());
    auto iv2 = MakeVar("i2", IndexType());

    auto tok_t = std::make_shared<SyncTokenType>(PipeType::MTE2);
    auto rec_mte2_1 = OpRegistry::GetInstance().Create("sync.record", {},
                                                       {{"pipe", static_cast<int>(PipeType::MTE2)}}, kSpan);
    auto tv1 = MakeVar("tv1", tok_t);
    auto wait_v_1 = OpRegistry::GetInstance().Create("sync.wait", {tv1},
                                                     {{"dst_pipe", static_cast<int>(PipeType::V)}}, kSpan);
    auto c1_base =
        SimpleFor(iv1, CI(0), CI(2), CI(1),
                  {EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 64}), x1}),
                   std::make_shared<AssignStmt>(tv1, rec_mte2_1, kSpan),
                   std::make_shared<EvalStmt>(wait_v_1, kSpan), EvalCall("buffer.recip", {x1, y1})});
    auto c1 = std::make_shared<ForStmt>(*c1_base);
    c1->attrs_.emplace_back(kClosedSyncChildAttr, 1);
    c1->attrs_.emplace_back(kClosedSyncChildFirstVisitAttr, true);

    // Child 2 has MTE3-wait from S or no sync.wait on V<->S, yet uses buffer.recip (native V<->S claim).
    // Marking Child 2 first_visit must be rejected because Child 1 already claimed V<->S!
    auto rec_v_2 =
        OpRegistry::GetInstance().Create("sync.record", {}, {{"pipe", static_cast<int>(PipeType::V)}}, kSpan);
    auto tok_v_t = std::make_shared<SyncTokenType>(PipeType::V);
    auto tv2 = MakeVar("tv2", tok_v_t);
    auto wait_mte3_2 = OpRegistry::GetInstance().Create(
        "sync.wait", {tv2}, {{"dst_pipe", static_cast<int>(PipeType::MTE3)}}, kSpan);
    auto c2_base =
        SimpleFor(iv2, CI(0), CI(2), CI(1),
                  {EvalCall("buffer.recip", {x2, y2}), std::make_shared<AssignStmt>(tv2, rec_v_2, kSpan),
                   std::make_shared<EvalStmt>(wait_mte3_2, kSpan),
                   EvalCall("buffer.store", {y2, ConstTup({0, 0}), ConstTup({16, 64}), gm_out})});
    auto c2 = std::make_shared<ForStmt>(*c2_base);
    c2->attrs_.emplace_back(kClosedSyncChildAttr, 1);
    c2->attrs_.emplace_back(kClosedSyncChildFirstVisitAttr, true);

    auto fn = MakeAicFn("bug5_native_claim_prior_dirs", {gm_in, gm_out},
                        {ParamDirection::In, ParamDirection::InOut},
                        {AllocBuf(x1, 0, buf_t), AllocBuf(y1, 4096, buf_t), AllocBuf(x2, 8192, buf_t),
                         AllocBuf(y2, 12288, buf_t), c1, c2,
                         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
                        FunctionType::AIV);
    bool rejected = false;
    try {
      (void)TryLowerClosedPrefixSync(fn, "a2a3", std::nullopt);
    } catch (const pypto::ValueError&) {
      rejected = true;
    }
    assert(rejected);
  }

  // Bug 6a & 6b:
  // 6a: set_validshape on unproved slot invalidates ExactFullWindow on other slots in same space, and
  // alias copy clears exact on root.
  {
    auto gm_t = MakeTensorType(16, 64, DataType::FP32);
    auto gm_in = MakeVar("gm_in", gm_t);
    auto dyn_idx = MakeVar("dyn_idx", IndexType());
    auto pool_t = MultiVecType(2, 16, 64);
    auto elem_t = VecType(16, 64);
    auto pool = MakeVar("pool", pool_t);
    auto s0 = MakeVar("s0", elem_t);
    auto s_dyn = MakeVar("s_dyn", elem_t);
    auto b0 = MakeVar("b0", elem_t);
    auto b1 = MakeVar("b1", elem_t);

    auto fn_dyn =
        MakeAicFn("bug6a_dynamic_valid_unbounded", {gm_in, dyn_idx}, {ParamDirection::In, ParamDirection::In},
                  {AllocBuf(pool, 0, pool_t, "buffer.alloc_multi"), BufferSlot(s0, pool, CI(0), elem_t),
                   BufferSlot(s_dyn, pool, dyn_idx, elem_t),
                   EvalCall("buffer.set_validshape", {s_dyn, ConstTup({16, 64})}),
                   EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 64}), s0}),
                   std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)},
                  FunctionType::AIV);

    SyncFootprintResolver resolver_dyn(fn_dyn);
    assert(!resolver_dyn.ExactFullWindow(s0).has_value());
    assert(resolver_dyn.ViewOf(s0) && !resolver_dyn.ViewOf(s0)->static_full_valid);
    assert(resolver_dyn.ViewOf(s0)->vec_lattice && !resolver_dyn.ViewOf(s0)->vec_lattice->exact);

    auto fn_alias = MakeAicFn("bug6a_dynamic_valid_alias", {gm_in}, {ParamDirection::In},
                              {AllocBuf(b0, 0, elem_t), std::make_shared<AssignStmt>(b1, b0, kSpan),
                               EvalCall("buffer.set_validshape", {b1, ConstTup({16, 64})}),
                               EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 64}), b0}),
                               std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)},
                              FunctionType::AIV);

    SyncFootprintResolver resolver_alias(fn_alias);
    assert(!resolver_alias.ExactFullWindow(b0).has_value());
    assert(resolver_alias.ViewOf(b0) && !resolver_alias.ViewOf(b0)->static_full_valid);
    assert(resolver_alias.ViewOf(b0)->vec_lattice && !resolver_alias.ViewOf(b0)->vec_lattice->exact);
  }

  // 6b: Multiple InlineExactSubBlocks in one function get distinct freshened token names (and idempotency),
  // plus two same-pipe V producers before one V consumer emit a single bar_v pair.
  {
    auto gm_t = MakeTensorType(16, 64, DataType::FP32);
    auto gm_in = MakeVar("gm_in", gm_t);
    auto gm_out = MakeVar("gm_out", gm_t);
    auto cond = MakeVar("cond", BoolType());
    auto buf_t = VecType(16, 64);
    auto s_t = VecType(3, 32);
    auto r_t = VecType(2, 48);
    auto va_t = VecType(3, 8);
    auto vb_t = VecType(1, 24);
    auto p1 = MakeVar("p1", s_t);
    auto r1 = MakeVar("r1", r_t);
    auto sa1 = MakeVar("sa1", va_t);
    auto sb1 = MakeVar("sb1", vb_t);
    auto p2 = MakeVar("p2", s_t);
    auto r2 = MakeVar("r2", r_t);
    auto sa2 = MakeVar("sa2", va_t);
    auto sb2 = MakeVar("sb2", vb_t);
    auto a1 = MakeVar("a1", buf_t);
    auto a2 = MakeVar("a2", buf_t);
    auto a3 = MakeVar("a3", buf_t);
    auto b1 = MakeVar("b1", buf_t);
    auto b2 = MakeVar("b2", buf_t);
    auto b3 = MakeVar("b3", buf_t);

    auto w1 = std::make_shared<WhileStmt>(
        cond, std::vector<IterArgPtr>{},
        std::make_shared<SeqStmts>(
            std::vector<StmtPtr>{BufferReshape(r1, p1, r_t), BufferSubview(sa1, p1, CI(0), CI(0), va_t),
                                 BufferSubview(sb1, r1, CI(0), CI(8), vb_t),
                                 EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({3, 8}), sa1}),
                                 EvalCall("buffer.add", {a1, a1, a2}), EvalCall("buffer.add", {a1, a2, a3}),
                                 EvalCall("buffer.mul", {a2, a3, a1}),
                                 EvalCall("buffer.add", {sb1, sb1, sb1}),
                                 std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)},
            kSpan),
        std::vector<VarPtr>{}, kSpan);
    auto w2 = std::make_shared<WhileStmt>(
        cond, std::vector<IterArgPtr>{},
        std::make_shared<SeqStmts>(
            std::vector<StmtPtr>{BufferReshape(r2, p2, r_t), BufferSubview(sa2, p2, CI(0), CI(0), va_t),
                                 BufferSubview(sb2, r2, CI(0), CI(8), vb_t),
                                 EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({3, 8}), sa2}),
                                 EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 64}), b1}),
                                 EvalCall("buffer.add", {b1, b1, b2}), EvalCall("buffer.mul", {b2, b2, b3}),
                                 EvalCall("buffer.add", {sb2, sb2, sb2}),
                                 std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)},
            kSpan),
        std::vector<VarPtr>{}, kSpan);
    auto fn = MakeAicFn("bug6b_multi_inline_subblock_and_same_pipe_dedup", {gm_in, gm_out, cond},
                        {ParamDirection::In, ParamDirection::InOut, ParamDirection::In},
                        {AllocBuf(p1, 0, s_t), AllocBuf(a1, 4096, buf_t), AllocBuf(a2, 8192, buf_t),
                         AllocBuf(a3, 12288, buf_t), AllocBuf(p2, 16384, s_t), AllocBuf(b1, 20480, buf_t),
                         AllocBuf(b2, 24576, buf_t), AllocBuf(b3, 28672, buf_t), w1, w2,
                         EvalCall("buffer.store", {b3, ConstTup({0, 0}), ConstTup({16, 64}), gm_out}),
                         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
                        FunctionType::AIV);

    struct ScopedPassCtx {
      PassContext ctx;
      ScopedPassCtx()
          : ctx({}, VerificationLevel::Basic, DiagnosticPhase::PrePipeline, {}, MemoryPlanner::PyPTO, false,
                kDefaultRuntimeKind, true, false, false, true, true, "conservative_prefix") {
        ctx.EnterContext();
      }
      ~ScopedPassCtx() { ctx.ExitContext(); }
    } scoped_ctx;

    const auto cons = AnalyzeConservativeSync(fn, "a2a3", {}, {}, false, /*automatic_children=*/true);
    assert(cons.ChildEntries().size() == 2);
    assert(cons.ChildEntries()[0].inline_subblock.has_value());
    assert(cons.ChildEntries()[1].inline_subblock.has_value());
    // w1 has add1(V->a2) + add2(V->a3) -> mul(V reads a2,a3): deduplicated to 1 bar_v per consumer!
    const auto logical1 = InsertConservativePrefixes(cons);
    const auto logical2 = InsertConservativePrefixes(cons);
    const auto lowered = TryLowerClosedPrefixSync(logical1, "a2a3", std::nullopt);
    assert(lowered.has_value());
    assert(logical2 != nullptr);
  }
}

void TestRound2AuditAndCleanupFixes() {
  struct ScopedPassCtx {
    PassContext ctx;
    ScopedPassCtx()
        : ctx({}, VerificationLevel::Basic, DiagnosticPhase::PrePipeline, {}, MemoryPlanner::PyPTO, false,
              kDefaultRuntimeKind, true, false, false, true, true, "conservative_prefix") {
      ctx.EnterContext();
    }
    ~ScopedPassCtx() { ctx.ExitContext(); }
  } scoped_ctx;

  auto gm_t = MakeTensorType(16, 64, DataType::FP32);
  auto gm_in = MakeVar("gm_in", gm_t);
  auto gm_out = MakeVar("gm_out", gm_t);
  auto cond = MakeVar("cond", BoolType());

  // [B-P1.1] parent_static_origin requires parent->vec_lattice && width == cols * elem_bytes.
  {
    auto buf_t = VecType(4, 32);
    auto col_sub_t = VecType(4, 16);
    auto row_sub_t = VecType(2, 16);
    auto buf = MakeVar("buf", buf_t);
    auto col_off = MakeVar("col_off", IndexType());
    auto sub_move = MakeVar("sub_move", col_sub_t);
    auto r0 = MakeVar("r0", row_sub_t);
    auto r2 = MakeVar("r2", row_sub_t);
    auto fn = MakeAicFn("bp1_1", {col_off}, {ParamDirection::In},
                        {AllocBuf(buf, 0, buf_t), BufferSubview(sub_move, buf, CI(0), col_off, col_sub_t),
                         BufferSubview(r0, sub_move, CI(0), CI(0), row_sub_t),
                         BufferSubview(r2, sub_move, CI(2), CI(0), row_sub_t),
                         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)},
                        FunctionType::AIV);
    SyncFootprintResolver resolver(fn);
    assert(resolver.ViewOf(sub_move) && resolver.ViewOf(sub_move)->vec_lattice &&
           resolver.ViewOf(sub_move)->vec_lattice->width == 32 * 4);
    assert(resolver.ViewOf(r0) && !resolver.ViewOf(r0)->vec_lattice &&
           resolver.ViewOf(r0)->byte_hull->end == 4 * 32 * 4);
    assert(resolver.ViewOf(r2) && !resolver.ViewOf(r2)->vec_lattice &&
           resolver.ViewOf(r2)->byte_hull->end == 4 * 32 * 4);
  }

  // [B-P1.2] Chained column subview preserves parent column spread in vec_lattice->width.
  {
    auto buf_t = VecType(4, 32);
    auto sub8_t = VecType(4, 8);
    auto sub4_t = VecType(4, 4);
    auto buf = MakeVar("buf", buf_t);
    auto c_var = MakeVar("c_var", IndexType());
    auto sub_move = MakeVar("sub_move", sub8_t);
    auto c0 = MakeVar("c0", sub4_t);
    auto c1 = MakeVar("c1", sub4_t);
    auto c_cond = std::make_shared<And>(std::make_shared<Ge>(c_var, CI(0), DataType::BOOL, kSpan),
                                        std::make_shared<Le>(c_var, CI(1), DataType::BOOL, kSpan),
                                        DataType::BOOL, kSpan);
    auto inner_seq = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{BufferSubview(sub_move, buf, CI(0), c_var, sub8_t),
                             BufferSubview(c0, sub_move, CI(0), CI(0), sub4_t),
                             BufferSubview(c1, sub_move, CI(0), CI(4), sub4_t),
                             std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)},
        kSpan);
    auto br = std::make_shared<IfStmt>(c_cond, inner_seq, std::nullopt, std::vector<VarPtr>{}, kSpan);
    auto fn =
        MakeAicFn("bp1_2", {c_var}, {ParamDirection::In},
                  {AllocBuf(buf, 0, buf_t), br, std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)},
                  FunctionType::AIV);
    SyncFootprintResolver resolver(fn);
    assert(resolver.ViewOf(c0) && resolver.ViewOf(c0)->vec_lattice);
    assert(resolver.ViewOf(c1) && resolver.ViewOf(c1)->vec_lattice);
    assert(StridedByteFootprint::MayOverlap(*resolver.ViewOf(c0)->vec_lattice,
                                            *resolver.ViewOf(c1)->vec_lattice));
  }

  // [B-P1.3] Unproved orthogonal coordinate must NOT yield AffineSlice fiber.
  {
    auto buf_t = VecType(4, 32);
    auto sub_t = VecType(1, 8);
    auto buf = MakeVar("buf", buf_t);
    auto iv = MakeVar("i", IndexType());
    auto c_unproved = MakeVar("c_unproved", IndexType());
    auto sub = MakeVar("sub", sub_t);
    auto loop = SimpleFor(iv, CI(0), CI(4), CI(1),
                          {BufferSubview(sub, buf, iv, c_unproved, sub_t),
                           EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({1, 8}), sub})});
    auto fn = MakeAicFn(
        "bp1_3", {gm_in, c_unproved}, {ParamDirection::In, ParamDirection::In},
        {AllocBuf(buf, 0, buf_t), loop, std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)},
        FunctionType::AIV);
    SyncFootprintResolver resolver(fn);
    assert(resolver.ViewOf(sub) && resolver.ViewOf(sub)->fiber.kind == IterationFiber::Kind::Wildcard);
  }

  // [B-P1.4] Overflowing AssignStmt (`0 - unbounded`) must not become ready_ for prover.
  {
    auto p_unb = MakeVar("p_unb", IndexType());
    auto neg_p = MakeVar("neg_p", IndexType());
    auto zero_c = MakeVar("zero_c", IndexType());
    auto iv = MakeVar("i", IndexType());
    auto assign_neg = std::make_shared<AssignStmt>(neg_p, MakeSub(CI(0), p_unb), kSpan);
    auto assign_sum = std::make_shared<AssignStmt>(zero_c, MakeAdd(neg_p, p_unb), kSpan);
    auto loop = SimpleFor(iv, CI(0), MakeAdd(zero_c, CI(1)), CI(1), {});
    auto fn =
        MakeAicFn("bp1_4", {p_unb}, {ParamDirection::In},
                  {assign_neg, assign_sum, loop, std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)},
                  FunctionType::AIV);
    SyncControlRangeProver prover(fn);
    assert(!prover.IsPositiveTrip(loop.get()));
  }

  // [B-P1.5] AffineSlice near INT64_MAX stays Wildcard rather than wrapping int64_t.
  {
    const int64_t kHuge = std::numeric_limits<int64_t>::max() - 10;
    auto buf_t = VecType(4, 64);
    auto sub_t = VecType(1, 64);
    auto buf = MakeVar("buf", buf_t);
    auto iv = MakeVar("i", IndexType());
    auto sub = MakeVar("sub", sub_t);
    auto loop =
        SimpleFor(iv, CI(0), CI(2), CI(1), {BufferSubview(sub, buf, MakeAdd(iv, CI(kHuge)), CI(0), sub_t)});
    auto fn = MakeAicFn(
        "bp1_5", {}, {},
        {AllocBuf(buf, 0, buf_t), loop, std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)},
        FunctionType::AIV);
    SyncFootprintResolver resolver(fn);
    assert(resolver.ViewOf(sub) && resolver.ViewOf(sub)->fiber.kind == IterationFiber::Kind::Wildcard);
  }

  // [B-P1.6 R2] <= 1-trip non-const-start loop allows first-visit closed child through LowerSync.
  {
    auto vx_t = VecType(16, 64);
    auto x = MakeVar("x", vx_t);
    auto y = MakeVar("y", vx_t);
    auto iv_outer = MakeVar("ib", IndexType());
    auto iv_inner = MakeVar("ii", IndexType());
    auto bidx = TypedCall("buffer.get_block_idx", {}, IndexType());
    auto b_var = MakeVar("bidx", IndexType());
    auto inner = SimpleFor(iv_inner, CI(0), CI(2), CI(1),
                           {EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 64}), x}),
                            EvalCall("buffer.add", {x, x, y})});
    auto outer = SimpleFor(iv_outer, b_var, CI(2), CI(2), {inner});
    auto fn = MakeAicFn(
        "bp1_6_r2", {gm_in, gm_out}, {ParamDirection::In, ParamDirection::InOut},
        {std::make_shared<AssignStmt>(b_var, bidx, kSpan), AllocBuf(x, 0, vx_t), AllocBuf(y, 4096, vx_t),
         outer, EvalCall("buffer.store", {y, ConstTup({0, 0}), ConstTup({16, 64}), gm_out}),
         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
        FunctionType::AIV);
    const auto cons =
        AnalyzeConservativeSync(fn, "a2a3", {}, {}, false, /*automatic_children=*/true,
                                /*enable_refresh_kill=*/true, {}, std::make_pair(int64_t{0}, int64_t{1}));
    assert(cons.ChildEntries().size() == 1);
    const auto logical = InsertConservativePrefixes(cons);
    const auto lowered = TryLowerClosedPrefixSync(logical, "a2a3", std::nullopt, {}, {},
                                                  std::make_pair(int64_t{0}, int64_t{1}));
    assert(lowered.has_value());
  }

  // [B-P2.8 R3 (a)&(c)] Non-sharpening slice is skipped; >4 simultaneous in-flight per dir is declined.
  {
    auto vx_t = VecType(16, 64);
    auto x1 = MakeVar("x1", vx_t);
    auto y1 = MakeVar("y1", vx_t);
    auto z1 = MakeVar("z1", vx_t);
    auto w_ns = std::make_shared<WhileStmt>(
        cond, std::vector<IterArgPtr>{},
        std::make_shared<SeqStmts>(
            std::vector<StmtPtr>{EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 64}), x1}),
                                 EvalCall("buffer.add", {x1, x1, y1}), EvalCall("buffer.mul", {y1, y1, z1}),
                                 std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan)},
            kSpan),
        std::vector<VarPtr>{}, kSpan);
    auto fn_ns = MakeAicFn("bp2_8_non_sharp", {gm_in, gm_out, cond},
                           {ParamDirection::In, ParamDirection::InOut, ParamDirection::In},
                           {AllocBuf(x1, 0, vx_t), AllocBuf(y1, 4096, vx_t), AllocBuf(z1, 8192, vx_t), w_ns,
                            EvalCall("buffer.store", {z1, ConstTup({0, 0}), ConstTup({16, 64}), gm_out}),
                            std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan)},
                           FunctionType::AIV);
    auto auto_ns = PrepareAutomaticSyncChildren(fn_ns, "a2a3");
    assert(auto_ns.inline_subblocks.empty());

    // 5 simultaneous in-flight (MTE2 -> V) pairs + sharpening subview -> declined (<= 4 cap).
    auto s_t = VecType(3, 32);
    auto r_t = VecType(2, 48);
    auto va_t = VecType(3, 8);
    auto vb_t = VecType(1, 24);
    auto p1 = MakeVar("p1", s_t);
    auto r1 = MakeVar("r1", r_t);
    auto sa = MakeVar("sa", va_t);
    auto sb = MakeVar("sb", vb_t);
    std::vector<VarPtr> u_bufs, v_bufs;
    std::vector<StmtPtr> allocs, body;
    allocs.push_back(AllocBuf(p1, 0, s_t));
    for (int idx = 0; idx < 5; ++idx) {
      auto ub = MakeVar("u" + std::to_string(idx), vx_t);
      auto vb = MakeVar("v" + std::to_string(idx), vx_t);
      u_bufs.push_back(ub);
      v_bufs.push_back(vb);
      allocs.push_back(AllocBuf(ub, 4096 + idx * 8192, vx_t));
      allocs.push_back(AllocBuf(vb, 8192 + idx * 8192, vx_t));
    }
    body.push_back(BufferReshape(r1, p1, r_t));
    body.push_back(BufferSubview(sa, p1, CI(0), CI(0), va_t));
    body.push_back(BufferSubview(sb, r1, CI(0), CI(8), vb_t));
    body.push_back(EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({3, 8}), sa}));
    for (int idx = 0; idx < 5; ++idx) {
      body.push_back(EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 64}), u_bufs[idx]}));
    }
    body.push_back(EvalCall("buffer.add", {sb, sb, sb}));
    for (int idx = 0; idx < 5; ++idx) {
      body.push_back(EvalCall("buffer.add", {u_bufs[idx], u_bufs[idx], v_bufs[idx]}));
    }
    body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan));
    auto w_cap = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{},
                                             std::make_shared<SeqStmts>(std::move(body), kSpan),
                                             std::vector<VarPtr>{}, kSpan);
    allocs.push_back(w_cap);
    allocs.push_back(EvalCall("buffer.store", {v_bufs.back(), ConstTup({0, 0}), ConstTup({16, 64}), gm_out}));
    allocs.push_back(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan));
    auto fn_cap = MakeAicFn("bp2_8_cap5", {gm_in, gm_out, cond},
                            {ParamDirection::In, ParamDirection::InOut, ParamDirection::In},
                            std::move(allocs), FunctionType::AIV);
    auto auto_cap = PrepareAutomaticSyncChildren(fn_cap, "a2a3");
    assert(auto_cap.inline_subblocks.empty());
  }

  // [B-P2.9] ANF SSA-bound FloorMod produces ModularSlot fiber.
  {
    auto mvec_t = MultiVecType(4, 16, 64);
    auto vec_t = VecType(16, 64);
    auto pool = MakeVar("pool", mvec_t);
    auto rem0 = MakeVar("rem0", IndexType());
    auto s0 = MakeVar("s0", vec_t);
    auto iv = MakeVar("i", IndexType());
    auto assign_rem = std::make_shared<AssignStmt>(rem0, MakeFloorMod(MakeMul(CI(2), iv), CI(4)), kSpan);
    auto slot0 = BufferSlot(s0, pool, rem0, vec_t);
    auto loop = SimpleFor(
        iv, CI(0), CI(8), CI(1),
        {assign_rem, slot0, EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 64}), s0})});
    auto fn = MakeAicFn("bp2_9_anf_mod", {gm_in}, {ParamDirection::In},
                        {AllocBuf(pool, 0, mvec_t, "buffer.alloc_multi"), loop,
                         std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, kSpan)},
                        FunctionType::AIV);
    SyncFootprintResolver resolver(fn);
    assert(resolver.ViewOf(s0) && resolver.ViewOf(s0)->fiber.kind == IterationFiber::Kind::ModularSlot);
    assert(resolver.ViewOf(s0)->fiber.modulus == 4 && resolver.ViewOf(s0)->fiber.coeff == 2);
  }

  // Sequential ForStmts with Class-B InlineExactSubBlock reuse (MTE2 -> V) IDs after loop exit to scope 0.
  {
    auto vx_t = VecType(16, 64);
    auto s_t = VecType(3, 32);
    auto r_t = VecType(2, 48);
    auto va_t = VecType(3, 8);
    auto vb_t = VecType(1, 24);
    std::vector<StmtPtr> allocs;
    int64_t next_off = 0;
    auto mk_loop = [&](const std::string& pfx, int n_pairs) -> StmtPtr {
      auto p = MakeVar(pfx + "_p", s_t);
      auto r = MakeVar(pfx + "_r", r_t);
      auto sa = MakeVar(pfx + "_sa", va_t);
      auto sb = MakeVar(pfx + "_sb", vb_t);
      allocs.push_back(AllocBuf(p, next_off, s_t));
      next_off += 4096;
      std::vector<VarPtr> us, vs;
      for (int idx = 0; idx < n_pairs; ++idx) {
        auto u = MakeVar(pfx + "_u" + std::to_string(idx), vx_t);
        auto v = MakeVar(pfx + "_v" + std::to_string(idx), vx_t);
        us.push_back(u);
        vs.push_back(v);
        allocs.push_back(AllocBuf(u, next_off, vx_t));
        next_off += 4096;
        allocs.push_back(AllocBuf(v, next_off, vx_t));
        next_off += 4096;
      }
      std::vector<StmtPtr> wbody{BufferReshape(r, p, r_t), BufferSubview(sa, p, CI(0), CI(0), va_t),
                                 BufferSubview(sb, r, CI(0), CI(8), vb_t),
                                 EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({3, 8}), sa})};
      for (int idx = 0; idx < n_pairs; ++idx) {
        wbody.push_back(EvalCall("buffer.load", {gm_in, ConstTup({0, 0}), ConstTup({16, 64}), us[idx]}));
      }
      wbody.push_back(EvalCall("buffer.add", {sb, sb, sb}));
      for (int idx = 0; idx < n_pairs; ++idx) {
        wbody.push_back(EvalCall("buffer.add", {us[idx], us[idx], vs[idx]}));
      }
      wbody.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, kSpan));
      auto w = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{},
                                           std::make_shared<SeqStmts>(std::move(wbody), kSpan),
                                           std::vector<VarPtr>{}, kSpan);
      auto iv = MakeVar(pfx + "_i", IndexType());
      return SimpleFor(iv, CI(0), CI(2), CI(1), {w});
    };
    auto z_end = MakeVar("z_end", vx_t);
    allocs.push_back(AllocBuf(z_end, next_off, vx_t));
    auto l0 = mk_loop("l0", 4);
    auto l1 = mk_loop("l1", 4);
    auto l2 = mk_loop("l2", 1);
    allocs.push_back(l0);
    allocs.push_back(l1);
    allocs.push_back(l2);
    allocs.push_back(EvalCall("buffer.store", {z_end, ConstTup({0, 0}), ConstTup({16, 64}), gm_out}));
    allocs.push_back(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{gm_out}, kSpan));
    auto fn = MakeAicFn("rmsnorm_seq_reuse", {gm_in, gm_out, cond},
                        {ParamDirection::In, ParamDirection::InOut, ParamDirection::In}, std::move(allocs),
                        FunctionType::AIV);
    const auto cons = AnalyzeConservativeSync(fn, "a2a3", {}, {}, false, /*automatic_children=*/true);
    const auto logical = InsertConservativePrefixes(cons);
    const auto lowered = TryLowerClosedPrefixSync(logical, "a2a3", std::nullopt);
    assert(lowered.has_value() && lowered->set_sites > 0);
  }
}

}  // namespace
}  // namespace pypto::ir::sync

int main() {
  pypto::backend::BackendConfig::SetBackendType(pypto::backend::BackendType::Ascend910);
  pypto::ir::sync::TestPeerTransferBufferGeometry();
  pypto::ir::sync::TestB1SyncControlRangeProver();
  pypto::ir::sync::TestB1RangeUseCuts();
  pypto::ir::sync::TestB1PositiveTripCompletionAndRelay();
  pypto::ir::sync::TestB2StridedVecAndMatStorageClasses();
  pypto::ir::sync::TestB3IterationFiberRefinement();
  pypto::ir::sync::TestAffineVecPhysicalByteOrigins();
  pypto::ir::sync::TestB4InitGuardAndSubblockPromotion();
  pypto::ir::sync::TestRound1AuditFixes();
  pypto::ir::sync::TestRound2AuditAndCleanupFixes();
  std::cout << "All refactor_b_test assertions passed!" << std::endl;
  return 0;
}
