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

/// AutoTileMatmulL0
/// ----------------
/// For each ``tile.matmul`` or ``tile.matmul_acc`` with static 2D operands,
/// picks an L0 tile shape ``(m, n, k)`` from the active ``BackendHandler``'s
/// L0 capacities (via ``utils::ChooseL0Tile``) and rewrites the call into a
/// K-loop.  The right (B) operand must be ``Mat``-resident; the left (A)
/// operand may be ``Mat`` (the QK pattern) or ``Vec`` (the fused-attention
/// ``score·V`` / PV pattern, where the softmax output crosses the cube↔vector
/// boundary resident in ``Vec``).  Tiling the Vec-fed PV matmul symmetrically
/// with QK makes its L0B right buffer a reusable sub-tile so ``MemoryReuse``
/// can alias it onto QK's freed L0B (peak L0B = ``max(QK, PV)`` instead of the
/// sum).  The K-loop has the shape:
///
///   * ``tile.matmul`` — the loop body branches on the iteration index
///     (``ko == 0``) so the first iteration uses ``tile.matmul`` (fresh
///     accumulator) and subsequent iterations use ``tile.matmul_acc``
///     (accumulating into the iter-arg).  The iter-arg init is an Acc-
///     resident ``tile.create`` placeholder so the iter-arg / yield /
///     return_var chain is Acc-typed end-to-end.
///   * ``tile.matmul_acc`` — every iteration is ``tile.matmul_acc``; the
///     iter-arg init is the caller-provided accumulator directly, so the
///     chain is uniform and no if-else is needed.
///
/// The K-loop is marked ``ForKind::Pipeline`` with ``pipeline_stages=2`` so
/// the downstream ``LowerPipelineLoops`` pass produces a 2-deep ping-pong
/// on the per-iter Mat→Left/Right extracts.
///
/// Operand extraction uses ``tile.extract(src, idx_row, idx_col, shape,
/// target_memory=Left|Right)`` directly — the SSA-form fusion of the older
/// ``tile.slice`` (Mat-resident result) + ``tile.mov`` (Mat→Left/Right) pair.
/// This (a) eliminates the intermediate Mat-resident slice tiles and their
/// MemRef allocations, and (b) lowers to ``pto.textract`` rather than
/// ``pto.subview``, sidestepping the latter's ``valid_row`` codegen
/// mismatch.
///
/// Layout for ``tile.matmul``:
///   c_init = tile.create([m, n], dtype, target_memory=Acc)  // placeholder
///   for ko in pl.pipeline(0, K, k, init_values=(c_init,), stage=2):
///     sa = tile.extract(x_mat, 0, ko, [m, k], target_memory=Left)
///     sb = tile.extract(y_mat, ko, 0, [k, n], target_memory=Right)
///     if ko == 0:
///       c1 = tile.matmul(sa, sb)             // fresh Acc
///       c_phi = pl.yield_(c1)                // if's return_var
///     else:
///       c2 = tile.matmul_acc(c_iter, sa, sb) // accumulate
///       c_phi = pl.yield_(c2)
///     yield c_phi
///
/// Layout for ``tile.matmul_acc`` (acc_init is the caller's accumulator):
///   for ko in pl.pipeline(0, K, k, init_values=(acc_init,), stage=2):
///     sa = tile.extract(x_mat, 0, ko, [m, k], target_memory=Left)
///     sb = tile.extract(y_mat, ko, 0, [k, n], target_memory=Right)
///     c_new = tile.matmul_acc(c_iter, sa, sb)
///     yield c_new
///
/// A fresh return_var typed identically to the iter-arg replaces the original
/// matmul's Var; uses of the original Var in the enclosing SeqStmts are
/// substituted by the mutator.
///
/// M/N tiling (output exceeds L0c)
/// -------------------------------
/// When ``ChooseL0Tile`` returns ``m < M`` or ``n < N`` the ``[M, N]`` output
/// Acc's physical footprint overflows L0c.  Capacity uses the backend's
/// accumulator-row alignment, which may be stricter than the logical M shape
/// (Ascend910B INT32 M=16 occupies 32 physical rows). For a fresh
/// ``tile.matmul`` / ``tile.matmul_bias``, the pass emits a
/// ``ceil(M/m) x ceil(N/n)`` grid and hands each ``[m_eff, n_eff]`` Acc result to
/// one or more placement strategies: direct-store to a 2D
/// ``tile.store(c, base, out)`` consumer, assembly into an on-chip Mat scratch
/// when every on-chip use is a later matmul operand, or both for a stored-and-
/// reused value. A Vec-resident left operand is staged into Mat once before the
/// grid. Each sub-tile uses the pipelined
/// K-loop above when K spans >= 2 L0 blocks, or — when ``k == K`` (the full K
/// fits L0a/L0b at once) — a single straight-line ``tile.matmul`` emitted inside
/// **nested pipelined loops** over the divisible interior so
/// ``LowerPipelineLoops`` double-buffers the operand extracts (see
/// ``BuildFullKPipelined``). The partial boundary is peeled into a
/// straight-line tail, so ``m`` / ``n`` need not divide ``M`` / ``N``.
///
/// Supported today:
///   * ``tile.matmul``, ``tile.matmul_acc``, and ``tile.matmul_bias``. Bias is
///     sliced along N and applied exactly once on the first K block of each
///     output sub-tile.
///   * K tiling (``m == M and n == N``) for ``tile.matmul`` and
///     ``tile.matmul_acc``; M/N tiling for fresh ``tile.matmul`` /
///     ``tile.matmul_bias`` with direct-store, Mat-scratch, or composite
///     placement, with a pipelined K-loop or a straight-line single-K-block
///     (``k == K``) per sub-tile.
///     The canonical frontend split-K create/pipeline/store form is M/N-tiled
///     outside its K loop: each output sub-tile completes the whole source K
///     reduction before the next one starts. A linear fresh-matmul ->
///     ``tile.matmul_acc`` chain is likewise rewritten at chain scope.
///     Arbitrary standalone
///     ``tile.matmul_acc`` (which would need slices of a caller-owned
///     accumulator) and mixed/non-matmul on-chip consumers are deferred with a
///     ``PerfHint``.
///   * A compatible f32-to-bf16/f16 ``tile.cast(mode="rint")`` feeding only
///     matmul operands is folded into the Acc-to-Mat FIXPIPE writeback.
///   * Any 16-aligned K.  When the chosen ``k`` does not divide ``K``
///     (``allow_k_boundary``) the K-loop peels a partial last block of width
///     ``K - (K/k)*k``; with K and k both 16-aligned the tail is itself 16-aligned
///     -- an ordinary matmul_acc block (ptoas requires 16-aligned tile cols).
///
/// When the chooser returns ``(M, N, K)``, the matmul itself needs no tiling
/// rewrite.  Its result may still participate in the fits-L0c chained
/// cast-fold, which remaps the compatible downcast to a full-window Mat
/// scratch.

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kPassName = "AutoTileMatmulL0";

ExprPtr MakeIndex(int64_t v, const Span& span) {
  return std::make_shared<ConstInt>(v, DataType::INDEX, span);
}

ExprPtr MakeIndexTuple(const std::vector<int64_t>& values, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(values.size());
  for (auto v : values) elements.push_back(MakeIndex(v, span));
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

int64_t AlignStaticExtent(int64_t extent, int64_t alignment, const Span& span) {
  INTERNAL_CHECK_SPAN(extent > 0 && alignment > 0, span)
      << "Internal error: tile extent/alignment must be positive, got " << extent << "/" << alignment;
  const int64_t remainder = extent % alignment;
  const int64_t increment = remainder == 0 ? 0 : alignment - remainder;
  INTERNAL_CHECK_SPAN(extent <= std::numeric_limits<int64_t>::max() - increment, span)
      << "Internal error: box-aligning tile extent " << extent << " by " << alignment << " overflows int64";
  return extent + increment;
}

/// True if `tile`'s 2D shape is static and its memory space is one of
/// `allowed`.  Operand-source residency check for the L0 tiling rewrite:
///
///   * The right (B) operand must be ``Mat`` — it is loaded from DDR into L1
///     and fed into L0B.
///   * The left (A) operand may be ``Mat`` (the QK pattern) *or* ``Vec`` (the
///     fused-attention ``score·V`` / PV pattern, where the softmax/``exp``
///     output crosses the cube↔vector boundary resident in ``Vec`` rather
///     than ``Mat``).
///
/// This is purely a residency/static-shape check.  A ``Vec`` left operand is
/// not extracted directly: ``BuildKLoopRewrite`` stages it into ``Mat`` first
/// via ``BuildMoveToMat``, so the per-iter ``tile.extract`` always slices from
/// a ``Mat`` source regardless of the original operand space.
bool IsStatic2DInSpaces(const TileTypePtr& tile, std::initializer_list<MemorySpace> allowed, int64_t& out_d0,
                        int64_t& out_d1) {
  if (!tile || tile->shape_.size() != 2) return false;
  auto mem = tile->GetMemorySpace();
  if (!mem.has_value()) return false;
  bool space_ok = false;
  for (auto space : allowed) {
    if (*mem == space) {
      space_ok = true;
      break;
    }
  }
  if (!space_ok) return false;
  auto a = As<ConstInt>(tile->shape_[0]);
  auto b = As<ConstInt>(tile->shape_[1]);
  if (!a || !b) return false;
  out_d0 = a->value_;
  out_d1 = b->value_;
  return true;
}

/// Element width in bytes for a tile dtype.  Returns 0 for sub-byte types
/// (INT4, FP4 et al.) which the cube path does not support; the caller emits
/// a ``PerfHint`` and skips in that case.
uint32_t DTypeBytes(const DataType& dt) {
  size_t bits = dt.GetBit();
  if (bits % 8 != 0) return 0;
  return static_cast<uint32_t>(bits / 8);
}

/// Build a ``tile.extract(source, idx_row, idx_col, [shape],
/// target_memory=target)`` AssignStmt — the Mat→Left/Right SSA-form
/// extract used inside the K-loop.  Offsets are passed as separate scalar
/// exprs (typically a ConstInt 0 for the static axis and the loop var
/// ``ko`` for the K axis).  The result tile is already in the destination
/// memory space, so no follow-up ``tile.mov`` is needed.  The source is
/// always Mat-resident — a Vec-fed left operand is first staged into Mat by
/// ``BuildMoveToMat`` (see ``BuildKLoopRewrite``).
AssignStmtPtr BuildExtract(const VarPtr& source, const std::vector<int64_t>& shape, const ExprPtr& index_row,
                           const ExprPtr& index_col, MemorySpace target, const std::string& name_hint,
                           const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<ExprPtr> args = {source, index_row, index_col, MakeIndexTuple(shape, span)};
  std::vector<std::pair<std::string, std::any>> kwargs = {{"target_memory", target}};
  auto call = reg.Create("tile.extract", args, kwargs, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

/// Build a ``tile.move(source, target_memory=Mat)`` AssignStmt that stages a
/// Vec-resident left operand into Mat (L1) *before* the K-loop, so the per-iter
/// ``tile.extract`` slices from Mat exactly like the QK (Mat-fed) path.
///
/// This matters for fused cube+vector roots (fused-attention PV / ``score·V``):
/// the softmax/``exp`` output reaches the matmul resident in ``Vec`` at the
/// cube↔vector boundary.  Keeping the boundary crossing a ``tile.move`` lets
/// ``ExpandMixedKernel`` recognise it (``CollectCVBoundaryMoves`` only matches
/// ``tile.move``) and lower it to the cross-core ``tpop_from_aiv`` handshake
/// (which lands the data in Mat — ``GetBoundaryTpopMemory(AIC) == Mat``).
/// Extracting straight from the Vec tile instead would leave the operand a
/// dangling cross-boundary free variable on the cube side.
AssignStmtPtr BuildMoveToMat(const VarPtr& source, const std::string& name_hint, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs = {{"target_memory", MemorySpace::Mat}};
  auto call = reg.Create("tile.move", {source}, kwargs, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

/// Build the ``tile.create([m, n], dtype, target_memory=Acc)`` placeholder
/// that initializes the iter-arg.  Acc keeps the iter-arg / yield / return_var
/// chain structurally consistent with the per-iter ``tile.matmul[_acc]``
/// outputs, so subsequent matmul_acc consumers (and any nested for-loops
/// initialised from this return_var) still see an Acc-typed accumulator and
/// can be tiled in turn.  ``tile.create``'s deduce_type honors ``Acc`` and
/// emits the Nz TileView ``(col_major, row_major, fractal=1024)`` that
/// matches matmul output, so iter_arg/yield TileViews line up.
AssignStmtPtr BuildAccInit(int64_t m, int64_t n, const DataType& dtype, const std::string& name_hint,
                           const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", dtype},
                                                          {"target_memory", MemorySpace::Acc}};
  auto call = reg.Create("tile.create", {MakeIndexTuple({m, n}, span)}, kwargs, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

struct AccInitValue {
  std::vector<StmtPtr> stmts;
  VarPtr value;
};

/// Build an Acc placeholder whose allocation may be box-padded while its valid
/// rectangle remains the logical output tile. The ordinary unpadded case keeps
/// the historical single ``tile.create`` form byte-for-byte.
AccInitValue BuildAccInitWithValidShape(int64_t physical_m, int64_t physical_n, ExprPtr valid_m,
                                        ExprPtr valid_n, const DataType& dtype, const std::string& name_hint,
                                        const Span& span) {
  INTERNAL_CHECK_SPAN(valid_m && valid_n, span)
      << "Internal error: accumulator initializer requires two valid extents";
  auto valid_m_const = As<ConstInt>(valid_m);
  auto valid_n_const = As<ConstInt>(valid_n);
  if (valid_m_const && valid_n_const && physical_m == valid_m_const->value_ &&
      physical_n == valid_n_const->value_) {
    auto init = BuildAccInit(physical_m, physical_n, dtype, name_hint, span);
    return AccInitValue{{init}, init->var_};
  }

  auto storage = BuildAccInit(physical_m, physical_n, dtype, name_hint + "_storage", span);
  auto& reg = OpRegistry::GetInstance();
  auto narrowed_call =
      reg.Create("tile.set_validshape", {storage->var_, std::move(valid_m), std::move(valid_n)}, span);
  auto narrowed_var = std::make_shared<Var>(name_hint, narrowed_call->GetType(), span);
  auto narrowed = std::make_shared<AssignStmt>(narrowed_var, narrowed_call, span);
  return AccInitValue{{storage, narrowed}, narrowed_var};
}

struct KLoopRewrite {
  AssignStmtPtr original;
  VarPtr lhs_src;                 ///< [M, K] left operand — Mat- or Vec-resident
  VarPtr rhs_src;                 ///< [K, N] right operand — Mat-resident
  VarPtr bias_src;                ///< optional [1, N] bias — Mat- or Bias-resident
  bool stage_lhs_to_mat = false;  ///< lhs is Vec-resident: stage Vec→Mat before the K-loop
  VarPtr acc_init = nullptr;      ///< Caller-provided accumulator for matmul_acc;
                                  ///< nullptr for plain matmul (Vec placeholder is built instead).
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  /// Static K origin within the full source operands. Ordinary rewrites use
  /// zero. A linear chain's fresh root may emit its first block explicitly,
  /// then run a uniform matmul_acc loop over the remaining K range.
  int64_t k_base = 0;
  /// Logical output window carried by the loop accumulator. These may be
  /// smaller than m/n when a boxed boundary operand is physically padded.
  ExprPtr valid_m = nullptr;
  ExprPtr valid_n = nullptr;
  /// Output sub-tile origin (row, col) within the [M, N] product. The per-iter
  /// extracts slice ``lhs[mi : mi + m, ko : ko + k]`` and ``rhs[ko : ko + k,
  /// ni : ni + n]``. Null means 0 — the K-only path (m == M, n == N) leaves
  /// these null so the emitted IR is identical to the un-tiled output case.
  ExprPtr mi = nullptr;
  ExprPtr ni = nullptr;
  /// Var-name prefix for the loop's locals. Empty means use the original
  /// matmul's name hint. M/N tiling sets a per-sub-tile prefix so unrolled
  /// sub-tiles get distinct names (the print/parse round-trip needs unique
  /// names within a scope).
  std::string name_base;
};

struct RewriteResult {
  std::vector<StmtPtr> stmts;  ///< [Optional init,] ForStmt replacing the original AssignStmt.
  VarPtr return_var;           ///< ForStmt's return_var; substituted into downstream uses.
};

/// Body of the K-loop for plain ``tile.matmul``: branches on ``ko == 0``
/// between ``tile.matmul`` (fresh Acc) and ``tile.matmul_acc`` (accumulating).
/// The ``IfStmt`` materializes a phi return_var that the outer yield carries
/// back to the iter-arg.
StmtPtr BuildMatmulBody(const VarPtr& ko_var, const IterArgPtr& c_iter, const AssignStmtPtr& sa,
                        const AssignStmtPtr& sb, const VarPtr& bias, const std::string& base,
                        const Span& sp) {
  auto& reg = OpRegistry::GetInstance();

  // Then-branch: fresh Acc tile from tile.matmul.
  auto c_then_call = bias ? reg.Create("tile.matmul_bias", {sa->var_, sb->var_, bias}, sp)
                          : reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
  auto c_then_var = std::make_shared<Var>(base + "_l0_c_first", c_then_call->GetType(), sp);
  auto c_then_assign = std::make_shared<AssignStmt>(c_then_var, c_then_call, sp);
  auto then_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_then_var}, sp);
  StmtPtr then_body = SeqStmts::Flatten(std::vector<StmtPtr>{c_then_assign, then_yield}, sp);

  // Else-branch: accumulate into the iter-arg.
  auto c_else_call = reg.Create("tile.matmul_acc", {ExprPtr(c_iter), sa->var_, sb->var_}, sp);
  auto c_else_var = std::make_shared<Var>(base + "_l0_c_acc", c_else_call->GetType(), sp);
  auto c_else_assign = std::make_shared<AssignStmt>(c_else_var, c_else_call, sp);
  auto else_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_else_var}, sp);
  StmtPtr else_body = SeqStmts::Flatten(std::vector<StmtPtr>{c_else_assign, else_yield}, sp);

  auto c_phi = std::make_shared<Var>(base + "_l0_c_phi", c_then_call->GetType(), sp);
  auto cond = MakeEq(ko_var, MakeIndex(0, sp), sp);
  auto if_stmt = std::make_shared<IfStmt>(cond, then_body, std::optional<StmtPtr>(else_body),
                                          std::vector<VarPtr>{c_phi}, sp);
  auto outer_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_phi}, sp);
  return SeqStmts::Flatten(std::vector<StmtPtr>{sa, sb, if_stmt, outer_yield}, sp);
}

/// Body of the K-loop for ``tile.matmul_acc``: every iteration accumulates
/// into ``c_iter`` via ``tile.matmul_acc``.  The first iteration's ``c_iter``
/// is the caller-supplied ``acc_init`` (threaded through ``init_values``), so
/// no if-else is needed — the accumulator chain is uniform.
StmtPtr BuildMatmulAccBody(const IterArgPtr& c_iter, const AssignStmtPtr& sa, const AssignStmtPtr& sb,
                           const std::string& base, const Span& sp) {
  auto& reg = OpRegistry::GetInstance();
  auto c_call = reg.Create("tile.matmul_acc", {ExprPtr(c_iter), sa->var_, sb->var_}, sp);
  auto c_var = std::make_shared<Var>(base + "_l0_c_acc", c_call->GetType(), sp);
  auto c_assign = std::make_shared<AssignStmt>(c_var, c_call, sp);
  auto outer_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_var}, sp);
  return SeqStmts::Flatten(std::vector<StmtPtr>{sa, sb, c_assign, outer_yield}, sp);
}

/// Build the replacement statements for one Mat-resident matmul or matmul_acc.
/// See the file-level comment for the emitted shape.
RewriteResult BuildKLoopRewrite(const KLoopRewrite& r) {
  const Span sp = r.original->span_;
  const std::string base = r.name_base.empty() ? r.original->var_->name_hint_ : r.name_base;
  const bool is_acc = r.acc_init != nullptr;
  auto& reg = OpRegistry::GetInstance();

  INTERNAL_CHECK_SPAN(r.k < r.K, sp) << "Internal error: BuildKLoopRewrite expects a tiled K (k < K), got k="
                                     << r.k << ", K=" << r.K;

  // K-block decomposition.  With allow_k_boundary the chosen k need not divide
  // K: the reduction is `num_full` full blocks of width k plus a final partial
  // block of width `k_eff = K - num_full*k` (k_eff == 0 when k divides K — the
  // legacy uniform loop).  K and k are 16-aligned, so k_eff is itself 16-aligned —
  // an ordinary matmul_acc block (ptoas requires 16-aligned tile cols).
  const int64_t num_full = r.K / r.k;
  const int64_t k_full = num_full * r.k;
  const int64_t k_eff = r.K - k_full;
  const bool has_tail = k_eff > 0;

  std::vector<StmtPtr> out;

  // Iter-arg init for the pipelined K-loop (only when there are >= 2 full
  // blocks).  Emitted FIRST — before the optional Vec->Mat staging below — so a
  // k that divides K yields byte-identical output to the pre-peel emitter.  For
  // matmul_acc the caller's accumulator is the init directly; for plain matmul a
  // fresh Acc-resident ``tile.create`` placeholder that the first iteration's
  // ``tile.matmul`` overwrites (the Nz TileView matches, so iter_arg / yield /
  // return_var stay Acc-typed).
  ExprPtr loop_init;
  TypePtr loop_iter_type;
  if (num_full >= 2) {
    if (is_acc) {
      loop_init = r.acc_init;
      loop_iter_type = r.acc_init->GetType();
    } else {
      auto acc_dtype = As<TileType>(r.original->var_->GetType())->dtype_;
      auto c_init =
          BuildAccInitWithValidShape(r.m, r.n, r.valid_m, r.valid_n, acc_dtype, base + "_l0_init", sp);
      for (auto& stmt : c_init.stmts) out.push_back(std::move(stmt));
      loop_init = c_init.value;
      loop_iter_type = c_init.value->GetType();
    }
  }

  // A Vec-resident left operand (fused-attention PV / ``score·V``) is staged into
  // Mat once, before any K block, so each extract slices from Mat exactly like
  // the QK path — and so ``ExpandMixedKernel`` can lower the Vec→Mat crossing via
  // its ``tile.move`` handshake (``CollectCVBoundaryMoves`` only matches
  // ``tile.move``).  Mat-resident left operands extract directly.
  VarPtr lhs_extract_src = r.lhs_src;
  if (r.stage_lhs_to_mat) {
    auto lhs_mat = BuildMoveToMat(r.lhs_src, base + "_l0_lmat", sp);
    out.push_back(lhs_mat);
    lhs_extract_src = lhs_mat->var_;
  }

  ExprPtr mi_off = r.mi ? r.mi : MakeIndex(0, sp);
  ExprPtr ni_off = r.ni ? r.ni : MakeIndex(0, sp);

  // Bias is applied exactly once, on the fresh first K block. A full Bias-
  // resident vector can be reused directly; an N sub-window (or a Mat-resident
  // source awaiting memory inference) is extracted into the architectural Bias
  // buffer once outside the K loop.
  VarPtr bias_operand;
  if (r.bias_src) {
    auto bias_ty = As<TileType>(r.bias_src->GetType());
    const bool already_full_bias = bias_ty && bias_ty->GetMemorySpace() == MemorySpace::Bias && r.n == r.N &&
                                   (!r.ni || (As<ConstInt>(r.ni) && As<ConstInt>(r.ni)->value_ == 0));
    if (already_full_bias) {
      bias_operand = r.bias_src;
    } else {
      auto bias = BuildExtract(r.bias_src, {1, r.n}, MakeIndex(0, sp), ni_off, MemorySpace::Bias,
                               base + "_l0_bias", sp);
      out.push_back(bias);
      bias_operand = bias->var_;
    }
  }

  // Emit one straight-line K block ``[m, kb] x [kb, n]`` at static K-offset
  // ``ko``, accumulating into ``acc_in`` (``tile.matmul_acc``) or starting fresh
  // (``tile.matmul`` when ``acc_in`` is null).  Used for the single-full-block
  // (num_full == 1) and partial-tail cases; the multi-block case pipelines below.
  auto emit_block = [&](int64_t ko, int64_t kb, const ExprPtr& acc_in, const std::string& tag) -> VarPtr {
    auto sa = BuildExtract(lhs_extract_src, {r.m, kb}, mi_off, MakeIndex(r.k_base + ko, sp),
                           MemorySpace::Left, base + "_l0_a" + tag, sp);
    auto sb = BuildExtract(r.rhs_src, {kb, r.n}, MakeIndex(r.k_base + ko, sp), ni_off, MemorySpace::Right,
                           base + "_l0_b" + tag, sp);
    ExprPtr call;
    if (acc_in) {
      call = reg.Create("tile.matmul_acc", {acc_in, sa->var_, sb->var_}, sp);
    } else if (bias_operand) {
      call = reg.Create("tile.matmul_bias", {sa->var_, sb->var_, bias_operand}, sp);
    } else {
      call = reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
    }
    auto cvar = std::make_shared<Var>(base + "_l0_c" + tag, call->GetType(), sp);
    out.push_back(sa);
    out.push_back(sb);
    out.push_back(std::make_shared<AssignStmt>(cvar, call, sp));
    return cvar;
  };

  // --- Full blocks: a pipelined K-loop when there are >= 2 of them, else a
  //     single straight-line block (a 1-trip pipeline loop would be degenerate).
  VarPtr main_var;
  if (num_full >= 2) {
    auto ko_var = std::make_shared<Var>(base + "_l0_ko", std::make_shared<ScalarType>(DataType::INDEX), sp);
    auto c_iter = std::make_shared<IterArg>(base + "_l0_c", loop_iter_type, loop_init, sp);
    ExprPtr ko_offset = r.k_base == 0 ? ExprPtr(ko_var) : MakeAdd(ko_var, MakeIndex(r.k_base, sp), sp);
    auto sa =
        BuildExtract(lhs_extract_src, {r.m, r.k}, mi_off, ko_offset, MemorySpace::Left, base + "_l0_a", sp);
    auto sb = BuildExtract(r.rhs_src, {r.k, r.n}, ko_offset, ni_off, MemorySpace::Right, base + "_l0_b", sp);
    StmtPtr body = is_acc ? BuildMatmulAccBody(c_iter, sa, sb, base, sp)
                          : BuildMatmulBody(ko_var, c_iter, sa, sb, bias_operand, base, sp);
    std::vector<std::pair<std::string, std::any>> attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2}};
    // Loop return var: an intermediate when a partial tail follows (named
    // distinctly so round-trip names stay unique), else the final result.
    auto rv =
        std::make_shared<Var>(has_tail ? base + "_l0_kmain" : base, loop_iter_type, r.original->var_->span_);
    auto for_stmt = std::make_shared<ForStmt>(
        ko_var, MakeIndex(0, sp), MakeIndex(k_full, sp), MakeIndex(r.k, sp), std::vector<IterArgPtr>{c_iter},
        body, std::vector<VarPtr>{rv}, sp, ForKind::Pipeline, std::move(attrs));
    out.push_back(for_stmt);
    main_var = rv;
  } else {
    // num_full == 1: a single straight-line full block (k < K checked above, so a
    // partial tail always follows).  This is a correctness guard, not a hot path:
    // the roofline chooser never selects k in (K/2, K) -- a near-full k is wall-
    // dominated (2x the MAD ceil-step of a divisor, and it loses the min-padding
    // tie-break), so num_full is >= 2 in practice.  Kept so the emitter stays
    // correct (no degenerate 1-trip pipeline) if the cost model ever changes.
    main_var = emit_block(/*ko=*/0, /*kb=*/r.k, is_acc ? ExprPtr(r.acc_init) : nullptr, "0");
  }

  // --- Partial tail: matmul_acc the [m, k_eff] x [k_eff, n] block onto the
  //     full-blocks accumulator (no-op when k divides K). ---
  VarPtr result_var = has_tail ? emit_block(k_full, k_eff, ExprPtr(main_var), "t") : main_var;
  return RewriteResult{std::move(out), result_var};
}

/// Operands + chosen L0 tile shape for a tileable matmul.  Produced by
/// ``AnalyzeMatmul``; the caller dispatches on ``needs_mn_tiling()`` to build
/// either the whole-output K-loop or the unrolled M/N grid of sub-tiles.
struct MatmulTiling {
  AssignStmtPtr assign;
  VarPtr lhs;       ///< [M, K] left operand — Mat (or Vec for the PV pattern; see stage_lhs_to_mat)
  VarPtr rhs;       ///< [K, N] right operand — Mat
  VarPtr bias;      ///< optional [1, N] bias for tile.matmul_bias
  VarPtr acc_init;  ///< caller-provided accumulator for matmul_acc; null for plain matmul
  bool stage_lhs_to_mat = false;
  int64_t M = 0, N = 0, K = 0;
  int64_t m = 0, n = 0, k = 0;
  /// Logical output extents. Ordinarily identical to M/N; a box-padded
  /// boundary matmul keeps the larger physical M/N in the operands/result.
  ExprPtr output_valid_m = nullptr;
  ExprPtr output_valid_n = nullptr;
  /// Chosen L0 stationarity (which operand is pinned single-buffered across the
  /// moving grid). Output-stationary unless the chooser picked A/B-stationary
  /// (k == K); BuildFullKPipelined sets the loop order + single-buffers the
  /// stationary (outer) operand accordingly.
  utils::Stationarity stationarity = utils::Stationarity::kOutputStationary;
  /// For full-K output-stationary (BuildFullKPipelined at k == K): which operand
  /// the chooser hoists to the outer loop — true = hold A (rows outer), false =
  /// hold B (cols outer). Set from L0TileResult::os_holds_a so the emitted hoist
  /// matches the bandwidth-weighted hoist the wall cost was scored under. Ignored
  /// for A/B-stationary (loop order comes from `stationarity`) and split-K.
  bool os_holds_a = true;
  /// True when the chooser picked dbC=2 (double-buffered L0C): the accumulator is
  /// budgeted at L0C/2 so two co-live [m, n] Acc tiles fit, and BuildFullKPipelined
  /// tags the moving loop with kPipelineDoubleBufferCAttr so CanonicalizeIOOrder
  /// floats the drains past the next matmul, keeping the two tiles co-live.  With
  /// Under PTOAS, InitMemRef keeps the overlapping-live-range buffers distinct
  /// and ptoas places them. Under the PyPTO opt-in, the flat depth-2 pipeline
  /// membership prevents MemoryReuse from coalescing them. Set from
  /// L0TileResult::double_buffer_c; only true for full-K tiles (see the assert
  /// in AnalyzeMatmul).
  bool double_buffer_c = false;
  [[nodiscard]] bool is_acc() const { return acc_init != nullptr; }
  /// True when the chosen L0 tile is smaller than the [M, N] output on either
  /// axis — the output Acc would overflow L0c, so the output must be tiled.
  [[nodiscard]] bool needs_mn_tiling() const { return m != M || n != N; }
};

/// Build the K-loop descriptor for one output sub-tile ``[mi : mi + m_eff,
/// ni : ni + n_eff]``.  Passing ``mi == ni == nullptr`` with ``m_eff == M`` and
/// ``n_eff == N`` yields the whole-output (K-only) case unchanged.
KLoopRewrite MakeKLoop(const MatmulTiling& t, ExprPtr mi, ExprPtr ni, int64_t m_eff, int64_t n_eff,
                       std::string name_base) {
  KLoopRewrite r;
  r.original = t.assign;
  r.lhs_src = t.lhs;
  r.rhs_src = t.rhs;
  r.bias_src = t.bias;
  r.stage_lhs_to_mat = t.stage_lhs_to_mat;
  r.acc_init = t.acc_init;
  r.M = t.M;
  r.N = t.N;
  r.K = t.K;
  r.m = m_eff;
  r.n = n_eff;
  r.k = t.k;
  auto local_valid_extent = [&](const ExprPtr& output_valid, const ExprPtr& offset,
                                int64_t physical_extent) -> ExprPtr {
    INTERNAL_CHECK_SPAN(output_valid, t.assign->span_)
        << "Internal error: matmul output requires a logical valid extent";
    if (!offset) return output_valid;

    // Constant output grids are by far the common case. Fold them here so the
    // generated set_validshape operands stay simple constants.
    if (auto valid_const = As<ConstInt>(output_valid)) {
      if (auto offset_const = As<ConstInt>(offset)) {
        return MakeIndex(std::clamp<int64_t>(valid_const->value_ - offset_const->value_, 0, physical_extent),
                         t.assign->span_);
      }
    }

    // Keep the zero-offset form canonical with InferWindowReadValidShape. This
    // also avoids asking the verifier to prove `max(valid, 0) - 0 == valid`.
    if (auto offset_const = As<ConstInt>(offset); offset_const && offset_const->value_ == 0) {
      return MakeMin(output_valid, MakeIndex(physical_extent, t.assign->span_), t.assign->span_);
    }

    // The logical window of a physical sub-tile is
    // clamp(output_valid - offset, 0, physical_extent). Spell the lower clamp
    // as max(output_valid, offset) - offset so UINT64 valid extents cannot
    // underflow when the sub-tile starts beyond the logical boundary.
    auto clamped_valid = MakeMax(output_valid, offset, t.assign->span_);
    auto remaining = MakeSub(clamped_valid, offset, t.assign->span_);
    return MakeMin(remaining, MakeIndex(physical_extent, t.assign->span_), t.assign->span_);
  };
  r.valid_m = local_valid_extent(t.output_valid_m, mi, m_eff);
  r.valid_n = local_valid_extent(t.output_valid_n, ni, n_eff);
  r.mi = std::move(mi);
  r.ni = std::move(ni);
  r.name_base = std::move(name_base);
  return r;
}

/// Decide whether `assign` is a Mat-resident matmul we know how to tile, and if
/// so which L0 tile shape to use.  Returns the tiling plan on success;
/// otherwise nullopt and (when useful) appends a PerfHint.  The caller
/// dispatches K-only vs M/N tiling on ``MatmulTiling::needs_mn_tiling()``.
std::optional<MatmulTiling> AnalyzeMatmul(
    const AssignStmtPtr& assign, std::vector<Diagnostic>& hints, bool force_output_stationary = false,
    std::optional<tile_view_semantics::BoxedTileAlignment> output_box_alignment = std::nullopt) {
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_) return std::nullopt;

  // Plain, accumulating, and bias matmuls share the same L0 tile chooser. Bias
  // is applied on the first K block only, then later blocks use matmul_acc.
  const bool is_matmul = IsOp(call, "tile.matmul");
  const bool is_matmul_acc = IsOp(call, "tile.matmul_acc");
  const bool is_matmul_bias = IsOp(call, "tile.matmul_bias");
  if (!is_matmul && !is_matmul_acc && !is_matmul_bias) return std::nullopt;

  // Operand layout: (lhs, rhs) for matmul; (acc, lhs, rhs) for matmul_acc.
  // Use ``AsVarLike`` for the operands so IterArg (Var subclass) is accepted —
  // this is the common case for the accumulator inside a pipelined K-loop.
  const size_t expected_arity = is_matmul ? 2u : 3u;
  if (call->args_.size() != expected_arity) return std::nullopt;
  const size_t lhs_idx = is_matmul_acc ? 1u : 0u;
  auto lhs = AsVarLike(call->args_[lhs_idx]);
  auto rhs = AsVarLike(call->args_[lhs_idx + 1u]);
  if (!lhs || !rhs) return std::nullopt;
  auto lhs_tile = As<TileType>(lhs->GetType());
  auto rhs_tile = As<TileType>(rhs->GetType());
  if (!lhs_tile || !rhs_tile) return std::nullopt;

  // For matmul_acc, ensure the caller's accumulator is a Var/IterArg with a
  // 2D TileType.  We accept both Acc- and Vec-typed accumulators: Vec is
  // common when the user pre-allocated the running accumulator with
  // ``pl.create_tensor`` / ``tile.create(target=Vec)`` and lets downstream
  // passes (``InferTileMemorySpace``) bridge to Acc.  We thread the
  // accumulator through the inner K-loop's iter-arg in either case.
  VarPtr acc_var;
  if (is_matmul_acc) {
    acc_var = AsVarLike(call->args_[0]);
    if (!acc_var) return std::nullopt;
    auto acc_tile = As<TileType>(acc_var->GetType());
    if (!acc_tile || acc_tile->shape_.size() != 2) return std::nullopt;
  }

  VarPtr bias_var;
  int64_t bias_n = 0;
  if (is_matmul_bias) {
    bias_var = AsVarLike(call->args_[2]);
    if (!bias_var) return std::nullopt;
    auto bias_tile = As<TileType>(bias_var->GetType());
    int64_t bias_m = 0;
    if (!IsStatic2DInSpaces(bias_tile, {MemorySpace::Mat, MemorySpace::Bias}, bias_m, bias_n) ||
        bias_m != 1) {
      return std::nullopt;
    }
  }

  // Operand source residency, with static 2D shapes.  The right (B) operand
  // must be Mat — it is loaded from DDR into L1 and fed into L0B.  The left (A)
  // operand may be Mat (the QK pattern) or Vec (the fused-attention PV /
  // ``score·V`` pattern, where the softmax/``exp`` output crosses the
  // cube↔vector boundary resident in Vec).  Other cases (Acc operands, a Vec
  // right operand, dynamic shapes) are out of scope; return silently.
  int64_t M = 0, K_lhs = 0, K_rhs = 0, N = 0;
  if (!IsStatic2DInSpaces(lhs_tile, {MemorySpace::Mat, MemorySpace::Vec}, M, K_lhs) ||
      !IsStatic2DInSpaces(rhs_tile, {MemorySpace::Mat}, K_rhs, N)) {
    return std::nullopt;
  }
  // K mismatch is an ill-typed matmul — the op verifier should have caught it
  // upstream.  Treat as an internal invariant.
  INTERNAL_CHECK(K_lhs == K_rhs) << "tile.matmul: K dimensions don't match (lhs K=" << K_lhs
                                 << ", rhs K=" << K_rhs << ")";
  INTERNAL_CHECK_SPAN(!is_matmul_bias || bias_n == N, assign->span_)
      << "Internal error: tile.matmul_bias bias N does not match rhs N";
  const int64_t K = K_lhs;

  uint32_t bytes_a = DTypeBytes(lhs_tile->dtype_);
  uint32_t bytes_b = DTypeBytes(rhs_tile->dtype_);
  // Output dtype is set by the matmul op's deduction (FP32 / INT32 today, but
  // future cube paths may add half-precision accumulation).  Read from the
  // call's result type rather than hardcoding so the chooser sees the actual
  // accumulator footprint.
  auto out_tile = As<TileType>(call->GetType());
  INTERNAL_CHECK(out_tile) << "Internal error: tile.matmul result is not a TileType";
  const auto output_valid_shape = tile_view_semantics::GetEffectiveTileView(*out_tile).valid_shape;
  INTERNAL_CHECK_SPAN(output_valid_shape.size() == 2, assign->span_)
      << "Internal error: tile.matmul result valid_shape must be 2D";
  uint32_t bytes_c = DTypeBytes(out_tile->dtype_);
  if (bytes_a == 0 || bytes_b == 0 || bytes_c == 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-003",
                       "tile.matmul: unsupported sub-byte dtype on operand or accumulator — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  // Prefer the active PassContext's BackendHandler (the production path runs
  // under PassPipeline::Run, which establishes a context).  Fall back to the
  // global default backend so direct callers — e.g. tests that call
  // PassManager strategies' run_passes() without wrapping in a PassContext —
  // still work; this mirrors the env-var fallback documented in
  // .claude/rules/pass-context-config.md.
  const auto* ctx = PassContext::Current();
  const auto* handler = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  INTERNAL_CHECK(handler) << "Internal error: BackendHandler is null";

  utils::L0TileConfig cfg;
  cfg.M = static_cast<int>(M);
  cfg.N = static_cast<int>(N);
  cfg.K = static_cast<int>(K);
  cfg.l0a_bytes = handler->GetL0aCapacityBytes();
  cfg.l0b_bytes = handler->GetL0bCapacityBytes();
  cfg.l0c_bytes = handler->GetL0cCapacityBytes();
  const auto cost_model = handler->GetL0CostModel();
  cfg.bw_a = cost_model.bw_l0a;
  cfg.bw_b = cost_model.bw_l0b;
  cfg.bw_drain = cost_model.bw_drain;
  cfg.drain_fixed_cycles = cost_model.drain_fixed_cycles;
  cfg.drain_row_cycles = cost_model.drain_row_cycles;
  cfg.drain_penalty_cycles = cost_model.drain_penalty_cycles;
  cfg.drain_c0_bytes = cost_model.drain_c0_bytes;
  cfg.mad_head = cost_model.mad_head_cycles;
  cfg.mad_k_fractal_bytes = cost_model.mad_k_fractal_bytes;
  cfg.mad_fp32_passes = cost_model.mad_fp32_passes;
  cfg.bytes_a = bytes_a;
  cfg.bytes_b = bytes_b;
  cfg.bytes_c = bytes_c;
  cfg.align_m = handler->GetL0FractalAlignment();
  cfg.align_n = handler->GetL0FractalAlignment();
  cfg.align_k = handler->GetL0FractalAlignment();
  cfg.l0c_align_m = handler->GetL0cMAlignment(out_tile->dtype_);
  if (output_box_alignment) {
    INTERNAL_CHECK_SPAN(output_box_alignment->rows > 0 && output_box_alignment->cols > 0 &&
                            output_box_alignment->rows <= std::numeric_limits<int>::max() &&
                            output_box_alignment->cols <= std::numeric_limits<int>::max(),
                        assign->span_)
        << "Internal error: canonical split-K operand boxing has an invalid alignment ["
        << output_box_alignment->rows << ", " << output_box_alignment->cols << "]";
    cfg.box_align_m = static_cast<int>(output_box_alignment->rows);
    cfg.box_align_n = static_cast<int>(output_box_alignment->cols);
  }
  cfg.min_m = handler->GetMinL0TileDim();
  cfg.min_n = handler->GetMinL0TileDim();
  cfg.min_k = handler->GetMinL0TileDim();
  // Operand-stationary (the chooser's allow_a/b_stationary; dbA/dbB derived):
  // pin the chosen operand in a SINGLE-buffered (full) L0 buffer across the moving
  // grid (requires k == K), streaming the other operand double-buffered.
  // BuildFullKPipelined realizes it by making the stationary (outer) loop
  // ForKind::Sequential — the held operand then uses the full L0 buffer the chooser
  // budgeted (no /2) and is loaded once per outer step (no re-stream across the
  // moving axis). The chooser adopts A/B-stationary only on a strictly lower wall,
  // so it stays output-stationary for compute-bound shapes.
  // Chained Mat-scratch producers pass force_output_stationary=true to turn these
  // off (see the #1908 guard at the fold site): the Mat-scratch offset-packing path
  // cannot yet pack a single-buffered A/B-stationary producer against the consumer
  // matmul's double-buffered operands, so those producers must stay output-stationary.
  cfg.allow_a_stationary = !force_output_stationary;
  cfg.allow_b_stationary = !force_output_stationary;
  // L0C double-buffering (dbC=2): the chooser budgets the accumulator at L0C/2 and
  // scores the drain-hidden wall so tile i's FIXPIPE drain overlaps tile i+1's
  // MAD. This needs two *co-live* L0C accumulators. Under PTOAS, MemoryReuse is
  // skipped, InitMemRef keeps the buffers distinct, and ptoas assigns their
  // physical offsets.
  //
  // Under the PyPTO planner dbC=2 is an experimental opt-in (PassContext flag,
  // default OFF). It now works there because the pipeline-membership tagger gives
  // the dbC accumulator a *flat depth-2* membership — only the moving (dbC) loop
  // tags it; enclosing loops skip it since the cube serializes MADs — so
  // MemoryReuse's capacity gate (#1475) allocates exactly the two co-live L0C
  // buffers instead of over-coalescing the pair (its former behaviour, which
  // shrank the tile to L0C/2 with no second buffer). Kept default-off pending
  // device validation of the numerics + drain-hidden win. The co-live emit is
  // gated on the chooser's `double_buffer_c` result below, which tags the moving
  // loop with kPipelineDoubleBufferCAttr.
  //
  // KNOWN-FRAGILE (experimental): this reads the memory planner from the *mutable*
  // mid-pipeline PassContext, so dbC=2 behaviour depends on run-time context state
  // rather than an immutable IR property. A nested PassContext once silently reset
  // it (fixed by propagating memory_planner + PassManager's fail-loud planner check),
  // but the underlying smell remains. The durable design is a first-class co-live /
  // no-coalesce Acc-buffer-pair IR property set once at emit and honoured by BOTH
  // planners. Tracked as a follow-up.
  const bool ptoas_planner = ctx && ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS;
  const bool pypto_dbc =
      ctx && ctx->GetMemoryPlanner() == MemoryPlanner::PyPTO && ctx->GetEnablePyptoL0cDoubleBuffer();
  cfg.allow_double_buffer_c = ptoas_planner || pypto_dbc;
  // tile.matmul_acc threads the caller's accumulator into the K-loop's
  // iter-arg, so each invocation reads C from L1 at start and writes back at
  // end (gamma_c = 2 in the chooser's traffic model).  Plain tile.matmul
  // starts from a fresh Acc placeholder so C is write-only (gamma_c = 1).
  cfg.c_read = is_matmul_acc;
  cfg.allow_padding = false;
  // Permit a non-divisor final K block: the chooser may return a k that does not
  // divide K, and BuildKLoopRewrite peels the partial last K iteration.  The peel
  // is only valid when K is 16-aligned — then the tail K - floor(K/k)*k is itself
  // 16-aligned (ptoas requires 16-aligned tile cols).  A non-16-aligned K has no
  // valid K-tiling (any tail or whole-K block has non-fractal cols), so skip it
  // here with a perf hint rather than emit invalid extracts (the pre-roofline path
  // also bailed on unsupported K).
  cfg.allow_k_boundary = true;
  if (K % cfg.align_k != 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-007",
                       "tile.matmul: K=" + std::to_string(K) + " is not a multiple of the cube fractal " +
                           std::to_string(cfg.align_k) +
                           " — non-16-aligned K is unsupported; left untouched.",
                       assign->span_);
    return std::nullopt;
  }

  utils::L0TileResult res;
  try {
    res = utils::ChooseL0Tile(cfg);
  } catch (const pypto::ValueError& e) {
    hints.emplace_back(
        DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-005",
        std::string("tile.matmul: ChooseL0Tile rejected configuration — left untouched. ") + e.what(),
        assign->span_);
    return std::nullopt;
  }

  // Already L0-sized — nothing to do.
  if (res.m == M && res.n == N && res.k == K) return std::nullopt;

  if (!res.perf_hint.empty()) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-008",
                       "tile.matmul: ChooseL0Tile fallback. " + res.perf_hint, assign->span_);
  }

  MatmulTiling t;
  t.assign = assign;
  t.lhs = lhs;
  t.rhs = rhs;
  t.bias = bias_var;
  // A Vec-resident left operand is staged into Mat before the K-loop (see
  // BuildMoveToMat); Mat-resident left operands extract directly.  The right
  // operand is always Mat (checked above), so it never needs staging.
  t.stage_lhs_to_mat = lhs_tile->GetMemorySpace() == MemorySpace::Vec;
  t.acc_init = acc_var;  // null for tile.matmul, set for tile.matmul_acc
  t.M = M;
  t.N = N;
  t.K = K;
  t.m = res.m;
  t.n = res.n;
  t.k = res.k;
  t.output_valid_m = output_valid_shape[0];
  t.output_valid_n = output_valid_shape[1];
  t.stationarity = res.stationarity;
  t.os_holds_a = res.os_holds_a;
  // dbC=2 is realized only by the full-K emitter: BuildFullKPipelined attaches
  // kPipelineDoubleBufferCAttr, BuildSplitKGrid never does.  The chooser already
  // guarantees dbC ⇒ k == K (l0_tile_chooser's require_full_k), and it applied the
  // L0C/2 accumulator budget on that promise.  Assert the invariant here rather
  // than silently clamping (`&& k == K`): a clamp would drop the attr but keep the
  // L0C/2 budget, shipping a shrunk single-buffer tile — the exact regression this
  // feature exists to avoid.  A future chooser change that sets dbC on a split-K
  // tile must fail loudly instead.
  INTERNAL_CHECK_SPAN(!res.double_buffer_c || res.k == K, assign->span_)
      << "Internal error: chooser set double_buffer_c on a split-K tile (k=" << res.k << ", K=" << K
      << "); dbC=2 requires the full-K emitter";
  t.double_buffer_c = res.double_buffer_c;
  return t;
}

/// Per-output-sub-tile origin offset ``base + delta``.  Folds the common
/// constant-``base`` case (almost always ``0``) to a single ConstInt so the
/// emitted store offsets stay literal and round-trip cleanly.
ExprPtr OffsetPlus(const ExprPtr& base, int64_t delta, const Span& sp) {
  if (auto ci = As<ConstInt>(base)) return MakeIndex(ci->value_ + delta, sp);
  if (delta == 0) return base;
  return MakeAdd(base, MakeIndex(delta, sp), sp);
}

/// ``base + delta`` where ``delta`` is an Expr — a static ConstInt in the
/// unrolled grid (folded like ``OffsetPlus``) or a loop variable in the
/// pipelined emitter (left as a ``MakeAdd``).
ExprPtr AddOffset(const ExprPtr& base, const ExprPtr& delta, const Span& sp) {
  if (auto cd = As<ConstInt>(delta)) {
    if (cd->value_ == 0) return base;  // base + 0 = base (folds even for a dynamic base)
    if (auto cb = As<ConstInt>(base)) return MakeIndex(cb->value_ + cd->value_, sp);
  }
  if (auto cb = As<ConstInt>(base)) {
    if (cb->value_ == 0) return delta;  // 0 + delta = delta
  }
  return MakeAdd(base, delta, sp);
}

/// Counts reads (uses) of every Var/IterArg across a statement list, excluding
/// AssignStmt LHS defs.  Built once per SeqStmts (see ``BuildSiblingIndex``) so
/// the M/N foldability check — "is the matmul result used exactly once?" — is
/// an O(1) lookup, keeping the pass O(N) overall rather than rescanning the
/// siblings for every oversized matmul (.claude/rules/pass-complexity.md).
/// ``VisitVarLike_`` covers both Var and IterArg (.claude/rules/ir-kind-traits.md).
class SiblingUseCounter : public IRVisitor {
 public:
  std::unordered_map<const Var*, int> counts;                   ///< all reads
  std::unordered_map<const Var*, int> matmul_operand_uses;      ///< reads at a matmul-operand position
  std::unordered_map<const Var*, int> matmul_accumulator_uses;  ///< reads as tile.matmul_acc arg 0
  std::unordered_map<const Var*, VarPtr> vars;                  ///< owning pointers for type inspection

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    vars.emplace(op.get(), op);
    ++counts[op.get()];
    if (in_matmul_operand_) ++matmul_operand_uses[op.get()];
    if (in_matmul_accumulator_) ++matmul_accumulator_uses[op.get()];
  }
  // Skip the LHS (a def); count only reads in the RHS value.
  void VisitStmt_(const AssignStmtPtr& op) override { VisitExpr(op->value_); }
  // A *direct* Var at a matmul OPERAND position (``tile.matmul`` args {0,1};
  // ``tile.matmul_acc`` args {1,2} — arg 0 is the Acc accumulator, NOT a matrix
  // operand) is a Mat-safe consumer use: the consumer K-tiles that operand, so an
  // L1/Mat scratch produced upstream is legal there.  Classifying by operand index
  // is essential — a scratch fed to ``matmul_acc`` arg 0 would be an illegal
  // Mat-for-Acc substitution and must stay deferred.
  void VisitExpr_(const CallPtr& op) override {
    const bool is_mm = IsOp(op, "tile.matmul");
    const bool is_acc = IsOp(op, "tile.matmul_acc");
    const bool is_bias = IsOp(op, "tile.matmul_bias");
    for (size_t i = 0; i < op->args_.size(); ++i) {
      const bool operand_pos = ((is_mm || is_bias) && (i == 0 || i == 1)) || (is_acc && (i == 1 || i == 2));
      const bool prev = in_matmul_operand_;
      const bool prev_acc = in_matmul_accumulator_;
      in_matmul_operand_ = operand_pos && (AsVarLike(op->args_[i]) != nullptr);
      in_matmul_accumulator_ = is_acc && i == 0 && (AsVarLike(op->args_[i]) != nullptr);
      VisitExpr(op->args_[i]);
      in_matmul_operand_ = prev;
      in_matmul_accumulator_ = prev_acc;
    }
  }

 private:
  bool in_matmul_operand_ = false;
  bool in_matmul_accumulator_ = false;
};

/// One-shot index over a SeqStmts' children, built lazily on the first
/// oversized matmul and reused for the rest so M/N folding stays O(N):
///   * ``use_counts[v]`` — number of reads of ``v`` (excluding defs).
///   * ``stores_of[v]`` — top-level 2D ``tile.store`` consumers of ``v``.
///   * ``matmul_operand_positions[v]`` — sibling indices of Mat-safe reads,
///     used to prove a stored-and-reused scratch is defined before every read.
/// Counts/sites reflect the original (pre-rewrite) siblings, which is what the
/// foldability check needs (a matmul result is freshly defined; its uses do
/// not change until we rewrite it).
struct SiblingIndex {
  std::unordered_map<const Var*, int> use_counts;
  std::unordered_map<const Var*, int> matmul_operand_uses;  ///< reads at a matmul-operand position
  std::unordered_map<const Var*, int> matmul_accumulator_uses;
  std::unordered_map<const Var*, std::vector<AssignStmtPtr>> accumulator_users;
  std::unordered_map<const Var*, std::vector<const AssignStmt*>> stores_of;
  std::unordered_map<const Var*, std::vector<size_t>> matmul_operand_positions;
  std::unordered_map<const Stmt*, size_t> positions;
  std::unordered_map<const Var*, size_t> def_positions;
  std::unordered_map<const Var*, const AssignStmt*> cast_of;  ///< ``cb = tile.cast(v, dtype)`` by source
};

SiblingIndex BuildSiblingIndex(const std::vector<StmtPtr>& stmts) {
  SiblingIndex idx;
  for (size_t position = 0; position < stmts.size(); ++position) {
    const auto& s = stmts[position];
    idx.positions.emplace(s.get(), position);
    SiblingUseCounter counter;
    counter.VisitStmt(s);
    for (const auto& [var, count] : counter.counts) idx.use_counts[var] += count;
    for (const auto& [var, count] : counter.matmul_operand_uses) {
      idx.matmul_operand_uses[var] += count;
      auto& positions = idx.matmul_operand_positions[var];
      positions.insert(positions.end(), static_cast<size_t>(count), position);
    }
    for (const auto& [var, count] : counter.matmul_accumulator_uses) {
      idx.matmul_accumulator_uses[var] += count;
    }
    auto as = std::dynamic_pointer_cast<const AssignStmt>(s);
    if (!as) continue;
    idx.def_positions.emplace(as->var_.get(), position);
    auto call = As<Call>(as->value_);
    if (!call || !call->op_) continue;
    // Record top-level 2D ``tile.store(src, offsets, out)`` by source operand.
    if (IsOp(call, "tile.store") && call->args_.size() == 3) {
      if (auto src = AsVarLike(call->args_[0])) idx.stores_of[src.get()].push_back(as.get());
    } else if (IsOp(call, "tile.cast") && !call->args_.empty()) {
      // Record ``cb = tile.cast(src, dtype)`` by source: a chained matmul whose
      // result is downcast (Acc f32 -> bf16/f16) before the consumer matmul is
      // foldable into a low-precision Mat scratch (the cast = FIXPIPE writeback).
      if (auto src = AsVarLike(call->args_[0])) idx.cast_of.emplace(src.get(), as.get());
    } else if (IsOp(call, "tile.matmul_acc") && call->args_.size() == 3) {
      if (auto acc = AsVarLike(call->args_[0])) idx.accumulator_users[acc.get()].push_back(as);
    }
  }
  return idx;
}

struct LinearMatmulChain {
  std::vector<MatmulTiling> stages;  ///< root matmul[/bias], then one or more matmul_acc stages
  std::vector<const Var*> continuation_defs;
  const Var* terminal = nullptr;
};

/// Follow a linear accumulator SSA chain rooted at a fresh matmul. Every
/// intermediate must be used exactly once, as argument 0 of one top-level
/// tile.matmul_acc. A common M/N grid is the component-wise minimum of all
/// stages' legal tiles; each stage keeps its independently chosen K blocking.
std::optional<LinearMatmulChain> AnalyzeLinearMatmulChain(const MatmulTiling& root, const SiblingIndex& index,
                                                          std::vector<Diagnostic>& hints) {
  if (root.is_acc()) return std::nullopt;
  LinearMatmulChain chain{{root}, {}, root.assign->var_.get()};
  const Var* current = root.assign->var_.get();
  auto root_position = index.def_positions.find(root.assign->var_.get());
  if (root_position == index.def_positions.end()) return std::nullopt;
  size_t previous_position = root_position->second;

  while (true) {
    auto users_it = index.accumulator_users.find(current);
    if (users_it == index.accumulator_users.end()) break;
    auto uses_it = index.use_counts.find(current);
    auto acc_uses_it = index.matmul_accumulator_uses.find(current);
    const int uses = uses_it == index.use_counts.end() ? 0 : uses_it->second;
    const int acc_uses = acc_uses_it == index.matmul_accumulator_uses.end() ? 0 : acc_uses_it->second;
    if (users_it->second.size() != 1 || uses != 1 || acc_uses != 1) return std::nullopt;

    const AssignStmtPtr& next_assign = users_it->second.front();
    auto position_it = index.positions.find(next_assign.get());
    if (position_it == index.positions.end() || position_it->second <= previous_position) return std::nullopt;
    auto next = AnalyzeMatmul(next_assign, hints, /*force_output_stationary=*/true);
    if (!next || !next->is_acc() || next->acc_init.get() != current || next->M != root.M ||
        next->N != root.N) {
      return std::nullopt;
    }
    next->double_buffer_c = false;  // the chain grid serializes its accumulation stages
    chain.stages.push_back(std::move(*next));
    chain.continuation_defs.push_back(next_assign->var_.get());
    current = next_assign->var_.get();
    chain.terminal = current;
    previous_position = position_it->second;
  }

  if (chain.stages.size() < 2) return std::nullopt;
  int64_t common_m = root.m;
  int64_t common_n = root.n;
  for (const auto& stage : chain.stages) {
    common_m = std::min(common_m, stage.m);
    common_n = std::min(common_n, stage.n);
  }
  for (auto& stage : chain.stages) {
    stage.m = common_m;
    stage.n = common_n;
    stage.stationarity = utils::Stationarity::kOutputStationary;
    stage.double_buffer_c = false;
  }
  return chain;
}

/// One folded M/N rewrite: the unrolled per-sub-tile K-loops + stores that
/// replace ``c = tile.matmul(...)`` together with its consumer store
/// ``out = tile.store(c, base, out)``.  The grid emits at the consumer-store
/// position; the store's LHS is remapped to ``return_var`` and the store
/// dropped.
struct MNFold {
  std::vector<StmtPtr> stmts;         ///< pipelined interior + tail / K-loops + per-sub-tile placement
  VarPtr return_var;                  ///< final output tensor value (replaces the store's result downstream)
  VarPtr store_result_var;            ///< the consumer store's LHS (remapped to return_var)
  const AssignStmt* store = nullptr;  ///< consumer/anchor statement to replace in the SeqStmts
  /// Optional on-chip materialization installed for downstream consumers when
  /// the logical result is both stored and reused.
  const Var* materialized_old_var = nullptr;
  VarPtr materialized_new_var = nullptr;
  /// Optional cast definition made redundant by a low-precision Mat scratch.
  const Var* dropped_def = nullptr;
};

/// Where each computed ``[m_eff, n_eff]`` Acc sub-tile is placed.  The M/N grid
/// builders (``BuildFullKPipelined`` interior+tail, ``BuildSplitKGrid`` K-loop
/// grid) are placement-agnostic: they compute each sub-tile's Acc result and
/// hand it to a ``SubtilePlacer``. ``DirectGmPlacer`` stores to a DDR output;
/// ``MatScratchPlacer`` assembles into an on-chip Mat scratch for a chained
/// matmul consumer. A placer threads one or more chained output Vars in
/// traversal order and yields the final state via ``PlaceAt``. Multiple state
/// elements let one logical matmul result be materialized to more than one
/// destination (for example GM plus an internal Mat scratch) without ever
/// constructing the oversized Acc value.
using PlacementState = std::vector<VarPtr>;

class SubtilePlacer {
 public:
  virtual ~SubtilePlacer() = default;
  /// Emit any prologue and return the initial chained state. The grid threads
  /// every state element through each ``PlaceAt`` and returns the final state.
  [[nodiscard]] virtual PlacementState Init(std::vector<StmtPtr>& stmts) = 0;
  /// Place ``sub`` (an ``[m, n]`` Acc result) into ``chain_in`` at output offsets
  /// ``(row_off, col_off)`` — both Exprs (static ConstInt in the unrolled grid,
  /// loop variables in the pipelined emitter).  Append the placement stmt and
  /// return the new chained Var.  Stateless so it works inside a loop body where
  /// the chain is a loop iter-arg.
  [[nodiscard]] virtual PlacementState PlaceAt(std::vector<StmtPtr>& stmts, const VarPtr& sub,
                                               const ExprPtr& row_off, const ExprPtr& col_off,
                                               const PlacementState& chain_in, int step) = 0;
};

CallPtr PreserveCallAttrs(const std::vector<std::pair<std::string, std::any>>& attrs,
                          const CallPtr& deduced) {
  if (attrs.empty()) return deduced;
  return std::make_shared<Call>(deduced->op_, deduced->args_, deduced->kwargs_, attrs, deduced->GetType(),
                                deduced->span_);
}

CallPtr PreserveCallAttrs(const CallPtr& original, const CallPtr& deduced) {
  return PreserveCallAttrs(original->attrs_, deduced);
}

/// Direct-store placement: ``out = tile.store(sub, [base_r + mi, base_c + ni],
/// out_prev)`` per sub-tile, chaining the DDR output tensor in SSA form.
class DirectGmPlacer : public SubtilePlacer {
 public:
  DirectGmPlacer(ExprPtr base_r, ExprPtr base_c, VarPtr out_in,
                 std::vector<std::pair<std::string, std::any>> store_kwargs,
                 std::vector<std::pair<std::string, std::any>> store_attrs, Span span)
      : base_r_(std::move(base_r)),
        base_c_(std::move(base_c)),
        out_in_(std::move(out_in)),
        out_base_(out_in_->name_hint_),
        kwargs_(std::move(store_kwargs)),
        attrs_(std::move(store_attrs)),
        sp_(std::move(span)) {}

  [[nodiscard]] PlacementState Init(std::vector<StmtPtr>& /*stmts*/) override { return {out_in_}; }

  [[nodiscard]] PlacementState PlaceAt(std::vector<StmtPtr>& stmts, const VarPtr& sub, const ExprPtr& row_off,
                                       const ExprPtr& col_off, const PlacementState& chain_in,
                                       int step) override {
    INTERNAL_CHECK_SPAN(chain_in.size() == 1, sp_)
        << "Internal error: DirectGmPlacer expects exactly one chained output";
    auto& reg = OpRegistry::GetInstance();
    auto offs = std::make_shared<MakeTuple>(
        std::vector<ExprPtr>{AddOffset(base_r_, row_off, sp_), AddOffset(base_c_, col_off, sp_)}, sp_);
    auto deduced = reg.Create("tile.store", {sub, offs, chain_in.front()}, kwargs_, sp_);
    auto scall = PreserveCallAttrs(attrs_, deduced);
    auto sv = std::make_shared<Var>(out_base_ + "_t" + std::to_string(step), scall->GetType(), sp_);
    stmts.push_back(std::make_shared<AssignStmt>(sv, scall, sp_));
    return {sv};
  }

 private:
  ExprPtr base_r_, base_c_;
  VarPtr out_in_;
  std::string out_base_;
  std::vector<std::pair<std::string, std::any>> kwargs_;
  std::vector<std::pair<std::string, std::any>> attrs_;
  Span sp_;
};

/// Mat-scratch placement (on-chip matmul consumers): keep the whole ``[M, N]``
/// result in an L1/Mat scratch instead of storing it to a DDR tensor, so a
/// matmul-operand consumer reads it on-chip.  ``Init`` creates the scratch (mirrors
/// ``BuildAccInit`` but in ``Mat``); each ``PlaceAt`` assembles a sub-tile in place:
/// ``scratch_{k+1} = tile.assemble(scratch_k, sub, [row_off, col_off])`` — Acc→Mat.
/// A low-precision bf16/f16 scratch lowers to FIXPIPE ``pto.tinsert``; a
/// supported same-dtype full-window assemble uses ``pto.subview`` + ``pto.tmov``.
/// ``tile.assemble`` is ``set_output_memory_inherit_input()``, so the chain shares
/// one Mat base before MemoryReuse runs (no full-scratch copy per insert).
///
/// ``tile.assemble``'s offset is a literal ``MakeTuple`` whose *elements* may be
/// loop variables (``ValidateIndexTupleElements`` only requires index-typed
/// elements, not constants), so this placer drives both the constant-offset
/// unrolled grid (``BuildSplitKGrid``, K-split) and the loop-variable pipelined
/// emitter (``BuildFullKPipelined``, full-K).
class MatScratchPlacer : public SubtilePlacer {
 public:
  MatScratchPlacer(int64_t big_m, int64_t big_n, DataType dtype, std::string base, Span span)
      : m_(big_m), n_(big_n), dtype_(dtype), base_(std::move(base)), sp_(std::move(span)) {}

  [[nodiscard]] PlacementState Init(std::vector<StmtPtr>& stmts) override {
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", dtype_},
                                                            {"target_memory", MemorySpace::Mat}};
    auto call = reg.Create("tile.create", {MakeIndexTuple({m_, n_}, sp_)}, kwargs, sp_);
    auto scratch = std::make_shared<Var>(base_, call->GetType(), sp_);
    stmts.push_back(std::make_shared<AssignStmt>(scratch, call, sp_));
    return {scratch};
  }

  [[nodiscard]] PlacementState PlaceAt(std::vector<StmtPtr>& stmts, const VarPtr& sub, const ExprPtr& row_off,
                                       const ExprPtr& col_off, const PlacementState& chain_in,
                                       int step) override {
    INTERNAL_CHECK_SPAN(chain_in.size() == 1, sp_)
        << "Internal error: MatScratchPlacer expects exactly one chained scratch";
    auto& reg = OpRegistry::GetInstance();
    auto offs = std::make_shared<MakeTuple>(std::vector<ExprPtr>{row_off, col_off}, sp_);
    auto call = reg.Create("tile.assemble", {chain_in.front(), sub, offs}, sp_);
    auto sv = std::make_shared<Var>(base_ + "_t" + std::to_string(step), call->GetType(), sp_);
    stmts.push_back(std::make_shared<AssignStmt>(sv, call, sp_));
    return {sv};
  }

 private:
  int64_t m_, n_;
  DataType dtype_;
  std::string base_;
  Span sp_;
};

/// Fan one computed Acc sub-tile out to several independent materializers.
/// Each child owns a disjoint slice of the loop-carried state. Current leaves
/// each own one element, but retaining widths makes nesting well-defined and
/// keeps the grid builders independent of destination count.
class CompositeSubtilePlacer : public SubtilePlacer {
 public:
  explicit CompositeSubtilePlacer(std::vector<std::unique_ptr<SubtilePlacer>> children, Span span)
      : children_(std::move(children)), sp_(std::move(span)) {}

  [[nodiscard]] PlacementState Init(std::vector<StmtPtr>& stmts) override {
    INTERNAL_CHECK_SPAN(!children_.empty(), sp_)
        << "Internal error: CompositeSubtilePlacer requires at least one destination";
    widths_.clear();
    PlacementState state;
    for (auto& child : children_) {
      auto child_state = child->Init(stmts);
      INTERNAL_CHECK_SPAN(!child_state.empty(), sp_)
          << "Internal error: a composite placement child returned empty state";
      widths_.push_back(child_state.size());
      state.insert(state.end(), child_state.begin(), child_state.end());
    }
    return state;
  }

  [[nodiscard]] PlacementState PlaceAt(std::vector<StmtPtr>& stmts, const VarPtr& sub, const ExprPtr& row_off,
                                       const ExprPtr& col_off, const PlacementState& chain_in,
                                       int step) override {
    INTERNAL_CHECK_SPAN(widths_.size() == children_.size(), sp_)
        << "Internal error: CompositeSubtilePlacer::Init must run before PlaceAt";
    PlacementState next;
    size_t begin = 0;
    for (size_t i = 0; i < children_.size(); ++i) {
      const size_t end = begin + widths_[i];
      INTERNAL_CHECK_SPAN(end <= chain_in.size(), sp_)
          << "Internal error: composite placement state is shorter than its child layout";
      PlacementState child_in(chain_in.begin() + static_cast<std::ptrdiff_t>(begin),
                              chain_in.begin() + static_cast<std::ptrdiff_t>(end));
      auto child_out = children_[i]->PlaceAt(stmts, sub, row_off, col_off, child_in, step);
      INTERNAL_CHECK_SPAN(child_out.size() == widths_[i], sp_)
          << "Internal error: composite placement child changed its state width";
      next.insert(next.end(), child_out.begin(), child_out.end());
      begin = end;
    }
    INTERNAL_CHECK_SPAN(begin == chain_in.size(), sp_)
        << "Internal error: composite placement state has unowned elements";
    return next;
  }

 private:
  std::vector<std::unique_ptr<SubtilePlacer>> children_;
  std::vector<size_t> widths_;
  Span sp_;
};

/// Count every syntactic Var/IterArg read in one function traversal while
/// excluding SSA definition sites. The canonical split-K pre-phase uses this
/// index to prove that its create and loop result have no consumers outside the
/// matched create/loop/store chain.
class CanonicalReadCounter : public IRVisitor {
 public:
  std::unordered_map<const Var*, size_t> counts;

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) ++counts[op.get()];
  }

  // The generic visitor follows an IterArg read through ``initValue_``.  That
  // is useful for recursive dependency walks, but this index counts direct SSA
  // edges: the initializer is visited once at the enclosing loop and an
  // IterArg use in the body must not count it again.
  void VisitExpr_(const IterArgPtr& op) override { VisitVarLike_(op); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->value_) VisitExpr(op->value_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (op->condition_) VisitExpr(op->condition_);
    if (op->then_body_) VisitStmt(op->then_body_);
    if (op->else_body_.has_value() && *op->else_body_) VisitStmt(*op->else_body_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->start_) VisitExpr(op->start_);
    if (op->stop_) VisitExpr(op->stop_);
    if (op->step_) VisitExpr(op->step_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    if (op->body_) VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (op->condition_) VisitExpr(op->condition_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    if (op->body_) VisitStmt(op->body_);
  }
};

/// Canonical frontend split-K reduction matched for loop-level M/N tiling:
///
///   init = tile.create([M, N])
///   for kb, (acc,) in pipeline(..., init_values=(init,)):
///     lhs = tile.load(A, [mi0, k0], [M, Kb], ...)
///     rhs = tile.load(B, [k0, ni0], [Kb, N], ...)
///     if first:
///       next = tile.matmul(lhs, rhs)
///     else:
///       next = tile.matmul_acc(acc, lhs, rhs)
///     yield next
///   out = tile.store(loop_result, [base_m, base_n], out)
///
/// The whole loop must move outside the output-tile grid: every [m, n]
/// accumulator completes all split-K iterations before the next output tile is
/// started. Merely slicing the matmul_acc call in place would leave the
/// impossible full [M, N] loop-carried accumulator live in L0C.
struct CanonicalSplitKAccMatch {
  AssignStmtPtr init;
  ForStmtPtr loop;
  AssignStmtPtr store;
  AssignStmtPtr lhs_load;
  AssignStmtPtr rhs_load;
  AssignStmtPtr matmul;
  AssignStmtPtr matmul_acc;
  VarPtr phi;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;  ///< K width of one source split-K iteration
};

struct CanonicalOutputWindow {
  int64_t valid_m = 0;
  int64_t valid_n = 0;
  int64_t physical_m = 0;
  int64_t physical_n = 0;
};

std::optional<std::pair<AssignStmtPtr, YieldStmtPtr>> MatchSingleAssignYield(const StmtPtr& body,
                                                                             const char* op_name) {
  auto seq = As<SeqStmts>(body);
  if (!seq || seq->stmts_.size() != 2) return std::nullopt;
  auto assign = As<AssignStmt>(seq->stmts_[0]);
  auto yield = As<YieldStmt>(seq->stmts_[1]);
  auto call = assign ? As<Call>(assign->value_) : nullptr;
  if (!assign || !yield || !call || !IsOp(call, op_name) || yield->value_.size() != 1 ||
      yield->value_[0].get() != assign->var_.get()) {
    return std::nullopt;
  }
  return std::make_pair(assign, yield);
}

bool IsStatic2DTuple(const ExprPtr& expr, int64_t dim0, int64_t dim1) {
  auto tuple = As<MakeTuple>(expr);
  if (!tuple || tuple->elements_.size() != 2) return false;
  auto first = As<ConstInt>(tuple->elements_[0]);
  auto second = As<ConstInt>(tuple->elements_[1]);
  return first && second && first->value_ == dim0 && second->value_ == dim1;
}

std::optional<CanonicalSplitKAccMatch> MatchCanonicalSplitKAcc(
    const AssignStmtPtr& init, const ForStmtPtr& loop, const AssignStmtPtr& store,
    const std::unordered_map<const Var*, size_t>& use_counts) {
  if (!init || !loop || !store || loop->kind_ != ForKind::Pipeline || loop->iter_args_.size() != 1 ||
      loop->return_vars_.size() != 1) {
    return std::nullopt;
  }

  auto init_call = As<Call>(init->value_);
  auto store_call = As<Call>(store->value_);
  if (!init_call || !IsOp(init_call, "tile.create") || !store_call || !IsOp(store_call, "tile.store") ||
      store_call->args_.size() != 3 || loop->iter_args_[0]->initValue_.get() != init->var_.get() ||
      store_call->args_[0].get() != loop->return_vars_[0].get()) {
    return std::nullopt;
  }
  auto store_offsets = As<MakeTuple>(store_call->args_[1]);
  if (!store_offsets || store_offsets->elements_.size() != 2 || !AsVarLike(store_call->args_[2])) {
    return std::nullopt;
  }
  auto init_use = use_counts.find(init->var_.get());
  auto result_use = use_counts.find(loop->return_vars_[0].get());
  // True read counts exclude definitions: the create feeds exactly one IterArg
  // initializer and the loop result feeds exactly one store. Any other read
  // would be invalidated when the triplet is replaced.
  if (init_use == use_counts.end() || init_use->second != 1 || result_use == use_counts.end() ||
      result_use->second != 1) {
    return std::nullopt;
  }

  auto body = As<SeqStmts>(loop->body_);
  if (!body || body->stmts_.size() < 2) return std::nullopt;
  const size_t if_pos = body->stmts_.size() - 2;
  auto if_stmt = As<IfStmt>(body->stmts_[if_pos]);
  auto outer_yield = As<YieldStmt>(body->stmts_.back());
  if (!if_stmt || !if_stmt->else_body_.has_value() || if_stmt->return_vars_.size() != 1 || !outer_yield ||
      outer_yield->value_.size() != 1 || outer_yield->value_[0].get() != if_stmt->return_vars_[0].get()) {
    return std::nullopt;
  }

  auto then_mm = MatchSingleAssignYield(if_stmt->then_body_, "tile.matmul");
  auto else_mm = MatchSingleAssignYield(*if_stmt->else_body_, "tile.matmul_acc");
  auto then_acc = MatchSingleAssignYield(if_stmt->then_body_, "tile.matmul_acc");
  auto else_plain = MatchSingleAssignYield(*if_stmt->else_body_, "tile.matmul");
  AssignStmtPtr matmul;
  AssignStmtPtr matmul_acc;
  if (then_mm && else_mm) {
    matmul = then_mm->first;
    matmul_acc = else_mm->first;
  } else if (then_acc && else_plain) {
    matmul = else_plain->first;
    matmul_acc = then_acc->first;
  } else {
    return std::nullopt;
  }

  auto mm_call = As<Call>(matmul->value_);
  auto acc_call = As<Call>(matmul_acc->value_);
  if (!mm_call || !acc_call || mm_call->args_.size() != 2 || acc_call->args_.size() != 3 ||
      acc_call->args_[0].get() != loop->iter_args_[0].get() ||
      mm_call->args_[0].get() != acc_call->args_[1].get() ||
      mm_call->args_[1].get() != acc_call->args_[2].get()) {
    return std::nullopt;
  }
  auto lhs = AsVarLike(mm_call->args_[0]);
  auto rhs = AsVarLike(mm_call->args_[1]);
  // One source load cannot simultaneously be narrowed as [m, K] and [K, n].
  // Reject the square self-matmul corner rather than mutate it ambiguously.
  if (!lhs || !rhs || lhs.get() == rhs.get()) return std::nullopt;

  std::unordered_map<const Var*, AssignStmtPtr> direct_defs;
  for (size_t i = 0; i < if_pos; ++i) {
    auto assign = As<AssignStmt>(body->stmts_[i]);
    if (!assign) return std::nullopt;
    direct_defs.emplace(assign->var_.get(), assign);
    // Duplicating a source split-K loop per output tile is legal only when all
    // definitions other than the two operand loads are scalar computations.
    // This excludes hidden tile side effects or unrelated loads instead of
    // silently multiplying them.
    if (assign->var_.get() != lhs.get() && assign->var_.get() != rhs.get() &&
        (As<Call>(assign->value_) || !As<ScalarType>(assign->var_->GetType()))) {
      return std::nullopt;
    }
  }
  auto lhs_it = direct_defs.find(lhs.get());
  auto rhs_it = direct_defs.find(rhs.get());
  if (lhs_it == direct_defs.end() || rhs_it == direct_defs.end()) return std::nullopt;
  auto lhs_load_call = As<Call>(lhs_it->second->value_);
  auto rhs_load_call = As<Call>(rhs_it->second->value_);
  if (!lhs_load_call || !rhs_load_call || !IsOp(lhs_load_call, "tile.load") ||
      !IsOp(rhs_load_call, "tile.load") || lhs_load_call->args_.size() != 4 ||
      rhs_load_call->args_.size() != 4) {
    return std::nullopt;
  }
  int64_t M = 0;
  int64_t K_lhs = 0;
  int64_t K_rhs = 0;
  int64_t N = 0;
  auto lhs_ty = As<TileType>(lhs->GetType());
  auto rhs_ty = As<TileType>(rhs->GetType());
  if (!IsStatic2DInSpaces(lhs_ty, {MemorySpace::Mat}, M, K_lhs) ||
      !IsStatic2DInSpaces(rhs_ty, {MemorySpace::Mat}, K_rhs, N) || K_lhs != K_rhs) {
    return std::nullopt;
  }
  // Replacing shape and valid_shape with the narrowed output window is sound
  // only for full rectangular loads. A padded or dynamic valid_shape requires
  // its own intersection arithmetic and stays on the conservative deferred
  // path for now.
  if (!As<MakeTuple>(lhs_load_call->args_[1]) || !As<MakeTuple>(rhs_load_call->args_[1]) ||
      !IsStatic2DTuple(lhs_load_call->args_[2], M, K_lhs) ||
      !IsStatic2DTuple(lhs_load_call->args_[3], M, K_lhs) ||
      !IsStatic2DTuple(rhs_load_call->args_[2], K_rhs, N) ||
      !IsStatic2DTuple(rhs_load_call->args_[3], K_rhs, N)) {
    return std::nullopt;
  }
  auto init_ty = As<TileType>(init->var_->GetType());
  auto init_m = init_ty && init_ty->shape_.size() == 2 ? As<ConstInt>(init_ty->shape_[0]) : nullptr;
  auto init_n = init_ty && init_ty->shape_.size() == 2 ? As<ConstInt>(init_ty->shape_[1]) : nullptr;
  if (!init_m || !init_n || init_m->value_ != M || init_n->value_ != N) return std::nullopt;

  return CanonicalSplitKAccMatch{
      init, loop, store, lhs_it->second, rhs_it->second, matmul, matmul_acc, if_stmt->return_vars_[0],
      M,    N,    K_lhs};
}

std::optional<tile_view_semantics::BoxedTileAlignment> GetCanonicalOutputBoxAlignment(
    const CanonicalSplitKAccMatch& match) {
  auto lhs_type = As<TileType>(match.lhs_load->var_->GetType());
  auto rhs_type = As<TileType>(match.rhs_load->var_->GetType());
  INTERNAL_CHECK_SPAN(lhs_type && rhs_type, match.loop->span_)
      << "Internal error: canonical split-K operand loads lost their TileTypes";

  const auto lhs_alignment = tile_view_semantics::GetBoxedTileAlignment(*lhs_type);
  const auto rhs_alignment = tile_view_semantics::GetBoxedTileAlignment(*rhs_type);
  if (!lhs_alignment || !rhs_alignment) return std::nullopt;

  return tile_view_semantics::BoxedTileAlignment{/*rows=*/lhs_alignment->rows,
                                                 /*cols=*/rhs_alignment->cols};
}

CanonicalOutputWindow BuildCanonicalOutputWindow(
    const CanonicalSplitKAccMatch& match, int64_t valid_m, int64_t valid_n,
    const tile_view_semantics::BoxedTileAlignment& output_box_alignment) {
  return CanonicalOutputWindow{
      /*valid_m=*/valid_m,
      /*valid_n=*/valid_n,
      /*physical_m=*/AlignStaticExtent(valid_m, output_box_alignment.rows, match.loop->span_),
      /*physical_n=*/AlignStaticExtent(valid_n, output_box_alignment.cols, match.loop->span_),
  };
}

/// Retile one DeepClone of the canonical source K loop. Definitions are fresh
/// already; this mutator narrows the two GM->Mat loads and re-deduces both MAD
/// calls plus the if/loop phi types for one [m, n] output tile.
class CanonicalSplitKRetiler : public IRMutator {
 public:
  CanonicalSplitKRetiler(const CanonicalSplitKAccMatch& match,
                         const std::unordered_map<const Var*, VarPtr>& clone_map, const VarPtr& init,
                         int64_t mi, int64_t ni, CanonicalOutputWindow window, std::string suffix)
      : mi_(mi), ni_(ni), window_(window), k_(match.K), suffix_(std::move(suffix)) {
    auto cloned = [&](const VarPtr& original) -> VarPtr {
      auto it = clone_map.find(original.get());
      INTERNAL_CHECK_SPAN(it != clone_map.end(), original->span_)
          << "Internal error: canonical split-K clone omitted definition " << original->name_hint_;
      return it->second;
    };

    lhs_load_ = cloned(match.lhs_load->var_).get();
    rhs_load_ = cloned(match.rhs_load->var_).get();
    matmul_ = cloned(match.matmul->var_).get();
    matmul_acc_ = cloned(match.matmul_acc->var_).get();

    auto old_iter_var = cloned(match.loop->iter_args_[0]);
    auto old_iter = std::dynamic_pointer_cast<const IterArg>(old_iter_var);
    INTERNAL_CHECK_SPAN(old_iter, match.loop->span_)
        << "Internal error: canonical split-K IterArg cloned as a plain Var";
    INTERNAL_CHECK_SPAN(init, match.loop->span_)
        << "Internal error: canonical split-K output tile has no accumulator initializer";
    auto acc_ty = init->GetType();
    auto new_iter = std::make_shared<IterArg>(old_iter->name_hint_ + suffix_, acc_ty, init, old_iter->span_);
    var_remap_[old_iter.get()] = new_iter;

    auto old_phi = cloned(match.phi);
    auto new_phi = std::make_shared<Var>(old_phi->name_hint_ + suffix_, acc_ty, old_phi->span_);
    var_remap_[old_phi.get()] = new_phi;

    auto old_return = cloned(match.loop->return_vars_[0]);
    auto new_return = std::make_shared<Var>(old_return->name_hint_ + suffix_, acc_ty, old_return->span_);
    var_remap_[old_return.get()] = new_return;
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    const Var* key = op->var_.get();
    if (key != lhs_load_ && key != rhs_load_ && key != matmul_ && key != matmul_acc_) {
      return IRMutator::VisitStmt_(op);
    }
    auto call = As<Call>(op->value_);
    INTERNAL_CHECK_SPAN(call, op->span_) << "Internal error: canonical split-K definition is not a Call";
    CallPtr rebuilt;
    if (key == lhs_load_ || key == rhs_load_) {
      rebuilt = RebuildLoad(call, /*lhs=*/key == lhs_load_);
    } else {
      std::vector<ExprPtr> args;
      args.reserve(call->args_.size());
      for (const auto& arg : call->args_) args.push_back(VisitExpr(arg));
      auto deduced = OpRegistry::GetInstance().Create(call->op_->name_, args, call->kwargs_, call->span_);
      rebuilt = PreserveCallAttrs(call, deduced);
    }
    auto var = std::make_shared<Var>(op->var_->name_hint_ + suffix_, rebuilt->GetType(), op->var_->span_);
    var_remap_[op->var_.get()] = var;
    return std::make_shared<AssignStmt>(var, rebuilt, op->span_, op->leading_comments_);
  }

 private:
  CallPtr RebuildLoad(const CallPtr& call, bool lhs) {
    std::vector<ExprPtr> args;
    args.reserve(call->args_.size());
    for (const auto& arg : call->args_) args.push_back(VisitExpr(arg));
    auto offsets = As<MakeTuple>(args[1]);
    INTERNAL_CHECK_SPAN(offsets && offsets->elements_.size() == 2, call->span_)
        << "Internal error: canonical split-K tile.load lost its 2D offsets";
    std::vector<ExprPtr> new_offsets = offsets->elements_;
    if (lhs) {
      new_offsets[0] = OffsetPlus(new_offsets[0], mi_, call->span_);
      args[2] = MakeIndexTuple({window_.physical_m, k_}, call->span_);
      args[3] = MakeIndexTuple({window_.valid_m, k_}, call->span_);
    } else {
      new_offsets[1] = OffsetPlus(new_offsets[1], ni_, call->span_);
      args[2] = MakeIndexTuple({k_, window_.physical_n}, call->span_);
      args[3] = MakeIndexTuple({k_, window_.valid_n}, call->span_);
    }
    args[1] = std::make_shared<MakeTuple>(std::move(new_offsets), call->span_);
    auto deduced = OpRegistry::GetInstance().Create(call->op_->name_, args, call->kwargs_, call->span_);
    return PreserveCallAttrs(call, deduced);
  }

  const Var* lhs_load_ = nullptr;
  const Var* rhs_load_ = nullptr;
  const Var* matmul_ = nullptr;
  const Var* matmul_acc_ = nullptr;
  int64_t mi_ = 0;
  int64_t ni_ = 0;
  CanonicalOutputWindow window_;
  int64_t k_ = 0;
  std::string suffix_;
};

struct CanonicalSplitKFold {
  std::vector<StmtPtr> stmts;
  VarPtr final_output;
  VarPtr old_store_result;
};

std::optional<CanonicalSplitKFold> TryFoldCanonicalSplitKAcc(const CanonicalSplitKAccMatch& match,
                                                             std::vector<Diagnostic>& hints) {
  const auto output_box_alignment = GetCanonicalOutputBoxAlignment(match);
  if (!output_box_alignment) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
                       "canonical split-K M/N tiling needs boxed Mat operand layouts with a supported static "
                       "fractal alignment; left untouched",
                       match.loop->span_);
    return std::nullopt;
  }
  // The source loop already realizes output-stationary accumulation across its
  // K blocks. Use the conservative output-stationary chooser regime for the
  // output grid. Account for the same Mat boxing that RebuildLoad will apply to
  // every physical output window, so chooser capacity cannot admit a logical
  // tile that becomes oversized after padding. The recursively visited
  // narrowed calls independently choose their legal inner K blocking.
  auto tiling = AnalyzeMatmul(match.matmul, hints, /*force_output_stationary=*/true, output_box_alignment);
  if (!tiling || !tiling->needs_mn_tiling()) return std::nullopt;

  auto store_call = As<Call>(match.store->value_);
  auto offsets = As<MakeTuple>(store_call->args_[1]);
  auto out_in = AsVarLike(store_call->args_[2]);
  INTERNAL_CHECK_SPAN(offsets && out_in, match.store->span_)
      << "Internal error: matched canonical split-K store became invalid";
  DirectGmPlacer placer(offsets->elements_[0], offsets->elements_[1], out_in, store_call->kwargs_,
                        store_call->attrs_, match.store->span_);
  std::vector<StmtPtr> stmts;
  PlacementState chain = placer.Init(stmts);
  INTERNAL_CHECK_SPAN(chain.size() == 1, match.store->span_)
      << "Internal error: canonical split-K direct store must have one placement result";
  auto out_ty = As<TileType>(match.matmul->var_->GetType());
  INTERNAL_CHECK_SPAN(out_ty, match.matmul->span_)
      << "Internal error: canonical split-K matmul result lost its TileType";
  const int64_t num_m = (match.M + tiling->m - 1) / tiling->m;
  const int64_t num_n = (match.N + tiling->n - 1) / tiling->n;
  // Output-sensitive expansion: each source statement is cloned once per
  // required output tile, matching BuildSplitKGrid. Work is O(input IR plus
  // emitted IR); there is no repeated scan of the surrounding program.
  int step = 0;
  for (int64_t nj = 0; nj < num_n; ++nj) {
    const int64_t ni = nj * tiling->n;
    const int64_t n_eff = std::min<int64_t>(tiling->n, match.N - ni);
    for (int64_t mj = 0; mj < num_m; ++mj) {
      const int64_t mi = mj * tiling->m;
      const int64_t m_eff = std::min<int64_t>(tiling->m, match.M - mi);
      const std::string suffix = "_mn" + std::to_string(step);
      const auto window = BuildCanonicalOutputWindow(match, m_eff, n_eff, *output_box_alignment);
      auto init = BuildAccInitWithValidShape(window.physical_m, window.physical_n,
                                             MakeIndex(window.valid_m, match.init->span_),
                                             MakeIndex(window.valid_n, match.init->span_), out_ty->dtype_,
                                             match.init->var_->name_hint_ + suffix, match.init->span_);
      for (auto& init_stmt : init.stmts) stmts.push_back(std::move(init_stmt));

      std::unordered_map<const Var*, ExprPtr> seed = {{match.init->var_.get(), init.value}};
      auto clone = DeepClone(match.loop, seed, /*clone_def_vars=*/true);
      auto cloned_loop = As<ForStmt>(clone.cloned_body);
      INTERNAL_CHECK_SPAN(cloned_loop, match.loop->span_)
          << "Internal error: canonical split-K loop clone is not a ForStmt";
      CanonicalSplitKRetiler retiler(match, clone.var_map, init.value, mi, ni, window, suffix);
      auto narrowed = As<ForStmt>(retiler.VisitStmt(cloned_loop));
      INTERNAL_CHECK_SPAN(narrowed, match.loop->span_)
          << "Internal error: canonical split-K retiling did not return a ForStmt";
      stmts.push_back(narrowed);
      chain = placer.PlaceAt(stmts, narrowed->return_vars_[0], MakeIndex(mi, match.store->span_),
                             MakeIndex(ni, match.store->span_), chain, step);
      ++step;
    }
  }
  return CanonicalSplitKFold{std::move(stmts), chain.front(), match.store->var_};
}

/// Rewrite every canonical create/loop/store triplet in one SeqStmts scan.
/// ``use_counts`` is built once over the original function, so matching all
/// nested sequences remains O(N) rather than recursively rescanning each
/// subtree. A running remap preserves an earlier store's output chain in later
/// siblings.
StmtPtr RewriteCanonicalSplitKSeq(const SeqStmtsPtr& seq,
                                  const std::unordered_map<const Var*, size_t>& use_counts,
                                  std::vector<Diagnostic>& hints) {
  if (!seq || seq->stmts_.size() < 3) return seq;
  std::vector<StmtPtr> out;
  out.reserve(seq->stmts_.size());
  std::unordered_map<const Var*, VarPtr> remap;
  bool changed = false;

  for (size_t i = 0; i < seq->stmts_.size();) {
    auto current = remap.empty() ? seq->stmts_[i] : transform_utils::Substitute(seq->stmts_[i], remap);
    if (i + 2 < seq->stmts_.size()) {
      auto next = remap.empty() ? seq->stmts_[i + 1] : transform_utils::Substitute(seq->stmts_[i + 1], remap);
      auto third =
          remap.empty() ? seq->stmts_[i + 2] : transform_utils::Substitute(seq->stmts_[i + 2], remap);
      auto match = MatchCanonicalSplitKAcc(As<AssignStmt>(current), As<ForStmt>(next), As<AssignStmt>(third),
                                           use_counts);
      if (match) {
        if (auto fold = TryFoldCanonicalSplitKAcc(*match, hints)) {
          for (auto& stmt : fold->stmts) out.push_back(std::move(stmt));
          remap[fold->old_store_result.get()] = fold->final_output;
          i += 3;
          changed = true;
          continue;
        }
      }
    }
    if (current.get() != seq->stmts_[i].get()) changed = true;
    out.push_back(std::move(current));
    ++i;
  }
  if (!changed) return seq;
  return SeqStmts::Flatten(std::move(out), seq->span_);
}

/// First phase of AutoTileMatmulL0: move canonical split-K reductions outside
/// their M/N output grid before dbC planning or ordinary call-level tiling.
class CanonicalSplitKPreMutator : public IRMutator {
 public:
  CanonicalSplitKPreMutator(const std::unordered_map<const Var*, size_t>& use_counts,
                            std::vector<Diagnostic>& hints)
      : use_counts_(use_counts), hints_(hints) {}

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // Match before recursively mutating children so definition identities still
    // refer to the original function-wide read index. The qualified base visit
    // then reaches canonical triplets in nested source regions; newly generated
    // loops do not match because their fresh Vars are absent from that index and
    // wait for the ordinary AutoTile phase to K-tile their narrowed calls.
    auto rewritten = As<SeqStmts>(RewriteCanonicalSplitKSeq(op, use_counts_, hints_));
    INTERNAL_CHECK_SPAN(rewritten, op->span_)
        << "Internal error: canonical split-K pre-phase did not preserve SeqStmts";
    return IRMutator::VisitStmt_(rewritten);
  }

 private:
  const std::unordered_map<const Var*, size_t>& use_counts_;
  std::vector<Diagnostic>& hints_;
};

FunctionPtr RewriteCanonicalSplitKAcc(const FunctionPtr& func, std::vector<Diagnostic>& hints) {
  CanonicalReadCounter counter;
  counter.VisitStmt(func->body_);
  CanonicalSplitKPreMutator mutator(counter.counts, hints);
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body == func->body_) return func;
  auto rewritten = MutableCopy(func);
  rewritten->body_ = new_body;
  return rewritten;
}

/// Emit one straight-line full-K sub-tile (no K-loop): extract the ``[m_eff, K]``
/// left and ``[K, n_eff]`` right panels, ``tile.matmul``, and hand the
/// ``[m_eff, n_eff]`` result to ``placer``.  ``m_eff`` / ``n_eff`` may be a
/// partial (< m / < n) remainder — this is the boundary-tile emitter for the
/// full-K grid's L-shaped tail (the divisible interior is pipelined instead).
/// Returns the chain after placement.
PlacementState EmitFullKTile(std::vector<StmtPtr>& stmts, const MatmulTiling& t, SubtilePlacer& placer,
                             const PlacementState& chain, int64_t mi, int64_t ni, int64_t m_eff,
                             int64_t n_eff, const std::string& base, int step) {
  const Span sp = t.assign->span_;
  auto& reg = OpRegistry::GetInstance();
  auto sa = BuildExtract(t.lhs, {m_eff, t.K}, MakeIndex(mi, sp), MakeIndex(0, sp), MemorySpace::Left,
                         base + "_ta" + std::to_string(step), sp);
  auto sb = BuildExtract(t.rhs, {t.K, n_eff}, MakeIndex(0, sp), MakeIndex(ni, sp), MemorySpace::Right,
                         base + "_tb" + std::to_string(step), sp);
  stmts.push_back(sa);
  stmts.push_back(sb);
  VarPtr bias_operand;
  if (t.bias) {
    auto bias_ty = As<TileType>(t.bias->GetType());
    if (bias_ty && bias_ty->GetMemorySpace() == MemorySpace::Bias && ni == 0 && n_eff == t.N) {
      bias_operand = t.bias;
    } else {
      auto bias = BuildExtract(t.bias, {1, n_eff}, MakeIndex(0, sp), MakeIndex(ni, sp), MemorySpace::Bias,
                               base + "_tbias" + std::to_string(step), sp);
      stmts.push_back(bias);
      bias_operand = bias->var_;
    }
  }
  auto c_call = bias_operand ? reg.Create("tile.matmul_bias", {sa->var_, sb->var_, bias_operand}, sp)
                             : reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
  auto c_var = std::make_shared<Var>(base + "_tc" + std::to_string(step), c_call->GetType(), sp);
  stmts.push_back(std::make_shared<AssignStmt>(c_var, c_call, sp));
  return placer.PlaceAt(stmts, c_var, MakeIndex(mi, sp), MakeIndex(ni, sp), chain, step);
}

/// Build the full-K (``k == K``) M/N grid as a pipelined **interior** plus a
/// straight-line **tail**, so the downstream ``LowerPipelineLoops`` double-buffers
/// both operand extracts (the latency win the pto-isa A2A3 cost model predicts:
/// hiding the per-sub-tile L1→L0 extract behind the cube keeps it fed).
///
///   # interior: the [0, full_m) x [0, full_n) region tiled by FULL m x n blocks
///   out = for mi in pipeline(0, full_m, m):       # outer (stationary) axis
///       A = extract(lhs, mi, 0, [m, K], Left)
///       out = for ni in pipeline(0, full_n, n):   # inner (moving) axis → B double-buffered
///           B = extract(rhs, 0, ni, [K, n], Right)
///           out = place(matmul(A, B), mi, ni, out)
///   # tail: straight-line partial tiles over the L-shaped boundary
///   for ni in 0..N:           out = place(matmul(A[M-full_m,K], B[K, n_eff]), full_m, ni, out)
///   for mi in 0..full_m:      out = place(matmul(A[m, K], B[K, N-full_n]), mi, full_n, out)
///
/// The interior pipelines with **exact trip counts** (``full_m`` / ``full_n`` are
/// multiples of ``m`` / ``n`` by construction), so no divisibility constraint on
/// ``M`` / ``N`` is needed — any aligned tile the chooser picks works, and the
/// partial boundary is peeled rather than forcing a tiny exact-divisor tile.
/// The outer-loop (stationary) axis is chosen to minimise total operand-extract
/// traffic over the interior grid: A-stationary (rows outer) costs
/// ``P*A + P*Q*B``, B-stationary (cols outer) ``P*Q*A + Q*B`` (``P`` / ``Q`` =
/// interior row/col blocks, ``A`` / ``B`` = the per-panel extract bytes), so the
/// stationary operand is re-extracted once per outer step.  Drives the same
/// ``SubtilePlacer`` as the split-K grid, so the direct-store placement comes
/// out double-buffered.
std::pair<std::vector<StmtPtr>, PlacementState> BuildFullKPipelined(const MatmulTiling& t,
                                                                    SubtilePlacer& placer) {
  const Span sp = t.assign->span_;
  auto& reg = OpRegistry::GetInstance();
  const std::string base = t.assign->var_->name_hint_;
  // A tile never exceeds the problem dims (you cannot tile M into blocks larger
  // than M); the chooser guarantees this, so the interior always has >= 1 block.
  INTERNAL_CHECK_SPAN(t.m <= t.M && t.n <= t.N, sp)
      << "Internal error: full-K tile must not exceed the problem dims; got M=" << t.M << " m=" << t.m
      << " N=" << t.N << " n=" << t.n;

  // Loop order + buffering. When the chooser picked an operand-stationary point
  // (k == K), the stationary operand is the OUTER (held) panel and is SINGLE-
  // buffered — the chooser budgeted it the full L0 buffer (no /2). A-stationary
  // -> A held -> rows outer; B-stationary -> B held -> cols outer. Output-
  // stationary double-buffers both and hoists one operand to the outer loop; the
  // chooser already made that bandwidth-weighted choice while scoring the wall
  // (LoadCycles' min-hoist) and recorded it in `os_holds_a`, so we obey it here
  // rather than re-derive from raw bytes — the two objectives disagree under the
  // ~130:85 L0A:L0B bandwidth ratio, and diverging would emit a different loop
  // order than the wall was scored under.
  const bool a_stationary = t.stationarity == utils::Stationarity::kAStationary;
  const bool b_stationary = t.stationarity == utils::Stationarity::kBStationary;
  const bool stationary_single_buffered = a_stationary || b_stationary;
  const bool row_outer = stationary_single_buffered ? a_stationary : t.os_holds_a;

  // Interior = the region tiled by FULL m x n blocks; the L-shaped partial
  // boundary beyond it is peeled into straight-line tiles below.
  const int64_t full_m = (t.M / t.m) * t.m;
  const int64_t full_n = (t.N / t.n) * t.n;

  std::vector<StmtPtr> stmts;
  PlacementState chain = placer.Init(stmts);
  INTERNAL_CHECK_SPAN(!chain.empty(), sp) << "Internal error: M/N placement state must not be empty";

  // --- Interior: nested pipelined loops over [0, full_m) x [0, full_n) ---
  {
    const int64_t outer_extent = row_outer ? full_m : full_n;
    const int64_t outer_step = row_outer ? t.m : t.n;
    const int64_t inner_extent = row_outer ? full_n : full_m;
    const int64_t inner_step = row_outer ? t.n : t.m;
    auto idx_type = std::make_shared<ScalarType>(DataType::INDEX);
    auto outer_var = std::make_shared<Var>(base + "_o", idx_type, sp);
    auto inner_var = std::make_shared<Var>(base + "_i", idx_type, sp);
    ExprPtr mi = row_outer ? ExprPtr(outer_var) : ExprPtr(inner_var);
    ExprPtr ni = row_outer ? ExprPtr(inner_var) : ExprPtr(outer_var);
    // Every output/scratch chain threads through both loops as an iter-arg; each
    // inner iter-arg is initialised from its matching outer iter-arg.
    std::vector<IterArgPtr> out_outer;
    std::vector<IterArgPtr> out_inner;
    PlacementState inner_state;
    out_outer.reserve(chain.size());
    out_inner.reserve(chain.size());
    inner_state.reserve(chain.size());
    for (size_t state_i = 0; state_i < chain.size(); ++state_i) {
      const std::string suffix = chain.size() == 1 ? "" : "_" + std::to_string(state_i);
      auto outer_arg =
          std::make_shared<IterArg>(base + "_oc" + suffix, chain[state_i]->GetType(), chain[state_i], sp);
      auto inner_arg =
          std::make_shared<IterArg>(base + "_ic" + suffix, chain[state_i]->GetType(), outer_arg, sp);
      out_outer.push_back(outer_arg);
      out_inner.push_back(inner_arg);
      inner_state.push_back(inner_arg);
    }
    auto sa = BuildExtract(t.lhs, {t.m, t.K}, mi, MakeIndex(0, sp), MemorySpace::Left, base + "_a", sp);
    auto sb = BuildExtract(t.rhs, {t.K, t.n}, MakeIndex(0, sp), ni, MemorySpace::Right, base + "_b", sp);
    const AssignStmtPtr& outer_extract = row_outer ? sa : sb;  // stationary panel
    const AssignStmtPtr& inner_extract = row_outer ? sb : sa;  // moving panel
    AssignStmtPtr bias_extract;
    VarPtr bias_operand;
    if (t.bias) {
      auto bias_ty = As<TileType>(t.bias->GetType());
      if (bias_ty && bias_ty->GetMemorySpace() == MemorySpace::Bias && t.n == t.N) {
        bias_operand = t.bias;
      } else {
        bias_extract =
            BuildExtract(t.bias, {1, t.n}, MakeIndex(0, sp), ni, MemorySpace::Bias, base + "_bias", sp);
        bias_operand = bias_extract->var_;
      }
    }
    auto c_call = bias_operand ? reg.Create("tile.matmul_bias", {sa->var_, sb->var_, bias_operand}, sp)
                               : reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
    auto c_var = std::make_shared<Var>(base + "_c", c_call->GetType(), sp);
    std::vector<StmtPtr> inner_body{inner_extract};
    if (bias_extract) inner_body.push_back(bias_extract);
    inner_body.push_back(std::make_shared<AssignStmt>(c_var, c_call, sp));
    PlacementState inner_chain = placer.PlaceAt(inner_body, c_var, mi, ni, inner_state, /*step=*/0);
    std::vector<ExprPtr> inner_yields(inner_chain.begin(), inner_chain.end());
    inner_body.push_back(std::make_shared<YieldStmt>(std::move(inner_yields), sp));
    // overlap_stores stays false: the one-accumulator schedule
    // (matmul_i, store_i, matmul_{i+1}, store_{i+1}) drains each L0C result before
    // the next matmul overwrites it.  dbC=2 (double_buffer_c) instead sets the
    // stronger double_buffer_c attr, which floats *both* stores below *both*
    // matmuls (matmul c, matmul c₁, store c, store c₁) so the two [m, n] Acc tiles
    // stay co-live; the chooser budgeted them at L0C/2 so both fit. Under PTOAS,
    // MemoryReuse is skipped and ptoas places the distinct live ranges. Under
    // the PyPTO opt-in, flat depth-2 pipeline membership keeps MemoryReuse from
    // coalescing the pair. In both cases tile i's FIXPIPE drain overlaps tile
    // i+1's MAD. The moving-operand extract is double-buffered (Load tier,
    // hoisted) in both schedules.
    std::vector<std::pair<std::string, std::any>> inner_attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2},
                                                                 {kPipelineOverlapStoresAttr, false}};
    // Only dbC=2 loops carry the attr (absent ⇒ false), so non-dbC=2 emit is
    // unchanged.  It lifts the moving-loop stores above both matmuls in
    // CanonicalizeIOOrder to keep the two L0C accumulators co-live.
    if (t.double_buffer_c) inner_attrs.emplace_back(kPipelineDoubleBufferCAttr, true);
    std::vector<VarPtr> inner_rvs;
    inner_rvs.reserve(chain.size());
    for (size_t state_i = 0; state_i < chain.size(); ++state_i) {
      const std::string suffix = chain.size() == 1 ? "" : "_" + std::to_string(state_i);
      inner_rvs.push_back(std::make_shared<Var>(base + "_irv" + suffix, chain[state_i]->GetType(), sp));
    }
    auto inner_for = std::make_shared<ForStmt>(inner_var, MakeIndex(0, sp), MakeIndex(inner_extent, sp),
                                               MakeIndex(inner_step, sp), out_inner,
                                               SeqStmts::Flatten(std::move(inner_body), sp), inner_rvs, sp,
                                               ForKind::Pipeline, std::move(inner_attrs));
    std::vector<ExprPtr> outer_yields(inner_rvs.begin(), inner_rvs.end());
    std::vector<StmtPtr> outer_body{outer_extract, inner_for,
                                    std::make_shared<YieldStmt>(std::move(outer_yields), sp)};
    // Operand-stationary: the outer loop carries the SINGLE-buffered stationary
    // panel, so it is Sequential — a Pipeline stage=2 outer would double-buffer the
    // held operand (2x its full-L0 budget -> overflow). The inner (moving) loop
    // stays pipelined. Output-stationary double-buffers both -> outer Pipeline.
    const ForKind outer_kind = stationary_single_buffered ? ForKind::Sequential : ForKind::Pipeline;
    std::vector<std::pair<std::string, std::any>> outer_attrs;
    if (!stationary_single_buffered) {
      outer_attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2}, {kPipelineOverlapStoresAttr, false}};
    }
    PlacementState outer_rvs;
    outer_rvs.reserve(chain.size());
    for (size_t state_i = 0; state_i < chain.size(); ++state_i) {
      const std::string suffix = chain.size() == 1 ? "" : "_" + std::to_string(state_i);
      outer_rvs.push_back(std::make_shared<Var>(base + "_orv" + suffix, chain[state_i]->GetType(), sp));
    }
    auto outer_for = std::make_shared<ForStmt>(
        outer_var, MakeIndex(0, sp), MakeIndex(outer_extent, sp), MakeIndex(outer_step, sp), out_outer,
        SeqStmts::Flatten(std::move(outer_body), sp), outer_rvs, sp, outer_kind, std::move(outer_attrs));
    stmts.push_back(outer_for);
    chain = std::move(outer_rvs);
  }

  // --- Tail: straight-line partial tiles for the L-shaped boundary ---
  // Bottom strip [full_m, M) x [0, N) (covers the corner), then right strip
  // [0, full_m) x [full_n, N).  Either is empty when its dim divides evenly.
  // step 0 is used by the interior placement; the tail continues from step 1.
  int tail_step = 1;
  for (int64_t ni = 0; full_m < t.M && ni < t.N; ni += t.n) {
    chain = EmitFullKTile(stmts, t, placer, chain, full_m, ni, t.M - full_m, std::min<int64_t>(t.n, t.N - ni),
                          base, tail_step++);
  }
  for (int64_t mi = 0; full_n < t.N && mi < full_m; mi += t.m) {
    chain = EmitFullKTile(stmts, t, placer, chain, mi, full_n, t.m, t.N - full_n, base, tail_step++);
  }
  return {std::move(stmts), chain};
}

/// Build the split-K M/N grid: ``ceil(M/m) x ceil(N/n)`` sub-tiles, each a
/// pipelined K-loop (``BuildKLoopRewrite``) over the ``[m_eff, n_eff]`` output,
/// handed to ``placer`` for placement.  Used when K spans >= 2 L0 blocks, so the
/// operand panel does not fit L0 and cannot stay resident across sub-tiles
/// (unlike the full-K pipelined path).  N-major traversal preserves the
/// historical sub-tile ordering / naming.
std::pair<std::vector<StmtPtr>, PlacementState> BuildSplitKGrid(const MatmulTiling& t,
                                                                SubtilePlacer& placer) {
  const Span sp = t.assign->span_;
  const std::string base = t.assign->var_->name_hint_;
  const int64_t num_m = (t.M + t.m - 1) / t.m;
  const int64_t num_n = (t.N + t.n - 1) / t.n;

  std::vector<StmtPtr> stmts;
  PlacementState chain = placer.Init(stmts);
  int step = 0;
  for (int64_t nj = 0; nj < num_n; ++nj) {
    const int64_t ni = nj * t.n;
    const int64_t n_eff = std::min<int64_t>(t.n, t.N - ni);
    for (int64_t mj = 0; mj < num_m; ++mj) {
      const int64_t mi = mj * t.m;
      const int64_t m_eff = std::min<int64_t>(t.m, t.M - mi);
      const std::string tbase = base + "_t" + std::to_string(step);

      auto inner = BuildKLoopRewrite(MakeKLoop(t, MakeIndex(mi, sp), MakeIndex(ni, sp), m_eff, n_eff, tbase));
      for (auto& s : inner.stmts) stmts.push_back(std::move(s));
      chain = placer.PlaceAt(stmts, inner.return_var, MakeIndex(mi, sp), MakeIndex(ni, sp), chain, step);
      ++step;
    }
  }
  return {std::move(stmts), chain};
}

struct PreparedMNTiling {
  MatmulTiling tiling;
  std::vector<StmtPtr> prologue;
};

bool ValidateBiasMNTiling(const MatmulTiling& t, std::vector<Diagnostic>& hints) {
  if (!t.bias || t.n == t.N) return true;
  auto bias_ty = As<TileType>(t.bias->GetType());
  if (!bias_ty || bias_ty->GetMemorySpace() != MemorySpace::Bias) return true;
  hints.emplace_back(
      DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
      "tile.matmul_bias needs N tiling, but its source is already resident in the architectural Bias "
      "buffer; forming an N sub-window would require an unsupported Bias-to-Bias extract, so the call is "
      "left untouched (keep the pre-inference bias source in Mat)",
      t.assign->span_);
  return false;
}

/// Stage a Vec-resident left operand into one full Mat tile before expanding
/// the M/N grid. Building the move in each output sub-tile would repeat the
/// cross-core transfer and create a different logical operand per grid cell;
/// the original value is one logical matrix and is staged exactly once.
/// `other_mat_bytes` accounts for a compiler-created output scratch that must
/// coexist with the staged operand. Unknown footprints fail closed.
std::optional<PreparedMNTiling> PrepareMNTilingOperands(const MatmulTiling& t, uint64_t other_mat_bytes,
                                                        std::vector<Diagnostic>& hints) {
  if (!ValidateBiasMNTiling(t, hints)) return std::nullopt;
  PreparedMNTiling prepared{t, {}};
  if (!t.stage_lhs_to_mat) return prepared;

  const Span sp = t.assign->span_;
  auto lhs_mat = BuildMoveToMat(t.lhs, t.assign->var_->name_hint_ + "_l0_lmat", sp);
  auto lhs_mat_ty = As<TileType>(lhs_mat->var_->GetType());
  const auto* ctx = PassContext::Current();
  const auto* handler = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  INTERNAL_CHECK_SPAN(handler, sp) << "Internal error: BackendHandler is null";
  auto lhs_bytes = utils::StaticPhysicalAllocationBytes(lhs_mat_ty, MemorySpace::Mat, handler);
  const uint64_t mat_capacity = handler->GetMatCapacityBytes();
  const bool addition_overflows =
      lhs_bytes && *lhs_bytes > std::numeric_limits<uint64_t>::max() - other_mat_bytes;
  if (!lhs_bytes || addition_overflows || *lhs_bytes + other_mat_bytes > mat_capacity) {
    hints.emplace_back(
        DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
        "tile.matmul with a Vec left operand needs M/N tiling, but staging the complete logical left "
        "operand in Mat together with its compiler-created output materialization cannot be proven to fit "
        "the backend Mat capacity; left untouched",
        sp);
    return std::nullopt;
  }

  prepared.tiling.lhs = lhs_mat->var_;
  prepared.tiling.stage_lhs_to_mat = false;
  prepared.prologue.push_back(lhs_mat);
  return prepared;
}

struct PreparedLinearChain {
  LinearMatmulChain chain;
  std::vector<StmtPtr> prologue;
};

std::optional<PreparedLinearChain> PrepareLinearChainOperands(const LinearMatmulChain& chain,
                                                              uint64_t other_mat_bytes,
                                                              std::vector<Diagnostic>& hints) {
  INTERNAL_CHECK(!chain.stages.empty()) << "Internal error: cannot prepare an empty matmul chain";
  PreparedLinearChain prepared{chain, {}};
  const Span sp = chain.stages.front().assign->span_;
  const auto* ctx = PassContext::Current();
  const auto* handler = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  INTERNAL_CHECK_SPAN(handler, sp) << "Internal error: BackendHandler is null";
  const uint64_t mat_capacity = handler->GetMatCapacityBytes();
  uint64_t required = other_mat_bytes;
  std::unordered_map<const Var*, VarPtr> staged;

  for (size_t i = 0; i < prepared.chain.stages.size(); ++i) {
    auto& stage = prepared.chain.stages[i];
    if (!ValidateBiasMNTiling(stage, hints)) return std::nullopt;
    if (!stage.stage_lhs_to_mat) continue;
    if (auto it = staged.find(stage.lhs.get()); it != staged.end()) {
      stage.lhs = it->second;
      stage.stage_lhs_to_mat = false;
      continue;
    }
    auto lhs_mat = BuildMoveToMat(stage.lhs, stage.assign->var_->name_hint_ + "_l0_lmat", sp);
    auto lhs_mat_ty = As<TileType>(lhs_mat->var_->GetType());
    auto bytes = utils::StaticPhysicalAllocationBytes(lhs_mat_ty, MemorySpace::Mat, handler);
    if (!bytes || required > std::numeric_limits<uint64_t>::max() - *bytes ||
        required + *bytes > mat_capacity) {
      hints.emplace_back(
          DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
          "linear matmul/matmul_acc chain needs M/N tiling, but its complete Vec operands and output "
          "materialization cannot be proven to fit Mat capacity; left untouched",
          sp);
      return std::nullopt;
    }
    required += *bytes;
    staged.emplace(stage.lhs.get(), lhs_mat->var_);
    stage.lhs = lhs_mat->var_;
    stage.stage_lhs_to_mat = false;
    prepared.prologue.push_back(lhs_mat);
  }
  return prepared;
}

/// Emit one static K block for a linear accumulator chain. ``acc == nullptr``
/// starts the chain with matmul[/bias]; otherwise the block accumulates in
/// place. The explicit K window lets a split-K root start with one fresh block
/// and feed a uniform matmul_acc loop, avoiding a fresh-vs-acc if-phi whose
/// conservative MemRef join would otherwise reserve a second full L0C buffer.
VarPtr EmitLinearChainKBlock(std::vector<StmtPtr>& stmts, const MatmulTiling& t, const VarPtr& acc,
                             int64_t mi, int64_t ni, int64_t m_eff, int64_t n_eff, int64_t k_offset,
                             int64_t k_extent, const std::string& base) {
  const Span sp = t.assign->span_;
  INTERNAL_CHECK_SPAN(!t.stage_lhs_to_mat, sp)
      << "Internal error: linear-chain Vec operands must be staged before grid expansion";
  INTERNAL_CHECK_SPAN(k_offset >= 0 && k_extent > 0 && k_offset <= t.K - k_extent, sp)
      << "Internal error: linear-chain K block [" << k_offset << ", " << k_offset + k_extent
      << ") is outside K=" << t.K;
  auto sa = BuildExtract(t.lhs, {m_eff, k_extent}, MakeIndex(mi, sp), MakeIndex(k_offset, sp),
                         MemorySpace::Left, base + "_a", sp);
  auto sb = BuildExtract(t.rhs, {k_extent, n_eff}, MakeIndex(k_offset, sp), MakeIndex(ni, sp),
                         MemorySpace::Right, base + "_b", sp);
  stmts.push_back(sa);
  stmts.push_back(sb);

  VarPtr bias_operand;
  if (t.bias) {
    auto bias_ty = As<TileType>(t.bias->GetType());
    if (bias_ty && bias_ty->GetMemorySpace() == MemorySpace::Bias && ni == 0 && n_eff == t.N) {
      bias_operand = t.bias;
    } else {
      auto bias = BuildExtract(t.bias, {1, n_eff}, MakeIndex(0, sp), MakeIndex(ni, sp), MemorySpace::Bias,
                               base + "_bias", sp);
      stmts.push_back(bias);
      bias_operand = bias->var_;
    }
  }

  auto& reg = OpRegistry::GetInstance();
  ExprPtr call;
  if (acc) {
    call = reg.Create("tile.matmul_acc", {acc, sa->var_, sb->var_}, sp);
  } else if (bias_operand) {
    call = reg.Create("tile.matmul_bias", {sa->var_, sb->var_, bias_operand}, sp);
  } else {
    call = reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
  }
  auto result = std::make_shared<Var>(base + "_c", call->GetType(), sp);
  stmts.push_back(std::make_shared<AssignStmt>(result, call, sp));
  return result;
}

std::pair<std::vector<StmtPtr>, PlacementState> BuildLinearChainGrid(const PreparedLinearChain& prepared,
                                                                     SubtilePlacer& placer) {
  const auto& chain = prepared.chain;
  INTERNAL_CHECK(!chain.stages.empty()) << "Internal error: cannot emit an empty matmul chain";
  const MatmulTiling& root = chain.stages.front();
  const Span sp = root.assign->span_;
  const int64_t num_m = (root.M + root.m - 1) / root.m;
  const int64_t num_n = (root.N + root.n - 1) / root.n;
  std::vector<StmtPtr> stmts = prepared.prologue;
  PlacementState state = placer.Init(stmts);
  int step = 0;
  for (int64_t nj = 0; nj < num_n; ++nj) {
    const int64_t ni = nj * root.n;
    const int64_t n_eff = std::min<int64_t>(root.n, root.N - ni);
    for (int64_t mj = 0; mj < num_m; ++mj) {
      const int64_t mi = mj * root.m;
      const int64_t m_eff = std::min<int64_t>(root.m, root.M - mi);
      VarPtr acc;
      for (size_t stage_i = 0; stage_i < chain.stages.size(); ++stage_i) {
        MatmulTiling stage = chain.stages[stage_i];
        stage.acc_init = acc;
        const std::string base =
            root.assign->var_->name_hint_ + "_t" + std::to_string(step) + "_s" + std::to_string(stage_i);
        if (stage.k < stage.K && !acc) {
          // A fresh split-K root starts outside the loop. The remainder is a
          // pure matmul_acc reduction pinned to that first result, so both the
          // PyPTO and PTOAS planners see one physical accumulator rather than
          // a conservative fresh/acc if-phi join plus its continuation.
          acc = EmitLinearChainKBlock(stmts, stage, nullptr, mi, ni, m_eff, n_eff,
                                      /*k_offset=*/0, /*k_extent=*/stage.k, base + "_first");
          const int64_t remaining_k = stage.K - stage.k;
          if (remaining_k <= stage.k) {
            acc = EmitLinearChainKBlock(stmts, stage, acc, mi, ni, m_eff, n_eff,
                                        /*k_offset=*/stage.k, /*k_extent=*/remaining_k, base + "_rest");
          } else {
            MatmulTiling remainder = stage;
            remainder.K = remaining_k;
            remainder.acc_init = acc;
            remainder.bias = nullptr;  // bias was applied by the fresh first block
            auto loop =
                MakeKLoop(remainder, MakeIndex(mi, sp), MakeIndex(ni, sp), m_eff, n_eff, base + "_rest");
            loop.k_base = stage.k;
            auto rewrite = BuildKLoopRewrite(loop);
            stmts.insert(stmts.end(), std::make_move_iterator(rewrite.stmts.begin()),
                         std::make_move_iterator(rewrite.stmts.end()));
            acc = rewrite.return_var;
          }
        } else if (stage.k < stage.K) {
          auto rewrite =
              BuildKLoopRewrite(MakeKLoop(stage, MakeIndex(mi, sp), MakeIndex(ni, sp), m_eff, n_eff, base));
          stmts.insert(stmts.end(), std::make_move_iterator(rewrite.stmts.begin()),
                       std::make_move_iterator(rewrite.stmts.end()));
          acc = rewrite.return_var;
        } else {
          acc = EmitLinearChainKBlock(stmts, stage, acc, mi, ni, m_eff, n_eff,
                                      /*k_offset=*/0, /*k_extent=*/stage.K, base);
        }
      }
      INTERNAL_CHECK_SPAN(acc, sp) << "Internal error: linear matmul chain produced no Acc result";
      state = placer.PlaceAt(stmts, acc, MakeIndex(mi, sp), MakeIndex(ni, sp), state, step);
      ++step;
    }
  }
  return {std::move(stmts), std::move(state)};
}

/// Try to fold a Mat-resident plain ``tile.matmul`` whose [M, N] output exceeds
/// L0c into a ``ceil(M/m) x ceil(N/n)`` grid of sub-tile matmuls, each computing
/// an ``[m, n]`` (partial on the boundary) Acc result.  Operands are already
/// Mat-resident, so only the output Acc overflows; sub-tiling keeps every Acc
/// tile within L0c. This helper handles the direct-store consumer:
///
///   * **Direct-store** — the sole consumer is a 2D ``tile.store(c, base, out)``:
///     each sub-tile stores straight to ``out[mi:, ni:]`` (the DDR-output case
///     our solver kernels need).  The store is folded in and emitted at the
///     store site.
///
/// The Mat-scratch alternative is handled earlier by ``TryFoldMatScratch``.
/// ``result_uses`` / ``store_stmt`` come from the precomputed SiblingIndex.
/// Returns nullopt (with a PerfHint) when neither placement applies — an
/// arbitrary ``matmul_acc`` with a caller-supplied [M, N] accumulator and
/// mixed/non-matmul on-chip consumers are deferred. A Vec left operand is
/// staged into Mat once before this grid. The
/// canonical frontend split-K create/pipeline/store form is handled earlier at
/// the enclosing-loop level.
std::optional<MNFold> TryFoldMNTiling(const MatmulTiling& t, int result_uses, const AssignStmt* store_stmt,
                                      std::vector<Diagnostic>& hints) {
  const Span sp = t.assign->span_;
  auto skip = [&](const std::string& msg) -> std::optional<MNFold> {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006", msg, sp);
    return std::nullopt;
  };

  if (t.is_acc()) {
    return skip(
        "oversized tile.matmul_acc does not match the canonical create -> split-K pipeline -> store "
        "form handled by loop-level M/N tiling; slicing this caller-owned [M, N] accumulator is "
        "unsupported, so the call is left untouched");
  }
  // K spans >= 2 L0 blocks → pipelined K-loop per sub-tile (BuildSplitKGrid);
  // k == K (full K fits L0a/L0b) → pipelined interior + straight-line partial
  // tail (BuildFullKPipelined).  Either grid drives the chosen SubtilePlacer;
  // neither requires m | M or n | N (the full-K tail peels the partial boundary).
  // The chooser returns k == K exactly when the full K reduction fits one L0
  // block (possibly an unaligned k < align under allow_k_boundary) — only then
  // is there no K-loop.  A non-divisor k with k < K < 2k still needs the
  // K-loop+peel, so test k == K rather than the integer-division proxy K/k < 2.
  const bool full_k = t.k == t.K;

  // Direct-store: the sole consumer is a 2D tile.store.  The grid is emitted
  // later at the store site, where the caller re-applies the then-current remap
  // — so a prior fold that redefined this output is rewritten correctly (a
  // stale-output SSA guard); resolving it here would miss folds emitted before
  // this one.
  if (store_stmt && result_uses == 1) {
    auto store_call = As<Call>(store_stmt->value_);
    INTERNAL_CHECK_SPAN(store_call, store_stmt->span_)
        << "Internal error: SiblingIndex store_of mapped a non-Call AssignStmt";
    auto offs = As<MakeTuple>(store_call->args_[1]);
    if (!offs || offs->elements_.size() != 2) {
      return skip("tile.store offsets are not a 2D tuple — M/N fold not applicable; left untouched");
    }
    auto out_in = AsVarLike(store_call->args_[2]);
    if (!out_in) {
      return skip(
          "tile.store target is not a simple tensor variable — M/N fold not applicable; left untouched");
    }
    DirectGmPlacer placer(offs->elements_[0], offs->elements_[1], out_in, store_call->kwargs_,
                          store_call->attrs_, sp);
    auto prepared = PrepareMNTilingOperands(t, /*other_mat_bytes=*/0, hints);
    if (!prepared) return std::nullopt;
    auto [grid, last_out] =
        full_k ? BuildFullKPipelined(prepared->tiling, placer) : BuildSplitKGrid(prepared->tiling, placer);
    std::vector<StmtPtr> stmts = std::move(prepared->prologue);
    stmts.insert(stmts.end(), std::make_move_iterator(grid.begin()), std::make_move_iterator(grid.end()));
    INTERNAL_CHECK_SPAN(last_out.size() == 1, sp)
        << "Internal error: direct-store M/N placement must return one tensor";
    return MNFold{std::move(stmts), last_out.front(), store_stmt->var_, store_stmt};
  }

  return skip(
      "tile.matmul output exceeds L0c but its result is not consumed by a single 2D tile.store "
      "(direct-store) — a result consumed on-chip (chained matmul / elementwise), stored-and-reused, "
      "or fed to a non-store consumer is deferred; left untouched");
}

/// True when a ``tile.cast`` may be folded into a cube FIXPIPE Acc->Mat writeback
/// (``pto.tinsert``) instead of a standalone Vector ``pto.tcvt``.  FIXPIPE narrows
/// only ``f32 -> bf16`` / ``f32 -> f16`` (the ``F322BF16`` / ``F322F16`` writeback
/// modes; an ``int32`` source would need a *scaled dequant*, not a plain cast) and
/// applies a single fixed tie rule: **round-to-nearest-even** — the pto-isa CPU
/// reference narrows via ``std::bfloat16_t`` and ``pto.tinsert`` carries no
/// ``rmode``.  So fold only an ``f32`` source cast to ``bf16``/``f16`` whose round
/// mode is ``RINT`` (round-half-to-even).  ``ROUND`` (round-half-*away*, the
/// frontend default) and the directional/truncating modes round ties differently,
/// so they must keep the Vector cast — it lowers to ``pto.tcvt``, the only path
/// that honors the requested ``rmode``.
bool CastFoldableToFixpipeMat(const CallPtr& cast, const TileTypePtr& src_ty, DataType dst_dtype) {
  if (!cast || !src_ty) return false;
  if (src_ty->dtype_ != DataType::FP32) return false;
  if (dst_dtype != DataType::BF16 && dst_dtype != DataType::FP16) return false;
  // ``tile.cast`` "mode" (see src/ir/op/tile_ops/unary.cpp): NONE(0), RINT(1),
  // ROUND(2), FLOOR(3), CEIL(4), TRUNC(5), ODD(6).  FIXPIPE's fixed narrowing is
  // round-half-to-even == RINT; only that mode matches.  A missing "mode" defaults
  // to the frontend's ROUND (ties away) — not foldable.
  constexpr int kRoundRint = 1;
  constexpr int kRoundRound = 2;
  const int mode = cast->GetKwarg<int>("mode", kRoundRound);
  return mode == kRoundRint;
}

struct PreparedMatScratch {
  PreparedMNTiling operands;
  uint64_t physical_bytes = 0;
};

std::optional<uint64_t> ValidateMatScratch(const MatmulTiling& t, DataType scratch_dtype,
                                           std::vector<Diagnostic>& hints) {
  const Span sp = t.assign->span_;
  const auto* ctx = PassContext::Current();
  const auto* handler = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  INTERNAL_CHECK_SPAN(handler, sp) << "Internal error: BackendHandler is null";

  if (handler->RequiresLowPrecisionMatScratch() && scratch_dtype != DataType::BF16 &&
      scratch_dtype != DataType::FP16) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-009",
                       "chained-matmul [" + std::to_string(t.M) + ", " + std::to_string(t.N) +
                           "] intermediate is " + scratch_dtype.ToString() +
                           "; this backend's oversized on-chip Mat scratch needs a bf16/f16 "
                           "intermediate (cast the matmul result to bf16 before the consumer "
                           "matmul, the cube's native operand precision) — left on the deferred path",
                       sp);
    return std::nullopt;
  }

  auto scratch_call =
      OpRegistry::GetInstance().Create("tile.create", {MakeIndexTuple({t.M, t.N}, sp)},
                                       {{"dtype", scratch_dtype}, {"target_memory", MemorySpace::Mat}}, sp);
  auto scratch_ty = As<TileType>(scratch_call->GetType());
  auto scratch_bytes = utils::StaticPhysicalAllocationBytes(scratch_ty, MemorySpace::Mat, handler);
  const uint64_t mat_capacity = handler->GetMatCapacityBytes();
  if (!scratch_bytes || *scratch_bytes > mat_capacity) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
                       "chained-matmul [" + std::to_string(t.M) + ", " + std::to_string(t.N) +
                           "] physical Mat scratch footprint " +
                           (scratch_bytes ? "(" + std::to_string(*scratch_bytes) + " bytes) exceeds"
                                          : "cannot be proven within") +
                           " Mat capacity (" + std::to_string(mat_capacity) +
                           " bytes); left on the deferred path",
                       sp);
    return std::nullopt;
  }

  return scratch_bytes;
}

/// Validate the backend's Acc->Mat path and physical Mat capacity, then prepare
/// any one-time Vec->Mat operand staging that must coexist with the scratch.
/// This is shared by Mat-only and GM+Mat composite materialization so their
/// legality contract cannot drift.
std::optional<PreparedMatScratch> PrepareMatScratch(const MatmulTiling& t, DataType scratch_dtype,
                                                    std::vector<Diagnostic>& hints) {
  auto scratch_bytes = ValidateMatScratch(t, scratch_dtype, hints);
  if (!scratch_bytes) return std::nullopt;
  auto operands = PrepareMNTilingOperands(t, *scratch_bytes, hints);
  if (!operands) return std::nullopt;
  return PreparedMatScratch{std::move(*operands), *scratch_bytes};
}

/// Try to fold a Mat-resident plain ``tile.matmul`` whose [M, N] output exceeds
/// L0c into a Mat-scratch grid when the result is consumed *entirely* at
/// matmul-operand positions (a chained matmul reads it on-chip).  Each sub-tile is
/// assembled into an L1/Mat scratch (``MatScratchPlacer``) instead of stored to a
/// DDR tensor, keeping the whole result on-chip; the caller remaps the matmul
/// result Var to the returned scratch Var.  Returns the grid stmts + scratch Var.
///
/// Both K-split (unrolled, constant offsets) and full-K (pipelined, loop-variable
/// offsets) are supported: ``tile.assemble`` only needs a literal ``MakeTuple``
/// offset whose *elements* may be loop variables (`ValidateIndexTupleElements`
/// requires index-typed elements, not constants). Arbitrary ``matmul_acc``
/// stays deferred; a Vec left operand is staged once before the grid. The
/// canonical split-K form is handled before this local call-level fold.
std::optional<std::pair<std::vector<StmtPtr>, VarPtr>> TryFoldMatScratch(const MatmulTiling& t,
                                                                         int result_uses, int operand_uses,
                                                                         DataType scratch_dtype,
                                                                         std::vector<Diagnostic>& hints) {
  const Span sp = t.assign->span_;
  // Arbitrary matmul_acc is deferred (the direct-store path emits its hint).
  // Canonical split-K is rewritten at the enclosing-loop level.
  if (t.is_acc()) return std::nullopt;
  // Every use must be a matmul operand: a non-operand use (store, elementwise,
  // matmul_acc accumulator) means substituting an upstream Mat scratch is illegal.
  if (result_uses < 1 || operand_uses != result_uses) return std::nullopt;
  auto prepared = PrepareMatScratch(t, scratch_dtype, hints);
  if (!prepared) return std::nullopt;
  const std::string base = t.assign->var_->name_hint_ + "_mat";
  MatScratchPlacer placer(t.M, t.N, scratch_dtype, base, sp);
  // K-split (K spans >= 2 L0 blocks) → unrolled per-sub-tile K-loop grid; full-K →
  // the pipelined interior + straight-line tail.  Both drive MatScratchPlacer,
  // which assembles each sub-tile into the L1/Mat scratch (tile.assemble accepts
  // constant or loop-variable offsets).  A non-divisor k with k < K < 2k still needs
  // the K-loop + peel (BuildSplitKGrid), so dispatch on k == K rather than the
  // integer-division proxy K/k < 2 — which would mis-route a split-K tile to the
  // full-K [m,K]/[K,n] emitter and blow the L0A/L0B budget.
  // dbC=2 works on the Mat-scratch path too: the Acc->Mat drain is `tile.assemble`,
  // which CanonicalizeIOOrder floats above the compute tier under the dbC attr (same
  // as tile.store for the direct-store path), keeping the two accumulators co-live.
  // BuildFullKPipelined attaches the attr when t.double_buffer_c; the split-K grid
  // never carries it.  (The Acc->Mat drain is cheaper than Acc->GM, so the hiding
  // upside is smaller here, but the mechanism is the same.)
  const bool full_k = t.k == t.K;
  auto [grid, scratch] = full_k ? BuildFullKPipelined(prepared->operands.tiling, placer)
                                : BuildSplitKGrid(prepared->operands.tiling, placer);
  std::vector<StmtPtr> stmts = std::move(prepared->operands.prologue);
  stmts.insert(stmts.end(), std::make_move_iterator(grid.begin()), std::make_move_iterator(grid.end()));
  INTERNAL_CHECK_SPAN(scratch.size() == 1, sp)
      << "Internal error: Mat-scratch M/N placement must return one scratch";
  return std::make_pair(std::move(stmts), scratch.front());
}

/// Materialize one oversized logical matmul result to both of the destinations
/// already expressed by the source program: a GM store and downstream Mat-safe
/// matmul operand reads. The compiler creates the Mat scratch and fans each
/// legal Acc sub-tile out to both sinks; the source program never needs to know
/// that M/N tiling or `tile.assemble` exists.
std::optional<MNFold> TryFoldStoredAndReused(const MatmulTiling& t, const AssignStmt* store_stmt,
                                             const Var* materialized_old_var, DataType scratch_dtype,
                                             const Var* dropped_def, const SiblingIndex& index,
                                             std::vector<Diagnostic>& hints) {
  if (t.is_acc() || !store_stmt || !materialized_old_var) return std::nullopt;
  const Span sp = t.assign->span_;

  auto uses_it = index.use_counts.find(materialized_old_var);
  auto operands_it = index.matmul_operand_uses.find(materialized_old_var);
  const int uses = uses_it == index.use_counts.end() ? 0 : uses_it->second;
  const int operand_uses = operands_it == index.matmul_operand_uses.end() ? 0 : operands_it->second;
  if (uses < 1 || uses != operand_uses) return std::nullopt;

  auto store_pos_it = index.positions.find(store_stmt);
  auto consumer_pos_it = index.matmul_operand_positions.find(materialized_old_var);
  INTERNAL_CHECK_SPAN(store_pos_it != index.positions.end(), sp)
      << "Internal error: stored-and-reused store is absent from its sibling index";
  INTERNAL_CHECK_SPAN(consumer_pos_it != index.matmul_operand_positions.end(), sp)
      << "Internal error: Mat-safe consumer positions are absent from their sibling index";
  for (size_t consumer_pos : consumer_pos_it->second) {
    if (consumer_pos <= store_pos_it->second) {
      hints.emplace_back(
          DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
          "tile.matmul output exceeds L0c and is both stored and reused on-chip, but an on-chip consumer "
          "precedes the store where the composite GM/Mat materialization would be emitted; left untouched",
          sp);
      return std::nullopt;
    }
  }

  auto store_call = As<Call>(store_stmt->value_);
  INTERNAL_CHECK_SPAN(store_call && IsOp(store_call, "tile.store") && store_call->args_.size() == 3,
                      store_stmt->span_)
      << "Internal error: stored-and-reused placement received a non-store consumer";
  auto offsets = As<MakeTuple>(store_call->args_[1]);
  auto out_in = AsVarLike(store_call->args_[2]);
  if (!offsets || offsets->elements_.size() != 2 || !out_in) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
                       "stored-and-reused tile.matmul needs a 2D tile.store with a simple tensor target; "
                       "left untouched",
                       sp);
    return std::nullopt;
  }

  auto prepared = PrepareMatScratch(t, scratch_dtype, hints);
  if (!prepared) return std::nullopt;

  std::vector<std::unique_ptr<SubtilePlacer>> children;
  children.push_back(std::make_unique<DirectGmPlacer>(offsets->elements_[0], offsets->elements_[1], out_in,
                                                      store_call->kwargs_, store_call->attrs_, sp));
  children.push_back(
      std::make_unique<MatScratchPlacer>(t.M, t.N, scratch_dtype, t.assign->var_->name_hint_ + "_mat", sp));
  CompositeSubtilePlacer placer(std::move(children), sp);

  const bool full_k = t.k == t.K;
  auto [grid, state] = full_k ? BuildFullKPipelined(prepared->operands.tiling, placer)
                              : BuildSplitKGrid(prepared->operands.tiling, placer);
  INTERNAL_CHECK_SPAN(state.size() == 2, sp)
      << "Internal error: GM+Mat materialization must return one state value per destination";
  std::vector<StmtPtr> stmts = std::move(prepared->operands.prologue);
  stmts.insert(stmts.end(), std::make_move_iterator(grid.begin()), std::make_move_iterator(grid.end()));
  return MNFold{std::move(stmts),     state[0], store_stmt->var_, store_stmt,
                materialized_old_var, state[1], dropped_def};
}

struct LinearChainFold {
  std::vector<StmtPtr> stmts;
  const AssignStmt* store = nullptr;
  VarPtr old_store_result;
  VarPtr new_store_result;
  const Var* materialized_old_var = nullptr;
  VarPtr materialized_new_var;
  const Var* dropped_def = nullptr;
};

std::optional<LinearChainFold> TryFoldLinearMatmulChain(const LinearMatmulChain& chain,
                                                        const SiblingIndex& index,
                                                        std::vector<Diagnostic>& hints) {
  INTERNAL_CHECK(!chain.stages.empty() && chain.terminal) << "Internal error: invalid linear matmul chain";
  const MatmulTiling& root = chain.stages.front();
  const Var* terminal = chain.terminal;
  auto terminal_ty = As<TileType>(terminal->GetType());
  INTERNAL_CHECK_SPAN(terminal_ty, root.assign->span_)
      << "Internal error: linear matmul chain terminal is not a tile";

  const int terminal_uses = index.use_counts.count(terminal) ? index.use_counts.at(terminal) : 0;
  const int terminal_operand_uses =
      index.matmul_operand_uses.count(terminal) ? index.matmul_operand_uses.at(terminal) : 0;
  auto stores_it = index.stores_of.find(terminal);
  const AssignStmt* store = stores_it != index.stores_of.end() && stores_it->second.size() == 1
                                ? stores_it->second.front()
                                : nullptr;

  const Var* cast_result = nullptr;
  const Var* dropped_cast = nullptr;
  DataType cast_dtype = terminal_ty->dtype_;
  int cast_uses = 0;
  int cast_operand_uses = 0;
  if (auto cast_it = index.cast_of.find(terminal); cast_it != index.cast_of.end()) {
    const Var* cb = cast_it->second->var_.get();
    auto cb_ty = As<TileType>(cb->GetType());
    auto cast_call = As<Call>(cast_it->second->value_);
    if (cb_ty && CastFoldableToFixpipeMat(cast_call, terminal_ty, cb_ty->dtype_)) {
      cast_result = cb;
      dropped_cast = cb;
      cast_dtype = cb_ty->dtype_;
      cast_uses = index.use_counts.count(cb) ? index.use_counts.at(cb) : 0;
      cast_operand_uses = index.matmul_operand_uses.count(cb) ? index.matmul_operand_uses.at(cb) : 0;
    }
  }

  enum class Destination { kDirect, kMat, kComposite };
  std::optional<Destination> destination;
  const Var* materialized_old = nullptr;
  const Var* drop = nullptr;
  DataType scratch_dtype = terminal_ty->dtype_;
  if (store && terminal_uses == 1) {
    destination = Destination::kDirect;
  } else if (!store && terminal_uses >= 1 && terminal_uses == terminal_operand_uses) {
    destination = Destination::kMat;
    materialized_old = terminal;
  } else if (!store && cast_result && terminal_uses == 1 && cast_uses >= 1 &&
             cast_uses == cast_operand_uses) {
    destination = Destination::kMat;
    materialized_old = cast_result;
    drop = dropped_cast;
    scratch_dtype = cast_dtype;
  } else if (store && terminal_operand_uses >= 1 && terminal_uses == terminal_operand_uses + 1) {
    destination = Destination::kComposite;
    materialized_old = terminal;
  } else if (store && cast_result && terminal_uses == 2 && terminal_operand_uses == 0 && cast_uses >= 1 &&
             cast_uses == cast_operand_uses) {
    destination = Destination::kComposite;
    materialized_old = cast_result;
    drop = dropped_cast;
    scratch_dtype = cast_dtype;
  }
  if (!destination) return std::nullopt;

  CallPtr store_call;
  MakeTuplePtr offsets;
  VarPtr out_in;
  if (store) {
    store_call = As<Call>(store->value_);
    offsets = store_call && store_call->args_.size() == 3 ? As<MakeTuple>(store_call->args_[1]) : nullptr;
    out_in = store_call && store_call->args_.size() == 3 ? AsVarLike(store_call->args_[2]) : nullptr;
    if (!store_call || !IsOp(store_call, "tile.store") || !offsets || offsets->elements_.size() != 2 ||
        !out_in) {
      return std::nullopt;
    }
  }

  uint64_t scratch_bytes = 0;
  if (*destination != Destination::kDirect) {
    auto validated = ValidateMatScratch(root, scratch_dtype, hints);
    if (!validated) return std::nullopt;
    scratch_bytes = *validated;
  }
  if (*destination == Destination::kComposite) {
    auto store_pos = index.positions.at(store);
    auto consumer_positions = index.matmul_operand_positions.find(materialized_old);
    if (consumer_positions == index.matmul_operand_positions.end()) return std::nullopt;
    for (size_t consumer_pos : consumer_positions->second) {
      if (consumer_pos <= store_pos) {
        hints.emplace_back(
            DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
            "linear matmul/matmul_acc chain is stored and reused, but an on-chip consumer precedes its "
            "GM/Mat materialization point; left untouched",
            root.assign->span_);
        return std::nullopt;
      }
    }
  }

  auto prepared = PrepareLinearChainOperands(chain, scratch_bytes, hints);
  if (!prepared) return std::nullopt;
  std::unique_ptr<SubtilePlacer> placer;
  if (*destination == Destination::kDirect) {
    placer = std::make_unique<DirectGmPlacer>(offsets->elements_[0], offsets->elements_[1], out_in,
                                              store_call->kwargs_, store_call->attrs_, root.assign->span_);
  } else if (*destination == Destination::kMat) {
    placer = std::make_unique<MatScratchPlacer>(root.M, root.N, scratch_dtype,
                                                root.assign->var_->name_hint_ + "_mat", root.assign->span_);
  } else {
    std::vector<std::unique_ptr<SubtilePlacer>> children;
    children.push_back(std::make_unique<DirectGmPlacer>(offsets->elements_[0], offsets->elements_[1], out_in,
                                                        store_call->kwargs_, store_call->attrs_,
                                                        root.assign->span_));
    children.push_back(std::make_unique<MatScratchPlacer>(
        root.M, root.N, scratch_dtype, root.assign->var_->name_hint_ + "_mat", root.assign->span_));
    placer = std::make_unique<CompositeSubtilePlacer>(std::move(children), root.assign->span_);
  }

  auto [stmts, state] = BuildLinearChainGrid(*prepared, *placer);
  const size_t expected_state = *destination == Destination::kComposite ? 2 : 1;
  INTERNAL_CHECK_SPAN(state.size() == expected_state, root.assign->span_)
      << "Internal error: linear-chain placement returned the wrong state width";
  LinearChainFold fold;
  fold.stmts = std::move(stmts);
  fold.store = store;
  if (store) {
    fold.old_store_result = store->var_;
    fold.new_store_result = state.front();
  }
  if (*destination == Destination::kMat) {
    fold.materialized_old_var = materialized_old;
    fold.materialized_new_var = state.front();
  } else if (*destination == Destination::kComposite) {
    fold.materialized_old_var = materialized_old;
    fold.materialized_new_var = state[1];
  }
  fold.dropped_def = drop;
  return fold;
}

/// Static physical footprint of a tile, rounded exactly as the active memory
/// allocator rounds one buffer in @p space.  Unknown/dynamic shapes and
/// arithmetic overflow return nullopt: an automatic capacity decision must
/// never guess low.
std::optional<uint64_t> StaticAlignedTileBytes(const TileTypePtr& tile, MemorySpace space,
                                               const MemoryAllocatorPolicy& policy,
                                               const backend::BackendHandler* handler) {
  if (!tile || tile->GetMemorySpace() != space) return std::nullopt;
  auto bytes = utils::StaticPhysicalAllocationBytes(tile, space, handler);
  if (!bytes) return std::nullopt;
  const uint64_t aligned = policy.AlignAddress(*bytes, space);
  if (aligned < *bytes) return std::nullopt;  // alignment arithmetic overflow
  return aligned;
}

/// Whole-function conservative L0C inventory after accounting for pipeline
/// replication.  LowerPipelineLoops gives every non-cube Acc producer one
/// physical-membership request per source stage (and the product of the stage
/// depths under nested pipelines).  Counting only the pre-lowering SSA value
/// would therefore underestimate the placement that MemoryReuse may preserve.
///
/// Cube matmul accumulators are normally serialized and left untagged, so they
/// need one slot.  An already-marked dbC pipeline is conservatively charged at
/// its full source depth; a newly selected candidate is charged separately by
/// BuildPipelineDbCPlan as one existing slot plus one extra ping-pong slot.
///
/// This remains an intentional upper bound: sequential values and independent
/// pipeline groups may later coalesce, but the automatic dbC plan must not
/// force an existing pipeline to shed buffering depth merely because its
/// post-lowering multiplicity was omitted here.
class AccFootprintCollector : public IRVisitor {
 public:
  AccFootprintCollector(const MemoryAllocatorPolicy& policy, const backend::BackendHandler* handler)
      : policy_(policy), handler_(handler) {}

  bool valid = true;
  uint64_t total_bytes = 0;

 protected:
  void VisitVarLike_(const VarPtr& op) override { Record(op, /*copies=*/1); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    uint64_t copies = 1;
    auto tile = As<TileType>(op->var_->GetType());
    if (tile && tile->GetMemorySpace() == MemorySpace::Acc) {
      auto call = As<Call>(op->value_);
      const bool is_cube_matmul = call && call->op_ && call->op_->name_.rfind("tile.matmul", 0) == 0;
      // The lowering tagger skips ordinary cube accumulators because the cube
      // serializes them. Every other Acc producer is replicated across all
      // enclosing source pipeline stages. An explicit dbC marker makes the
      // cube accumulator replicated too; charging the full depth is safe even
      // when CanonicalizeIOOrder later rotates it over only two slots.
      if (!is_cube_matmul || explicit_dbc_depth_ != 0) copies = pipeline_depth_;
    }
    Record(op->var_, copies);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    const uint64_t saved_pipeline_depth = pipeline_depth_;
    const int saved_explicit_dbc_depth = explicit_dbc_depth_;
    if (op && op->kind_ == ForKind::Pipeline) {
      const int stages = op->GetAttr<int>(kPipelineStagesAttr, 0);
      if (stages <= 0 ||
          pipeline_depth_ > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(stages)) {
        valid = false;
      } else {
        pipeline_depth_ *= static_cast<uint64_t>(stages);
      }
      if (op->GetAttr<bool>(kPipelineDoubleBufferCAttr, false)) ++explicit_dbc_depth_;
    }
    IRVisitor::VisitStmt_(op);
    pipeline_depth_ = saved_pipeline_depth;
    explicit_dbc_depth_ = saved_explicit_dbc_depth;
  }

 private:
  struct Entry {
    uint64_t bytes = 0;
    uint64_t copies = 0;
  };

  void Record(const VarPtr& var, uint64_t copies) {
    if (!valid || !var || copies == 0) return;
    auto tile = As<TileType>(var->GetType());
    if (!tile || tile->GetMemorySpace() != MemorySpace::Acc) return;
    auto bytes = StaticAlignedTileBytes(tile, MemorySpace::Acc, policy_, handler_);
    if (!bytes || copies > std::numeric_limits<uint64_t>::max() / *bytes) {
      valid = false;
      return;
    }
    auto [it, inserted] = entries_.try_emplace(var.get(), Entry{*bytes, 0});
    if (!inserted && it->second.bytes != *bytes) {
      valid = false;
      return;
    }
    if (copies <= it->second.copies) return;
    const uint64_t added_copies = copies - it->second.copies;
    if (added_copies > std::numeric_limits<uint64_t>::max() / *bytes) {
      valid = false;
      return;
    }
    const uint64_t added_bytes = added_copies * *bytes;
    if (total_bytes > std::numeric_limits<uint64_t>::max() - added_bytes) {
      valid = false;
      return;
    }
    it->second.copies = copies;
    total_bytes += added_bytes;
  }

  const MemoryAllocatorPolicy& policy_;
  const backend::BackendHandler* handler_ = nullptr;
  uint64_t pipeline_depth_ = 1;
  int explicit_dbc_depth_ = 0;
  std::unordered_map<const Var*, Entry> entries_;
};

/// True for any call whose result occupies L0C. This future-proofs the
/// recognizer against new MAD-family operations: a candidate body may have one
/// Acc producer in total, and that producer must be the plain matmul selected
/// below.
bool ProducesAcc(const CallPtr& call) {
  auto tile = call ? As<TileType>(call->GetType()) : nullptr;
  return tile && tile->GetMemorySpace() == MemorySpace::Acc;
}

/// A per-iteration Mat->L0 transfer that the pipeline already replicates.  This
/// proves the "moving" side of the stationary-panel pattern; an arbitrary local
/// assignment is not sufficient evidence that the operand is a prefetchable
/// pipeline input.
bool IsMovingMatmulOperandProducer(const CallPtr& call, MemorySpace target_space) {
  if (!call || call->args_.empty()) return false;
  auto out = As<TileType>(call->GetType());
  auto source = As<TileType>(call->args_[0]->GetType());
  if (!out || out->GetMemorySpace() != target_space || !source ||
      source->GetMemorySpace() != MemorySpace::Mat) {
    return false;
  }
  return IsOp(call, "tile.extract") || IsOp(call, "tile.move");
}

enum class PipelineAccumulatorDrainPath {
  DirectGm,
  MatScratch,
};

struct PipelineAccumulatorCandidate {
  uint64_t aligned_acc_bytes = 0;
  int64_t trip_count = 0;
  PipelineAccumulatorDrainPath drain_path = PipelineAccumulatorDrainPath::DirectGm;
};

/// Two L0C slots need at least two complete compute/drain pairs before their
/// fill/drain bubble is reliably amortized.  The device sweep for #2131 found
/// the two-iteration (one-pair) form tied, while every direct-store case with
/// four or more iterations won across M/N/K, operand side, and Acc size.
constexpr int64_t kMinAutoDirectGmDbCTripCount = 4;

/// Acc->Mat has a cheaper drain and therefore needs more work and a larger
/// accumulator before the overlap repays the two-slot schedule.  Device
/// calibration found 8/16 KiB Mat-scratch accumulators regressed or tied while
/// independent 32/40 KiB cases won at eight iterations.  Keep this path's
/// first automatic region deliberately conservative and backend-relative:
/// one Acc tile must occupy at least one quarter of L0C, and the loop must
/// provide at least four complete compute/drain pairs.
constexpr int64_t kMinAutoMatScratchDbCTripCount = 8;
constexpr uint64_t kMinAutoMatScratchAccL0cDenominator = 4;

bool IsPipelineAccumulatorProfitable(const PipelineAccumulatorCandidate& candidate, uint64_t l0c_bytes) {
  if (candidate.drain_path == PipelineAccumulatorDrainPath::DirectGm) {
    return candidate.trip_count >= kMinAutoDirectGmDbCTripCount;
  }
  const uint64_t min_acc_bytes = l0c_bytes / kMinAutoMatScratchAccL0cDenominator +
                                 (l0c_bytes % kMinAutoMatScratchAccL0cDenominator != 0);
  return candidate.trip_count >= kMinAutoMatScratchDbCTripCount &&
         candidate.aligned_acc_bytes >= min_acc_bytes;
}

/// Recognize an already-L0, directly-drained matmul in a software pipeline and
/// return the extra accumulator slot it would need.
///
/// This is the user-authored-pipeline counterpart of the M/N tiler's dbC mode:
///
///   for n in pl.pipeline(..., stage=2):
///     b_l0 = ... Right
///     c_l0 = tile.matmul(a_l0, b_l0)  # Acc
///     out  = tile.store(c_l0, ..., out)
///
/// Success proves the local schedule and dataflow.  Whole-function L0C capacity
/// is checked separately by ``BuildPipelineDbCPlan`` before any loop is marked.
/// ``LowerPipelineLoops`` gives adjacent Acc clones two rotating memberships,
/// and ``CanonicalizeIOOrder`` emits depth-two chunks
/// ``matmul, matmul, drain, drain``.  A user-selected pipeline depth greater
/// than two still controls operand prefetch depth; L0C remains a two-slot
/// ping-pong rather than growing to one accumulator per stage.
///
/// Keep the recognition deliberately conservative:
///   * static pipeline with stage >= 2 and a trip count divisible by the stage
///     count (no separately lowered tail group);
///   * no nested control flow in the candidate body;
///   * exactly one cube MAD, and it is plain ``tile.matmul`` over Left/Right;
///   * the selected moving operand is a recognized per-iteration Mat->L0
///     transfer; the stationary operand is defined outside the loop and is not
///     an IterArg;
///   * no other Acc definition/read or store-like operation in the body;
///   * one canonical loop-carried ``tile.store`` or ``tile.assemble`` chain:
///     the drain targets IterArg i and its result is yielded at index i.
///
/// The returned path, trip count, and Acc footprint are filtered by
/// ``IsPipelineAccumulatorProfitable`` after the backend L0C capacity is known.
///
/// The direct-body scans are disjoint across nested loops, so this remains
/// linear in program size.
std::optional<PipelineAccumulatorCandidate> AnalyzePipelineAccumulator(
    const ForStmtPtr& loop, const MemoryAllocatorPolicy& policy, const backend::BackendHandler* handler) {
  const int stages = loop ? loop->GetAttr<int>(kPipelineStagesAttr, 0) : 0;
  const int64_t trip_count = loop ? transform_utils::EvalConstTripCount(loop) : -1;
  if (!loop || loop->kind_ != ForKind::Pipeline || loop->HasAttr(kPipelineOverlapStoresAttr) ||
      loop->HasAttr(kPipelineDoubleBufferCAttr) || stages < 2 || trip_count < stages ||
      trip_count % stages != 0) {
    return std::nullopt;
  }

  std::vector<StmtPtr> body_stmts;
  if (auto seq = As<SeqStmts>(loop->body_)) {
    body_stmts = seq->stmts_;
  } else {
    body_stmts.push_back(loop->body_);
  }

  AssignStmtPtr candidate;
  CallPtr candidate_call;
  std::unordered_set<const Var*> direct_defs;
  std::unordered_map<const Var*, CallPtr> direct_call_defs;
  int acc_producers = 0;
  for (const auto& stmt : body_stmts) {
    // A nested region could contain another use or cube operation that needs a
    // joint schedule. Defer instead of trying to summarize it locally.
    if (As<ForStmt>(stmt) || As<IfStmt>(stmt) || As<WhileStmt>(stmt) || As<ScopeStmt>(stmt)) {
      return std::nullopt;
    }
    auto assign = As<AssignStmt>(stmt);
    if (assign) direct_defs.insert(assign->var_.get());
    auto call = transform_utils::GetCallFromStmt(stmt);
    if (assign && call) direct_call_defs.emplace(assign->var_.get(), call);
    if (!ProducesAcc(call)) continue;
    ++acc_producers;
    if (assign && IsOp(call, "tile.matmul")) {
      candidate = assign;
      candidate_call = call;
    }
  }
  if (acc_producers != 1 || !candidate || !candidate_call || candidate_call->args_.size() != 2) {
    return std::nullopt;
  }

  auto lhs = AsVarLike(candidate_call->args_[0]);
  auto rhs = AsVarLike(candidate_call->args_[1]);
  auto lhs_ty = lhs ? As<TileType>(lhs->GetType()) : nullptr;
  auto rhs_ty = rhs ? As<TileType>(rhs->GetType()) : nullptr;
  int64_t M = 0;
  int64_t K_lhs = 0;
  int64_t K_rhs = 0;
  int64_t N = 0;
  if (!IsStatic2DInSpaces(lhs_ty, {MemorySpace::Left}, M, K_lhs) ||
      !IsStatic2DInSpaces(rhs_ty, {MemorySpace::Right}, K_rhs, N) || K_lhs != K_rhs) {
    return std::nullopt;
  }

  std::unordered_set<const Var*> iter_args;
  for (const auto& iter_arg : loop->iter_args_) iter_args.insert(iter_arg.get());
  // Match the stationary-panel construction from #2131.  A direct definition
  // proves movement only when it is the result of a Mat->Left/Right transfer;
  // the other operand must be a true outside-loop value, not loop-carried state.
  const bool lhs_moves = direct_defs.count(lhs.get()) != 0;
  const bool rhs_moves = direct_defs.count(rhs.get()) != 0;
  if (lhs_moves == rhs_moves) return std::nullopt;
  const VarPtr& moving = lhs_moves ? lhs : rhs;
  const VarPtr& stationary = lhs_moves ? rhs : lhs;
  const MemorySpace moving_space = lhs_moves ? MemorySpace::Left : MemorySpace::Right;
  auto producer = direct_call_defs.find(moving.get());
  if (producer == direct_call_defs.end() || !IsMovingMatmulOperandProducer(producer->second, moving_space) ||
      direct_defs.count(stationary.get()) != 0 || iter_args.count(stationary.get()) != 0) {
    return std::nullopt;
  }

  auto acc_ty = As<TileType>(candidate->var_->GetType());
  int64_t acc_m = 0;
  int64_t acc_n = 0;
  if (!IsStatic2DInSpaces(acc_ty, {MemorySpace::Acc}, acc_m, acc_n) || acc_m != M || acc_n != N) {
    return std::nullopt;
  }
  auto acc_bytes = StaticAlignedTileBytes(acc_ty, MemorySpace::Acc, policy, handler);
  if (!acc_bytes) return std::nullopt;

  // Count all uses in the direct body, excluding assignment LHS definitions.
  // Compound statements were rejected above, so this scan cannot descend into
  // a child region and re-count work owned by another loop.
  SiblingUseCounter uses;
  for (const auto& stmt : body_stmts) uses.VisitStmt(stmt);
  auto use_it = uses.counts.find(candidate->var_.get());
  if (use_it == uses.counts.end() || use_it->second != 1) return std::nullopt;
  // Any other Acc read is loop-carried/external L0C state whose lifetime and
  // aliasing would be changed by the loop-wide drain reorder.
  for (const auto& [raw, var] : uses.vars) {
    if (raw == candidate->var_.get()) continue;
    auto tile = As<TileType>(var->GetType());
    if (tile && tile->GetMemorySpace() == MemorySpace::Acc) return std::nullopt;
  }
  // Likewise reject every other Acc definition, including non-MAD data
  // movement such as tile.extract(..., target_memory=Acc).
  for (const auto& stmt : body_stmts) {
    auto assign = As<AssignStmt>(stmt);
    if (!assign || assign.get() == candidate.get()) continue;
    auto tile = As<TileType>(assign->var_->GetType());
    if (tile && tile->GetMemorySpace() == MemorySpace::Acc) return std::nullopt;
  }

  AssignStmtPtr drain_assign;
  CallPtr drain_call;
  int store_like_calls = 0;
  for (const auto& stmt : body_stmts) {
    auto assign = As<AssignStmt>(stmt);
    auto call = transform_utils::GetCallFromStmt(stmt);
    if (!call) continue;
    const bool is_store = IsOp(call, "tile.store");
    const bool is_assemble = IsOp(call, "tile.assemble");
    if (!is_store && !is_assemble && !IsOp(call, "tile.write")) continue;
    ++store_like_calls;
    if (!assign || call->args_.size() != 3) return std::nullopt;
    const size_t source_index = is_assemble ? 1 : 0;
    auto source = AsVarLike(call->args_[source_index]);
    if (!source || source.get() != candidate->var_.get()) return std::nullopt;
    if (drain_assign) return std::nullopt;
    drain_assign = assign;
    drain_call = call;
  }
  if (store_like_calls != 1 || !drain_assign || !drain_call) return std::nullopt;

  // Canonical loop-carried drain: target IterArg i -> drain result -> yield i.
  auto yield = transform_utils::GetLastYieldStmt(loop->body_);
  if (!yield || yield->value_.size() != loop->iter_args_.size()) return std::nullopt;
  std::optional<size_t> yield_index;
  for (size_t i = 0; i < yield->value_.size(); ++i) {
    auto value = AsVarLike(yield->value_[i]);
    if (value && value.get() == drain_assign->var_.get()) {
      if (yield_index) return std::nullopt;
      yield_index = i;
    }
  }
  if (!yield_index || *yield_index >= loop->iter_args_.size()) return std::nullopt;
  const bool is_assemble = IsOp(drain_call, "tile.assemble");
  const size_t target_index = is_assemble ? 0 : 2;
  auto target = AsVarLike(drain_call->args_[target_index]);
  if (!target || target.get() != loop->iter_args_[*yield_index].get()) return std::nullopt;
  if (uses.counts[drain_assign->var_.get()] != 1 || uses.counts[target.get()] != 1) {
    return std::nullopt;
  }
  if (is_assemble) {
    auto target_ty = As<TileType>(target->GetType());
    auto result_ty = As<TileType>(drain_assign->var_->GetType());
    if (!target_ty || target_ty->GetMemorySpace() != MemorySpace::Mat || !result_ty ||
        result_ty->GetMemorySpace() != MemorySpace::Mat) {
      return std::nullopt;
    }
  }
  return PipelineAccumulatorCandidate{
      *acc_bytes, trip_count,
      is_assemble ? PipelineAccumulatorDrainPath::MatScratch : PipelineAccumulatorDrainPath::DirectGm};
}

/// Select all structurally eligible loops only when their worst-case combined
/// L0C footprint fits.  The base footprint counts every Acc SSA value once; the
/// extra term adds one slot per loop this pass will mark.  This intentionally
/// over-approximates liveness but guarantees that the loop-wide schedule change
/// cannot turn a previously fitting function into an L0C overflow.
std::unordered_set<const ForStmt*> BuildPipelineDbCPlan(const FunctionPtr& func) {
  const auto* ctx = PassContext::Current();
  const MemoryPlanner planner = ctx ? ctx->GetMemoryPlanner() : MemoryPlanner::PyPTO;
  // #2131 explicitly targets the PyPTO planner. PTOAS already gives the
  // reproduced loop four distinct Acc placements and showed no measurable
  // benefit from this source-level marker.
  if (planner != MemoryPlanner::PyPTO) return {};

  // Profitability and capacity are backend-specific. Direct pass invocation
  // without a configured backend must leave an already-L0 pipeline unchanged,
  // just as it did before this recognizer existed.
  if (!pypto::backend::BackendConfig::IsConfigured()) return {};

  auto policy = pypto::backend::GetBackend()->CreateMemoryAllocatorPolicy();
  if (!policy) return {};

  const auto* handler = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  const uint64_t l0c_bytes = handler ? handler->GetL0cCapacityBytes() : 0;
  if (!handler || l0c_bytes == 0) return {};

  AccFootprintCollector footprint(*policy, handler);
  footprint.VisitFunction(func);
  if (!footprint.valid) return {};

  class CandidateCollector : public IRVisitor {
   public:
    CandidateCollector(const MemoryAllocatorPolicy& policy, const backend::BackendHandler* handler)
        : policy_(policy), handler_(handler) {}
    std::unordered_map<const ForStmt*, PipelineAccumulatorCandidate> candidates;

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      if (auto candidate = AnalyzePipelineAccumulator(op, policy_, handler_)) {
        candidates.emplace(op.get(), *candidate);
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    const MemoryAllocatorPolicy& policy_;
    const backend::BackendHandler* handler_ = nullptr;
  } candidates(*policy, handler);
  candidates.VisitStmt(func->body_);
  if (candidates.candidates.empty()) return {};

  for (auto it = candidates.candidates.begin(); it != candidates.candidates.end();) {
    if (!IsPipelineAccumulatorProfitable(it->second, l0c_bytes)) {
      it = candidates.candidates.erase(it);
    } else {
      ++it;
    }
  }
  if (candidates.candidates.empty()) return {};

  uint64_t worst_case = footprint.total_bytes;
  if (ctx && ctx->GetEnablePyptoL0cDoubleBuffer()) {
    // The chooser may also double-buffer any other Acc result in this function;
    // reserve a second slot for the full inventory rather than trying to
    // duplicate its profitability decision here.
    if (worst_case > std::numeric_limits<uint64_t>::max() - footprint.total_bytes) return {};
    worst_case += footprint.total_bytes;
  } else {
    for (const auto& entry : candidates.candidates) {
      const auto& candidate = entry.second;
      if (worst_case > std::numeric_limits<uint64_t>::max() - candidate.aligned_acc_bytes) return {};
      worst_case += candidate.aligned_acc_bytes;
    }
  }
  if (worst_case > l0c_bytes) return {};

  std::unordered_set<const ForStmt*> plan;
  for (const auto& entry : candidates.candidates) plan.insert(entry.first);
  return plan;
}

class AutoTileMutator : public IRMutator {
 public:
  explicit AutoTileMutator(std::unordered_set<const ForStmt*> pipeline_dbc_plan)
      : pipeline_dbc_plan_(std::move(pipeline_dbc_plan)) {}

  std::vector<Diagnostic> hints;

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    const bool should_double_buffer_c = pipeline_dbc_plan_.count(op.get()) != 0;
    // Recurse first: nested pipelines make their own local dbC decision. An
    // enclosing pipeline does not inherit the marker and therefore does not
    // multiply the accumulator buffering depth.
    auto visited = IRMutator::VisitStmt_(op);
    auto loop = As<ForStmt>(visited);
    if (!should_double_buffer_c || !loop) return visited;

    auto result = MutableCopy(loop);
    result->attrs_ =
        StripAttr(StripAttr(loop->attrs_, kPipelineOverlapStoresAttr), kPipelineDoubleBufferCAttr);
    result->attrs_.emplace_back(kPipelineOverlapStoresAttr, false);
    result->attrs_.emplace_back(kPipelineDoubleBufferCAttr, true);
    return result;
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // Per-SeqStmts substitution map: when we rewrite ``c = tile.matmul(...)``
    // into a ForStmt with a fresh return_var, subsequent statements in the
    // same SeqStmts that referenced ``c`` need to be redirected to that
    // return_var.  Scoped to this SeqStmts so substitutions don't leak into
    // sibling regions.
    std::unordered_map<const Var*, VarPtr> remap;
    // Defs to drop: a downcast ``cb = tile.cast(c, bf16)`` whose matmul+cast was
    // folded into a low-precision Mat scratch — the scratch already holds the
    // bf16 intermediate, so the now same-dtype cast is a dead no-op to remove.
    std::unordered_set<const Var*> dropped;
    // M/N tiling folds a later consumer/anchor into the per-sub-tile rewrite.
    // We drop the matmul at its own position and emit the sub-tile stmts where
    // that anchor was (preserving the order of any statements between them),
    // keyed by the anchor statement's identity. Usually this is a store; a
    // Mat-only linear accumulator chain uses its terminal matmul_acc so every
    // later-stage operand remains defined before the replacement.
    std::unordered_map<const Stmt*, MNFold> pending_folds;
    // Use counts + store-consumer sites across this SeqStmts, built lazily on
    // the first oversized matmul and reused — O(N) total, no rescan per matmul.
    std::optional<SiblingIndex> sibling_index;
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      const StmtPtr& child = op->stmts_[i];

      // A consumer/anchor folded into a prior M/N rewrite: emit the sub-tile
      // stmts in the anchor's original position and drop the anchor itself.
      // Apply the now-current remap so the folded stores' output-tensor chain
      // start (and any other operands) reflect rewrites installed between the
      // matmul and this store — in particular a prior fold that redefined the
      // output this store fed from.  Without this, a fold built before that
      // remap existed (e.g. when the matmuls are defined in the reverse order
      // of their stores) would keep a stale, now-undefined output Var.
      //
      // Exclude this fold's *own* store-result var: its rewrite targets only
      // downstream uses, never the fold's internal chain start.  For an
      // output-param store the input tensor and the store result are the same
      // SSA var, so applying that one entry would rewrite the chain start onto
      // the fold's final output — a self-referential use-before-def.
      if (auto it = pending_folds.find(child.get()); it != pending_folds.end()) {
        decltype(remap)::node_type self;
        if (it->second.store_result_var) self = remap.extract(it->second.store_result_var.get());
        for (auto& s : it->second.stmts) {
          out.push_back(remap.empty() ? s : transform_utils::Substitute(s, remap));
        }
        if (!self.empty()) remap.insert(std::move(self));  // restore for downstream uses
        changed = true;
        continue;
      }

      // A downcast whose matmul+cast was folded into a Mat scratch (its def is in
      // ``dropped``): skip it — the scratch already holds the bf16 intermediate and
      // re-emitting ``cb = tile.cast(scratch_bf16, bf16)`` is an invalid no-op cast.
      if (auto as = std::dynamic_pointer_cast<const AssignStmt>(child); as && dropped.count(as->var_.get())) {
        changed = true;
        continue;
      }

      // Apply the running remap to redirect prior rewrites' downstream uses.
      StmtPtr current = remap.empty() ? child : transform_utils::Substitute(child, remap);

      // Fits-L0c chained-matmul cast-fold: rewrite ``cb = tile.cast(src_acc,
      // bf16/f16)`` into a full-window Acc->Mat scratch (``tile.create`` +
      // ``tile.assemble``) when every use of ``cb`` is a matmul operand.  This
      // routes the f32->bf16 downcast through the cube FIXPIPE (``pto.tinsert``)
      // instead of the Vector (``pto.tcvt``) — the fits-L0c analogue of the
      // oversized per-sub-tile Mat-scratch fold (``TryFoldMatScratch``).  The
      // matmul producing ``src`` is K-tiled (or left untouched) by the dispatch
      // below; here we only redirect the cast's result onto a Mat scratch and
      // drop the now-dead cast.  Oversized chains never reach here — their cast
      // is dropped via ``dropped`` at the matmul site (see ``TryFoldMatScratch``
      // remap above), so this only fires for results that fit L0c.
      if (auto cast_as = std::dynamic_pointer_cast<const AssignStmt>(current)) {
        auto cast = As<Call>(cast_as->value_);
        auto cb_ty = As<TileType>(cast_as->var_->GetType());
        if (cast && IsOp(cast, "tile.cast") && !cast->args_.empty() && cb_ty) {
          auto src = AsVarLike(cast->args_[0]);
          auto src_ty = src ? As<TileType>(src->GetType()) : nullptr;
          const bool src_acc = src_ty && src_ty->memory_space_ == MemorySpace::Acc;
          // Only fold what FIXPIPE can reproduce: f32 -> bf16/f16, round-to-nearest.
          const bool fixpipe_castable = CastFoldableToFixpipeMat(cast, src_ty, cb_ty->dtype_);
          // Static [M, N] — a fits-L0c chained matmul result is always static here.
          auto m_ci = cb_ty->shape_.size() == 2 ? As<ConstInt>(cb_ty->shape_[0]) : nullptr;
          auto n_ci = cb_ty->shape_.size() == 2 ? As<ConstInt>(cb_ty->shape_[1]) : nullptr;
          // Only fold when the Acc result fits L0c.  An oversized result is
          // Case 2's domain: ``TryFoldMatScratch`` folds the cast into per-sub-tile
          // assembles (and drops it), or defers it when the scratch exceeds Mat
          // capacity — in which case the cast must stay (we must not collapse an
          // oversized [M, N] into one impossible full-window assemble here).
          const auto* bh_ctx = PassContext::Current();
          const auto* bh = bh_ctx ? bh_ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
          const uint64_t l0c_bytes = bh ? bh->GetL0cCapacityBytes() : 0;
          const auto acc_bytes = src_acc && m_ci && n_ci
                                     ? utils::StaticPhysicalAllocationBytes(src_ty, MemorySpace::Acc, bh)
                                     : std::optional<uint64_t>{};
          if (src_acc && m_ci && n_ci && l0c_bytes && acc_bytes && *acc_bytes <= l0c_bytes) {
            if (!sibling_index) sibling_index = BuildSiblingIndex(op->stmts_);
            const Var* cb = cast_as->var_.get();
            auto uc = sibling_index->use_counts.find(cb);
            auto mo = sibling_index->matmul_operand_uses.find(cb);
            const int cb_uses = uc == sibling_index->use_counts.end() ? 0 : uc->second;
            const int cb_mm = mo == sibling_index->matmul_operand_uses.end() ? 0 : mo->second;
            // A chained cast: every use of ``cb`` is a matmul operand, so the bf16
            // value can live entirely in Mat (a store / elementwise consumer could
            // not read it there, so those keep the Vector cast path regardless).
            if (cb_uses >= 1 && cb_uses == cb_mm) {
              if (fixpipe_castable) {
                MatScratchPlacer placer(m_ci->value_, n_ci->value_, cb_ty->dtype_, src->name_hint_ + "_mat",
                                        cast_as->span_);
                PlacementState scratch = placer.Init(out);
                PlacementState cmat = placer.PlaceAt(out, src, MakeIndex(0, cast_as->span_),
                                                     MakeIndex(0, cast_as->span_), scratch, /*step=*/0);
                INTERNAL_CHECK_SPAN(cmat.size() == 1, cast_as->span_)
                    << "Internal error: fits-L0c Mat-scratch fold must return one scratch";
                remap[cb] = cmat.front();  // the consumer matmul now reads the Mat scratch
                changed = true;
                continue;  // drop the dead tile.cast
              }
              // Chained cast that fits L0c but FIXPIPE cannot reproduce (a non-f32
              // accumulator, or a round mode other than round-half-to-even): keep
              // the standalone Vector cast and warn — it stays a cube->vector->cube
              // round-trip that overflows the Vec buffer at large [M, N].
              hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-010",
                                 "chained-matmul [" + std::to_string(m_ci->value_) + ", " +
                                     std::to_string(n_ci->value_) + "] cast to " + cb_ty->dtype_.ToString() +
                                     " cannot fold onto the cube FIXPIPE (which narrows f32->bf16/f16 with "
                                     "round-half-to-even only); kept on the Vector path (pto.tcvt), a "
                                     "cube->vector->cube round-trip that may overflow the Vec buffer at this "
                                     "[M, N] — cast an f32 result with mode=\"rint\" to keep it on the cube",
                                 cast_as->span_);
              // fall through: the tile.cast stays (Vector pto.tcvt)
            }
          }
        }
      }

      // Check if this is a matmul we rewrite *at this SeqStmts level*.  We
      // try this before recursive visitation so the rewrite — which produces
      // a sequence of stmts — lands in this enclosing SeqStmts.  Recursive
      // visitation happens after rewrite-rejection so nested matmuls inside
      // ForStmt bodies still get rewritten by the recursive visit.
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(current)) {
        if (auto tiling = AnalyzeMatmul(assign, hints)) {
          if (!tiling->needs_mn_tiling()) {
            // Whole output fits L0c — tile K only.  k < K here (k == K with
            // m == M, n == N needs no matmul tiling and was skipped by
            // AnalyzeMatmul); the chooser may return a non-divisor k that
            // BuildKLoopRewrite peels.
            INTERNAL_CHECK_SPAN(tiling->k < tiling->K, tiling->assign->span_)
                << "Internal error: K-only tiling expects k < K (K=" << tiling->K << ", k=" << tiling->k
                << ")";
            auto rewrite = BuildKLoopRewrite(
                MakeKLoop(*tiling, /*mi=*/nullptr, /*ni=*/nullptr, tiling->m, tiling->n, /*name_base=*/""));
            remap[assign->var_.get()] = rewrite.return_var;
            for (auto& s : rewrite.stmts) out.push_back(std::move(s));
            changed = true;
            continue;
          }
          // Output exceeds L0c — tile M/N by folding the consumer store, found
          // via the raw (un-substituted) SiblingIndex: the matmul's result is
          // freshly defined here, so its use count / store site are never
          // affected by the running remap.
          if (!sibling_index) sibling_index = BuildSiblingIndex(op->stmts_);

          // A fresh root may feed a linear sequence of tile.matmul_acc calls.
          // Rewrite the whole logical reduction atomically so no oversized
          // caller-owned intermediate Acc survives between stages.
          if (!tiling->is_acc()) {
            if (auto chain = AnalyzeLinearMatmulChain(*tiling, *sibling_index, hints)) {
              if (auto chain_fold = TryFoldLinearMatmulChain(*chain, *sibling_index, hints)) {
                for (const Var* continuation : chain->continuation_defs) dropped.insert(continuation);
                if (chain_fold->dropped_def) dropped.insert(chain_fold->dropped_def);
                if (chain_fold->materialized_old_var) {
                  remap[chain_fold->materialized_old_var] = chain_fold->materialized_new_var;
                }
                if (chain_fold->store) {
                  remap[chain_fold->old_store_result.get()] = chain_fold->new_store_result;
                  MNFold pending{std::move(chain_fold->stmts), chain_fold->new_store_result,
                                 chain_fold->old_store_result, chain_fold->store};
                  pending_folds.emplace(static_cast<const Stmt*>(chain_fold->store), std::move(pending));
                } else {
                  const AssignStmt* terminal = chain->stages.back().assign.get();
                  MNFold pending{std::move(chain_fold->stmts), nullptr, nullptr, terminal};
                  pending_folds.emplace(static_cast<const Stmt*>(terminal), std::move(pending));
                }
                changed = true;
                continue;
              }
            }
          }

          const Var* result = assign->var_.get();
          auto uc_it = sibling_index->use_counts.find(result);
          const int source_result_uses = uc_it == sibling_index->use_counts.end() ? 0 : uc_it->second;
          auto mo_it = sibling_index->matmul_operand_uses.find(result);
          const int source_operand_uses =
              mo_it == sibling_index->matmul_operand_uses.end() ? 0 : mo_it->second;
          const auto stores_it = sibling_index->stores_of.find(result);
          const bool has_one_store =
              stores_it != sibling_index->stores_of.end() && stores_it->second.size() == 1;
          const AssignStmt* store_stmt = has_one_store ? stores_it->second.front() : nullptr;
          // Mat-scratch dtype + remap target. Default: the matmul result itself at
          // its own dtype. Chained-matmul-with-downcast — `c -> tile.cast(c,
          // bf16/f16) -> matmul` — fuses the cast (the cube FIXPIPE writeback,
          // pto.tinsert) into the scratch: the scratch holds the bf16/f16
          // intermediate, the per-sub-tile assemble downcasts Acc f32 -> Mat bf16,
          // and the cast result is remapped to the scratch so the consumer matmul
          // reads it on-chip (the cast op then goes dead).
          auto result_tile_ty = As<TileType>(result->GetType());
          DataType scratch_dtype = result_tile_ty->dtype_;
          const Var* remap_target = result;
          const Var* extra_remap = nullptr;
          const Var* cast_result = nullptr;
          const Var* foldable_cast_def = nullptr;
          DataType cast_scratch_dtype = scratch_dtype;
          int cast_uses = 0;
          int cast_operand_uses = 0;
          if (auto cast_it = sibling_index->cast_of.find(result); cast_it != sibling_index->cast_of.end()) {
            const Var* cb = cast_it->second->var_.get();
            auto cb_ty = As<TileType>(cb->GetType());
            auto cast_call = As<Call>(cast_it->second->value_);
            // Fold the downcast into the FIXPIPE Acc->Mat writeback only when FIXPIPE
            // can reproduce it: f32 (the matmul Acc) -> bf16/f16, round-to-nearest.
            // Otherwise keep the standalone Vector cast (e.g. an int accumulator, or a
            // directional/truncating round mode FIXPIPE has no `rmode` for).
            if (cb_ty && CastFoldableToFixpipeMat(cast_call, result_tile_ty, cb_ty->dtype_)) {
              auto cb_uc = sibling_index->use_counts.find(cb);
              auto cb_mo = sibling_index->matmul_operand_uses.find(cb);
              const int cb_uses = cb_uc == sibling_index->use_counts.end() ? 0 : cb_uc->second;
              const int cb_mm = cb_mo == sibling_index->matmul_operand_uses.end() ? 0 : cb_mo->second;
              cast_result = cb;
              foldable_cast_def = cb;
              cast_scratch_dtype = cb_ty->dtype_;
              cast_uses = cb_uses;
              cast_operand_uses = cb_mm;
              // `c`'s sole use is the cast, whose result is consumed entirely as
              // matmul operands — fold the matmul+cast into a low-precision scratch.
              if (source_result_uses == 1 && cb_uses >= 1 && cb_uses == cb_mm) {
                scratch_dtype = cb_ty->dtype_;
                remap_target = cb;     // consumer matmul reads the scratch
                extra_remap = result;  // c -> scratch too; the cast op goes dead
              }
            }
          }
          // Mat-scratch: result consumed entirely on-chip at matmul-operand
          // positions — assemble the sub-tiles into an L1/Mat scratch and remap the
          // matmul result (or its downcast) to it.  Emitted at the matmul site
          // (like the K-only rewrite), with no store to defer.  Checked before the
          // direct-store fold so its hints stay clean.
          // #1908 guard: a chained Mat-scratch producer must stay output-stationary.
          // The Mat-scratch offset-packing path cannot yet pack an A/B-stationary
          // producer (its held operand is a monolithic single-buffered [m,K]/[K,n]
          // panel in the full L0 buffer) against the consumer matmul's double-buffered
          // operands, so an A/B-stationary chained producer overflows at
          // AllocateMemoryAddr. OS is always a legal tile and the oversized producer
          // must be tiled (deferring would overflow L0c), so re-choose OS-only for the
          // fold rather than defer. Remove once offset packing lands.
          const MatmulTiling* fold_tiling = &*tiling;
          std::optional<MatmulTiling> os_tiling;
          if (tiling->stationarity != utils::Stationarity::kOutputStationary) {
            std::vector<Diagnostic> discard;  // the first AnalyzeMatmul already emitted the hints
            os_tiling = AnalyzeMatmul(assign, discard, /*force_output_stationary=*/true);
            if (os_tiling) fold_tiling = &*os_tiling;
          }

          // Stored-and-reused: preserve the programmer's ordinary GM store and
          // on-chip matmul dataflow by materializing each legal Acc sub-tile to
          // both destinations. A foldable cast is absorbed into the Mat writeback.
          const Var* composite_target = nullptr;
          const Var* composite_drop = nullptr;
          DataType composite_dtype = result_tile_ty->dtype_;
          if (has_one_store && cast_result && source_result_uses == 2 && source_operand_uses == 0 &&
              cast_uses >= 1 && cast_uses == cast_operand_uses) {
            composite_target = cast_result;
            composite_drop = foldable_cast_def;
            composite_dtype = cast_scratch_dtype;
          } else if (has_one_store && source_operand_uses >= 1 &&
                     source_result_uses == source_operand_uses + 1) {
            composite_target = result;
          }
          if (composite_target) {
            if (auto fold = TryFoldStoredAndReused(*fold_tiling, store_stmt, composite_target,
                                                   composite_dtype, composite_drop, *sibling_index, hints)) {
              remap[fold->store_result_var.get()] = fold->return_var;
              remap[fold->materialized_old_var] = fold->materialized_new_var;
              if (fold->dropped_def) dropped.insert(fold->dropped_def);
              pending_folds.emplace(static_cast<const Stmt*>(fold->store), std::move(*fold));
              changed = true;
              continue;
            }
          }

          const int scratch_uses = extra_remap ? cast_uses : source_result_uses;
          const int scratch_operand_uses = extra_remap ? cast_operand_uses : source_operand_uses;
          if (auto ms =
                  TryFoldMatScratch(*fold_tiling, scratch_uses, scratch_operand_uses, scratch_dtype, hints)) {
            for (auto& s : ms->first) out.push_back(std::move(s));
            remap[remap_target] = ms->second;
            if (extra_remap) {
              remap[extra_remap] = ms->second;  // c -> scratch (cast reads the scratch)
              dropped.insert(remap_target);     // ... and drop the now-dead cast def
            }
            changed = true;
            continue;
          }
          if (auto fold = TryFoldMNTiling(*tiling, source_result_uses, store_stmt, hints)) {
            remap[fold->store_result_var.get()] = fold->return_var;
            pending_folds.emplace(static_cast<const Stmt*>(fold->store), std::move(*fold));
            changed = true;
            continue;  // drop the matmul; sub-tile stmts emit at the store site
          }
          // M/N tiling not applicable — fall through and leave it untouched.
        }
      }
      auto visited = VisitStmt(current);
      if (visited.get() != child.get()) changed = true;
      out.push_back(visited);
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

 private:
  std::unordered_set<const ForStmt*> pipeline_dbc_plan_;
};

FunctionPtr TransformFunction(const FunctionPtr& func, std::vector<Diagnostic>& hints) {
  if (!func || !func->body_) return func;
  if (!IsInCoreType(func->func_type_)) return func;
  // Canonical loop-carried split-K output tiling changes both the Acc inventory
  // and loop identities. Rewrite it first so the #2131 dbC capacity/placement
  // plan is computed from the exact IR the ordinary AutoTile phase will visit.
  auto canonical = RewriteCanonicalSplitKAcc(func, hints);
  AutoTileMutator mutator(BuildPipelineDbCPlan(canonical));
  auto new_body = mutator.VisitStmt(canonical->body_);
  for (auto& d : mutator.hints) hints.push_back(std::move(d));
  if (new_body == canonical->body_) return canonical;
  auto new_func = MutableCopy(canonical);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass AutoTileMatmulL0() {
  auto run = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool any_change = false;
    std::vector<Diagnostic> hints;
    for (const auto& [gvar, func] : program->functions_) {
      auto new_func = TransformFunction(func, hints);
      if (new_func != func) any_change = true;
      new_functions.emplace(gvar, new_func);
    }
    if (!hints.empty()) EmitDiagnostics(hints, kPassName);
    if (!any_change) return program;
    auto new_program = MutableCopy(program);
    new_program->functions_ = std::move(new_functions);
    return new_program;
  };
  return CreateProgramPass(run, kPassName, kAutoTileMatmulL0Properties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
