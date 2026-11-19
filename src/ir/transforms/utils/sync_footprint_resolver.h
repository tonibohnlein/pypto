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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_FOOTPRINT_RESOLVER_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_FOOTPRINT_RESOLVER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/utils/sync_loop_trip_proof.h"

namespace pypto::ir::sync {

// Native scalar flattening needs actual inherited layout provenance.
void RequireBufferSyncScalarLayout(const FunctionPtr& owner);

// Canonical storage root shared by Exact and Conservative engines.
struct SyncCanonicalRoot {
  size_t id = 0;  // 1-based; 0 = space wildcard / unproved
  MemorySpace space = MemorySpace::DDR;
  bool placed = false;  // True for constant-addressed local allocations
  int64_t address = 0;  // Physical base byte address
  int64_t bytes = 0;    // Total reservation bytes across all slots
  int64_t slot_count = 1;
  int64_t slot_bytes = 0;
  int64_t rows = 0;
  int64_t columns = 0;
  BufferTypePtr canonical_layout;  // Canonical BufferType (nullptr for Vec byte space / DDR)
  ExprPtr gm_root;                 // Formal Tensor parameter or MemRef::base_ for DDR
  bool touched_by_set_validshape = false;
};

enum class SyncViewKind : uint8_t {
  Unknown,         // Unaddressed alloc, unproved slot index, carried/branch handle
  PlacedLocal,     // Direct buffer.alloc / buffer.alloc_multi
  Slot,            // buffer.slot(multi_buf, idx) with proved idx in [0, slot_count)
  VecSubview,      // buffer.subview(vec_parent, (r_off, c_off), [valid]) — chained pitches preserved
  Reshape,         // buffer.reshape(parent) — Vec contiguous or Matrix NZ<->ZN relabel
  ExternalTensor,  // Qualified DDR formal Tensor or plain alias
};

// Periodic 2D strided byte footprint in Vec:
// union_{r=0}^{rows-1} [base + r*pitch, base + r*pitch + width).
struct StridedByteFootprint {
  int64_t base = 0;
  int64_t pitch = 0;
  int64_t rows = 1;
  int64_t width = 0;
  bool exact = false;  // True when every byte in the lattice is unconditionally accessed

  std::optional<backend::BufferSyncByteWindow> ByteHull() const;
  bool SameGeometry(const StridedByteFootprint& other) const {
    return base == other.base && pitch == other.pitch && rows == other.rows && width == other.width;
  }
  bool operator==(const StridedByteFootprint& other) const {
    return SameGeometry(other) && exact == other.exact;
  }
  bool operator<(const StridedByteFootprint& other) const {
    return std::tie(base, pitch, rows, width, exact) <
           std::tie(other.base, other.pitch, other.rows, other.width, other.exact);
  }
  static bool MayOverlap(const StridedByteFootprint& a, const StridedByteFootprint& b);
};

// Canonical 2D box footprint in Mat/Left/Right/Acc (after undoing `axis_swapped`).
struct MatrixBoxFootprint {
  size_t root_id = 0;
  int64_t slot_lo = 0;
  int64_t slot_hi = 0;  // Inclusive [slot_lo, slot_hi]
  int64_t row_begin = 0;
  int64_t row_end = 0;  // Half-open [row_begin, row_end) in canonical root coordinates
  int64_t col_begin = 0;
  int64_t col_end = 0;  // Half-open [col_begin, col_end) in canonical root coordinates
  bool exact = false;   // True when box-aligned & single-slot full overwrite/read

  bool SameGeometry(const MatrixBoxFootprint& other) const {
    return root_id == other.root_id && slot_lo == other.slot_lo && slot_hi == other.slot_hi &&
           row_begin == other.row_begin && row_end == other.row_end && col_begin == other.col_begin &&
           col_end == other.col_end;
  }
  bool operator==(const MatrixBoxFootprint& other) const {
    return SameGeometry(other) && exact == other.exact;
  }
  bool operator<(const MatrixBoxFootprint& other) const {
    return std::tie(root_id, slot_lo, slot_hi, row_begin, row_end, col_begin, col_end, exact) <
           std::tie(other.root_id, other.slot_lo, other.slot_hi, other.row_begin, other.row_end,
                    other.col_begin, other.col_end, other.exact);
  }
  static bool MayOverlap(const MatrixBoxFootprint& a, const MatrixBoxFootprint& b,
                         const std::vector<SyncCanonicalRoot>& roots);
};

// Intra-loop iteration fiber wrt an enclosing sequential ForStmt `frame`.
struct IterationFiber {
  enum class Kind : uint8_t { Wildcard, ModularSlot, AffineSlice, CompletedPast };
  Kind kind = Kind::Wildcard;
  const ForStmt* frame = nullptr;
  int64_t modulus = 0;     // ModularSlot: slot = (coeff * iv_norm + offset) mod modulus
  int64_t coeff = 0;       // Step per normalized trip of `frame` (slots or bytes/rows)
  int64_t offset = 0;      // Residue in [0, modulus) or static start offset
  int64_t extent = 0;      // Slice width (1 for ModularSlot)
  uint8_t slice_axis = 0;  // 0 = byte / matrix row, 1 = matrix col
  ExprPtr symbolic_base;

  static IterationFiber Wildcard() { return {}; }
};

// O(1) per-SSA-handle resolved view provenance.
struct SyncResolvedView {
  SyncViewKind kind = SyncViewKind::Unknown;
  MemorySpace space = MemorySpace::DDR;
  size_t root_id = 0;
  BufferTypePtr type;
  // Conservative invocation-wide 1D byte hull [begin, end) within `space`.
  // Nullopt iff unplaced or unproved slot selector (preserving workspace-separation refusals).
  std::optional<backend::BufferSyncByteWindow> byte_hull;
  int64_t base_byte_offset = 0;  // Static/min byte offset in `space` (Vec) or slot 0
  int64_t row_pitch_bytes = 0;   // Inherited physical row stride in bytes (Vec)
  std::optional<SymbolicIntRange> slot_selector;
  ExprPtr raw_slot_expr;
  std::array<SymbolicIntRange, 2> view_offsets{};
  bool axis_swapped = false;         // True when handle views `root_id` through NZ <-> ZN transpose
  bool valid_matrix_coords = false;  // True when handle coordinates map to canonical root_id rows/cols
  bool static_full_valid = false;    // Full static valid_shape & untouched by buffer.set_validshape
  std::optional<StridedByteFootprint> vec_lattice;
  std::optional<MatrixBoxFootprint> matrix_box;
  IterationFiber fiber;
};

// Unified per-operand payload effect resolved at a call site.
struct ResolvedOperandFootprint {
  MemorySpace space = MemorySpace::DDR;
  bool read = false;
  bool write = false;
  bool candidate_full_overwrite = false;
  size_t root_id = 0;
  ExprPtr gm_root;
  std::optional<backend::BufferSyncByteWindow> byte_hull;
  std::optional<StridedByteFootprint> vec_lattice;
  std::optional<MatrixBoxFootprint> matrix_box;
  IterationFiber fiber;
};

// Single O(N) unified storage, view, slot, and subtile resolver shared by
// Exact import (PlacedSyncImport), Conservative dataflow (ConservativeSyncDataflow),
// and PhysicalSyncAllocation (SyncNativeClaims).
class SyncFootprintResolver {
 public:
  explicit SyncFootprintResolver(const FunctionPtr& owner, const SyncControlRangeProver* prover = nullptr,
                                 bool run_scalar_layout_check = true, std::string target = "");

  const std::vector<SyncCanonicalRoot>& Roots() const { return roots_; }
  const SyncCanonicalRoot* Root(size_t root_id) const;
  const SyncResolvedView* ViewOf(const ExprPtr& expr) const;
  const SyncResolvedView* ViewOf(const Expr* expr) const;

  std::optional<backend::BufferSyncByteWindow> ByteHull(const ExprPtr& expr) const;
  std::optional<backend::BufferSyncByteWindow> ExactFullWindow(const ExprPtr& expr) const;
  void RequireVecCopyPlacement(const backend::BufferSyncOperation& operation) const;
  void RequireWorkspaceSeparation(const backend::BufferSyncOperation& operation) const;

  ResolvedOperandFootprint ResolveOperand(const Stmt* cut, const CallPtr& call, size_t arg_idx,
                                          BufferAccess mode, MemorySpace space,
                                          bool enable_refresh_kill = true, bool disjoint_gm = false) const;

  bool OverlapsDynamicValid(MemorySpace space, const backend::BufferSyncByteWindow& window) const;
  const SyncControlRangeProver& Prover() const { return *prover_; }

 private:
  void PrescanDynamicValidTargets(const StmtPtr& stmt);
  void WalkFunction(const FunctionPtr& owner);
  void WalkStmt(const StmtPtr& stmt);
  void ResolveAssign(const AssignStmtPtr& assign);
  bool IsDynamicValidTouched(const Expr* expr) const;

  std::unique_ptr<SyncControlRangeProver> owned_prover_;
  const SyncControlRangeProver* prover_ = nullptr;
  std::string target_;

  std::vector<SyncCanonicalRoot> roots_;
  std::unordered_map<const Expr*, SyncResolvedView> views_;
  std::unordered_map<const Expr*, ExprPtr> gm_roots_;
  std::unordered_set<const Expr*> dynamic_valid_targets_;
  std::unordered_set<MemorySpace> dynamic_valid_unbounded_spaces_;
  std::vector<std::pair<MemorySpace, backend::BufferSyncByteWindow>> dynamic_valid_windows_;
};

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_FOOTPRINT_RESOLVER_H_
