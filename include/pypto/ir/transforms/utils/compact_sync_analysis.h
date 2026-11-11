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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_COMPACT_SYNC_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_COMPACT_SYNC_ANALYSIS_H_
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/utils/finite_sync_analysis.h"

namespace pypto::ir::sync {
// Inclusive caller-established bounds, keyed by the original formal parameter
// index. Supplying a bound is a proof contract, not a specialization sample.
using SyncParameterBounds = std::map<size_t, std::pair<int64_t, int64_t>>;
using SyncTranslation = std::map<size_t, int64_t>;
std::optional<int64_t> SyncCoordinateMaximum(const ExprPtr& expression);
// Collect all call/submit sites in a closed program and propagate through its
// acyclic call graph. Unknown sites invalidate the respective parameter;
// recursive components and their descendants remain unqualified.
struct SyncParameterWork {
  size_t expression_visits = 0, range_visits = 0;
  size_t call_graph_expression_visits = 0;
  size_t call_graph_nodes = 0, call_graph_edges = 0, processed_functions = 0;
};
struct SyncInvocationEvidence {
  std::map<std::string, SyncParameterBounds> parameters;
  // Inclusive logical block-index hull, proved from every incoming SPMD launch.
  // Separate from formal ABI positions; absence is an unavailable launch proof.
  std::map<std::string, std::pair<int64_t, int64_t>> block_indices;
};
SyncInvocationEvidence CollectSyncInvocationEvidence(const ProgramPtr& program,
                                                     SyncParameterWork* work = nullptr);
std::map<std::string, SyncParameterBounds> CollectSyncParameterBounds(const ProgramPtr& program,
                                                                      SyncParameterWork* work = nullptr);

// Research provider contract: static placed local writes, affine read
// rectangles, disjoint ND GM arguments, sequential constant-bound loops.
// Matrix/GM geometry uses logical element coordinates in a retained layout,
// never an absolute GM address. Placed dense Vec views instead share a physical
// byte domain (byte_coordinates), so typed aliases cannot hide overlapping bytes.
struct SyncStorage {
  std::string name;
  MemorySpace space;
  int64_t address, rows, columns, bytes;
  BufferTypePtr layout;           // Null for ND GM or the Vec byte domain.
  bool byte_coordinates = false;  // Vec: (row=0, column=physical byte address).
  // Common invocation-invariant translation of EVERY access in this domain.
  // Analysis uses relative coordinates; exports must bind these original params.
  SyncTranslation row_parameters, column_parameters;
  // Logical launch coordinate is not a formal ABI parameter. These common
  // translations require an explicit launch valuation at point-query time.
  int64_t row_block = 0, column_block = 0;
  // Original explicit INDEX formal naming a dynamic leading allocation extent.
  // rows retains its proved upper bound; accesses must fit its lower bound.
  // Boundary queries require the actual extent binding, never a sampled size.
  std::optional<size_t> row_extent_parameter;
  // Uniform physical slots: address + slot*slot_bytes. rows/columns describe
  // one slot; bytes covers the whole reservation. No slot list is materialized.
  int64_t slot_count = 1, slot_bytes = 0;
};
struct SyncAccess {
  size_t storage;
  bool write;
  int64_t row, column, row_step, column_step, rows, columns;
  bool after_first = false;     // Accumulator read, absent during initialization.
  bool exact_footprint = true;  // False: sound enclosure, selectors describe the selected model.
  // Selected slot at normalized visit i is (slot_first+i*slot_step) mod the
  // storage's slot_count. Selection is checked in its original machine type.
  int64_t slot_first = 0, slot_step = 0;
  // Original normalized coordinate controlling row_step/column_step. Null
  // means invariant geometry. It may be an enclosing frame of a finite leaf.
  ForStmtPtr coordinate_frame;
  // Exact half-open physical byte range within the canonical slot. Missing
  // means this geometry has no admitted interval export, never an empty effect.
  std::optional<std::pair<int64_t, int64_t>> physical_interval;
};
struct SyncSite {
  CallPtr original;
  CallPtr continuation;  // Original else-arm; never replaced by the first arm.
  PipeType pipe;
  std::vector<SyncAccess> accesses;
  // Qualified manual a2a3 Mmad accumulator footprint; no event-order promise.
  std::optional<size_t> protected_accumulator;
  bool initializes_accumulator = false;
  // Non-null: first-visit initialization spans this enclosing source loop
  // through the leaf's innermost frame. Original guards/arms remain owned.
  ForStmtPtr initialization_scope;
};
struct SyncLoopFrame {
  ForStmtPtr loop;
  int64_t count, start, step;
  ExprPtr start_expression;  // Ready invariant start; count is uniformly proved.
};
struct SyncSegment {
  ForStmtPtr original_loop;
  int64_t count = 1, start = 0, step = 1;
  std::vector<SyncSite> sites;
  // Perfect numerical repetition: invariant conflict-class membership, no reset
  // branch. Reads may move within whole-write classes; point selectors retain
  // their inner counter reset. Frames are outermost first; a checked mixed-radix ordinal is an
  // exact occurrence bijection, not an unfolding or a new storage abstraction.
  // Empty for the original single-loop representation.
  std::vector<SyncLoopFrame> nested_frames;
  // Positive: the innermost access/slot coordinate restarts every reset_period visits.
  // count remains the total mixed-radix visit count across enclosing frames.
  int64_t reset_period = 0;
};
// Original contiguous region tree, distinct from shared numerical summaries.
// A leaf names one segment definition. A nonleaf is a sequence, optionally
// repeated by its original loop frame. Each leaf has exactly one source path.
struct SyncRegion {
  std::optional<size_t> leaf;
  std::vector<SyncRegion> children;
  std::optional<SyncLoopFrame> repetition;
  // Qualified first-touch initializer (leaf, site) of this repetition body.
  // Later visits may share its internal mathematics with distinct entry proof.
  std::optional<std::pair<size_t, size_t>> startup;
};
struct SyncStorageClass {
  size_t storage;
  int64_t row, column, rows, columns;
};
struct CompactSyncInput {
  FunctionPtr owner;
  std::string hardware_target;
  std::vector<SyncStorage> storage;
  std::vector<SyncSegment> segments;
  SyncParameterBounds parameter_bounds;
  std::optional<std::pair<int64_t, int64_t>> block_bounds;
  std::optional<SyncRegion> region_tree;
  size_t coordinate_nodes = 0, coordinate_flatten_visits = 0, coordinate_coefficient_entries = 0;
  size_t interval_access_visits = 0, qualified_intervals = 0;
};
// Explicit qualification, not a default alias assumption. Unsupported input
// throws a documented ValueError; no effects are skipped. Enclosures are explicit
// in SyncAccess and remain selected-model, not exact physical, footprints.
CompactSyncInput ImportPlacedSync(
    const FunctionPtr& function, bool disjoint_gm, const std::string& target = "a2a3",
    const SyncParameterBounds& parameter_bounds = {},
    const std::optional<std::pair<int64_t, int64_t>>& block_bounds = std::nullopt);

using Occurrence = std::tuple<size_t, int64_t, size_t>;  // segment, visit, site
using Crossing = std::pair<Occurrence, Occurrence>;
// A Cartesian domain over named original loops. Each axis advances the two
// endpoint leaves by their own mixed-radix visit stride; these need not match.
struct ScopedDemandAxis {
  ForStmtPtr frame;
  int64_t count, source_stride, target_stride;
};
struct ScopedDemandFamily {
  Occurrence source, target;
  std::vector<ScopedDemandAxis> axes;
};
struct ScopedSyncDemands {
  std::vector<ScopedDemandFamily> families;
  size_t visited_source_nodes = 0, visited_merge_nodes = 0, coordinate_entries = 0;
};
struct DemandFamily {
  size_t source, target;
  int64_t distance;
};
// Sufficient disjointness of translated all-access envelopes within one
// segment invocation. Coordinates use that domain's qualified physical units;
// the owning segment/frame is not interchangeable with an enclosing reset.
struct VisitOwnedStorage {
  int64_t row_step, column_step, row_begin, row_end, column_begin, column_end;
};
struct RotatingSlotInfo {
  int64_t stride = 0, gcd = 1, period = 1, inverse = 0;
};
struct CompactSegmentResult {
  int64_t count;
  std::vector<PipeType> pipes;
  std::vector<DemandFamily> generators, demands;
  FiniteSyncResult body;  // One body: cell summaries and finite numerical rows.
  // Minimum nonnegative visit displacement for each event pair. A negative
  // value is unavailable reachability, not an unavailable storage domain.
  std::vector<std::vector<int64_t>> distance;
  // The rotating strategy retains two pipe-prefix indices, never all pairs.
  // Nonnegative scaled shortest distances are wide to preserve exactness.
  // Missing entries mean unreachable, not an unavailable query capability.
  bool rotating = false;
  // Numerical query representation, independent of storage recognition.
  bool core_frontiers = false;
  int64_t internal_scope_count = 0, repetitions = 1;
  std::array<std::vector<std::optional<__int128>>, 2 * kPipeCount> frontier_distances;
  std::vector<int64_t> ranks;
  std::map<size_t, RotatingSlotInfo> slot_info;
  std::map<size_t, VisitOwnedStorage> visit_owned;
  std::map<size_t, std::vector<std::pair<size_t, size_t>>> storage_accesses;
  std::array<int64_t, kPipeCount> pipe_sizes{};
  size_t rotating_fragments = 0, modular_groups = 0, sort_comparisons = 0, sweep_visits = 0;
  size_t atom_tests = 0, shortest_path_runs = 0, shortest_path_edge_visits = 0;
  size_t retained_frontier_entries = 0, quotient_edges = 0;
  std::optional<__int128> Threshold(size_t source, bool sc, size_t target, bool tc) const;
  bool Precedes(int64_t a, size_t source, bool sc, int64_t b, size_t target, bool tc) const;
};
struct PointBoundary {
  std::optional<Occurrence> first_writer, last_writer;
  std::array<std::optional<Occurrence>, kPipeCount> first_readers{}, last_readers{};
};
struct SyncStorageAccessIndex {
  // Region/site/access indices, never borrowed pointers into a moving owner.
  std::vector<std::tuple<size_t, size_t, size_t>> accesses;
  bool written = false, enclosing_motion = false;
};
class NumericalSyncHierarchy;
class CompactSyncResult {
 public:
  CompactSyncInput input;
  std::vector<CompactSegmentResult> segments;
  std::vector<SyncStorageClass> storage_classes;
  std::vector<SyncStorageAccessIndex> storage_access_index;
  std::vector<Crossing> crossing_generators, crossing_demands;
  size_t analyzed_occurrences = 0, cells = 0, graph_edges = 0;
  size_t class_tests = 0, class_lookup_tests = 0, port_pair_tests = 0, crossing_path_searches = 0;
  std::shared_ptr<const NumericalSyncHierarchy> hierarchy;
  size_t interface_cells = 0, interface_slot_visits = 0, interface_ports = 0, interface_entries = 0;
  size_t interface_queries = 0, merge_nodes = 0, crossing_tests = 0;
  size_t interface_access_visits = 0, bridge_candidates = 0, port_normalization_visits = 0;
  size_t omitted_bridge_domains = 0;
  size_t source_region_nodes = 0, source_coordinate_entries = 0;
  mutable size_t scoped_export_builds = 0;
  mutable size_t point_selector_access_visits = 0;
  mutable size_t hierarchy_query_steps = 0;
  bool HasHierarchy() const { return static_cast<bool>(hierarchy); }
  std::vector<Crossing> ExpandRetainedDemands(size_t max_payloads) const;
  // Finite crossing records only; shared repeated scopes need a separate endpoint map.
  // Unavailable is distinct from an available empty crossing inventory.
  std::optional<std::vector<Crossing>> ExportCrossingDemands() const;
  // Complete compact demand export over original source scopes. Null means
  // unavailable; a non-null empty inventory proves no modeled demands. This
  // is not a physical reuse certificate or permission to flatten scope axes.
  std::shared_ptr<const ScopedSyncDemands> ExportScopedDemands() const;
  // Sufficient uniform period-one reuse for an OWNED scoped family. A false
  // result means unproved reuse, not scarcity. Never accepts detached schemas.
  bool ScopedConsecutiveReuse(size_t family) const;
  bool ScopedAdjacentBarrier(size_t family) const;
  bool ReferenceBefore(const Occurrence& a, const Occurrence& b) const;
  mutable size_t scoped_reuse_queries = 0;
  // Exact point answer within one retained leaf scope; coordinates are local
  // to that scope, not the repeated use of its source definition.
  PointBoundary LeafBoundary(size_t region, size_t storage, int64_t row, int64_t column) const;
  double import_seconds = 0;
  double partition_seconds = 0, reduction_seconds = 0, composition_seconds = 0;
  bool Precedes(const Occurrence& a, bool sc, const Occurrence& b, bool tc) const;
  // Out-of-domain -> nullopt. Covered but untouched point -> empty boundary.
  std::optional<PointBoundary> Boundary(size_t storage, int64_t row, int64_t column) const;
  std::optional<PointBoundary> BoundaryAt(size_t storage, int64_t row, int64_t column,
                                          const std::map<size_t, int64_t>& parameters,
                                          std::optional<int64_t> block_index = std::nullopt) const;
  // Explicit original physical slot. Legacy Boundary cannot omit this coordinate
  // for a multi-slot domain. Nullopt means outside/unavailable, never absence.
  std::optional<PointBoundary> BoundarySlot(size_t storage, int64_t slot, int64_t row, int64_t column,
                                            const std::map<size_t, int64_t>& parameters = {},
                                            std::optional<int64_t> block_index = std::nullopt) const;

 private:
  mutable std::shared_ptr<const ScopedSyncDemands> scoped_demands_;
  mutable std::map<size_t, bool> scoped_reuse_;
  friend CompactSyncResult AnalyzeCompactSync(CompactSyncInput input, const std::string& composition_order);
  std::optional<PointBoundary> RelativeBoundary(size_t storage, int64_t row, int64_t column,
                                                std::optional<size_t> leaf = std::nullopt) const;
  using Event = std::pair<Occurrence, bool>;
  std::map<Event, size_t> nodes_;
  std::vector<std::vector<size_t>> graph_;
  size_t Node(const Event& event);
  bool Path(size_t a, size_t b, std::optional<std::pair<size_t, size_t>> skip = std::nullopt) const;
};
// Common retained owner with representation-specific qualified construction.
// The legacy no-slot route uses bounded two-body reduction and a research
// boundary graph with O(P^2) edges/BFS queries (P is retained port count).
// Explicit-slot routes retain numerical periodic frontiers and indexed chain
// interfaces; shared reset repetition is analyzed without unfolding visits.
// Geometry normalization and complete interface output costs are separate from
// the compact reduction bounds. Unsupported premises remain explicit refusals.
CompactSyncResult AnalyzeCompactSync(CompactSyncInput input,
                                     const std::string& composition_order = "balanced");
struct SyncInvocationCase {
  int64_t block_first, block_last;
  std::shared_ptr<const CompactSyncResult> mathematics;
};
// Shallow exact case owner for ready invocation-only activation. Cases share
// original IR; each numerical result retains its own admitted absence pattern.
// No-valuation queries cannot choose an arbitrary representative case.
struct SyncInvocationAnalysis {
  FunctionPtr owner;
  SyncParameterBounds parameter_bounds;
  std::pair<int64_t, int64_t> block_bounds;
  std::vector<SyncInvocationCase> cases;
  size_t coordinate_nodes = 0, coordinate_pieces = 0;
  std::shared_ptr<const CompactSyncResult> SelectCase(int64_t block_index,
                                                      const std::map<size_t, int64_t>& parameters) const;
};
SyncInvocationAnalysis AnalyzeSyncInvocation(const FunctionPtr& function, bool disjoint_gm,
                                             const std::string& target, const SyncParameterBounds& parameters,
                                             std::pair<int64_t, int64_t> block_bounds);
struct LogicalSyncRealization {
  FunctionPtr function;
  size_t record_sites = 0, wait_sites = 0, guarded_wait_sites = 0, carried_tokens = 0;
  size_t name_scan_visits = 0, name_scan_nodes = 0;
};
// Research-only direct token realization of this retained owner. Never reruns
// analysis or mutates it. Tokens are non-consuming; this is not flag allocation.
LogicalSyncRealization RealizeCompactSync(const CompactSyncResult& analysis);

// One physical handoff family; endpoint ordinals refer to the original loops.
// A same-pipe family has no IDs. Crossings have count=1 and can span segments.
struct PhysicalSyncFamily {
  size_t source_region, source_site, target_region, target_site;
  int64_t source_first, target_first, count;
  std::vector<int> ids;  // Logical record ordinal modulo size.
  // Empty on the original flat route; otherwise exact cached source-frame axes.
  // This inventory is constructed/consumed with the same retained case owner.
  std::vector<ScopedDemandAxis> axes;
};
struct PhysicalSyncRealization {
  FunctionPtr function;
  std::vector<PhysicalSyncFamily> families;
  size_t set_sites = 0, wait_sites = 0, barrier_sites = 0, guard_sites = 0;
  size_t reuse_queries = 0, reserved_lanes = 0;
  std::vector<std::tuple<PipeType, PipeType, int>> native_reserved_ids;  // Fixed macro owners.
  bool finite_minimum = true;
};
// All cases are qualified before one original-payload traversal. Retain each
// exact mathematical owner alongside its allocation/cut evidence; family leaf
// indices must never be interpreted in a different invocation case.
struct PhysicalSyncInvocation {
  SyncInvocationAnalysis analysis;
  FunctionPtr function;
  // Allocation/cut evidence only: per-case function fields are unset. Only
  // this invocation's function is an executable original-payload program.
  std::vector<PhysicalSyncRealization> cases;
};
PhysicalSyncInvocation RealizePhysicalInvocation(const SyncInvocationAnalysis& analysis,
                                                 const std::string& target,
                                                 const std::optional<std::vector<int>>& eligible_ids);
// Qualified direct realization of the retained owner, NOT general token-IR
// LowerSync. Optional IDs restrict (never enlarge) the target's eligible pools.
// Requires fresh entry, closed notification exit and the declared manual native
// mode. Throws on unsupported placement, unproved reuse or fixed-plan scarcity.
PhysicalSyncRealization RealizePhysicalSync(const CompactSyncResult& analysis, const std::string& target,
                                            const std::optional<std::vector<int>>& eligible_ids);
}  // namespace pypto::ir::sync
#endif  // PYPTO_IR_TRANSFORMS_UTILS_COMPACT_SYNC_ANALYSIS_H_
