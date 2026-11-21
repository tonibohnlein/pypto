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
#include <array>
#include <map>
#include <memory>
#include <set>
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
  // Original Call and Submit edges, shared with mixed-unit ownership qualification.
  std::map<std::string, std::set<std::string>> callees;
  std::map<std::string, SyncParameterBounds> parameters;
  // Inclusive logical block-index hull, proved from every incoming SPMD launch.
  // Separate from formal ABI positions; absence is an unavailable launch proof.
  std::map<std::string, std::pair<int64_t, int64_t>> block_indices;
  // Launch cardinality is independent evidence: an index hull alone cannot
  // establish one count when different callers launch different sizes.
  std::map<std::string, std::pair<int64_t, int64_t>> block_counts;
};
SyncInvocationEvidence CollectSyncInvocationEvidence(const ProgramPtr& program,
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
  // False means geometry/selectors for this original domain are unavailable,
  // never an empty footprint. Unavailable extents need not contain numeric bounds.
  bool boundary_available = true;
};
// Checked source-coordinate DAG. Loop nodes denote the ORIGINAL induction
// value, not a flattened visit ordinal. A dynamic start is a dependency in
// left; all other node operands precede their users. This retained geometry is
// distinct from a proof that a storage root has no cross-visit conflicts.
struct SyncCoordinateNode {
  enum class Kind {
    Constant,
    Parameter,
    Block,
    Loop,
    Add,
    Sub,
    Scale,
    ModularShift,
    FloorDivide,
    Remainder,
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    And,
    Or,
    Not,
    Select,
  };
  Kind kind;
  int64_t upper = 0, factor = 0, start = 0, step = 0, lower = 0;
  size_t left = 0, right = 0, parameter = 0, condition = 0;
  ForStmtPtr frame;
  bool constant = false, dynamic_start = false;
};
struct SyncCoordinateProgram {
  std::vector<SyncCoordinateNode> nodes;
};
struct SyncRowMajorTileMap {
  ForStmtPtr frame;
  size_t tile_coordinate = 0;
  std::optional<size_t> frame_start_root;
  int64_t row_tiles = 0, column_tiles = 0;
  int64_t row_scale = 0, column_scale = 0;
};
struct SyncSwizzledTileMap {
  ForStmtPtr frame;
  size_t tile_coordinate = 0;
  std::optional<size_t> frame_start_root;
  int64_t row_tiles = 0, column_tiles = 0;
  int64_t row_scale = 0, column_scale = 0;
  int64_t group_tiles = 0, group_size = 0;
  bool col_major_groups = true;
  bool snake = false;
};
struct SyncExternalFootprint {
  std::shared_ptr<const SyncCoordinateProgram> program;
  size_t row = 0, column = 0;
  std::optional<SyncRowMajorTileMap> row_major_tiles;
  std::optional<SyncSwizzledTileMap> swizzled_tiles;
};
struct SyncAccess {
  size_t storage;
  bool write;
  int64_t row, column, row_step, column_step, rows, columns;
  bool after_first = false;     // Accumulator read, absent during initialization.
  bool exact_footprint = true;  // False: sound enclosure, selectors describe the selected model.
  // Selected slot at the controlling frame's normalized visit i is
  // (slot_first+i*slot_step) mod slot_count. This frame is independent of the
  // within-slot address frame below; an ancestor may select a resident slot
  // consumed by an inner loop. Null means a constant selection. Selection is
  // checked in its original machine type, with no enclosing-frame substitution.
  int64_t slot_first = 0, slot_step = 0;
  ForStmtPtr slot_coordinate_frame;
  // Original normalized coordinate controlling row_step/column_step. Null
  // means invariant geometry. It may be an enclosing frame of a finite leaf.
  ForStmtPtr coordinate_frame;
  // Exact half-open physical byte range within the canonical slot. Missing
  // means this geometry has no admitted interval export, never an empty effect.
  std::optional<std::pair<int64_t, int64_t>> physical_interval;
  // Exact original external rectangle. When present, row/column/step are NOT
  // its geometry and may not be used as an affine selector or an empty answer.
  std::shared_ptr<const SyncExternalFootprint> external;
};
// Hardware accumulator protection needs both the physical conflict domain
// and the original layout-compatible window. Partial byte overlap does not
// establish an accumulator continuation.
struct SyncAccumulatorBinding {
  size_t storage, layout_identity;
  bool operator==(const SyncAccumulatorBinding& other) const {
    return storage == other.storage && layout_identity == other.layout_identity;
  }
  bool operator!=(const SyncAccumulatorBinding& other) const { return !(*this == other); }
};

struct SyncSite {
  CallPtr original;
  CallPtr continuation;  // Original else-arm; never replaced by the first arm.
  PipeType pipe;
  std::vector<SyncAccess> accesses;
  // Qualified manual a2a3 Mmad accumulator footprint; no event-order promise.
  std::optional<SyncAccumulatorBinding> protected_accumulator;
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
// A qualified read-only external projection removes no internal conflict:
// external memory is distinct from local spaces and every external effect reads.
// It does not provide an exact external storage interface.
// DisjointReadOnlyProjection omits only qualified input roots under the
// supplied disjoint-GM contract. Geometry for every mutable root is retained.
enum class SyncExternalPolicy { Geometry, ReadOnlyProjection, DisjointReadOnlyProjection };
// Checked change of invocation coordinates. Only Qualify can construct this
// owner; it retains original IR and independent launch-cardinality evidence.
// Payload expressions and loop frames always remain in original coordinates.
class SyncInvocationCoordinate {
 public:
  static std::shared_ptr<const SyncInvocationCoordinate> Qualify(const FunctionPtr& owner,
                                                                 const SyncParameterBounds& parameters,
                                                                 std::pair<int64_t, int64_t> blocks,
                                                                 std::optional<int64_t> launch_count);
  const FunctionPtr& Owner() const { return owner_; }
  const ExprPtr& Expression() const { return expression_; }
  const ExprPtr& Definition() const { return definition_; }
  const SyncParameterBounds& Parameters() const { return parameters_; }
  int64_t Count() const { return count_; }
  bool SupportsDefinition(const ExprPtr& expression) const { return supporting_.count(expression.get()); }
  std::optional<int64_t> Map(int64_t original_block, const std::map<size_t, int64_t>& parameters) const;

 private:
  SyncInvocationCoordinate(FunctionPtr owner, ExprPtr expression, ExprPtr definition,
                           SyncParameterBounds parameters, size_t rotation, int64_t count,
                           std::set<const Expr*> supporting)
      : owner_(std::move(owner)),
        expression_(std::move(expression)),
        definition_(std::move(definition)),
        parameters_(std::move(parameters)),
        rotation_(rotation),
        count_(count),
        supporting_(std::move(supporting)) {}
  const FunctionPtr owner_;
  const ExprPtr expression_, definition_;
  const SyncParameterBounds parameters_;
  const size_t rotation_;
  const int64_t count_;
  const std::set<const Expr*> supporting_;
};
struct CompactSyncInput {
  FunctionPtr owner;
  std::string hardware_target;
  std::vector<SyncStorage> storage;
  std::set<MemorySpace> unavailable_storage_spaces;
  std::vector<ExprPtr> projected_readonly_external_roots;
  std::vector<SyncSegment> segments;
  SyncParameterBounds parameter_bounds;
  std::optional<std::pair<int64_t, int64_t>> block_bounds;
  std::shared_ptr<const SyncInvocationCoordinate> invocation_coordinate;
  std::optional<SyncRegion> region_tree;
  std::map<const IfStmt*, bool> static_branches;
  size_t coordinate_nodes = 0, coordinate_flatten_visits = 0, coordinate_coefficient_entries = 0;
  size_t interval_access_visits = 0, qualified_intervals = 0;
};
// Explicit qualification, not a default alias assumption. Unsupported input
// throws a documented ValueError; no effects are skipped. Enclosures are explicit
// in SyncAccess and remain selected-model, not exact physical, footprints.
CompactSyncInput ImportPlacedSync(
    const FunctionPtr& function, bool disjoint_gm, const std::string& target = "a2a3",
    const SyncParameterBounds& parameter_bounds = {},
    const std::optional<std::pair<int64_t, int64_t>>& block_bounds = std::nullopt,
    std::shared_ptr<const SyncInvocationCoordinate> coordinate = nullptr,
    SyncExternalPolicy external_policy = SyncExternalPolicy::Geometry);
CompactSyncInput ImportReadonlyExternalSync(
    const FunctionPtr& function, const std::string& target, const SyncParameterBounds& parameter_bounds = {},
    const std::optional<std::pair<int64_t, int64_t>>& block_bounds = std::nullopt);

using Occurrence = std::tuple<size_t, int64_t, size_t>;  // segment, visit, site
using Crossing = std::pair<Occurrence, Occurrence>;
// A Cartesian domain over named original loops. Each axis advances the two
// endpoint leaves by their own mixed-radix visit stride; these need not match.
struct ScopedDemandAxis {
  ForStmtPtr frame;
  int64_t count, source_stride, target_stride;
  // Original normalized coordinate advances. Periodic families must not be
  // interpreted by the older contiguous-axis realization qualifier.
  int64_t source_step = 1, target_step = 1;
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
  // Repeated strategies retain two pipe-prefix indices, never all pairs.
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
  // Interval extraction work, separate from legacy atom incidences. Tree
  // entries count total allocated nodes across groups, not peak live bytes.
  size_t interval_domains = 0, interval_fragments = 0, interval_endpoints = 0;
  size_t interval_queries = 0, interval_updates = 0, interval_node_visits = 0, interval_tree_entries = 0;
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
  bool written = false, enclosing_motion = false, external_geometry = false;
};
class NumericalSyncHierarchy;
class SyncBoundaryGraph;
// Construction diagnostics are not premises of any order or export proof.
// Query counters may change after publication without invalidating evidence.
struct CompactSyncDiagnostics {
  size_t analyzed_occurrences = 0, cells = 0, graph_edges = 0;
  size_t class_tests = 0, class_lookup_tests = 0, port_pair_tests = 0, crossing_path_searches = 0;
  size_t collective_domains = 0, collective_witnesses = 0, collective_access_visits = 0;
  size_t interface_cells = 0, interface_slot_visits = 0, interface_ports = 0, interface_entries = 0;
  size_t interface_queries = 0, merge_nodes = 0, crossing_tests = 0;
  size_t interface_access_visits = 0, bridge_candidates = 0, port_normalization_visits = 0;
  size_t omitted_bridge_domains = 0;
  size_t source_region_nodes = 0, source_coordinate_entries = 0;
  size_t phase_variants = 0, finite_microbody_occurrences = 0;
  size_t scoped_export_builds = 0;
  size_t point_selector_access_visits = 0;
  size_t hierarchy_query_steps = 0;
  size_t scoped_export_attempts = 0;
  double import_seconds = 0;
  double partition_seconds = 0, reduction_seconds = 0, composition_seconds = 0;
};
// Proof-bearing data. Only internal builders obtain mutable access. Published
// results expose it through a const view; nested retained graph handles are const.
struct CompactSyncState {
  CompactSyncInput input;
  std::vector<CompactSegmentResult> segments;
  std::vector<SyncStorageClass> storage_classes;
  std::vector<SyncStorageAccessIndex> storage_access_index;
  std::vector<Crossing> crossing_generators, crossing_demands;
  std::shared_ptr<const NumericalSyncHierarchy> hierarchy;
  std::shared_ptr<const SyncBoundaryGraph> boundary_order;
};
class CompactSyncBuilder;
// Read-only queries are also usable while an internal builder is constructing
// its hierarchy. This view has no cached export: only publication may memoize it.
class CompactSyncView {
 public:
  const CompactSyncState& State() const { return *state_; }
  CompactSyncDiagnostics& Diagnostics() const { return *diagnostics_; }
  bool HasHierarchy() const { return static_cast<bool>(State().hierarchy); }
  // Finite crossing records only; shared repeated scopes need a separate endpoint map.
  // Unavailable is distinct from an available empty crossing inventory.
  std::optional<std::vector<Crossing>> ExportCrossingDemands() const;
  bool ReferenceBefore(const Occurrence& a, const Occurrence& b) const;
  bool Precedes(const Occurrence& a, bool sc, const Occurrence& b, bool tc) const;

 protected:
  // A view cannot be sliced or assigned through a published result's base:
  // that would detach its premises from the publication-only export cache.
  CompactSyncView(const CompactSyncView&) = default;
  CompactSyncView(CompactSyncView&&) = default;
  CompactSyncView& operator=(const CompactSyncView&) = default;
  CompactSyncView& operator=(CompactSyncView&&) = default;
  explicit CompactSyncView(std::shared_ptr<const CompactSyncState> state,
                           std::shared_ptr<CompactSyncDiagnostics> diagnostics)
      : state_(std::move(state)), diagnostics_(std::move(diagnostics)) {}
  std::shared_ptr<const CompactSyncState> state_;
  std::shared_ptr<CompactSyncDiagnostics> diagnostics_;
};
// A published result and its cache share one immutable premise inventory.
// Copies share both; changing caller input requires a newly analyzed owner.
class CompactSyncResult final : public CompactSyncView {
 public:
  // Complete compact demand export over original source scopes. Null means
  // unavailable; a non-null empty inventory proves no modeled demands. This
  // is not a physical reuse certificate or permission to flatten scope axes.
  std::shared_ptr<const ScopedSyncDemands> ExportScopedDemands() const;

 private:
  friend class CompactSyncBuilder;
  struct ScopedExportCache {
    bool attempted = false;
    std::shared_ptr<const ScopedSyncDemands> value;
  };
  CompactSyncResult(std::shared_ptr<const CompactSyncState> state,
                    std::shared_ptr<CompactSyncDiagnostics> diagnostics)
      : CompactSyncView(std::move(state), std::move(diagnostics)),
        scoped_export_(std::make_shared<ScopedExportCache>()) {}
  std::shared_ptr<ScopedExportCache> scoped_export_;
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
  std::shared_ptr<const SyncInvocationCoordinate> coordinate;
};
SyncInvocationAnalysis AnalyzeSyncInvocation(
    const FunctionPtr& function, bool disjoint_gm, const std::string& target,
    const SyncParameterBounds& parameters, std::pair<int64_t, int64_t> block_bounds,
    std::optional<int64_t> launch_count = std::nullopt,
    SyncExternalPolicy external_policy = SyncExternalPolicy::Geometry);
struct LogicalSyncRealization {
  FunctionPtr function;
  size_t record_sites = 0, wait_sites = 0, guarded_wait_sites = 0, carried_tokens = 0;
  size_t name_scan_visits = 0, name_scan_nodes = 0;
};
// Logical token producer for this retained owner, used by InsertSync. Never reruns
// analysis or mutates it. Tokens are non-consuming; this is not flag allocation.
LogicalSyncRealization RealizeCompactSync(const CompactSyncResult& analysis);

// Immutable original-frame context shared by endpoint domains in one region.
// Blocks are the admitted invocation case, not a new run-time assumption.
struct SyncEndpointFrame {
  FunctionPtr owner;
  std::optional<std::pair<int64_t, int64_t>> blocks;
  size_t region;
  std::vector<SyncLoopFrame> loops;
};
struct SyncEndpointDomain {
  std::shared_ptr<const SyncEndpointFrame> frame;
  // Half-open normalized digit intervals, in original outer-to-inner order.
  std::vector<std::pair<int64_t, int64_t>> ranges;
  // Exact progression in each interval, anchored at its first digit. Empty
  // means unit steps. An interval hull alone never establishes activity.
  std::vector<int64_t> steps;
};
// One physical handoff family; endpoint ordinals refer to the original loops.
// A same-pipe family has no IDs. Crossings have count=1 and can span segments.
struct PhysicalSyncFamily {
  size_t source_region, source_site, target_region, target_site;
  int64_t source_first, target_first, count;
  std::vector<int> ids;  // Logical record ordinal modulo size.
  // Empty on the original flat route; otherwise exact cached source-frame axes.
  // This inventory is constructed/consumed with the same retained case owner.
  std::vector<ScopedDemandAxis> axes;
  std::shared_ptr<const SyncEndpointDomain> source_domain, target_domain;
  // Stable identities in the retained unassigned protocol; never physical IDs.
  std::vector<size_t> protocol_uses;
};
struct DirectSyncOrder;
struct PhysicalSyncRealization;
// Allocator-produced immutable assignment. Families and their ID tables are moved
// once, never copied by schedule or diagnostic consumers. Emission is synchronous:
// the exact, unchanged input used for allocation must outlive ValidateFor/emission.
// Retaining this handle permits diagnostic reads, not later reuse of borrowed input.
class ScopedSyncProtocol;
class ScopedPaletteCertificate;
struct AssignedScopedPalette {
  std::shared_ptr<const ScopedPaletteCertificate> certificate;
  std::vector<int> ids;
};
struct AssignedRawPalette {
  std::optional<size_t> palette;
  std::vector<size_t> pieces;
};
class AssignedSyncPlan {
 public:
  const std::vector<PhysicalSyncFamily>& Families() const;
  const std::shared_ptr<const ScopedSyncProtocol>& Protocol() const;
  const std::vector<size_t>& ProtocolAssignments() const;
  const std::vector<AssignedScopedPalette>& Palettes() const;
  const std::vector<AssignedRawPalette>& RawPalettes() const;
  void ValidateFor(const CompactSyncInput& input, const std::string& target) const;

 private:
  struct State;
  std::shared_ptr<const State> state_;
  AssignedSyncPlan(const CompactSyncInput& input, std::vector<PhysicalSyncFamily> families,
                   std::map<std::pair<PipeType, PipeType>, std::vector<int>> eligible,
                   std::vector<std::tuple<PipeType, PipeType, int>> reservations,
                   std::shared_ptr<const ScopedSyncProtocol> protocol,
                   std::vector<AssignedScopedPalette> palettes);
  friend PhysicalSyncRealization AllocateDirectSync(
      const CompactSyncInput&, std::vector<PhysicalSyncFamily>, const DirectSyncOrder&, const std::string&,
      const std::optional<std::vector<int>>&, const std::map<std::pair<PipeType, PipeType>, std::set<int>>&);
};
// Diagnostic callers default to no payload duplication. The ordinary pass
// supplies its bounded automatic policy explicitly. Both use the same compiler.
// Forced modes bypass profitability, never qualification
// or these finite growth limits. Control bounds count expanded IR occurrences,
// including cloned scalar definitions and payload operand expressions. These
// caps are compiler policy, not device-derived weights or performance bounds.
struct SyncSchedulePolicy {
  std::string mode = "auto";
  size_t max_extra_payload_sites = 4096;
  size_t max_extra_control_nodes = 65536;
  bool no_clone = true;
  bool boundary_motion = true;
};
struct SyncScheduleCandidateReceipt {
  std::string loop, mode, status, reason;
  size_t payload_sites = 0, control_node_bound = 0, command_sites = 0;
  // Arithmetic, comparisons, Boolean operators, decisions, loop headers, SSA
  // coordinate and shared-value bindings. Unresolved mixed decisions make
  // profitability unproved; exact means exact modeled generated control for
  // this standalone candidate, before any certified parent-gap composition.
  std::vector<std::array<int64_t, 6>> model_cost;
  bool model_comparable = false, model_exact = false, selected = false;
};
// A parent-gap composition preserves ordered commands and can only shrink the
// combined forest. Counts are guards, expression occurrences and statements.
// Original selection costs/budgets remain scoped to standalone candidates.
struct SyncScheduleParentGapReceipt {
  std::string loop;
  size_t command_sites = 0;
  std::array<size_t, 3> before{}, after{};
};
struct ScopedPaletteResult;
struct SyncAllocationAttemptReceipt {
  PipeType source, target;
  std::string status, reason;
  size_t reuse_queries = 0;
};
struct PhysicalSyncRealization {
  std::vector<SyncAllocationAttemptReceipt> allocation_attempts;
  std::vector<std::shared_ptr<const ScopedPaletteResult>> palette_attempts;
  FunctionPtr function;
  std::shared_ptr<const AssignedSyncPlan> assignment;
  size_t set_sites = 0, wait_sites = 0, barrier_sites = 0, guard_sites = 0;
  // Per-case command sites after local prefix sharing, before invocation-case
  // coalescing. The invocation diagnostic separately counts final emitted IR.
  size_t shared_guard_sites = 0, control_statement_visits = 0;
  size_t shared_composed_guard_sites = 0;  // Sharing composed cases and adjacent command gaps.
  size_t simplified_guard_sites = 0, shared_control_values = 0;
  // Invocation-wide domain/materialization work is charged to the first case.
  // Expression visits count our inspections, not internals of the arith analyzer.
  size_t cut_domain_interval_visits = 0, cut_case_comparisons = 0, coalesced_cut_cases = 0;
  // Includes rejected decision-tree candidates. Comparisons charge whole opaque
  // command trees separately from visits to their case/sequence inventories.
  size_t case_sequence_visits = 0, case_sequence_comparisons = 0, factored_case_cuts = 0;
  size_t shared_disjoint_case_cuts = 0;
  size_t endpoint_domain_entries = 0, endpoint_predicates_constructed = 0, normalized_endpoint_bounds = 0;
  size_t control_descriptors = 0, materialized_control_statements = 0;
  size_t invocation_scheduled_loops = 0, invocation_schedule_groups = 0, invocation_schedule_visits = 0,
         invocation_schedule_comparisons = 0, invocation_schedule_model_saving = 0;
  size_t schedule_name_scans = 0, schedule_name_visits = 0;
  std::vector<SyncScheduleCandidateReceipt> schedule_choices;
  std::vector<SyncScheduleParentGapReceipt> schedule_parent_gaps;
  size_t schedule_inspected_descriptors = 0, schedule_inspected_expressions = 0;
  size_t schedule_emitted_payload_sites = 0, schedule_emitted_control_nodes = 0;
  // Growth debits are shared across retained children of the same owner.
  size_t schedule_payload_growth_debit = 0, schedule_control_growth_debit = 0;
  size_t regular_schedule_candidates = 0, regular_schedule_loops = 0, regular_schedule_boundaries = 0;
  size_t regular_schedule_body_copies = 0, regular_schedule_payload_copies = 0,
         regular_schedule_model_saving = 0;
  size_t assigned_schedule_endpoints = 0;  // Symbolic uses before ID/guard expansion.
  size_t schedule_candidates = 0, scheduled_loops = 0, schedule_boundary_commands = 0,
         schedule_size_refusals = 0;
  size_t schedule_body_visits = 0, schedule_descriptor_visits = 0;
  size_t schedule_original_nodes = 0, schedule_candidate_nodes = 0;
  size_t control_expression_visits = 0;
  // Statement cuts and capture edges visited when building closed-prefix suffixes.
  size_t closed_prefix_suffix_visits = 0;
  size_t reuse_queries = 0, reserved_lanes = 0;
  std::vector<std::tuple<PipeType, PipeType, int>> native_reserved_ids;  // Fixed macro owners.
  bool finite_minimum = true;
};
}  // namespace pypto::ir::sync
#endif  // PYPTO_IR_TRANSFORMS_UTILS_COMPACT_SYNC_ANALYSIS_H_
