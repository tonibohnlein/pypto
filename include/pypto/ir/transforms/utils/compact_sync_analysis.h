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
// Collect all direct call/submit sites. Unknown sites invalidate the respective
// parameter evidence; neither SSA names nor tensor dimensions imply a bound.
struct SyncParameterWork {
  size_t expression_visits = 0, range_visits = 0;
};
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
};
struct SyncAccess {
  size_t storage;
  bool write;
  int64_t row, column, row_step, column_step, rows, columns;
  bool after_first = false;     // Accumulator read, absent during initialization.
  bool exact_footprint = true;  // False: sound enclosure, selectors describe the selected model.
};
struct SyncSite {
  CallPtr original;
  CallPtr continuation;  // Original else-arm; never replaced by the first arm.
  PipeType pipe;
  std::vector<SyncAccess> accesses;
  // Qualified manual a2a3 Mmad accumulator footprint; no event-order promise.
  std::optional<size_t> protected_accumulator;
  bool initializes_accumulator = false;
};
struct SyncSegment {
  ForStmtPtr original_loop;
  int64_t count = 1, start = 0, step = 1;
  std::vector<SyncSite> sites;
};
struct CompactSyncInput {
  FunctionPtr owner;
  std::string hardware_target;
  std::vector<SyncStorage> storage;
  std::vector<SyncSegment> segments;
  SyncParameterBounds parameter_bounds;
  size_t coordinate_nodes = 0, coordinate_flatten_visits = 0, coordinate_coefficient_entries = 0;
};
// Explicit qualification, not a default alias assumption. Unsupported input
// throws a documented ValueError; no effects are skipped. Enclosures are explicit
// in SyncAccess and remain selected-model, not exact physical, footprints.
CompactSyncInput ImportPlacedSync(const FunctionPtr& function, bool disjoint_gm,
                                  const std::string& target = "a2a3",
                                  const SyncParameterBounds& parameter_bounds = {});

using Occurrence = std::tuple<size_t, int64_t, size_t>;  // segment, visit, site
using Crossing = std::pair<Occurrence, Occurrence>;
struct DemandFamily {
  size_t source, target;
  int64_t distance;
};
struct CompactSegmentResult {
  int64_t count;
  std::vector<PipeType> pipes;
  std::vector<DemandFamily> generators, demands;
  FiniteSyncResult body;  // One body: cell summaries and finite numerical rows.
  // Minimum nonnegative visit displacement for each event pair. A negative
  // value is unavailable reachability, not an unavailable storage domain.
  std::vector<std::vector<int64_t>> distance;
  bool Precedes(int64_t a, size_t source, bool sc, int64_t b, size_t target, bool tc) const;
};
struct PointBoundary {
  std::optional<Occurrence> first_writer, last_writer;
  std::array<std::optional<Occurrence>, kPipeCount> first_readers{}, last_readers{};
};
class CompactSyncResult {
 public:
  CompactSyncInput input;
  std::vector<CompactSegmentResult> segments;
  std::vector<Crossing> crossing_generators, crossing_demands;
  size_t analyzed_occurrences = 0, cells = 0, graph_edges = 0;
  size_t class_tests = 0, class_lookup_tests = 0, port_pair_tests = 0, crossing_path_searches = 0;
  double import_seconds = 0;
  double partition_seconds = 0, reduction_seconds = 0, composition_seconds = 0;
  bool Precedes(const Occurrence& a, bool sc, const Occurrence& b, bool tc) const;
  // Out-of-domain -> nullopt. Covered but untouched point -> empty boundary.
  std::optional<PointBoundary> Boundary(size_t storage, int64_t row, int64_t column) const;
  std::optional<PointBoundary> BoundaryAt(size_t storage, int64_t row, int64_t column,
                                          const std::map<size_t, int64_t>& parameters) const;

 private:
  friend CompactSyncResult AnalyzeCompactSync(CompactSyncInput input);
  std::optional<PointBoundary> RelativeBoundary(size_t storage, int64_t row, int64_t column) const;
  using Event = std::pair<Occurrence, bool>;
  std::map<Event, size_t> nodes_;
  std::vector<std::vector<size_t>> graph_;
  size_t Node(const Event& event);
  bool Path(size_t a, size_t b, std::optional<std::pair<size_t, size_t>> skip = std::nullopt) const;
};
// Bounded two-body reduction: every mutable class is visited by a writer in
// every body, so storage generators span at most one visit. Alternative paths
// lie in the same two-body interval. No trip count is unfolded. The research
// boundary graph uses O(P^2) edges and BFS queries; it is not a claim of a
// linear general regional implementation. P is retained boundary port count.
CompactSyncResult AnalyzeCompactSync(CompactSyncInput input);
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
  std::vector<int> ids;  // Source ordinal relative to source_first, modulo size.
};
struct PhysicalSyncRealization {
  FunctionPtr function;
  std::vector<PhysicalSyncFamily> families;
  size_t set_sites = 0, wait_sites = 0, barrier_sites = 0, guard_sites = 0;
  size_t reuse_queries = 0, reserved_lanes = 0;
  std::vector<std::tuple<PipeType, PipeType, int>> native_reserved_ids;  // Fixed macro owners.
  bool finite_minimum = true;
};
// Qualified direct realization of the retained owner, NOT general token-IR
// LowerSync. Optional IDs restrict (never enlarge) the target's eligible pools.
// Requires fresh entry, closed notification exit and the declared manual native
// mode. Throws on unsupported placement, unproved reuse or fixed-plan scarcity.
PhysicalSyncRealization RealizePhysicalSync(const CompactSyncResult& analysis, const std::string& target,
                                            const std::optional<std::vector<int>>& eligible_ids);
}  // namespace pypto::ir::sync
#endif  // PYPTO_IR_TRANSFORMS_UTILS_COMPACT_SYNC_ANALYSIS_H_
