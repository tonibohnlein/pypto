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

/**
 * @file testing.cpp
 * @brief Implementation of Python bindings for testing utilities
 *
 * This module provides internal testing utilities that should not be used
 * in production code. It is exposed as pypto.testing in Python.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>  // NOLINT(misc-include-cleaner) -- registers shared_ptr casters
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner) -- registers std::string casters
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <any>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../module.h"
#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"
#include "pypto/ir/transforms/dsa/reuse_penalty_recognizer.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/finite_sync_analysis.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/scoped_token_families.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

nb::dict DescribeCompactSync(const ir::sync::CompactSyncResult& result) {
  nb::dict out;
  out["interval_access_visits"] = result.input.interval_access_visits;
  out["collective_domains"] = result.collective_domains;
  out["collective_witnesses"] = result.collective_witnesses;
  out["collective_access_visits"] = result.collective_access_visits;
  out["qualified_intervals"] = result.input.qualified_intervals;
  out["guarantee"] = "minimum demands for declared footprints and core native chains";
  out["realization"] = "not requested";
  out["analysis_available"] = result.segments.size() == result.input.segments.size();
  out["parameter_bounds"] = result.input.parameter_bounds;
  out["coordinate_nodes"] = result.input.coordinate_nodes;
  out["coordinate_flatten_visits"] = result.input.coordinate_flatten_visits;
  out["coordinate_coefficient_entries"] = result.input.coordinate_coefficient_entries;
  nb::list storage, regions;
  for (const auto& s : result.input.storage) {
    nb::dict item;
    item["name"] = s.name;
    item["space"] = s.space;
    item["address"] = s.address;
    item["rows"] = s.rows;
    item["row_extent_parameter"] = s.row_extent_parameter;
    item["columns"] = s.columns;
    item["bytes"] = s.bytes;
    item["slot_count"] = s.slot_count;
    item["slot_bytes"] = s.slot_bytes;
    item["byte_coordinates"] = s.byte_coordinates;
    item["row_parameters"] = s.row_parameters;
    item["column_parameters"] = s.column_parameters;
    item["row_block"] = s.row_block;
    item["column_block"] = s.column_block;
    item["block_origin"] = result.input.block_bounds ? result.input.block_bounds->first : 0;
    storage.append(item);
  }
  for (size_t r = 0; r < result.input.segments.size(); ++r) {
    const auto& source = result.input.segments[r];
    const auto* analysis = r < result.segments.size() ? &result.segments[r] : nullptr;
    nb::dict region;
    nb::list sites, families, generators;
    region["count"] = source.count;
    region["start"] = source.start;
    region["step"] = source.step;
    region["reset_period"] = source.reset_period;
    region["loop"] = source.original_loop;
    nb::list frames;
    for (const auto& frame : source.nested_frames)
      frames.append(nb::make_tuple(frame.loop, frame.count, frame.start, frame.step));
    region["nested_frames"] = frames;
    nb::list start_expressions;
    for (const auto& frame : source.nested_frames) start_expressions.append(frame.start_expression);
    region["frame_start_expressions"] = start_expressions;
    for (const auto& s : source.sites) {
      nb::dict site;
      nb::list accesses, precision, slots, access_frames, intervals;
      site["original"] = s.original;
      site["continuation"] = s.continuation;
      site["initialization_scope"] = s.initialization_scope;
      site["pipe"] = s.pipe;
      for (const auto& a : s.accesses) {
        access_frames.append(a.coordinate_frame);
        intervals.append(nb::cast(a.physical_interval));
        slots.append(nb::make_tuple(a.slot_first, a.slot_step));
        precision.append(a.exact_footprint ? "exact" : "enclosure");
        accesses.append(nb::make_tuple(a.storage, a.write, a.row, a.column, a.row_step, a.column_step, a.rows,
                                       a.columns, a.after_first));
      }
      site["physical_intervals"] = intervals;
      site["slot_selections"] = slots;
      site["access_precision"] = precision;
      site["accesses"] = accesses;
      site["access_frames"] = access_frames;
      sites.append(site);
    }
    if (analysis) {
      for (auto f : analysis->demands) families.append(nb::make_tuple(f.source, f.target, f.distance));
      for (auto f : analysis->generators) generators.append(nb::make_tuple(f.source, f.target, f.distance));
    }
    region["sites"] = sites;
    region["demands"] = analysis ? nb::object(families) : nb::object(nb::none());
    region["generators"] = analysis ? nb::object(generators) : nb::object(nb::none());
    region["distance_entries"] = analysis ? analysis->distance.size() * analysis->distance.size() : 0;
    if (analysis) {
      region["rotating"] = analysis->rotating;
      region["core_frontiers"] = analysis->core_frontiers;
      region["internal_scope_count"] = analysis->internal_scope_count;
      region["repetitions"] = analysis->repetitions;
      nb::dict ownership;
      for (const auto& [storage, proof] : analysis->visit_owned) {
        ownership[nb::int_(storage)] = nb::make_tuple(proof.row_step, proof.column_step, proof.row_begin,
                                                      proof.row_end, proof.column_begin, proof.column_end);
      }
      region["visit_owned"] = ownership;
      region["rotating_fragments"] = analysis->rotating_fragments;
      region["atom_tests"] = analysis->atom_tests;
      region["interval_domains"] = analysis->interval_domains;
      region["interval_fragments"] = analysis->interval_fragments;
      region["interval_endpoints"] = analysis->interval_endpoints;
      region["interval_queries"] = analysis->interval_queries;
      region["interval_updates"] = analysis->interval_updates;
      region["interval_node_visits"] = analysis->interval_node_visits;
      region["interval_tree_entries"] = analysis->interval_tree_entries;
      region["quotient_edges"] = analysis->quotient_edges;
      region["modular_groups"] = analysis->modular_groups;
      region["sort_comparisons"] = analysis->sort_comparisons;
      region["sweep_visits"] = analysis->sweep_visits;
      region["shortest_path_runs"] = analysis->shortest_path_runs;
      region["shortest_path_edge_visits"] = analysis->shortest_path_edge_visits;
      region["retained_frontier_entries"] = analysis->retained_frontier_entries;
    }
    regions.append(region);
  }
  out["storage"] = storage;
  out["regions"] = regions;
  out["crossing_demands"] =
      result.HasHierarchy() ? nb::object(nb::none()) : nb::cast(result.crossing_demands);
  out["crossing_generators"] =
      result.HasHierarchy() ? nb::object(nb::none()) : nb::cast(result.crossing_generators);
  out["analyzed_occurrences"] = result.analyzed_occurrences;
  out["hardware_target"] = result.input.hardware_target;
  out["cells"] = result.cells;
  out["boundary_graph_edges"] = result.graph_edges;
  out["class_tests"] = result.class_tests;
  out["class_lookup_tests"] = result.class_lookup_tests;
  out["port_pair_tests"] = result.port_pair_tests;
  out["crossing_path_searches"] = result.crossing_path_searches;
  out["import_seconds"] = result.import_seconds;
  out["partition_seconds"] = result.partition_seconds;
  out["reduction_seconds"] = result.reduction_seconds;
  out["composition_seconds"] = result.composition_seconds;
  out["hierarchical"] = result.HasHierarchy();
  out["source_region_nodes"] = result.source_region_nodes;
  out["source_coordinate_entries"] = result.source_coordinate_entries;
  out["scoped_export_builds"] = result.scoped_export_builds;
  out["scoped_reuse_queries"] = result.scoped_reuse_queries;
  out["point_selector_access_visits"] = result.point_selector_access_visits;
  out["interface_cells"] = result.interface_cells;
  out["omitted_bridge_domains"] = result.omitted_bridge_domains;
  out["interface_slot_visits"] = result.interface_slot_visits;
  out["interface_ports"] = result.interface_ports;
  out["interface_entries"] = result.interface_entries;
  out["interface_queries"] = result.interface_queries;
  out["merge_nodes"] = result.merge_nodes;
  out["crossing_tests"] = result.crossing_tests;
  out["hierarchy_query_steps"] = result.hierarchy_query_steps;
  out["interface_access_visits"] = result.interface_access_visits;
  out["bridge_candidates"] = result.bridge_candidates;
  out["port_normalization_visits"] = result.port_normalization_visits;
  return out;
}
nb::object CompactBoundary(const ir::sync::CompactSyncResult& result, size_t storage, int64_t row,
                           int64_t col, const std::map<size_t, int64_t>& parameters,
                           std::optional<int64_t> block_index) {
  auto b = result.BoundaryAt(storage, row, col, parameters, block_index);
  if (!b) return nb::none();
  nb::dict out;
  out["first_writer"] = b->first_writer;
  out["last_writer"] = b->last_writer;
  out["first_readers"] = b->first_readers;
  out["last_readers"] = b->last_readers;
  return out;
}
nb::object CompactSlotBoundary(const ir::sync::CompactSyncResult& result, size_t storage, int64_t slot,
                               int64_t row, int64_t col, const std::map<size_t, int64_t>& parameters,
                               std::optional<int64_t> block_index) {
  auto boundary = result.BoundarySlot(storage, slot, row, col, parameters, block_index);
  if (!boundary) return nb::none();
  nb::dict out;
  out["first_writer"] = boundary->first_writer;
  out["last_writer"] = boundary->last_writer;
  out["first_readers"] = boundary->first_readers;
  out["last_readers"] = boundary->last_readers;
  return out;
}

/// Numerical provider seam: tests supply qualified occurrences and compare
/// against independently constructed event graphs, not compiler generators.
nb::dict AnalyzeFiniteSyncForTesting(
    const std::vector<std::tuple<ir::PipeType, bool, std::vector<uint64_t>, std::vector<uint64_t>>>&
        occurrences,
    const std::vector<uint64_t>& covered_cells, bool exact_accesses,
    const std::vector<std::pair<size_t, size_t>>& prerequisites,
    const std::vector<std::pair<size_t, size_t>>& native_order,
    const std::vector<std::tuple<size_t, bool, size_t, bool>>& queries,
    const std::vector<uint64_t>& boundary_cells) {
  ir::sync::FiniteSyncInput input;
  input.covered_cells = covered_cells;
  input.exact_accesses = exact_accesses;
  input.prerequisites = prerequisites;
  input.native_order = native_order;
  for (const auto& [pipe, present, reads, writes] : occurrences) {
    input.occurrences.push_back({pipe, present, reads, writes});
  }
  const auto result = ir::sync::AnalyzeFiniteSync(input);
  nb::dict output;
  output["exact_accesses"] = result.exact_accesses;
  output["generators"] = nb::cast(result.generators);
  output["demands"] = nb::cast(result.demands);
  output["starts"] = nb::cast(result.starts);
  output["completions"] = nb::cast(result.completions);
  nb::list answers;
  for (const auto& [a, ac, b, bc] : queries) answers.append(result.Precedes(a, ac, b, bc));
  output["queries"] = answers;
  nb::list boundaries;
  for (uint64_t cell : boundary_cells) {
    const auto boundary = result.Boundary(cell);
    if (!boundary) {
      boundaries.append(nb::none());
      continue;
    }
    nb::dict entry;
    entry["first_writer"] = nb::cast(boundary->first_writer);
    entry["last_writer"] = nb::cast(boundary->last_writer);
    entry["first_readers"] = nb::cast(boundary->first_readers);
    entry["last_readers"] = nb::cast(boundary->last_readers);
    boundaries.append(entry);
  }
  output["boundaries"] = boundaries;
  nb::dict work;
  work["access_entries"] = result.work.access_entries;
  work["cell_visits"] = result.work.cell_visits;
  work["reader_visits"] = result.work.reader_visits;
  work["witnesses"] = result.work.witnesses;
  work["candidate_tests"] = result.work.candidate_tests;
  work["demand_joins"] = result.work.demand_joins;
  output["work"] = work;
  return output;
}

/// Validate malformed registrations without installing them in the singleton.
/// This exercises the exact checks used by registry creation and import.
void ValidateBufferOpContractForTesting(
    ir::OpIRStage stage, std::optional<size_t> arity, const std::vector<ir::ExprPtr>& args,
    const ir::TypePtr& result_type,
    const std::vector<std::tuple<size_t, bool, ir::BufferAccess, ir::BufferAccess>>& effects,
    const std::vector<std::tuple<size_t, ir::BufferResultBehavior, std::optional<size_t>>>& results,
    bool internal_only) {
  ir::OpRegistryEntry entry;
  entry.set_ir_stage(stage).set_internal_only(internal_only).no_argument();
  if (arity.has_value()) entry.set_output_arity(*arity);
  for (size_t i = 0; i < args.size(); ++i) entry.add_argument(std::to_string(i), "Test argument");
  for (const auto& [index, non_memory, data, metadata] : effects) {
    if (non_memory) {
      CHECK(data == ir::BufferAccess::None && metadata == ir::BufferAccess::None)
          << "A non-memory operand cannot declare memory effects";
      entry.set_buffer_non_memory_arg(index);
    } else {
      entry.set_buffer_arg_effect(index, data, metadata);
    }
  }
  for (const auto& [index, behavior, alias_arg] : results) {
    entry.set_buffer_result_behavior(index, behavior, alias_arg);
  }
  entry.ValidateIRStage();
  entry.ValidateCall(args, result_type, ir::Span::unknown());
}

/// Exercise typing-mode registration failures without modifying the singleton.
void ValidateOpTypeRegistrationForTesting(ir::OpIRStage stage, bool internal_only,
                                          const std::vector<std::string>& typing_modes) {
  ir::OpRegistryEntry entry;
  entry.set_ir_stage(stage).set_internal_only(internal_only).no_argument();
  if (stage == ir::OpIRStage::Buffer) {
    entry.set_output_arity(0).set_buffer_result_behavior(ir::BufferResultBehavior::None);
  }
  for (const auto& mode : typing_modes) {
    if (mode == "deduced") {
      entry.f_deduce_type(
          [](const std::vector<ir::ExprPtr>&, const std::vector<std::pair<std::string, std::any>>&) {
            return ir::GetVoidType();
          });
    } else if (mode == "explicit") {
      entry.f_validate_explicit_type([](const std::vector<ir::ExprPtr>&,
                                        const std::vector<std::pair<std::string, std::any>>&,
                                        const ir::TypePtr&) {});
    } else {
      CHECK(false) << "Unknown test typing mode '" << mode << "'";
    }
  }
  entry.ValidateIRStage();
}

// ============================================================================
// Helper functions to demonstrate error raising from C++
// ============================================================================

/**
 * @brief Raise a ValueError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_value_error(const std::string& message) { throw pypto::ValueError(message); }

/**
 * @brief Raise a TypeError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_type_error(const std::string& message) { throw pypto::TypeError(message); }

/**
 * @brief Raise a RuntimeError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_runtime_error(const std::string& message) { throw pypto::RuntimeError(message); }

/**
 * @brief Raise a NotImplementedError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_not_implemented_error(const std::string& message) {
  throw pypto::NotImplementedError(message);
}

/**
 * @brief Raise an IndexError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_index_error(const std::string& message) { throw pypto::IndexError(message); }

/**
 * @brief Raise a generic Error from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_generic_error(const std::string& message) { throw pypto::Error(message); }

/**
 * @brief Raise an AssertionError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_assertion_error(const std::string& message) { throw pypto::AssertionError(message); }

/**
 * @brief Raise an InternalError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_internal_error(const std::string& message) { throw pypto::InternalError(message); }

[[noreturn]] void raise_internal_error_with_span(const std::string& message, const std::string& filename,
                                                 int line, int col) {
  ir::Span span(filename, line, col);
  INTERNAL_CHECK_SPAN(false, span) << message;
}

/**
 * @brief Throw the PyPTO exception named by `kind`
 *
 * Sole throw site for `rethrow_with_message` below, so tests can assert that the
 * reported stack trace still names this function after the rethrow.
 *
 * @param kind Exception class name, e.g. "InternalError"
 * @param message Error message to include in the exception
 */
[[noreturn]] void RaiseByKind(const std::string& kind, const std::string& message) {
  if (kind == "ValueError") throw pypto::ValueError(message);
  if (kind == "TypeError") throw pypto::TypeError(message);
  if (kind == "RuntimeError") throw pypto::RuntimeError(message);
  if (kind == "NotImplementedError") throw pypto::NotImplementedError(message);
  if (kind == "IndexError") throw pypto::IndexError(message);
  if (kind == "AssertionError") throw pypto::AssertionError(message);
  if (kind == "InternalError") throw pypto::InternalError(message);
  if (kind == "Error") throw pypto::Error(message);
  throw pypto::ValueError("Unknown exception kind for testing: " + kind);
}

/**
 * @brief Raise `kind`, then route it through Error::RethrowWithMessage
 *
 * Exercises the trace-preserving typed rethrow that OpRegistry::CreateImpl depends on:
 * the exception reaching Python must keep the concrete type and the frames captured at
 * the original throw inside RaiseByKind, while carrying the replacement message.
 *
 * @param kind Exception class name to raise
 * @param original Message given to the original exception
 * @param replacement Message the rethrown exception should carry instead
 */
[[noreturn]] void rethrow_with_message(const std::string& kind, const std::string& original,
                                       const std::string& replacement) {
  try {
    RaiseByKind(kind, original);
  } catch (const pypto::Error& e) {
    e.RethrowWithMessage(replacement);
  }
  // `RethrowWithMessage` is `[[noreturn]]`, but the call above is virtual and an override
  // is not required to carry the attribute, so the compiler still treats this point as
  // reachable. Mark it explicitly instead of letting a `[[noreturn]]` function fall off
  // its end.
  INTERNAL_UNREACHABLE << "Error::RethrowWithMessage returned for kind '" << kind << "'";
}

/**
 * @brief Return DSA-RP recognizer output without running placement.
 *
 * This intentionally lives in the internal testing module: production callers
 * consume the same recognizer through MemRefDsaAdapter, while unit tests need
 * to distinguish edge construction from solver tie-breaking.
 */
nb::list RecognizeDsaReusePenaltiesForTesting(const ir::FunctionPtr& func) {
  const ir::dsa_adapter::AllocationPlan plan = ir::dsa_adapter::BuildDsaAllocationPlan(func);
  const auto penalties =
      ir::dsa_adapter::RecognizeReusePenalties(func, plan, *backend::BackendConfig::GetBackend());

  nb::list result;
  for (const auto& penalty : penalties) {
    INTERNAL_CHECK(penalty.first_interval < plan.intervals.size());
    INTERNAL_CHECK(penalty.second_interval < plan.intervals.size());
    nb::dict edge;
    edge["first_interval"] = penalty.first_interval;
    edge["second_interval"] = penalty.second_interval;
    edge["first_name"] = plan.intervals[penalty.first_interval].variable->name_hint_;
    edge["second_name"] = plan.intervals[penalty.second_interval].variable->name_hint_;
    edge["cost"] = penalty.cost;
    result.append(std::move(edge));
  }
  return result;
}

/**
 * @brief Return exact backend pipe inference for a Call, or None.
 */
nb::object TryInferPipeForTesting(const ir::CallPtr& call) {
  const auto pipe = backend::BackendConfig::GetBackend()->TryInferPipe(call);
  if (!pipe) return nb::none();
  return nb::int_(static_cast<int>(*pipe));
}

/**
 * @brief Return an operation's registered execution-memory-access evidence.
 */
std::string GetExecutionMemoryAccessEvidenceForTesting(const std::string& op_name) {
  const auto& registry = ir::OpRegistry::GetInstance();
  CHECK(registry.IsRegistered(op_name)) << "Unknown operation '" << op_name << "'";
  switch (registry.GetEntry(op_name).GetExecutionMemoryAccessEvidence()) {
    case ir::ExecutionMemoryAccessEvidence::Unknown:
      return "unknown";
    case ir::ExecutionMemoryAccessEvidence::Functional:
      return "functional";
    case ir::ExecutionMemoryAccessEvidence::NoAccess:
      return "no_access";
  }
  INTERNAL_UNREACHABLE << "Unknown execution-memory-access evidence";
}

/**
 * @brief Spell a CoreAffinity as the lowercase string the Python tests assert on.
 */
const char* CoreAffinityToString(ir::core_affinity::CoreAffinity affinity) {
  switch (affinity) {
    case ir::core_affinity::CoreAffinity::CUBE:
      return "cube";
    case ir::core_affinity::CoreAffinity::VECTOR:
      return "vector";
    case ir::core_affinity::CoreAffinity::SHARED:
      return "shared";
    case ir::core_affinity::CoreAffinity::MIXED:
      return "mixed";
  }
  INTERNAL_UNREACHABLE << "Unknown core affinity";
}

/**
 * @brief Return an operation's explicitly declared core affinity, or None.
 *
 * None means the op declares no affinity and ClassifyCallAffinity derives it
 * from the call (memory spec, operand tiles, result tile).
 */
nb::object GetDeclaredCoreAffinityForTesting(const std::string& op_name) {
  const auto& registry = ir::OpRegistry::GetInstance();
  CHECK(registry.IsRegistered(op_name)) << "Unknown operation '" << op_name << "'";
  const auto affinity = registry.GetEntry(op_name).GetCoreAffinity();
  if (!affinity) return nb::none();
  return nb::str(CoreAffinityToString(*affinity));
}

/**
 * @brief Return the core affinity ClassifyCallAffinity derives for a Call.
 *
 * Unlike get_declared_core_affinity this is the *effective* placement: it runs
 * the full classification chain (declared affinity, then the dynamic special
 * cases, then output memory spec, first tile argument, and result tile memory),
 * so it depends on how far the call has been lowered.
 */
std::string ClassifyCallAffinityForTesting(const ir::CallPtr& call) {
  return CoreAffinityToString(ir::core_affinity::ClassifyCallAffinity(call));
}

/**
 * @brief Return whether an operation must not run on a second core.
 *
 * The set_no_duplicate() axis: replicating such an op onto the other lane of a
 * mixed kernel changes what the program means. Read by LowerAutoVectorSplit's
 * pl.split_aiv region placement stamp.
 */
bool IsNoDuplicateOpForTesting(const std::string& op_name) {
  const auto& registry = ir::OpRegistry::GetInstance();
  CHECK(registry.IsRegistered(op_name)) << "Unknown operation '" << op_name << "'";
  return registry.GetEntry(op_name).IsNoDuplicate();
}

// ============================================================================
// Module binding
// ============================================================================

nb::dict DescribePhysicalSync(const ir::sync::PhysicalSyncRealization& result) {
  nb::dict out;
  out["function"] = result.function;
  out["set_sites"] = result.set_sites;
  out["wait_sites"] = result.wait_sites;
  out["barrier_sites"] = result.barrier_sites;
  out["guard_sites"] = result.guard_sites;
  out["reuse_queries"] = result.reuse_queries;
  out["reserved_lanes"] = result.reserved_lanes;
  out["native_reserved_ids"] = result.native_reserved_ids;
  out["finite_minimum"] = result.finite_minimum;
  nb::list families;
  for (const auto& f : result.families) {
    nb::dict row;
    row["source"] = nb::make_tuple(f.source_region, f.source_first, f.source_site);
    row["target"] = nb::make_tuple(f.target_region, f.target_first, f.target_site);
    row["count"] = f.count;
    row["ids"] = nb::cast(f.ids);
    nb::list axes;
    for (const auto& axis : f.axes)
      axes.append(nb::make_tuple(axis.frame, axis.count, axis.source_stride, axis.target_stride));
    row["axes"] = axes;
    families.append(row);
  }
  out["families"] = families;
  out["guarantee"] = "equal selected order under qualified direct-command model";
  out["native_mode"] = "manual a2a3; fresh flags at entry; closed notifications at exit";
  return out;
}

void BindTesting(nb::module_& m) {
  // Create a protected submodule for testing utilities
  // This will be accessible as pypto.testing in Python
  nb::module_ testing = m.def_submodule("testing", "Internal testing utilities (do not use in production)");

  testing.def("validate_buffer_op_contract", &ValidateBufferOpContractForTesting, nb::arg("stage"),
              nb::arg("arity").none(), nb::arg("args"), nb::arg("result_type"), nb::arg("effects"),
              nb::arg("results"), nb::arg("internal_only") = true,
              "Validate a local operator contract without changing the global registry");
  testing.def("validate_op_type_registration", &ValidateOpTypeRegistrationForTesting, nb::arg("stage"),
              nb::arg("internal_only"), nb::arg("typing_modes"),
              "Validate local operator typing modes without changing the global registry");
  testing.def(
      "validate_buffer_call",
      [](const ir::CallPtr& call) { ir::OpRegistry::GetInstance().ValidateBufferCall(call); },
      nb::arg("call"), "Validate a stored Buffer call against its registered schema");

  // Register error-raising helper functions
  testing.def("raise_value_error", &raise_value_error, nb::arg("message"),
              "Raise a ValueError from C++ for testing error handling");

  testing.def("raise_type_error", &raise_type_error, nb::arg("message"),
              "Raise a TypeError from C++ for testing error handling");

  testing.def("raise_runtime_error", &raise_runtime_error, nb::arg("message"),
              "Raise a RuntimeError from C++ for testing error handling");

  testing.def("raise_not_implemented_error", &raise_not_implemented_error, nb::arg("message"),
              "Raise a NotImplementedError from C++ for testing error handling");

  testing.def("raise_index_error", &raise_index_error, nb::arg("message"),
              "Raise an IndexError from C++ for testing error handling");

  testing.def("raise_generic_error", &raise_generic_error, nb::arg("message"),
              "Raise a generic Error from C++ for testing error handling");

  testing.def("raise_assertion_error", &raise_assertion_error, nb::arg("message"),
              "Raise an AssertionError from C++ for testing error handling");

  testing.def("raise_internal_error", &raise_internal_error, nb::arg("message"),
              "Raise an InternalError from C++ for testing error handling");

  testing.def("raise_internal_error_with_span", &raise_internal_error_with_span, nb::arg("message"),
              nb::arg("filename"), nb::arg("line"), nb::arg("col"),
              "Raise an InternalError with IR source span for testing");

  testing.def("rethrow_with_message", &rethrow_with_message, nb::arg("kind"), nb::arg("original"),
              nb::arg("replacement"),
              "Raise `kind` and rethrow it via Error::RethrowWithMessage for testing");

  testing.def("recognize_dsa_reuse_penalties", &RecognizeDsaReusePenaltiesForTesting, nb::arg("function"),
              "Return recognized DSA-RP edges without running placement");

  nb::class_<ir::sync::CompactSyncResult>(testing, "CompactSyncAnalysis")
      .def("export_crossing_demands", &ir::sync::CompactSyncResult::ExportCrossingDemands,
           "Export retained crossings for a flat regional use; unavailable is distinct from empty")
      .def(
          "realize",
          [](const ir::sync::CompactSyncResult& analysis) {
            const auto result = ir::sync::RealizeCompactSync(analysis);
            nb::dict out;
            out["function"] = result.function;
            out["record_sites"] = result.record_sites;
            out["wait_sites"] = result.wait_sites;
            out["guarded_wait_sites"] = result.guarded_wait_sites;
            out["carried_tokens"] = result.carried_tokens;
            out["name_scan_visits"] = result.name_scan_visits;
            out["name_scan_nodes"] = result.name_scan_nodes;
            out["guarantee"] = "equal selected order under the admitted logical-token model";
            return out;
          },
          "Realize retained demands with logical completion tokens; no physical allocation")
      .def(
          "realize_physical",
          [](const ir::sync::CompactSyncResult& analysis, const std::string& target,
             const std::optional<std::vector<int>>& eligible_ids) {
            const auto result = ir::sync::RealizePhysicalSync(analysis, target, eligible_ids);
            return DescribePhysicalSync(result);
          },
          nb::arg("target"), nb::arg("eligible_ids") = nb::none(),
          "Realize retained demands with certified physical handoffs; no scarcity repair")
      .def("describe", &DescribeCompactSync, "Inspect retained analysis without additional construction")
      .def("expand_retained_demands", &ir::sync::CompactSyncResult::ExpandRetainedDemands,
           nb::arg("max_payloads"), "Bounded audit expansion of retained records; never reruns analysis")
      .def(
          "export_scoped_demands",
          [](const ir::sync::CompactSyncResult& result) -> nb::object {
            const auto owned = result.ExportScopedDemands();
            if (!owned) return nb::none();
            nb::dict out;
            nb::list families;
            for (const auto& family : owned->families) {
              nb::dict item;
              item["source"] = family.source;
              item["target"] = family.target;
              nb::list axes;
              for (const auto& axis : family.axes)
                axes.append(nb::make_tuple(axis.frame, axis.count, axis.source_stride, axis.target_stride));
              item["axes"] = axes;
              families.append(item);
            }
            out["families"] = families;
            out["visited_source_nodes"] = owned->visited_source_nodes;
            out["visited_merge_nodes"] = owned->visited_merge_nodes;
            out["coordinate_entries"] = owned->coordinate_entries;
            return out;
          },
          "Export cached compact demand families in original source frames; not a physical certificate")
      .def("scoped_consecutive_reuse", &ir::sync::CompactSyncResult::ScopedConsecutiveReuse,
           nb::arg("family"), "Certify all carry classes for an owned original scoped family")
      .def("precedes", &ir::sync::CompactSyncResult::Precedes, nb::arg("source"),
           nb::arg("source_completion"), nb::arg("target"), nb::arg("target_completion"),
           "Query original segment/visit/site events")
      .def("boundary", &CompactBoundary, nb::arg("storage"), nb::arg("row"), nb::arg("column"),
           nb::arg("parameters") = std::map<size_t, int64_t>{}, nb::arg("block_index") = nb::none(),
           "Exact modeled point selector, or None for unavailable domain")
      .def("boundary_slot", &CompactSlotBoundary, nb::arg("storage"), nb::arg("slot"), nb::arg("row"),
           nb::arg("column"), nb::arg("parameters") = std::map<size_t, int64_t>{},
           nb::arg("block_index") = nb::none(),
           "Exact modeled point selector with explicit physical slot identity");
  testing.def(
      "inspect_scoped_token_families",
      [](const ir::FunctionPtr& function, const std::string& target) {
        ir::CheckLogicalSync(function);
        auto stripped = ir::sync::StripStructuredLogicalSync(function, true);
        const auto input = ir::sync::ImportPlacedSync(stripped.function, true, target);
        const auto families = ir::sync::ParseScopedTokenFamilies(function, input, stripped.loops);
        nb::dict out;
        out["function"] = stripped.function;
        nb::list rows, regions;
        for (const auto& family : families) {
          nb::dict row;
          row["source"] = nb::make_tuple(family.source_region, family.source_first, family.source_site);
          row["target"] = nb::make_tuple(family.target_region, family.target_first, family.target_site);
          row["count"] = family.count;
          nb::list axes;
          for (const auto& axis : family.axes)
            axes.append(nb::make_tuple(axis.frame, axis.count, axis.source_stride, axis.target_stride));
          row["axes"] = axes;
          rows.append(row);
        }
        for (const auto& segment : input.segments) {
          nb::dict region;
          nb::list sites, frames;
          for (const auto& site : segment.sites)
            sites.append(nb::make_tuple(site.original, site.continuation));
          for (const auto& frame : segment.nested_frames)
            frames.append(nb::make_tuple(frame.loop, frame.count, frame.start, frame.step));
          region["sites"] = sites;
          region["frames"] = frames;
          regions.append(region);
        }
        out["families"] = rows;
        out["regions"] = regions;
        return out;
      },
      nb::arg("function"), nb::arg("target") = "a2a3",
      "Inspect actual scoped token handoffs under the explicit disjoint-GM test contract; no allocation");
  nb::class_<ir::sync::SyncInvocationAnalysis>(testing, "SyncInvocationAnalysis")
      .def(
          "realize_physical",
          [](const ir::sync::SyncInvocationAnalysis& analysis, const std::string& target,
             const std::optional<std::vector<int>>& eligible_ids) {
            const auto result = ir::sync::RealizePhysicalInvocation(analysis, target, eligible_ids);
            nb::dict out;
            out["function"] = result.function;
            nb::list cases;
            size_t sets = 0, waits = 0, barriers = 0, guards = 0;
            for (size_t i = 0; i < result.cases.size(); ++i) {
              auto row = DescribePhysicalSync(result.cases[i]);
              row["block_bounds"] =
                  nb::make_tuple(analysis.cases[i].block_first, analysis.cases[i].block_last);
              cases.append(row);
              sets += result.cases[i].set_sites;
              waits += result.cases[i].wait_sites;
              barriers += result.cases[i].barrier_sites;
              guards += result.cases[i].guard_sites;
            }
            out["cases"] = cases;
            out["set_sites"] = sets;
            out["wait_sites"] = waits;
            out["barrier_sites"] = barriers;
            out["guard_sites"] = guards;
            return out;
          },
          nb::arg("target"), nb::arg("eligible_ids") = nb::none(),
          "Realize all qualified cases into one original payload tree transactionally")
      .def("select_case", &ir::sync::SyncInvocationAnalysis::SelectCase, nb::arg("block_index"),
           nb::arg("parameters"), "Retained case for a fully bound admitted invocation, or None")
      .def(
          "describe",
          [](const ir::sync::SyncInvocationAnalysis& result) {
            nb::dict out;
            out["block_bounds"] = result.block_bounds;
            out["parameter_bounds"] = result.parameter_bounds;
            out["coordinate_nodes"] = result.coordinate_nodes;
            out["coordinate_pieces"] = result.coordinate_pieces;
            nb::list cases;
            for (const auto& item : result.cases) {
              nb::dict row;
              row["block_bounds"] = nb::make_tuple(item.block_first, item.block_last);
              row["analysis"] = DescribeCompactSync(*item.mathematics);
              cases.append(row);
            }
            out["cases"] = cases;
            return out;
          },
          "Inspect exhaustive invocation partition without constructing exports");
  testing.def("analyze_sync_invocation", &ir::sync::AnalyzeSyncInvocation, nb::arg("function"),
              nb::arg("disjoint_gm"), nb::arg("target"), nb::arg("parameter_bounds"), nb::arg("block_bounds"),
              "Analyze an exhaustive analytic partition of a proved launch domain");
  testing.def(
      "describe_placed_sync_input",
      [](const ir::FunctionPtr& function, bool disjoint_gm, const std::string& target,
         const ir::sync::SyncParameterBounds& bounds,
         const std::optional<std::pair<int64_t, int64_t>>& block_bounds) {
        ir::sync::CompactSyncResult imported;
        imported.input = ir::sync::ImportPlacedSync(function, disjoint_gm, target, bounds, block_bounds);
        auto description = DescribeCompactSync(imported);
        description["guarantee"] = "qualified access description; demand analysis not requested";
        description["analysis_available"] = false;
        return description;
      },
      nb::arg("function"), nb::arg("disjoint_gm"), nb::arg("target") = "a2a3",
      nb::arg("parameter_bounds") = ir::sync::SyncParameterBounds{}, nb::arg("block_bounds") = nb::none(),
      "Inspect normalized physical access and slot certificates without constructing demands");
  testing.def(
      "analyze_placed_sync",
      [](const ir::FunctionPtr& function, bool disjoint_gm, const std::string& target,
         const ir::sync::SyncParameterBounds& bounds, const std::string& composition_order,
         const std::optional<std::pair<int64_t, int64_t>>& block_bounds) {
        auto start = std::chrono::steady_clock::now();
        auto input = ir::sync::ImportPlacedSync(function, disjoint_gm, target, bounds, block_bounds);
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        auto result = ir::sync::AnalyzeCompactSync(std::move(input), composition_order);
        result.import_seconds = seconds;
        return result;
      },
      nb::arg("function"), nb::arg("disjoint_gm"), nb::arg("target") = "a2a3",
      nb::arg("parameter_bounds") = ir::sync::SyncParameterBounds{},
      nb::arg("composition_order") = "balanced", nb::arg("block_bounds") = nb::none(),
      "Read-only research analysis of qualified placed Buffer IR under a named hardware contract");

  testing.def(
      "sync_parameter_bounds",
      [](const ir::ProgramPtr& program) { return ir::sync::CollectSyncParameterBounds(program); },
      nb::arg("program"), "Inspect proved caller coordinate ranges without analysis or realization");
  testing.def(
      "sync_invocation_evidence",
      [](const ir::ProgramPtr& program) {
        ir::sync::SyncParameterWork work;
        const auto evidence = ir::sync::CollectSyncInvocationEvidence(program, &work);
        nb::dict out;
        out["parameters"] = evidence.parameters;
        out["block_indices"] = evidence.block_indices;
        out["call_graph_expression_visits"] = work.call_graph_expression_visits;
        out["call_graph_nodes"] = work.call_graph_nodes;
        out["call_graph_edges"] = work.call_graph_edges;
        out["processed_functions"] = work.processed_functions;
        out["expression_visits"] = work.expression_visits;
        out["range_visits"] = work.range_visits;
        return out;
      },
      nb::arg("program"), "Read-only scalar and launch evidence; missing facts remain unavailable");
  testing.def(
      "sync_parameter_bounds_profile",
      [](const ir::ProgramPtr& program) {
        ir::sync::SyncParameterWork work;
        auto bounds = ir::sync::CollectSyncParameterBounds(program, &work);
        nb::dict out;
        out["bounds"] = bounds;
        out["expression_visits"] = work.expression_visits;
        out["range_visits"] = work.range_visits;
        return out;
      },
      nb::arg("program"), "Inspect caller proof traversal work in each bound environment");
  testing.def("analyze_finite_sync", &AnalyzeFiniteSyncForTesting, nb::arg("occurrences"),
              nb::arg("covered_cells"), nb::arg("exact_accesses"), nb::arg("prerequisites"),
              nb::arg("native_order"), nb::arg("queries"), nb::arg("boundary_cells"),
              "Analyze supplied finite physical classes without command insertion");
  testing.def("try_infer_pipe", &TryInferPipeForTesting, nb::arg("call"),
              "Return the exact backend pipe for a Call, or None");

  testing.def("get_execution_memory_access_evidence", &GetExecutionMemoryAccessEvidenceForTesting,
              nb::arg("op_name"), "Return an operation's execution-memory-access evidence");

  testing.def("get_declared_core_affinity", &GetDeclaredCoreAffinityForTesting, nb::arg("op_name"),
              "Return an operation's explicitly declared core affinity, or None");

  testing.def("is_no_duplicate_op", &IsNoDuplicateOpForTesting, nb::arg("op_name"),
              "Return whether an operation must not run on a second core");

  testing.def("classify_call_affinity", &ClassifyCallAffinityForTesting, nb::arg("call"),
              "Return the core affinity ClassifyCallAffinity derives for a Call");
}

}  // namespace python
}  // namespace pypto
