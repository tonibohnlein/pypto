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

#include <algorithm>
#include <any>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../module.h"
#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"
#include "pypto/ir/transforms/dsa/reuse_penalty_recognizer.h"
#include "pypto/ir/transforms/utils/buffer_active_domain_analysis.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/finite_sync_analysis.h"
#include "pypto/ir/transforms/utils/parent_stmt_analysis.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/sync.h"
#include "python/bindings/modules/testing_sync_support.h"
#include "src/ir/transforms/utils/compact_sync_builder.h"
#include "src/ir/transforms/utils/conservative_sync_dataflow.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/physical_sync_commands.h"
#include "src/ir/transforms/utils/retained_sync_child.h"
#include "src/ir/transforms/utils/scoped_sync_palette.h"
#include "src/ir/transforms/utils/scoped_token_families.h"
#include "src/ir/transforms/utils/scoped_token_order.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace nb = nanobind;

namespace pypto::ir::sync {
// Diagnostic access returns the existing immutable shared result; it never
// rebuilds mathematics or transfers its owner to another region context.
struct RetainedSyncChildDiagnostics {
  static CompactSyncResult Mathematics(const EventfulSyncChild& child) { return child.mathematics_; }
};
}  // namespace pypto::ir::sync

namespace pypto {
namespace python {

nb::dict DescribeCompactSync(const ir::sync::CompactSyncView& result) {
  nb::dict out;
  nb::list unavailable_spaces;
  for (const auto space : result.State().input.unavailable_storage_spaces) unavailable_spaces.append(space);
  out["unavailable_storage_spaces"] = unavailable_spaces;
  out["interval_access_visits"] = result.State().input.interval_access_visits;
  out["collective_domains"] = result.Diagnostics().collective_domains;
  out["collective_witnesses"] = result.Diagnostics().collective_witnesses;
  out["collective_access_visits"] = result.Diagnostics().collective_access_visits;
  out["qualified_intervals"] = result.State().input.qualified_intervals;
  out["guarantee"] = "minimum demands for declared footprints and core native chains";
  out["realization"] = "not requested";
  out["analysis_available"] = result.State().segments.size() == result.State().input.segments.size();
  out["parameter_bounds"] = result.State().input.parameter_bounds;
  out["coordinate_nodes"] = result.State().input.coordinate_nodes;
  out["coordinate_flatten_visits"] = result.State().input.coordinate_flatten_visits;
  out["coordinate_coefficient_entries"] = result.State().input.coordinate_coefficient_entries;
  nb::list storage, regions, coordinate_programs;
  std::map<const ir::sync::SyncCoordinateProgram*, size_t> coordinate_program_ids;
  auto external_geometry = [&](const ir::sync::SyncAccess& access) -> nb::object {
    if (!access.external) return nb::none();
    const auto& footprint = *access.external;
    auto [entry, fresh] =
        coordinate_program_ids.emplace(footprint.program.get(), coordinate_program_ids.size());
    if (fresh) {
      nb::list nodes;
      for (const auto& node : footprint.program->nodes) {
        using Kind = ir::sync::SyncCoordinateNode::Kind;
        const char* kind = nullptr;
        switch (node.kind) {
          case Kind::Constant:
            kind = "constant";
            break;
          case Kind::Parameter:
            kind = "parameter";
            break;
          case Kind::Block:
            kind = "block";
            break;
          case Kind::Loop:
            kind = "loop";
            break;
          case Kind::Add:
            kind = "add";
            break;
          case Kind::Sub:
            kind = "sub";
            break;
          case Kind::Scale:
            kind = "scale";
            break;
          case Kind::ModularShift:
            kind = "modular_shift";
            break;
          case Kind::FloorDivide:
            kind = "divide";
            break;
          case Kind::Remainder:
            kind = "remainder";
            break;
          case Kind::Equal:
            kind = "equal";
            break;
          case Kind::NotEqual:
            kind = "not_equal";
            break;
          case Kind::LessThan:
            kind = "less_than";
            break;
          case Kind::LessEqual:
            kind = "less_equal";
            break;
          case Kind::GreaterThan:
            kind = "greater_than";
            break;
          case Kind::GreaterEqual:
            kind = "greater_equal";
            break;
          case Kind::And:
            kind = "and";
            break;
          case Kind::Or:
            kind = "or";
            break;
          case Kind::Not:
            kind = "not";
            break;
          case Kind::Select:
            kind = "select";
            break;
        }
        INTERNAL_CHECK(kind) << "Unknown retained coordinate node";
        nb::dict value;
        value["kind"] = kind;
        value["lower"] = node.lower;
        value["upper"] = node.upper;
        value["factor"] = node.factor;
        value["start"] = node.start;
        value["step"] = node.step;
        value["left"] = node.left;
        value["right"] = node.right;
        value["condition"] = node.condition;
        value["parameter"] = node.parameter;
        value["frame"] = node.frame;
        value["constant"] = node.constant;
        value["dynamic_start"] = node.dynamic_start;
        nodes.append(value);
      }
      coordinate_programs.append(nodes);
    }
    nb::dict value;
    value["program"] = entry->second;
    value["row"] = footprint.row;
    value["column"] = footprint.column;
    value["row_major_tiles"] = nb::none();
    if (footprint.row_major_tiles) {
      const auto& proof = *footprint.row_major_tiles;
      nb::dict row_major;
      row_major["frame"] = proof.frame;
      row_major["tile_coordinate"] = proof.tile_coordinate;
      row_major["frame_start_root"] = proof.frame_start_root;
      row_major["row_tiles"] = proof.row_tiles;
      row_major["column_tiles"] = proof.column_tiles;
      row_major["row_scale"] = proof.row_scale;
      row_major["column_scale"] = proof.column_scale;
      value["row_major_tiles"] = row_major;
    }
    value["swizzled_tiles"] = nb::none();
    if (footprint.swizzled_tiles) {
      const auto& proof = *footprint.swizzled_tiles;
      nb::dict swizzled;
      swizzled["frame"] = proof.frame;
      swizzled["tile_coordinate"] = proof.tile_coordinate;
      swizzled["frame_start_root"] = proof.frame_start_root;
      swizzled["row_tiles"] = proof.row_tiles;
      swizzled["column_tiles"] = proof.column_tiles;
      swizzled["row_scale"] = proof.row_scale;
      swizzled["column_scale"] = proof.column_scale;
      swizzled["group_tiles"] = proof.group_tiles;
      swizzled["group_size"] = proof.group_size;
      swizzled["col_major_groups"] = proof.col_major_groups;
      swizzled["snake"] = proof.snake;
      value["swizzled_tiles"] = swizzled;
    }
    return value;
  };
  for (const auto& s : result.State().input.storage) {
    nb::dict item;
    item["name"] = s.name;
    item["boundary_available"] = s.boundary_available;
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
    item["block_origin"] = result.State().input.block_bounds ? result.State().input.block_bounds->first : 0;
    storage.append(item);
  }
  for (size_t r = 0; r < result.State().input.segments.size(); ++r) {
    const auto& source = result.State().input.segments[r];
    const auto* analysis = r < result.State().segments.size() ? &result.State().segments[r] : nullptr;
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
      nb::list accesses, precision, slots, slot_frames, access_frames, intervals, external;
      site["original"] = s.original;
      site["continuation"] = s.continuation;
      site["initialization_scope"] = s.initialization_scope;
      site["pipe"] = s.pipe;
      for (const auto& a : s.accesses) {
        external.append(external_geometry(a));
        access_frames.append(a.coordinate_frame);
        intervals.append(nb::cast(a.physical_interval));
        slots.append(nb::make_tuple(a.slot_first, a.slot_step));
        slot_frames.append(a.slot_coordinate_frame);
        precision.append(a.exact_footprint ? "exact" : "enclosure");
        accesses.append(nb::make_tuple(a.storage, a.write, a.row, a.column, a.row_step, a.column_step, a.rows,
                                       a.columns, a.after_first));
      }
      site["external_geometry"] = external;
      site["physical_intervals"] = intervals;
      site["slot_selections"] = slots;
      site["slot_frames"] = slot_frames;
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
    region["distance_entries"] = 0;
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
  out["coordinate_programs"] = coordinate_programs;
  out["storage"] = storage;
  out["regions"] = regions;
  out["crossing_demands"] =
      result.HasHierarchy() ? nb::object(nb::none()) : nb::cast(result.State().crossing_demands);
  out["crossing_generators"] =
      result.HasHierarchy() ? nb::object(nb::none()) : nb::cast(result.State().crossing_generators);
  out["analyzed_occurrences"] = result.Diagnostics().analyzed_occurrences;
  out["phase_variants"] = result.Diagnostics().phase_variants;
  out["finite_microbody_occurrences"] = result.Diagnostics().finite_microbody_occurrences;
  out["hardware_target"] = result.State().input.hardware_target;
  out["cells"] = result.Diagnostics().cells;
  out["boundary_graph_edges"] = result.Diagnostics().graph_edges;
  out["class_tests"] = result.Diagnostics().class_tests;
  out["class_lookup_tests"] = result.Diagnostics().class_lookup_tests;
  out["port_pair_tests"] = result.Diagnostics().port_pair_tests;
  out["crossing_path_searches"] = result.Diagnostics().crossing_path_searches;
  out["import_seconds"] = result.Diagnostics().import_seconds;
  out["partition_seconds"] = result.Diagnostics().partition_seconds;
  out["reduction_seconds"] = result.Diagnostics().reduction_seconds;
  out["composition_seconds"] = result.Diagnostics().composition_seconds;
  out["hierarchical"] = result.HasHierarchy();
  out["source_region_nodes"] = result.Diagnostics().source_region_nodes;
  out["source_coordinate_entries"] = result.Diagnostics().source_coordinate_entries;
  out["scoped_export_builds"] = result.Diagnostics().scoped_export_builds;
  out["scoped_export_attempts"] = result.Diagnostics().scoped_export_attempts;
  out["point_selector_access_visits"] = result.Diagnostics().point_selector_access_visits;
  out["interface_cells"] = result.Diagnostics().interface_cells;
  out["omitted_bridge_domains"] = result.Diagnostics().omitted_bridge_domains;
  out["interface_slot_visits"] = result.Diagnostics().interface_slot_visits;
  out["interface_ports"] = result.Diagnostics().interface_ports;
  out["interface_entries"] = result.Diagnostics().interface_entries;
  out["interface_queries"] = result.Diagnostics().interface_queries;
  out["merge_nodes"] = result.Diagnostics().merge_nodes;
  out["crossing_tests"] = result.Diagnostics().crossing_tests;
  out["hierarchy_query_steps"] = result.Diagnostics().hierarchy_query_steps;
  out["interface_access_visits"] = result.Diagnostics().interface_access_visits;
  out["bridge_candidates"] = result.Diagnostics().bridge_candidates;
  out["port_normalization_visits"] = result.Diagnostics().port_normalization_visits;
  return out;
}
nb::object CompactBoundary(const ir::sync::CompactSyncResult& result, size_t storage, int64_t row,
                           int64_t col, const std::map<size_t, int64_t>& parameters,
                           std::optional<int64_t> block_index) {
  auto b = sync_testing::BoundaryAt(result, storage, row, col, parameters, block_index);
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
  auto boundary = sync_testing::BoundarySlot(result, storage, slot, row, col, parameters, block_index);
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
  entry.ValidateCall(args, result_type, ir::Span::unknown(), {});
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

// Diagnostics count final emitted sites separately from per-case construction.
// if_sites includes original payload guards. The invocation result also
// compares this count with the original owner to report its added guards.
class PhysicalControlCensus : public ir::IRVisitor {
 public:
  size_t sets = 0, waits = 0, barriers = 0, branches = 0;
  void VisitStmt_(const ir::IfStmtPtr& statement) override {
    ++branches;
    ir::IRVisitor::VisitStmt_(statement);
  }
  void VisitStmt_(const ir::EvalStmtPtr& statement) override {
    const auto call = ir::As<ir::Call>(statement->expr_);
    if (ir::IsOp(call, "system.set_flag")) ++sets;
    if (ir::IsOp(call, "system.wait_flag")) ++waits;
    if (ir::IsOp(call, "system.bar_m") || ir::IsOp(call, "system.bar_v")) ++barriers;
    ir::IRVisitor::VisitStmt_(statement);
  }
};

size_t DescribeEmittedPhysicalControl(nb::dict& out, const ir::FunctionPtr& function) {
  PhysicalControlCensus emitted;
  emitted.VisitFunction(function);
  out["set_sites"] = emitted.sets;
  out["wait_sites"] = emitted.waits;
  out["barrier_sites"] = emitted.barriers;
  out["if_sites"] = emitted.branches;
  return emitted.branches;
}

nb::dict DescribeSchedulePolicy(const ir::sync::SyncSchedulePolicy& policy) {
  nb::dict out;
  out["mode"] = policy.mode;
  out["max_extra_payload_sites"] = policy.max_extra_payload_sites;
  out["max_extra_control_nodes"] = policy.max_extra_control_nodes;
  out["no_clone"] = policy.no_clone;
  out["boundary_motion"] = policy.boundary_motion;
  return out;
}
nb::dict DescribeSchedule(const ir::sync::PhysicalSyncRealization& result) {
  nb::dict out;
  nb::list choices;
  for (const auto& choice : result.schedule_choices) {
    nb::dict row;
    row["loop"] = choice.loop;
    row["mode"] = choice.mode;
    row["status"] = choice.status;
    row["reason"] = choice.reason;
    row["payload_sites"] = choice.payload_sites;
    row["control_node_bound"] = choice.control_node_bound;
    row["command_sites"] = choice.command_sites;
    nb::list costs;
    for (const auto& cost : choice.model_cost)
      costs.append(nb::make_tuple(cost[0], cost[1], cost[2], cost[3], cost[4], cost[5]));
    row["model_cost"] = costs;
    row["model_scope"] = "standalone_candidate_before_parent_gap_composition";
    row["model_comparable"] = choice.model_comparable;
    row["model_exact"] = choice.model_exact;
    row["selected"] = choice.selected;
    choices.append(row);
  }
  out["choices"] = choices;
  nb::list gaps;
  for (const auto& gap : result.schedule_parent_gaps) {
    nb::dict row;
    row["loop"] = gap.loop;
    row["command_sites"] = gap.command_sites;
    row["before"] = nb::cast(gap.before);
    row["after"] = nb::cast(gap.after);
    row["count_order"] = nb::make_tuple("guards", "expressions", "statements");
    row["scope"] = "parent_gap_after_candidate_selection";
    gaps.append(row);
  }
  out["parent_gaps"] = gaps;
  out["inspected_descriptors"] = result.schedule_inspected_descriptors;
  out["inspected_expressions"] = result.schedule_inspected_expressions;
  out["emitted_payload_sites"] = result.schedule_emitted_payload_sites;
  out["emitted_control_nodes"] = result.schedule_emitted_control_nodes;
  return out;
}
nb::dict DescribeScopedPalette(const ir::sync::ScopedPaletteResult& result);

nb::dict DescribePhysicalSync(const ir::sync::PhysicalSyncRealization& result) {
  nb::dict out;
  out["function"] = result.function;
  out["schedule"] = DescribeSchedule(result);
  out["set_sites"] = result.set_sites;
  out["wait_sites"] = result.wait_sites;
  out["barrier_sites"] = result.barrier_sites;
  out["guard_sites"] = result.guard_sites;
  out["constructed_guard_sites"] =
      result.guard_sites + result.shared_guard_sites + result.simplified_guard_sites;
  out["shared_guard_sites"] = result.shared_guard_sites;
  out["shared_composed_guard_sites"] = result.shared_composed_guard_sites;
  out["control_statement_visits"] = result.control_statement_visits;
  out["closed_prefix_suffix_visits"] = result.closed_prefix_suffix_visits;
  out["simplified_guard_sites"] = result.simplified_guard_sites;
  out["shared_control_values"] = result.shared_control_values;
  out["cut_domain_interval_visits"] = result.cut_domain_interval_visits;
  out["cut_case_comparisons"] = result.cut_case_comparisons;
  out["coalesced_cut_cases"] = result.coalesced_cut_cases;
  out["case_sequence_visits"] = result.case_sequence_visits;
  out["case_sequence_comparisons"] = result.case_sequence_comparisons;
  out["factored_case_cuts"] = result.factored_case_cuts;
  out["shared_disjoint_case_cuts"] = result.shared_disjoint_case_cuts;
  out["endpoint_domain_entries"] = result.endpoint_domain_entries;
  out["normalized_endpoint_bounds"] = result.normalized_endpoint_bounds;
  out["endpoint_predicates_constructed"] = result.endpoint_predicates_constructed;
  out["control_descriptors"] = result.control_descriptors;
  out["invocation_scheduled_loops"] = result.invocation_scheduled_loops;
  out["invocation_schedule_groups"] = result.invocation_schedule_groups;
  out["invocation_schedule_visits"] = result.invocation_schedule_visits;
  out["invocation_schedule_comparisons"] = result.invocation_schedule_comparisons;
  out["invocation_schedule_model_saving"] = result.invocation_schedule_model_saving;
  out["schedule_name_scans"] = result.schedule_name_scans;
  out["schedule_name_visits"] = result.schedule_name_visits;
  out["regular_schedule_candidates"] = result.regular_schedule_candidates;
  out["regular_schedule_loops"] = result.regular_schedule_loops;
  out["regular_schedule_boundaries"] = result.regular_schedule_boundaries;
  out["regular_schedule_body_copies"] = result.regular_schedule_body_copies;
  out["regular_schedule_payload_copies"] = result.regular_schedule_payload_copies;
  out["regular_schedule_model_saving"] = result.regular_schedule_model_saving;
  out["assigned_schedule_endpoints"] = result.assigned_schedule_endpoints;
  out["schedule_candidates"] = result.schedule_candidates;
  out["scheduled_loops"] = result.scheduled_loops;
  out["schedule_boundary_commands"] = result.schedule_boundary_commands;
  out["schedule_size_refusals"] = result.schedule_size_refusals;
  out["schedule_body_visits"] = result.schedule_body_visits;
  out["schedule_descriptor_visits"] = result.schedule_descriptor_visits;
  out["schedule_original_nodes"] = result.schedule_original_nodes;
  out["schedule_candidate_nodes"] = result.schedule_candidate_nodes;

  out["materialized_control_statements"] = result.materialized_control_statements;
  out["control_expression_visits"] = result.control_expression_visits;
  out["reuse_queries"] = result.reuse_queries;
  out["reserved_lanes"] = result.reserved_lanes;
  out["native_reserved_ids"] = result.native_reserved_ids;
  out["finite_minimum"] = result.finite_minimum;
  nb::list families;
  static const std::vector<ir::sync::PhysicalSyncFamily> empty_families;
  const auto& family_list = result.assignment ? result.assignment->Families() : empty_families;
  for (const auto& f : family_list) {
    nb::dict row;
    row["source"] = nb::make_tuple(f.source_region, f.source_first, f.source_site);
    row["target"] = nb::make_tuple(f.target_region, f.target_first, f.target_site);
    row["count"] = f.count;
    row["ids"] = nb::cast(f.ids);
    row["mapping_kind"] = "family-record";
    if (result.assignment && !result.assignment->RawPalettes().empty() && !f.protocol_uses.empty()) {
      const auto& raw = result.assignment->RawPalettes().at(f.protocol_uses.front());
      if (raw.palette) {
        row["mapping_kind"] = "publication-rank";
        row["palette"] = *raw.palette;
      }
    }
    nb::list axes;
    for (const auto& axis : f.axes)
      axes.append(nb::make_tuple(axis.frame, axis.count, axis.source_stride, axis.target_stride));
    row["axes"] = axes;
    families.append(row);
  }
  out["families"] = families;
  nb::list palettes;
  if (result.assignment) {
    for (const auto& palette : result.assignment->Palettes()) {
      nb::dict row;
      const auto& schedule = *palette.certificate->Schedule();
      row["direction"] = nb::make_tuple(schedule.SourcePipe(), schedule.TargetPipe());
      row["ids"] = palette.ids;
      row["publications"] = schedule.Publications();
      row["rank_pieces"] = schedule.Pieces().size();
      palettes.append(row);
    }
  }
  out["palettes"] = palettes;
  nb::list attempts;
  for (const auto& attempt : result.palette_attempts) attempts.append(DescribeScopedPalette(*attempt));
  out["palette_attempts"] = attempts;
  nb::list allocation_attempts;
  for (const auto& attempt : result.allocation_attempts) {
    nb::dict row;
    row["direction"] = nb::make_tuple(attempt.source, attempt.target);
    row["status"] = attempt.status;
    row["reason"] = attempt.reason;
    row["reuse_queries"] = attempt.reuse_queries;
    allocation_attempts.append(row);
  }
  out["allocation_attempts"] = allocation_attempts;
  out["guarantee"] = "equal selected order under qualified direct-command model";
  out["native_mode"] = "manual a2a3; fresh flags at entry; closed notifications at exit";
  return out;
}

nb::dict DescribeScopedPalette(const ir::sync::ScopedPaletteResult& result) {
  nb::dict out;
  out["status"] = ir::sync::ScopedPaletteStatusName(result.status);
  out["reason"] = result.reason;
  out["attempted_budgets"] = result.attempted_budgets;
  out["publications"] = result.schedule ? result.schedule->Publications() : 0;
  out["grammar_nodes"] = result.work.grammar_nodes;
  out["phase_visits"] = result.work.phase_visits;
  out["domain_visits"] = result.work.domain_visits;
  out["transition_queries"] = result.work.transition_queries;
  out["routing_entries"] = result.work.routing_entries;
  out["boundary_node_visits"] = result.work.boundary_node_visits;
  out["proof_axis_entries"] = result.work.proof_axis_entries;
  nb::list pieces;
  auto digits = [](const std::vector<ir::sync::ScopedPublicationDigit>& input) {
    std::vector<std::tuple<int64_t, int64_t, int64_t>> values;
    for (const auto& d : input) values.emplace_back(d.first, d.step, d.count);
    return values;
  };
  if (result.schedule) {
    for (const auto& piece : result.schedule->Pieces()) {
      nb::dict entry;
      entry["raw_use"] = piece.raw_use;
      entry["rank_base"] = piece.rank_base;
      entry["source"] = digits(piece.source);
      entry["target"] = digits(piece.target);
      entry["coefficients"] = piece.rank_coefficients;
      pieces.append(entry);
    }
  }
  out["pieces"] = pieces;
  return out;
}

void BindTesting(nb::module_& m) {
  // Create a protected submodule for testing utilities
  // This will be accessible as pypto.testing in Python
  nb::module_ testing = m.def_submodule("testing", "Internal testing utilities (do not use in production)");
  testing.def(
      "normalize_buffer_active_domains",
      [](const ir::FunctionPtr& function) {
        ir::BufferActiveDomainWork work;
        const auto result = ir::NormalizeBufferActiveDomains(function, &work);
        nb::dict counts;
        counts["definition_statements"] = work.definition_statements;
        counts["domain_statements"] = work.domain_statements;
        counts["metadata_statements"] = work.metadata_statements;
        counts["metadata_expression_states"] = work.metadata_expression_states;
        counts["metadata_types"] = work.metadata_types;
        counts["scalar_stability_nodes"] = work.scalar_stability_nodes;
        counts["rewritten_statements"] = work.rewritten_statements;
        counts["rewrite_expression_dispatches"] = work.rewrite_expression_dispatches;
        nb::dict output;
        output["candidate"] = result.Candidate();
        output["changed"] = result.Candidate() != result.Original();
        output["work"] = counts;
        return output;
      },
      nb::arg("function"),
      "Inspect the existing prefix normalization candidate and work; this is not whole Buffer admission");
  nb::class_<ir::sync::SyncSchedulePolicy>(testing, "SyncSchedulePolicy")
      .def(
          "__init__",
          [](ir::sync::SyncSchedulePolicy* self, std::string mode, size_t payloads, size_t controls,
             bool no_clone, bool boundary_motion) {
            new (self)
                ir::sync::SyncSchedulePolicy{std::move(mode), payloads, controls, no_clone, boundary_motion};
          },
          nb::arg("mode") = "auto", nb::arg("max_extra_payload_sites") = 4096,
          nb::arg("max_extra_control_nodes") = 65536, nb::arg("no_clone") = true,
          nb::arg("boundary_motion") = true);

  testing.def(
      "share_sync_command_guards",
      [](const ir::StmtPtr& commands) {
        PhysicalControlCensus census;
        census.VisitStmt(commands);
        ir::sync::PhysicalSyncRealization counters;
        counters.guard_sites = census.branches;
        const auto output = sync_testing::ShareGuards({commands}, counters);
        return std::make_shared<ir::SeqStmts>(output, commands->span_);
      },
      nb::arg("commands"),
      "Test the emitter helper on pure coordinate-guarded physical commands; not arbitrary IR");
  testing.def(
      "inspect_sync_scalar_plan",
      [](const ir::StmtPtr& commands) {
        using ir::sync::physical::ControlForest;
        using ir::sync::physical::ControlSequence;
        using ir::sync::physical::FreezeCommandForest;
        using ir::sync::physical::FrozenCommandForest;
        ir::sync::PhysicalSyncRealization work;
        ControlForest forest{sync_testing::ImportCommands(commands, work)};
        const auto plan = FreezeCommandForest(forest, work);
        std::map<std::string, size_t> result;
        const auto expressions = work.control_expression_visits;
        result["same_instance"] = FreezeCommandForest(forest, work) == plan;
        auto copy = forest;
        result["copy_reuses"] = FreezeCommandForest(copy, work) == plan;
        result["reuse_expression_visits"] = work.control_expression_visits - expressions;
        auto moved = std::move(copy);
        result["move_reuses"] = FreezeCommandForest(moved, work) == plan;
        moved.push_back(ControlSequence({}, commands->span_, work));
        const auto extended = FreezeCommandForest(moved, work);
        result["append_invalidates"] = extended != plan;
        std::swap(moved.front(), moved.back());
        result["reorder_invalidates"] = FreezeCommandForest(moved, work) != extended;
        moved.clear();
        result["clear_invalidates"] = FreezeCommandForest(moved, work)->commands.empty();
        auto count = [&](auto&& self, const FrozenCommandForest& scope) -> void {
          for (const auto& command : scope.commands) {
            for (size_t i = 0; i < command.scalar_cost.size(); ++i)
              result["cost_" + std::to_string(i)] += command.scalar_cost[i];
            if (command.body) self(self, *command.body);
            if (command.alternative) self(self, *command.alternative);
          }
        };
        count(count, *plan);
        return result;
      },
      nb::arg("commands"));
  testing.def("materialize_sync_command_values", &sync_testing::MaterializeValues, nb::arg("owner"),
              nb::arg("commands"), "Emitter-only pure command value sharing");
  testing.def(
      "emit_sync_command_cases",
      [](const ir::ExprPtr& block, const std::vector<std::pair<int64_t, int64_t>>& possible,
         const std::vector<std::tuple<int64_t, int64_t, ir::StmtPtr>>& supplied) {
        std::vector<sync_testing::CommandCase> cases;
        for (const auto& [lo, hi, body] : supplied) cases.push_back({{lo, hi}, body});
        ir::sync::PhysicalSyncRealization work;
        return std::make_shared<ir::SeqStmts>(
            sync_testing::EmitCases(block, possible, cases, work, ir::Span::unknown()), ir::Span::unknown());
      },
      nb::arg("block"), nb::arg("possible"), nb::arg("cases"),
      "Emitter-only case aggregation under a supplied possible cut domain; unknown means full domain");

  testing.def(
      "inspect_sync_invocation_groups",
      [](const ir::FunctionPtr& owner,
         const std::vector<std::tuple<int64_t, int64_t, ir::StmtPtr>>& supplied) {
        std::vector<sync_testing::CommandCase> cases;
        for (const auto& [lo, hi, body] : supplied) cases.push_back({{lo, hi}, body});
        auto [groups, work] = sync_testing::InspectGroups(owner, cases);
        nb::dict out;
        out["descriptor_nodes"] = work.control_descriptors;
        out["comparisons"] = work.invocation_schedule_comparisons;
        out["groups"] = nb::cast(groups);
        return out;
      },
      nb::arg("owner"), nb::arg("cases"),
      "Inspect structural complete-schedule groups only; no profitability or protocol certification");

  testing.def(
      "emit_sync_boundary_commands",
      [](const ir::FunctionPtr& owner,
         const std::vector<std::tuple<ir::CallPtr, ir::StmtPtr, ir::StmtPtr>>& cuts, bool enabled,
         bool validate_scope) {
        return DescribePhysicalSync(sync_testing::EmitBoundaries(owner, cuts, enabled, validate_scope));
      },
      nb::arg("owner"), nb::arg("cuts"), nb::arg("enabled") = true, nb::arg("validate_scope") = false,
      "Emitter-only boundary schedule for qualified original cuts; does not qualify a protocol");

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

  testing.def(
      "check_sync_assignment_contract",
      [](const ir::FunctionPtr& function, const std::string& change) {
        auto analysis = ir::sync::AnalyzeCompactSync(ir::sync::ImportPlacedSync(function, true));
        auto logical = ir::sync::RealizeCompactSync(analysis).function;
        auto stripped = ir::sync::StripStructuredLogicalSync(logical, true);
        auto input = ir::sync::ImportSyncPayloads(stripped.function, "a2a3");
        auto protocol = ir::sync::ParseScopedTokenProtocol(logical, std::move(input), stripped.loops);
        auto order = ir::sync::PrepareScopedTokenOrder(protocol);
        const auto& context = order->Input();
        const auto result =
            ir::sync::AllocateDirectSync(context, order->Families(), order->View(), "a2a3", std::nullopt);
        const auto& assignment = result.assignment;
        INTERNAL_CHECK(assignment) << "Missing completed assignment";
        const auto copy = result;
        nb::dict out;
        out["shared_family_storage"] = &result.assignment->Families() == &copy.assignment->Families();
        out["families"] = result.assignment->Families().size();
        if (change == "preassigned_ids") {
          auto families = order->Families();  // Deliberately invalid diagnostic input.
          CHECK(!families.empty()) << "Expected a nonempty diagnostic plan";
          families.front().ids = {99};
          ir::sync::AllocateDirectSync(context, std::move(families), order->View(), "a2a3", std::nullopt);
        } else if (change == "none" || change == "target") {
          assignment->ValidateFor(context, change == "target" ? "a5" : "a2a3");
        } else {
          // Production input is immutable and privately owned by the protocol.
          // Copies, including edited copies, cannot borrow its assignment.
          auto foreign = context;
          if (change == "owner")
            foreign.owner = nullptr;
          else if (change == "domain")
            foreign.block_bounds = std::make_pair(123, 456);
          else if (change == "frame")
            ++foreign.segments.front().count;
          else
            CHECK(change == "foreign_input") << "Unknown assignment diagnostic mutation";
          assignment->ValidateFor(foreign, "a2a3");
        }
        return out;
      },
      nb::arg("function"), nb::arg("change") = "none",
      "Test immutable assignment storage and rejection of foreign or changed synchronous contexts");

  nb::class_<ir::sync::CompactSyncResult>(testing, "CompactSyncAnalysis")
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
             const std::optional<std::vector<int>>& eligible_ids,
             const ir::sync::SyncSchedulePolicy& policy) {
            const auto result = sync_testing::RealizeDemandPlan(analysis, target, eligible_ids, policy);
            auto out = DescribePhysicalSync(result);
            out["schedule_policy"] = DescribeSchedulePolicy(policy);
            nb::dict case_sites;
            case_sites["set_sites"] = result.set_sites;
            case_sites["wait_sites"] = result.wait_sites;
            case_sites["barrier_sites"] = result.barrier_sites;
            case_sites["guard_sites"] = result.guard_sites;
            out["case_sites"] = case_sites;
            DescribeEmittedPhysicalControl(out, result.function);
            return out;
          },
          nb::arg("target"), nb::arg("eligible_ids") = nb::none(), nb::kw_only(),
          nb::arg("schedule_policy") = ir::sync::SyncSchedulePolicy{},
          "Test retained-demand insertion through the production token lowering pipeline")
      .def(
          "describe", [](const ir::sync::CompactSyncResult& result) { return DescribeCompactSync(result); },
          "Inspect retained analysis without additional construction")
      .def(
          "expand_retained_demands",
          [](const ir::sync::CompactSyncResult& result, size_t maximum) {
            return sync_testing::ExpandRetainedDemands(result, maximum);
          },
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
      .def(
          "precedes",
          [](const ir::sync::CompactSyncResult& result, const ir::sync::Occurrence& a, bool sc,
             const ir::sync::Occurrence& b, bool tc) { return result.Precedes(a, sc, b, tc); },
          nb::arg("source"), nb::arg("source_completion"), nb::arg("target"), nb::arg("target_completion"),
          "Query original segment/visit/site events")
      .def("boundary", &CompactBoundary, nb::arg("storage"), nb::arg("row"), nb::arg("column"),
           nb::arg("parameters") = std::map<size_t, int64_t>{}, nb::arg("block_index") = nb::none(),
           "Exact modeled point selector, or None for unavailable domain")
      .def("boundary_slot", &CompactSlotBoundary, nb::arg("storage"), nb::arg("slot"), nb::arg("row"),
           nb::arg("column"), nb::arg("parameters") = std::map<size_t, int64_t>{},
           nb::arg("block_index") = nb::none(),
           "Exact modeled point selector with explicit physical slot identity");
  nb::class_<ir::sync::ScopedPublicationSchedule>(testing, "ScopedPublicationSchedule")
      .def_prop_ro("publications", &ir::sync::ScopedPublicationSchedule::Publications)
      .def("certify", [](std::shared_ptr<const ir::sync::ScopedPublicationSchedule> schedule,
                         const ir::sync::ScopedTokenOrder& order, int64_t budget) {
        return DescribeScopedPalette(ir::sync::CertifyScopedPalette(order, std::move(schedule), budget));
      });
  nb::class_<ir::sync::ScopedTokenOrder>(testing, "ScopedTokenOrder")
      .def("precedes", &ir::sync::ScopedTokenOrder::Precedes)
      .def("uniform", &ir::sync::ScopedTokenOrder::Uniform)
      .def("publication_schedule",
           [](const ir::sync::ScopedTokenOrder& order, ir::PipeType source, ir::PipeType target) {
             const auto result = ir::sync::BuildScopedPublicationSchedule(order, source, target);
             CHECK(result.schedule) << result.reason;
             return result.schedule;
           })
      .def("palette",
           [](const ir::sync::ScopedTokenOrder& order, ir::PipeType source, ir::PipeType target,
              int64_t budget) {
             // Diagnostic one-shot composition of the actual retained services.
             auto built = ir::sync::BuildScopedPublicationSchedule(order, source, target);
             if (!built.schedule) return DescribeScopedPalette(built);
             auto result = ir::sync::CertifyScopedPalette(order, built.schedule, budget);
             result.work.grammar_nodes += built.work.grammar_nodes;
             result.work.phase_visits += built.work.phase_visits;
             result.work.domain_visits += built.work.domain_visits;
             result.work.rank_pieces += built.work.rank_pieces;
             result.work.routing_entries += built.work.routing_entries;
             return DescribeScopedPalette(result);
           })
      .def_prop_ro("region_frames",
                   [](const ir::sync::ScopedTokenOrder& order) {
                     std::vector<std::vector<ir::ForStmtPtr>> result;
                     for (const auto& segment : order.Input().segments) {
                       std::vector<ir::ForStmtPtr> frames;
                       for (const auto& frame : segment.nested_frames) frames.push_back(frame.loop);
                       result.push_back(std::move(frames));
                     }
                     return result;
                   })
      .def("prove_transition",
           [](const ir::sync::ScopedTokenOrder& order, const ir::sync::Occurrence& consumer,
              const ir::sync::Occurrence& producer,
              const std::vector<std::tuple<ir::ForStmtPtr, int64_t, int64_t, int64_t, int64_t, int64_t>>&
                  axes) {
             ir::sync::ScopedEventTransition query{consumer, producer, {}};
             for (const auto& [frame, count, source, target, source_step, target_step] : axes)
               query.axes.push_back({frame, count, source, target, source_step, target_step});
             return order.ProveTransition(query);
           })
      .def("validate_token_owner",
           [](const ir::sync::ScopedTokenOrder& order, const ir::FunctionPtr& function) {
             CHECK(order.Protocol()->Tokens() == function)
                 << "Unassigned synchronization protocol belongs to different token IR";
           })
      .def_prop_ro("summary_nodes", &ir::sync::ScopedTokenOrder::SummaryNodes)
      .def_prop_ro("interface_entries", &ir::sync::ScopedTokenOrder::InterfaceEntries)
      .def_prop_ro("witness_attempts", &ir::sync::ScopedTokenOrder::WitnessAttempts)
      .def_prop_ro("candidate_visits", &ir::sync::ScopedTokenOrder::CandidateVisits)
      .def_prop_ro("binary_probes", &ir::sync::ScopedTokenOrder::BinaryProbes)
      .def("reuse",
           [](const ir::sync::ScopedTokenOrder& order, size_t i) { return order.View().scoped_reuse(i); })
      .def("barrier",
           [](const ir::sync::ScopedTokenOrder& order, size_t i) { return order.View().scoped_barrier(i); })
      .def(
          "realize_legacy",
          [](const ir::sync::ScopedTokenOrder& order, const std::optional<std::vector<int>>& eligible_ids) {
            const auto prepared = ir::sync::PrepareScopedTokenOrder(order.Protocol());
            auto view = prepared->View();
            view.scoped_palette = {};
            return DescribePhysicalSync(ir::sync::RealizeDirectSync(prepared->Input(), prepared->Families(),
                                                                    view, prepared->Input().hardware_target,
                                                                    eligible_ids));
          },
          nb::arg("eligible_ids") = nb::none())
      .def(
          "realize",
          [](const ir::sync::ScopedTokenOrder& order, const std::string& target,
             const std::optional<std::vector<int>>& eligible_ids) {
            const auto prepared = ir::sync::PrepareScopedTokenOrder(order.Protocol());
            return ir::sync::RealizeDirectSync(prepared->Input(), prepared->Families(), prepared->View(),
                                               target, eligible_ids)
                .function;
          },
          nb::arg("target") = "a2a3", nb::arg("eligible_ids") = nb::none());
  testing.def("check_generated_logical_sync", &ir::sync::CheckGeneratedLogicalSync, nb::arg("function"),
              "Check insertion-owned logical IR using the production internal-error boundary");
  testing.def(
      "lower_sync_schedule",
      [](const ir::FunctionPtr& function, const std::string& target,
         const ir::sync::SyncParameterBounds& parameters, std::optional<std::pair<int64_t, int64_t>> blocks,
         const ir::sync::SyncSchedulePolicy& policy, const std::optional<std::vector<int>>& eligible_ids,
         std::optional<int64_t> launch_count) {
        const auto result =
            ir::sync::LowerLogicalSync(function, target, eligible_ids, parameters, blocks, policy,
                                       ir::sync::SyncExternalPolicy::Geometry, launch_count);
        nb::dict out;
        out["function"] = result.function;
        out["set_sites"] = result.set_sites;
        out["wait_sites"] = result.wait_sites;
        nb::list cases;
        for (const auto& item : result.physical_cases) cases.append(DescribePhysicalSync(item));
        out["physical_cases"] = cases;
        out["schedule"] = DescribeSchedule(result.physical_cases.empty() ? ir::sync::PhysicalSyncRealization{}
                                                                         : result.physical_cases.front());
        out["schedule_policy"] = DescribeSchedulePolicy(policy);
        return out;
      },
      nb::arg("function"), nb::arg("target") = "a2a3",
      nb::arg("parameter_bounds") = ir::sync::SyncParameterBounds{}, nb::arg("block_bounds") = nb::none(),
      nb::kw_only(), nb::arg("schedule_policy") = ir::sync::SyncSchedulePolicy{},
      nb::arg("eligible_ids") = nb::none(), nb::arg("launch_count") = nb::none(),
      "Run the production actual-token lowering with explicit current invocation evidence");
  testing.def(
      "analyze_scoped_token_order",
      [](const ir::FunctionPtr& function, const std::string& target,
         const ir::sync::SyncParameterBounds& parameters, std::optional<std::pair<int64_t, int64_t>> blocks) {
        ir::CheckLogicalSync(function);
        auto stripped = ir::sync::StripStructuredLogicalSync(function, true);
        auto input = ir::sync::ImportSyncPayloads(stripped.function, target, parameters, blocks);
        auto families = ir::sync::ParseScopedTokenProtocol(function, input, stripped.loops);
        return std::make_shared<ir::sync::ScopedTokenOrder>(std::move(families));
      },
      nb::arg("function"), nb::arg("target") = "a2a3",
      nb::arg("parameter_bounds") = ir::sync::SyncParameterBounds{}, nb::arg("block_bounds") = nb::none());
  testing.def(
      "inspect_scoped_token_families",
      [](const ir::FunctionPtr& function, const std::string& target,
         const ir::sync::SyncParameterBounds& parameters, std::optional<std::pair<int64_t, int64_t>> blocks,
         bool prepare) {
        ir::CheckLogicalSync(function);
        auto stripped = ir::sync::StripStructuredLogicalSync(function, true);
        const auto input = ir::sync::ImportSyncPayloads(stripped.function, target, parameters, blocks);
        ir::sync::ScopedTokenWork work;
        auto protocol = ir::sync::ParseScopedTokenProtocol(function, input, stripped.loops, &work);
        auto families = protocol->Families();
        const auto raw_count = families.size();
        if (prepare) families = ir::sync::PrepareScopedTokenOrder(protocol, &work)->Families();
        nb::dict out;
        nb::list cuts;
        for (const auto& cut : protocol->Cuts())
          cuts.append(nb::make_tuple(cut.region, cut.site, cut.before, cut.after));
        out["protocol_cuts"] = cuts;
        out["raw_family_count"] = raw_count;
        out["prepared"] = prepare;
        out["function"] = stripped.function;
        nb::dict counts;
        counts["read_statement_visits"] = work.statements;
        counts["waits_inspected"] = work.waits_inspected;
        counts["inactive_waits"] = work.inactive_waits;
        counts["families"] = work.families;
        counts["equation_visits"] = work.equation_visits;
        counts["capture_specializations"] = work.capture_specializations;
        counts["union_candidates"] = work.union_candidates;
        counts["union_merges"] = work.union_merges;
        counts["union_passes"] = work.union_passes;
        counts["union_reuse_checks"] = work.union_reuse_checks;
        counts["union_reuse_refusals"] = work.union_reuse_refusals;
        out["work"] = counts;
        nb::list rows, regions;
        for (const auto& family : families) {
          nb::dict row;
          row["source"] = nb::make_tuple(family.source_region, family.source_first, family.source_site);
          row["target"] = nb::make_tuple(family.target_region, family.target_first, family.target_site);
          row["count"] = family.count;
          row["protocol_uses"] = nb::cast(family.protocol_uses);
          row["source_domain"] = nb::cast(family.source_domain->ranges);
          row["target_domain"] = nb::cast(family.target_domain->ranges);
          row["source_steps"] = nb::cast(family.source_domain->steps);
          row["target_steps"] = nb::cast(family.target_domain->steps);
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
      nb::arg("parameter_bounds") = ir::sync::SyncParameterBounds{}, nb::arg("block_bounds") = nb::none(),
      nb::kw_only(), nb::arg("prepare") = false,
      "Inspect actual scoped token handoffs under the explicit disjoint-GM test contract; no allocation");
  nb::class_<ir::sync::SyncInvocationAnalysis>(testing, "SyncInvocationAnalysis")
      .def(
          "realize_physical",
          [](const ir::sync::SyncInvocationAnalysis& analysis, const std::string& target,
             const std::optional<std::vector<int>>& eligible_ids,
             const ir::sync::SyncSchedulePolicy& policy) {
            const auto result = sync_testing::RealizeDemandInvocation(analysis, target, eligible_ids, policy);
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
            out["schedule"] = DescribeSchedule(result.cases.front());
            out["schedule_policy"] = DescribeSchedulePolicy(policy);
            nb::dict case_sites;
            case_sites["set_sites"] = sets;
            case_sites["wait_sites"] = waits;
            case_sites["barrier_sites"] = barriers;
            case_sites["guard_sites"] = guards;
            out["case_sites"] = case_sites;
            const size_t emitted_branches = DescribeEmittedPhysicalControl(out, result.function);
            PhysicalControlCensus original;
            original.VisitFunction(analysis.owner);
            INTERNAL_CHECK(emitted_branches >= original.branches)
                << "Internal error: invocation payload guards changed";
            out["guard_sites"] = emitted_branches - original.branches;
            return out;
          },
          nb::arg("target"), nb::arg("eligible_ids") = nb::none(), nb::kw_only(),
          nb::arg("schedule_policy") = ir::sync::SyncSchedulePolicy{},
          "Test supplied-demand case adapter; does not exercise actual-token interpretation")
      .def("select_case", &sync_testing::SelectInvocationCase, nb::arg("block_index"), nb::arg("parameters"),
           "Retained case for a fully bound admitted invocation, or None")
      .def(
          "describe",
          [](const ir::sync::SyncInvocationAnalysis& result) {
            nb::dict out;
            out["block_bounds"] = result.block_bounds;
            out["parameter_bounds"] = result.parameter_bounds;
            out["effective_coordinate"] =
                result.coordinate ? nb::cast(result.coordinate->Expression()) : nb::none();
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
  testing.def(
      "analyze_sync_invocation",
      [](const ir::FunctionPtr& function, bool disjoint_gm, const std::string& target,
         const ir::sync::SyncParameterBounds& parameters, std::pair<int64_t, int64_t> blocks,
         std::optional<int64_t> launch_count) {
        return ir::sync::AnalyzeSyncInvocation(function, disjoint_gm, target, parameters, blocks,
                                               launch_count);
      },
      nb::arg("function"), nb::arg("disjoint_gm"), nb::arg("target"), nb::arg("parameter_bounds"),
      nb::arg("block_bounds"), nb::kw_only(), nb::arg("launch_count") = nb::none(),
      "Analyze an exhaustive analytic partition of a proved launch domain");
  testing.def(
      "describe_buffer_sync_operation",
      [](const ir::CallPtr& call, ir::FunctionType core, const std::string& target) {
        const auto qualified = backend::QualifyBufferSyncOperation(call, core, target);
        auto name = [](ir::BufferAccess access) {
          switch (access) {
            case ir::BufferAccess::None:
              return "none";
            case ir::BufferAccess::Read:
              return "read";
            case ir::BufferAccess::Write:
              return "write";
            case ir::BufferAccess::ReadWrite:
              return "read_write";
          }
          INTERNAL_CHECK(false) << "Internal error: invalid Buffer access mode";
          return "";
        };
        nb::list arguments;
        for (size_t i = 0; i < qualified.ArgumentEffects().size(); ++i) {
          const auto& effect = qualified.ArgumentEffects()[i];
          nb::dict argument;
          argument["operand"] = qualified.Operation()->args_[i];
          argument["data"] = name(effect.data);
          argument["metadata"] = name(effect.metadata);
          argument["non_memory"] = effect.non_memory;
          arguments.append(argument);
        }
        nb::dict result;
        result["operation"] = qualified.Operation();
        result["pipe"] = qualified.Implementation().Pipe();
        result["arguments"] = arguments;
        nb::list implicit;
        for (const auto& effect : qualified.Implementation().ImplicitEffects()) {
          nb::dict item;
          item["space"] = effect.space;
          item["begin"] = effect.begin;
          item["end"] = effect.end;
          item["data"] = name(effect.data);
          item["provenance"] = effect.provenance;
          item["exact_footprint"] = effect.exact_footprint;
          implicit.append(item);
        }
        result["implicit_effects"] = implicit;
        result["footprint_available"] = false;
        result["readiness_available"] = false;
        result["resource_assignment_available"] = false;
        return result;
      },
      nb::arg("call"), nb::arg("core"), nb::arg("target") = "a2a3",
      "Inspect static payload effects without loop, coordinate, geometry or readiness analysis");
  testing.def(
      "analyze_conservative_sync",
      [](const ir::FunctionPtr& function, const std::string& target,
         const std::vector<std::shared_ptr<const ir::sync::NativeSyncChild>>& children,
         bool retain_incoming_diagnostics, bool automatic_children, bool enable_refresh_kill) {
        const auto analysis =
            ir::sync::AnalyzeConservativeSync(function, target, children, {}, retain_incoming_diagnostics,
                                              automatic_children, enable_refresh_kill);
        auto facts = [](const ir::sync::ConservativeAccessSet& accesses) {
          nb::list result;
          for (const auto& [space, pipe, write, domain] : accesses) {
            nb::dict fact;
            fact["space"] = ir::MemorySpaceToString(space);
            fact["pipe"] = pipe;
            fact["write"] = write;
            fact["domain"] = domain;
            result.append(fact);
          }
          return result;
        };
        class StatementParents : public ir::ParentStmtAnalysis {
         public:
          void VisitExpr(const ir::ExprPtr&) override {}
        };
        StatementParents parents;
        parents.BuildMap(analysis.Function());
        nb::list sites;
        size_t control_output_nodes = 0;
        for (size_t index = 0; index < analysis.Sites().size(); ++index) {
          const auto& site = analysis.Sites()[index];
          nb::dict record;
          record["cut"] = site.cut;
          record["operation"] = site.operation.Operation();
          record["pipe"] = site.operation.Implementation().Pipe();
          std::vector<ir::StmtPtr> control;
          for (auto parent = parents.GetParent(site.cut); parent; parent = parents.GetParent(parent)) {
            if (ir::As<ir::IfStmt>(parent) || ir::As<ir::ForStmt>(parent) || ir::As<ir::WhileStmt>(parent))
              control.push_back(parent);
          }
          std::reverse(control.begin(), control.end());
          control_output_nodes += control.size();
          record["control"] = control;
          record["effects"] = facts(site.effects);
          if (analysis.HasIncomingDiagnostics())
            record["incoming"] = facts(site.incoming);
          else
            record["incoming"] = nb::none();
          record["source_pipes"] = site.source_pipes;
          record["required_source_pipes"] = site.required_source_pipes;
          record["post_destination"] = site.post_destination ? nb::cast(*site.post_destination) : nb::none();
          record["entry_only_source_pipes"] =
              site.loop_entry ? site.loop_entry->entry_only : std::vector<ir::PipeType>{};
          record["backedge_only_source_pipes"] =
              site.loop_entry ? site.loop_entry->backedge_only : std::vector<ir::PipeType>{};
          record["empty_entry_source_pipes"] =
              site.loop_entry ? site.loop_entry->empty_entry_sources : std::vector<ir::PipeType>{};
          nb::list refreshed;
          for (const auto& [space, domain] : site.refreshed_domains) {
            nb::dict entry;
            entry["space"] = ir::MemorySpaceToString(space);
            entry["domain"] = domain;
            refreshed.append(entry);
          }
          record["refreshed_domains"] = refreshed;
          sites.append(record);
        }
        const auto& cost = analysis.Cost();
        nb::dict counters;
        counters["statements"] = cost.statements;
        counters["expression_visits"] = cost.expression_visits;
        counters["dataflow_visits"] = cost.dataflow_visits;
        counters["retained_facts"] = cost.retained_facts;
        counters["services"] = cost.services;
        counters["baseline_services"] = cost.baseline_services;
        counters["suppressed_services"] = cost.suppressed_services;
        counters["certificate_visits"] = cost.certificate_visits;
        counters["certificate_omissions"] = cost.certificate_omissions;
        counters["killed_facts"] = cost.killed_facts;
        counters["parent_links"] = cost.parent_links;
        counters["enclosure_lookups"] = cost.enclosure_lookups;
        counters["storage_domains"] = cost.storage_domains;
        counters["conflict_summary_facts"] = cost.conflict_summary_facts;
        counters["conflict_queries"] = cost.conflict_queries;
        counters["incoming_diagnostic_facts"] = cost.incoming_diagnostic_facts;
        counters["control_snapshot_facts"] = cost.control_snapshot_facts;
        counters["control_merge_facts"] = cost.control_merge_facts;
        counters["diagnostic_control_output_nodes"] = control_output_nodes;
        counters["range_indexed_definitions"] = cost.range_indexed_definitions;
        counters["range_footprint_queries"] = cost.range_footprint_queries;
        counters["range_fallback_definitions"] = cost.range_fallback_definitions;
        nb::dict result;
        result["function"] = analysis.Function();
        result["sites"] = sites;
        nb::list child_entries;
        for (const auto& child : analysis.ChildEntries()) {
          nb::dict entry;
          entry["scope"] = child.scope;
          if (analysis.HasIncomingDiagnostics())
            entry["incoming"] = facts(child.incoming);
          else
            entry["incoming"] = nb::none();
          entry["effects"] = facts(child.effects);
          entry["services"] = child.services;
          entry["required_services"] = child.required_services;
          child_entries.append(entry);
        }
        result["children"] = child_entries;
        const auto& selection = analysis.ChildSelection();
        nb::list attempts;
        for (const auto& attempt : selection.attempts) {
          nb::dict record;
          record["scope"] = attempt.scope;
          record["stage"] = ir::sync::EventfulSyncChildStageName(attempt.stage);
          record["reason"] = attempt.reason;
          record["mathematics"] = attempt.mathematics ? nb::cast(*attempt.mathematics) : nb::none();
          record["child"] = attempt.child ? nb::cast(attempt.child) : nb::none();
          record["scope_nodes"] = attempt.binding_work.scope_nodes;
          record["expression_nodes"] = attempt.binding_work.expression_nodes;
          record["binding_definitions"] = attempt.binding_work.binding_definitions;
          record["ancestor_links"] = attempt.binding_work.ancestor_links;
          record["logical_index_nodes"] = attempt.binding_work.logical_index_nodes;
          record["effect_statements"] = attempt.effect_work.statements;
          record["effect_expressions"] = attempt.effect_work.expressions;
          record["native_qualifications"] = attempt.effect_work.native_qualifications;
          record["logical_record_sites"] = attempt.logical_record_sites;
          record["logical_wait_sites"] = attempt.logical_wait_sites;
          record["physical_qualification"] = "pending_whole_plan";
          nb::dict seconds;
          for (const auto& [phase, elapsed] : attempt.seconds)
            seconds[ir::sync::EventfulSyncChildStageName(phase)] = elapsed;
          record["seconds"] = seconds;
          attempts.append(record);
        }
        result["child_attempts"] = attempts;
        counters["child_discovery_nodes"] = selection.discovery_nodes;
        counters["child_index_nodes"] = selection.index_nodes;
        counters["child_parameter_nodes"] = selection.parameter_nodes;
        result["exit_effects"] =
            analysis.HasIncomingDiagnostics() ? facts(analysis.ExitEffects()) : nb::none();
        result["cost"] = counters;
        nb::list domains;
        for (const auto& domain : analysis.StorageDomains()) {
          nb::dict record;
          record["id"] = domain.id;
          record["space"] = ir::MemorySpaceToString(domain.space);
          record["enclosure"] = domain.enclosure
                                    ? nb::cast(std::make_pair(domain.enclosure->begin, domain.enclosure->end))
                                    : nb::none();
          record["original_root"] = domain.original_root ? nb::cast(domain.original_root) : nb::none();
          record["exact_selector"] = false;
          domains.append(record);
        }
        result["storage_domains"] = domains;
        result["incoming_diagnostics_available"] = analysis.HasIncomingDiagnostics();
        result["disjoint_gm"] = analysis.SuppliedDisjointGM();
        result["gm_alias_contract"] = analysis.SuppliedDisjointGM() ? "supplied_disjoint_roots" : "may_alias";
        result["precision"] =
            analysis.HasRefreshKill() ? "storage_domain_refresh_kill" : "storage_domain_no_kill";
        result["exact_storage_selectors"] = false;
        result["executable_protocol"] = false;
        return result;
      },
      nb::arg("function"), nb::arg("target") = "a2a3", nb::kw_only(),
      nb::arg("children") = std::vector<std::shared_ptr<const ir::sync::NativeSyncChild>>{},
      nb::arg("retain_incoming_diagnostics") = true, nb::arg("automatic_children") = false,
      nb::arg("enable_refresh_kill") = true,
      "Qualify conservative effects/readiness; incoming history is an optional diagnostic export");
  testing.def(
      "insert_conservative_prefixes",
      [](const ir::FunctionPtr& function, const std::string& target,
         const std::vector<std::shared_ptr<const ir::sync::NativeSyncChild>>& children) {
        return ir::sync::InsertConservativePrefixes(
            ir::sync::AnalyzeConservativeSync(function, target, children));
      },
      nb::arg("function"), nb::arg("target") = "a2a3", nb::kw_only(),
      nb::arg("children") = std::vector<std::shared_ptr<const ir::sync::NativeSyncChild>>{},
      "Explicit research request for conservative logical prefix protocols; production dispatch unchanged");
  testing.def("inspect_eventful_sync_child", &ir::sync::RetainedSyncChildDiagnostics::Mathematics,
              nb::arg("child"), "Inspect the child's actual retained immutable mathematics without analysis");
  testing.def(
      "export_native_sync_child",
      [](const ir::sync::CompactSyncResult& mathematics, const ir::ForStmtPtr& scope,
         const std::string& target) {
        const auto result = ir::sync::ExportNativeSyncChild(mathematics, scope, target);
        nb::dict out;
        out["status"] = ir::sync::NativeSyncChildStatusName(result.status);
        out["reason"] = result.reason;
        out["child"] = result.child;
        out["source_nodes"] = result.source_nodes;
        out["summary_nodes"] = result.summary_nodes;
        out["demands"] = result.demands;
        out["operations"] = result.operations;
        return out;
      },
      nb::arg("mathematics"), nb::arg("scope"), nb::arg("target") = "a2a3",
      "Request a context-bound native-only child from existing retained numerical mathematics");
  testing.def(
      "describe_placed_sync_input",
      [](const ir::FunctionPtr& function, bool disjoint_gm, const std::string& target,
         const ir::sync::SyncParameterBounds& bounds,
         const std::optional<std::pair<int64_t, int64_t>>& block_bounds) {
        ir::sync::CompactSyncBuilder imported;
        imported.State().input =
            ir::sync::ImportPlacedSync(function, disjoint_gm, target, bounds, block_bounds);
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
        result.Diagnostics().import_seconds = seconds;
        return result;
      },
      nb::arg("function"), nb::arg("disjoint_gm"), nb::arg("target") = "a2a3",
      nb::arg("parameter_bounds") = ir::sync::SyncParameterBounds{},
      nb::arg("composition_order") = "balanced", nb::arg("block_bounds") = nb::none(),
      "Read-only research analysis of qualified placed Buffer IR under a named hardware contract");

  testing.def(
      "sync_parameter_bounds",
      [](const ir::ProgramPtr& program) {
        return ir::sync::CollectSyncInvocationEvidence(program).parameters;
      },
      nb::arg("program"), "Inspect proved caller coordinate ranges without analysis or realization");
  testing.def(
      "sync_invocation_evidence",
      [](const ir::ProgramPtr& program) {
        ir::sync::SyncParameterWork work;
        const auto evidence = ir::sync::CollectSyncInvocationEvidence(program, &work);
        nb::dict out;
        out["parameters"] = evidence.parameters;
        out["block_indices"] = evidence.block_indices;
        out["block_counts"] = evidence.block_counts;
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
        auto bounds = ir::sync::CollectSyncInvocationEvidence(program, &work).parameters;
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
