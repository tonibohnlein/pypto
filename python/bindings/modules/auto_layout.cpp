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

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>            // NOLINT(misc-include-cleaner) -- registers std::map casters
#include <nanobind/stl/optional.h>       // NOLINT(misc-include-cleaner) -- registers std::optional casters
#include <nanobind/stl/pair.h>           // NOLINT(misc-include-cleaner) -- registers std::pair casters
#include <nanobind/stl/shared_ptr.h>     // NOLINT(misc-include-cleaner) -- registers std::shared_ptr casters
#include <nanobind/stl/string.h>         // NOLINT(misc-include-cleaner) -- registers std::string casters
#include <nanobind/stl/unordered_map.h>  // NOLINT(misc-include-cleaner) -- registers std::unordered_map casters
#include <nanobind/stl/unordered_set.h>  // NOLINT(misc-include-cleaner) -- registers std::unordered_set casters
#include <nanobind/stl/vector.h>         // NOLINT(misc-include-cleaner) -- registers std::vector casters

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "pypto/ir/transforms/auto_layout/extractor.h"
#include "pypto/ir/transforms/auto_layout/model.h"
#include "pypto/ir/transforms/auto_layout/solver.h"
#include "pypto/ir/transforms/auto_layout/writer.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

namespace al = pypto::ir::auto_layout;

void BindAutoLayout(nb::module_& m) {
  nb::module_ mod =
      m.def_submodule("auto_layout", "Native C++ AutoLayout model, IR extractor, solver, and writer");

  auto spec_cls = nb::class_<al::LayoutSpec>(mod, "LayoutSpec");
  nb::enum_<al::LayoutSpec::Kind>(spec_cls, "Kind")
      .value("Tensor", al::LayoutSpec::Kind::kTensor)
      .value("Tile", al::LayoutSpec::Kind::kTile);

  spec_cls.def(nb::init<>())
      .def_rw("kind", &al::LayoutSpec::kind)
      .def_rw("tensor_layout", &al::LayoutSpec::tensor_layout)
      .def_rw("blayout", &al::LayoutSpec::blayout)
      .def_rw("slayout", &al::LayoutSpec::slayout)
      .def_rw("fractal", &al::LayoutSpec::fractal)
      .def_rw("tag", &al::LayoutSpec::tag)
      .def_static("tensor", &al::LayoutSpec::Tensor, nb::arg("layout"))
      .def_static("nd", &al::LayoutSpec::ND)
      .def_static("dn", &al::LayoutSpec::DN)
      .def_static("nz", &al::LayoutSpec::NZ)
      .def_static("tile", &al::LayoutSpec::Tile, nb::arg("blayout"), nb::arg("slayout"),
                  nb::arg("fractal") = 512, nb::arg("tag") = "")
      .def_static("row_major_flat", &al::LayoutSpec::RowMajorFlat)
      .def_static("col_major_flat", &al::LayoutSpec::ColMajorFlat)
      .def_static("row_bcast_flat", &al::LayoutSpec::RowBcastFlat)
      .def_static("fractal_nz", &al::LayoutSpec::FractalNZ)
      .def_static("fractal_zn", &al::LayoutSpec::FractalZN)
      .def_static("fractal_acc_nz", &al::LayoutSpec::FractalAccNZ)
      .def("is_tensor", &al::LayoutSpec::IsTensor)
      .def("is_tile", &al::LayoutSpec::IsTile)
      .def("short_name", &al::LayoutSpec::ShortName)
      .def("__eq__", [](const al::LayoutSpec& a, const al::LayoutSpec& b) { return a == b; })
      .def("__ne__", [](const al::LayoutSpec& a, const al::LayoutSpec& b) { return a != b; })
      .def("__lt__", [](const al::LayoutSpec& a, const al::LayoutSpec& b) { return a < b; })
      .def("__hash__", [](const al::LayoutSpec& s) { return al::LayoutSpecHash{}(s); })
      .def("__repr__", [](const al::LayoutSpec& s) { return "<LayoutSpec " + s.ShortName() + ">"; });

  nb::class_<al::ValueNode>(mod, "ValueNode")
      .def(nb::init<>())
      .def_rw("name", &al::ValueNode::name)
      .def_rw("shape", &al::ValueNode::shape)
      .def_rw("dtype", &al::ValueNode::dtype)
      .def_rw("memory_space", &al::ValueNode::memory_space)
      .def_rw("current_layout", &al::ValueNode::current_layout)
      .def_rw("admissible_layouts", &al::ValueNode::admissible_layouts)
      .def_rw("producer_op_id", &al::ValueNode::producer_op_id)
      .def_rw("is_reusable_weight", &al::ValueNode::is_reusable_weight)
      .def_rw("is_external_output", &al::ValueNode::is_external_output)
      .def_prop_ro("num_elements", &al::ValueNode::NumElements)
      .def_prop_ro("byte_size", &al::ValueNode::ByteSize);

  nb::class_<al::OperatorConfig>(mod, "OperatorConfig")
      .def(nb::init<>())
      .def_rw("op_id", &al::OperatorConfig::op_id)
      .def_rw("op_name", &al::OperatorConfig::op_name)
      .def_rw("input_layouts", &al::OperatorConfig::input_layouts)
      .def_rw("output_layout", &al::OperatorConfig::output_layout)
      .def_rw("op_cost_cycles", &al::OperatorConfig::op_cost_cycles)
      .def_rw("notes", &al::OperatorConfig::notes)
      .def("__eq__", [](const al::OperatorConfig& a, const al::OperatorConfig& b) { return a == b; })
      .def("__ne__", [](const al::OperatorConfig& a, const al::OperatorConfig& b) { return a != b; });

  nb::class_<al::OperationNode>(mod, "OperationNode")
      .def(nb::init<>())
      .def_rw("op_id", &al::OperationNode::op_id)
      .def_rw("op_name", &al::OperationNode::op_name)
      .def_rw("inputs", &al::OperationNode::inputs)
      .def_rw("output", &al::OperationNode::output)
      .def_rw("invocations", &al::OperationNode::invocations)
      .def_rw("configs", &al::OperationNode::configs);

  nb::class_<al::LayoutCostModel>(mod, "LayoutCostModel")
      .def(nb::init<>())
      .def_rw("expected_uses", &al::LayoutCostModel::expected_uses)
      .def_rw("backend_type", &al::LayoutCostModel::backend_type)
      .def_rw("resolved_arch_hint", &al::LayoutCostModel::resolved_arch_hint)
      .def_rw("mte2_nz2nz_bytes_per_cycle", &al::LayoutCostModel::mte2_nz2nz_bytes_per_cycle)
      .def_rw("mte2_nd2nz_bytes_per_cycle", &al::LayoutCostModel::mte2_nd2nz_bytes_per_cycle)
      .def_rw("mte2_nd2nz_contiguous_bytes_per_cycle",
              &al::LayoutCostModel::mte2_nd2nz_contiguous_bytes_per_cycle)
      .def_rw("mte2_nd2nz_row_penalty_cycles", &al::LayoutCostModel::mte2_nd2nz_row_penalty_cycles)
      .def_rw("mte2_nd2nz_sub512b_penalty_cycles", &al::LayoutCostModel::mte2_nd2nz_sub512b_penalty_cycles)
      .def_rw("mte2_l2_reread_discount", &al::LayoutCostModel::mte2_l2_reread_discount)
      .def_rw("l2_cache_line_bytes", &al::LayoutCostModel::l2_cache_line_bytes)
      .def_rw("mte2_vec_bytes_per_cycle", &al::LayoutCostModel::mte2_vec_bytes_per_cycle)
      .def_rw("mte2_vec_strided_bytes_per_cycle", &al::LayoutCostModel::mte2_vec_strided_bytes_per_cycle)
      .def_rw("mte3_vec_bytes_per_cycle", &al::LayoutCostModel::mte3_vec_bytes_per_cycle)
      .def_rw("mte3_vec_strided_bytes_per_cycle", &al::LayoutCostModel::mte3_vec_strided_bytes_per_cycle)
      .def_rw("vec_strided_burst_penalty_cycles", &al::LayoutCostModel::vec_strided_burst_penalty_cycles)
      .def_rw("mte1_l1_to_l0_bytes_per_cycle", &al::LayoutCostModel::mte1_l1_to_l0_bytes_per_cycle)
      .def_rw("mte1_l0a_bytes_per_cycle", &al::LayoutCostModel::mte1_l0a_bytes_per_cycle)
      .def_rw("mte1_l0b_bytes_per_cycle", &al::LayoutCostModel::mte1_l0b_bytes_per_cycle)
      .def_rw("mte1_transpose_penalty_cycles", &al::LayoutCostModel::mte1_transpose_penalty_cycles)
      .def_rw("cube_macs_per_cycle", &al::LayoutCostModel::cube_macs_per_cycle)
      .def_rw("cube_issue_overhead_cycles", &al::LayoutCostModel::cube_issue_overhead_cycles)
      .def_rw("mad_fp32_passes", &al::LayoutCostModel::mad_fp32_passes)
      .def_rw("vec_elements_per_cycle", &al::LayoutCostModel::vec_elements_per_cycle)
      .def_rw("vec_reshape_metadata_cycles", &al::LayoutCostModel::vec_reshape_metadata_cycles)
      .def_rw("vec_transpose_fixed_cycles", &al::LayoutCostModel::vec_transpose_fixed_cycles)
      .def_rw("vec_vbrcb_fixed_cycles", &al::LayoutCostModel::vec_vbrcb_fixed_cycles)
      .def_rw("vec_vbrcb_cycles_per_row", &al::LayoutCostModel::vec_vbrcb_cycles_per_row)
      .def_rw("vec_row_expand_issue_cycles", &al::LayoutCostModel::vec_row_expand_issue_cycles)
      .def_rw("vec_unsupported_conversion_cycles", &al::LayoutCostModel::vec_unsupported_conversion_cycles)
      .def_rw("fixpipe_acc_to_nd_bytes_per_cycle", &al::LayoutCostModel::fixpipe_acc_to_nd_bytes_per_cycle)
      .def_rw("fixpipe_acc_to_nz_bytes_per_cycle", &al::LayoutCostModel::fixpipe_acc_to_nz_bytes_per_cycle)
      .def_rw("fixpipe_drain_fixed_cycles", &al::LayoutCostModel::fixpipe_drain_fixed_cycles)
      .def_rw("fixpipe_odd_n1_penalty_cycles", &al::LayoutCostModel::fixpipe_odd_n1_penalty_cycles)
      .def_rw("gm_pack_bytes_per_cycle", &al::LayoutCostModel::gm_pack_bytes_per_cycle)
      .def_rw("gm_pack_fixed_cycles", &al::LayoutCostModel::gm_pack_fixed_cycles)
      .def_rw("onchip_relayout_bytes_per_cycle", &al::LayoutCostModel::onchip_relayout_bytes_per_cycle)
      .def_rw("allow_external_input_relayout", &al::LayoutCostModel::allow_external_input_relayout)
      .def_rw("allow_external_output_relayout", &al::LayoutCostModel::allow_external_output_relayout)
      .def("target_arch", &al::LayoutCostModel::TargetArch)
      .def("gm_access_granularity_bytes", &al::LayoutCostModel::GmAccessGranularityBytes)
      .def("supports_acc_nz_store", &al::LayoutCostModel::SupportsAccNzStore, nb::arg("src_node"),
           nb::arg("dst_node"))
      .def("estimate_vbrcb_cost", &al::LayoutCostModel::EstimateVbrcbCost, nb::arg("rows"))
      .def("is_valid_load_config", &al::LayoutCostModel::IsValidLoadConfig, nb::arg("src_node"),
           nb::arg("dst_node"), nb::arg("src_layout"), nb::arg("dst_layout"))
      .def("is_valid_vec_store_config", &al::LayoutCostModel::IsValidVecStoreConfig, nb::arg("src_node"),
           nb::arg("dst_node"), nb::arg("src_layout"), nb::arg("dst_layout"))
      .def("is_valid_move_config", &al::LayoutCostModel::IsValidMoveConfig, nb::arg("src_node"),
           nb::arg("dst_node"), nb::arg("src_layout"), nb::arg("dst_layout"))
      .def("effective_mte2_load_multiplier", &al::LayoutCostModel::EffectiveMte2LoadMultiplier,
           nb::arg("src_node"), nb::arg("dst_node"), nb::arg("per_run_inv"))
      .def("estimate_load_cost", &al::LayoutCostModel::EstimateLoadCost, nb::arg("src_node"),
           nb::arg("dst_node"), nb::arg("src_layout"), nb::arg("dst_layout"))
      .def("estimate_move_cost", &al::LayoutCostModel::EstimateMoveCost, nb::arg("src_node"),
           nb::arg("dst_node"), nb::arg("src_layout"), nb::arg("dst_layout"))
      .def("estimate_matmul_cost", &al::LayoutCostModel::EstimateMatmulCost, nb::arg("lhs_node"),
           nb::arg("rhs_node"), nb::arg("out_node"))
      .def("estimate_store_cost", &al::LayoutCostModel::EstimateStoreCost, nb::arg("src_node"),
           nb::arg("dst_node"), nb::arg("dst_layout"), nb::arg("src_layout") = std::nullopt)
      .def("supports_vec_tile_conversion", &al::LayoutCostModel::SupportsVecTileConversion,
           nb::arg("val_node"), nb::arg("src_layout"), nb::arg("dst_layout"))
      .def("estimate_conversion_cost", &al::LayoutCostModel::EstimateConversionCost, nb::arg("val_node"),
           nb::arg("src_layout"), nb::arg("dst_layout"));

  nb::class_<al::LayoutProblemInstance>(mod, "LayoutProblemInstance")
      .def(nb::init<>())
      .def_rw("values", &al::LayoutProblemInstance::values)
      .def_rw("value_order", &al::LayoutProblemInstance::value_order)
      .def_rw("operations", &al::LayoutProblemInstance::operations)
      .def_rw("cost_model", &al::LayoutProblemInstance::cost_model)
      .def_rw("custom_conversion_costs", &al::LayoutProblemInstance::custom_conversion_costs)
      .def("add_value", &al::LayoutProblemInstance::AddValue, nb::arg("node"))
      .def("find_value", &al::LayoutProblemInstance::FindValue, nb::arg("name"),
           nb::rv_policy::reference_internal)
      .def("estimate_conversion_cost", &al::LayoutProblemInstance::EstimateConversionCost,
           nb::arg("val_node"), nb::arg("src_layout"), nb::arg("dst_layout"))
      .def(
          "set_custom_conversion_cost",
          [](al::LayoutProblemInstance& self, const std::string& value_name, const al::LayoutSpec& src,
             const al::LayoutSpec& dst,
             double cost) { self.custom_conversion_costs[value_name][{src, dst}] = cost; },
          nb::arg("value_name"), nb::arg("src_layout"), nb::arg("dst_layout"), nb::arg("cost"));

  nb::class_<al::LayoutAssignment>(mod, "LayoutAssignment")
      .def(nb::init<>())
      .def_rw("value_layouts", &al::LayoutAssignment::value_layouts)
      .def_rw("op_configs", &al::LayoutAssignment::op_configs);

  nb::class_<al::CostBreakdown>(mod, "CostBreakdown")
      .def(nb::init<>())
      .def_rw("assignment", &al::CostBreakdown::assignment)
      .def_rw("total_cost_cycles", &al::CostBreakdown::total_cost_cycles)
      .def_rw("operator_cost_cycles", &al::CostBreakdown::operator_cost_cycles)
      .def_rw("conversion_cost_cycles", &al::CostBreakdown::conversion_cost_cycles)
      .def_rw("per_op_costs", &al::CostBreakdown::per_op_costs)
      .def_rw("per_value_conversion_costs", &al::CostBreakdown::per_value_conversion_costs);

  nb::class_<al::LayoutSolverConfig>(mod, "LayoutSolverConfig")
      .def(nb::init<>())
      .def_rw("max_evaluations", &al::LayoutSolverConfig::max_evaluations)
      .def_rw("max_greedy_sweeps", &al::LayoutSolverConfig::max_greedy_sweeps)
      .def_rw("max_treewidth", &al::LayoutSolverConfig::max_treewidth)
      .def_rw("max_bag_states", &al::LayoutSolverConfig::max_bag_states)
      .def_rw("tolerance", &al::LayoutSolverConfig::tolerance)
      .def("validate", &al::LayoutSolverConfig::Validate);

  nb::class_<al::LayoutConversion>(mod, "LayoutConversion")
      .def(nb::init<>())
      .def_rw("value_name", &al::LayoutConversion::value_name)
      .def_rw("source_layout", &al::LayoutConversion::source_layout)
      .def_rw("target_layout", &al::LayoutConversion::target_layout)
      .def_rw("cost_cycles", &al::LayoutConversion::cost_cycles)
      .def_rw("consumer_op_ids", &al::LayoutConversion::consumer_op_ids)
      .def_rw("is_boundary", &al::LayoutConversion::is_boundary);

  nb::class_<al::TreewidthStats>(mod, "TreewidthStats")
      .def(nb::init<>())
      .def_rw("num_values", &al::TreewidthStats::num_values)
      .def_rw("num_operations", &al::TreewidthStats::num_operations)
      .def_rw("num_movable_operations", &al::TreewidthStats::num_movable_operations)
      .def_rw("dataflow_treewidth", &al::TreewidthStats::dataflow_treewidth)
      .def_rw("primal_treewidth", &al::TreewidthStats::primal_treewidth)
      .def_rw("active_primal_treewidth", &al::TreewidthStats::active_primal_treewidth)
      .def_rw("max_bag_size", &al::TreewidthStats::max_bag_size)
      .def_rw("max_active_bag_size", &al::TreewidthStats::max_active_bag_size)
      .def_rw("max_bag_domain_product", &al::TreewidthStats::max_bag_domain_product)
      .def_rw("total_dp_states", &al::TreewidthStats::total_dp_states)
      .def_rw("elimination_order", &al::TreewidthStats::elimination_order)
      .def_rw("bags", &al::TreewidthStats::bags);

  nb::class_<al::LayoutSolverStats>(mod, "LayoutSolverStats")
      .def(nb::init<>())
      .def_rw("evaluated_moves", &al::LayoutSolverStats::evaluated_moves)
      .def_rw("moves_applied", &al::LayoutSolverStats::moves_applied)
      .def_rw("greedy_sweeps", &al::LayoutSolverStats::greedy_sweeps)
      .def_rw("initial_cost_cycles", &al::LayoutSolverStats::initial_cost_cycles)
      .def_rw("termination_reason", &al::LayoutSolverStats::termination_reason)
      .def_rw("treewidth", &al::LayoutSolverStats::treewidth)
      .def_rw("active_treewidth", &al::LayoutSolverStats::active_treewidth);

  nb::class_<al::LayoutSolverResult>(mod, "LayoutSolverResult")
      .def(nb::init<>())
      .def_rw("breakdown", &al::LayoutSolverResult::breakdown)
      .def_rw("stats", &al::LayoutSolverResult::stats)
      .def_rw("conversions", &al::LayoutSolverResult::conversions)
      .def_rw("writeback_issues", &al::LayoutSolverResult::writeback_issues)
      .def_rw("treewidth_stats", &al::LayoutSolverResult::treewidth_stats)
      .def_prop_ro("assignment", &al::LayoutSolverResult::Assignment)
      .def_prop_ro("total_cost_cycles", &al::LayoutSolverResult::TotalCostCycles)
      .def_prop_ro("writeback_assignment", &al::LayoutSolverResult::WritebackAssignment);

  auto indexed_cls = nb::class_<al::IndexedProblem>(mod, "IndexedProblem");
  nb::class_<al::IndexedProblem::ValueInfo>(indexed_cls, "ValueInfo")
      .def_ro("name", &al::IndexedProblem::ValueInfo::name)
      .def_ro("producer_op", &al::IndexedProblem::ValueInfo::producer_op)
      .def_ro("participants", &al::IndexedProblem::ValueInfo::participants)
      .def_ro("layouts", &al::IndexedProblem::ValueInfo::layouts)
      .def_ro("default_source_idx", &al::IndexedProblem::ValueInfo::default_source_idx)
      .def_ro("boundary_demand_mask", &al::IndexedProblem::ValueInfo::boundary_demand_mask)
      .def_ro("boundary_counts", &al::IndexedProblem::ValueInfo::boundary_counts)
      .def_ro("conv_costs", &al::IndexedProblem::ValueInfo::conv_costs);
  nb::class_<al::IndexedProblem::OpConfigUse>(indexed_cls, "OpConfigUse")
      .def_ro("value_id", &al::IndexedProblem::OpConfigUse::value_id)
      .def_ro("demand_mask", &al::IndexedProblem::OpConfigUse::demand_mask)
      .def_ro("layout_counts", &al::IndexedProblem::OpConfigUse::layout_counts);
  nb::class_<al::IndexedProblem::OpInfo>(indexed_cls, "OpInfo")
      .def_ro("op_id", &al::IndexedProblem::OpInfo::op_id)
      .def_ro("op_name", &al::IndexedProblem::OpInfo::op_name)
      .def_ro("invocations", &al::IndexedProblem::OpInfo::invocations)
      .def_ro("output_value", &al::IndexedProblem::OpInfo::output_value)
      .def_ro("costs", &al::IndexedProblem::OpInfo::costs)
      .def_ro("output_layout_indices", &al::IndexedProblem::OpInfo::output_layout_indices)
      .def_ro("config_uses", &al::IndexedProblem::OpInfo::config_uses);
  indexed_cls.def(nb::init<const al::LayoutProblemInstance&>(), nb::keep_alive<1, 2>(), nb::arg("instance"))
      .def_prop_ro("num_values", &al::IndexedProblem::NumValues)
      .def_prop_ro("num_ops", &al::IndexedProblem::NumOps)
      .def_prop_ro("values", &al::IndexedProblem::Values)
      .def_prop_ro("ops", &al::IndexedProblem::Ops)
      .def_prop_ro("topo_order", &al::IndexedProblem::TopoOrder)
      .def("find_op_by_id", &al::IndexedProblem::FindOpById, nb::arg("op_id"))
      .def("find_value_by_name", &al::IndexedProblem::FindValueByName, nb::arg("name"))
      .def("initial_choices", &al::IndexedProblem::InitialChoices, nb::arg("initial"))
      .def("instance", &al::IndexedProblem::Instance, nb::rv_policy::reference_internal);

  nb::class_<al::IncrementalState>(mod, "IncrementalState")
      .def(nb::init<const al::IndexedProblem&, std::vector<uint16_t>>(), nb::keep_alive<1, 2>(),
           nb::arg("problem"), nb::arg("choices"))
      .def("gain", &al::IncrementalState::Gain, nb::arg("op_idx"), nb::arg("new_choice"))
      .def("apply", &al::IncrementalState::Apply, nb::arg("op_idx"), nb::arg("new_choice"))
      .def("to_assignment", &al::IncrementalState::ToAssignment)
      .def_prop_ro("cost", &al::IncrementalState::Cost)
      .def_prop_ro("choices", &al::IncrementalState::Choices)
      .def_prop_ro("sources", &al::IncrementalState::Sources)
      .def_prop_ro("demand_masks", &al::IncrementalState::DemandMasks)
      .def_prop_ro("counts", &al::IncrementalState::Counts)
      .def_prop_ro("source_costs", &al::IncrementalState::SourceCosts)
      .def("problem", &al::IncrementalState::Problem, nb::rv_policy::reference_internal);

  auto orch_cls = nb::class_<al::OrchCallAnalysis>(mod, "OrchCallAnalysis");
  nb::class_<al::OrchCallAnalysis::CallSite>(orch_cls, "CallSite")
      .def_ro("arg_names", &al::OrchCallAnalysis::CallSite::arg_names)
      .def_ro("multiplicity", &al::OrchCallAnalysis::CallSite::multiplicity);
  orch_cls.def_ro("call_counts", &al::OrchCallAnalysis::call_counts)
      .def_ro("call_arg_names", &al::OrchCallAnalysis::call_arg_names)
      .def_ro("call_sites", &al::OrchCallAnalysis::call_sites)
      .def_ro("call_order", &al::OrchCallAnalysis::call_order)
      .def_ro("param_steps", &al::OrchCallAnalysis::param_steps)
      .def_ro("reshaped_names", &al::OrchCallAnalysis::reshaped_names)
      .def_ro("nz_ineligible_names", &al::OrchCallAnalysis::nz_ineligible_names)
      .def_ro("internal_buffers", &al::OrchCallAnalysis::internal_buffers);

  mod.def("c0_elements", &al::C0Elements, nb::arg("dtype"));
  mod.def("try_c0_elements", &al::TryC0Elements, nb::arg("dtype"));
  mod.def("dtype_byte_width_safe", &al::DTypeByteWidthSafe, nb::arg("dtype"));
  mod.def("dtype_byte_width", &al::DTypeByteWidth, nb::arg("dtype"));
  mod.def("is_nz_compatible_shape", &al::IsNzCompatibleShape, nb::arg("shape"), nb::arg("dtype"));
  mod.def("analyze_orchestration", &al::AnalyzeOrchestration, nb::arg("program"));
  mod.def("effective_tile_layout", &al::EffectiveTileLayout, nb::arg("tile_type"));
  mod.def("extract_layout_instance", &al::ExtractLayoutInstance, nb::arg("program"),
          nb::arg("cost_model") = al::LayoutCostModel{},
          nb::arg("reusable_tensors") = std::unordered_set<std::string>{});
  mod.def("apply_layout_assignment", &al::ApplyLayoutAssignment, nb::arg("program"), nb::arg("assignment"));
  mod.def("sync_call_return_type", &al::SyncCallReturnType, nb::arg("call_type"), nb::arg("ret_types"));
  mod.def("operator_cost_multiplier", &al::OperatorCostMultiplier, nb::arg("instance"), nb::arg("op"));
  mod.def("evaluate_assignment", &al::EvaluateAssignment, nb::arg("instance"), nb::arg("assignment"));
  mod.def("enumerate_assignments", &al::EnumerateAssignments, nb::arg("instance"),
          nb::arg("deduplicate") = true);
  mod.def("compute_treewidth", &al::ComputeTreewidth, nb::arg("instance"));
  mod.def("solve_greedy", &al::SolveGreedy, nb::arg("instance"), nb::arg("initial") = std::nullopt,
          nb::arg("config") = al::LayoutSolverConfig{});
  mod.def("solve_treewidth_dp", &al::SolveTreewidthDP, nb::arg("instance"),
          nb::arg("config") = al::LayoutSolverConfig{});
  mod.def("solve_auto_layout", &al::SolveAutoLayout, nb::arg("instance"), nb::arg("initial") = std::nullopt,
          nb::arg("config") = al::LayoutSolverConfig{});
}

}  // namespace python
}  // namespace pypto
