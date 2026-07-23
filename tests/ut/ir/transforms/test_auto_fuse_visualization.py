# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the plan-driven AutoFuse Graphviz visualizations."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_VISUALIZER_PATH = _ROOT / "3rdparty" / "pto-fusebox" / "scripts" / "visualize.py"
_SPEC = importlib.util.spec_from_file_location("fusebox_visualize", _VISUALIZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
visualize = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = visualize
_SPEC.loader.exec_module(visualize)


def _l0_plan(target: str = "acc") -> dict:
    return {
        "tile": [32, 32, 16],
        "stationarity": "output",
        "output_stationary_holds_a": False,
        "buffer_depths": [2, 2, 1],
        "output_target": target,
        "k_loop": {"chunk": 16, "full_chunks": 1, "tail": 0, "pipeline_stages": 1},
        "estimated_traffic_bytes": 4096,
        "estimated_cost_cycles": 100,
        "padded_compute_volume": 16384,
        "phases": {
            "load_cycles": 20,
            "mad_cycles": 70,
            "init_cycles": 90,
            "rolled_cycles": 0,
            "tail_cycles": 0,
            "drain_cycles": 10,
            "wall_cycles": 100,
        },
    }


@pytest.fixture
def vector_problem() -> dict:
    return {
        "widths": [8192, 1, 8192],
        "heights": [8, 8, 8],
        "dtypes": ["FP32", "FP32", "FP32"],
        "inputs": [[0], [0, 1]],
        "outputs": [[1], [2]],
        "op_types": ["Reduction", "Pointwise"],
        "vector_primitive_families": ["row_sum", "div"],
        "required_outputs": [2],
        "num_cube_cores": 24,
        "num_vector_cores": 48,
        "l1_capacity": 524288,
        "cube_capacity": 131072,
        "vec_capacity": 196608,
        "cube_freq_hz": 1.85e9,
        "bw_gm_l1": 135,
        "bw_l0c_gm": 70,
        "bw_gm_ub": 100,
        "bw_ub_gm": 188,
    }


@pytest.fixture
def vector_solution() -> dict:
    return {
        "subgraphs": [[0, 1]],
        "granularities": [[8192, 8, 0]],
        "parts": [[1, 1]],
        "splits": [1],
        "cores": [1],
        "op_order": [[0, 1]],
        "seq_k": [[184, 184]],
        "subgraph_latencies": [1234.5],
        "tensors_to_retain": [[]],
        "vector_stream": [
            {
                "kind": "softmax_flash",
                "work_units": 1,
                "full_peak_ub_bytes": 262144,
                "chunk_peak_ub_bytes": 49152,
                "stream_band_count": 6,
                "axis": 1,
                "free_tile": 8,
                "free_tile_alloc": 16,
                "extent": 8192,
                "chunk": 184,
                "full_chunks": 44,
                "tail": 96,
                "stream_passes": 2,
                "tile": [8, 8192],
                "strip": [8, 184],
                "strip_grid": [1, 45],
                "overlap_granted": True,
                "reduction_split": {
                    "kind": "none",
                    "factor": 1,
                    "partial_extent": 0,
                    "seed": {"present": False},
                },
                "body": {"first_chunk": 0, "trip_count": 0, "pipeline_stages": 1},
                "stats": {"first_chunk": 1, "trip_count": 43, "pipeline_stages": 2},
                "apply": {"first_chunk": 0, "trip_count": 44, "pipeline_stages": 2},
                "serial_phases": {
                    "stats_init": {"present": True, "chunk_index": 0, "extent": 184},
                    "stats_tail": {"present": True, "chunk_index": 44, "extent": 96},
                    "apply_tail": {"present": True, "chunk_index": 44, "extent": 96},
                    "finalize": {"present": False, "chunk_index": 0, "extent": 0},
                },
                "p4_work": {
                    "generated": True,
                    "stats_init": {
                        "generated": True,
                        "primitives": [
                            {"kind": "row_max", "wide": 1, "thin": 0, "stream_starts": 0},
                            {"kind": "row_sum", "wide": 1, "thin": 0, "stream_starts": 0},
                        ],
                    },
                    "stats_update": {
                        "generated": True,
                        "primitives": [
                            {"kind": "row_max", "wide": 1, "thin": 0, "stream_starts": 0},
                            {"kind": "exp", "wide": 1, "thin": 1, "stream_starts": 0},
                        ],
                    },
                    "finalize": {"generated": False, "primitives": []},
                },
            }
        ],
        "cube_schedule": [None],
    }


@pytest.fixture
def cube_problem() -> dict:
    return {
        "widths": [64, 64, 64, 64, 64],
        "heights": [64, 64, 64, 64, 64],
        "dtypes": ["BF16"] * 5,
        "inputs": [[0, 1], [2, 3]],
        "outputs": [[2], [4]],
        "op_types": ["MatMul", "MatMul"],
        "required_outputs": [4],
        "num_cube_cores": 24,
        "num_vector_cores": 48,
        "l1_capacity": 524288,
        "cube_capacity": 131072,
        "vec_capacity": 196608,
        "cube_freq_hz": 1.85e9,
        "bw_gm_l1": 135,
        "bw_l0c_gm": 70,
        "bw_gm_ub": 100,
        "bw_ub_gm": 188,
    }


def _cube_matmul(instance: int, lhs: int, rhs: int, output: int, lhs_producer: int, sink: bool) -> dict:
    return {
        "instance": instance,
        "op": instance,
        "lhs_producer": lhs_producer,
        "rhs_producer": -1,
        "is_sink": sink,
        "lhs_ephemeral": lhs_producer >= 0,
        "rhs_ephemeral": False,
        "output_ephemeral": not sink,
        "contraction": 64,
        "effective_contraction": 64,
        "accumulator_dtype": "fp32",
        "storage_dtype": "bf16",
        "lhs": {
            "tensor": lhs,
            "height_binding": "spatial_m",
            "width_binding": "sequential_k",
            "height": 64,
            "width": 64,
        },
        "rhs": {
            "tensor": rhs,
            "height_binding": "sequential_k",
            "width_binding": "spatial_n",
            "height": 64,
            "width": 64,
        },
        "output": {
            "tensor": output,
            "height_binding": "spatial_m",
            "width_binding": "spatial_n",
            "height": 64,
            "width": 64,
        },
        "k_loop": {"l1_window_k": 64, "chunk": 16, "full_chunks": 4, "tail": 0, "pipeline_stages": 2},
        "output_tile": [32, 32],
        "output_grid": [2, 2],
        "output_variants": [
            {
                "shape": [32, 32],
                "count": 4,
                "l0_init": _l0_plan(),
                "l0_rolled": _l0_plan(),
                "l0_tail": None,
            }
        ],
        "final_drain": {
            "required": True,
            "target_l1": not sink,
            "atomic": False,
            "valid_rows": 32,
            "valid_cols": 32,
            "tile_count": 4,
            "bytes": 8192,
            "cycles": 64,
        },
    }


@pytest.fixture
def cube_solution() -> dict:
    return {
        "subgraphs": [[0, 1]],
        "granularities": [[64, 64, 64]],
        "parts": [[1, 1]],
        "splits": [1],
        "cores": [1],
        "op_order": [[0, 1]],
        "seq_k": [[64, 64]],
        "subgraph_latencies": [2468.0],
        "tensors_to_retain": [[]],
        "vector_stream": [None],
        "cube_schedule": [
            {
                "emit_compatible": True,
                "spatial_policy": "uniform",
                "spatial_tiles": 1,
                "split_k": 1,
                "work_units": 1,
                "peak_l1_bytes": 65536,
                "split_merge_policy": "none",
                "first_partial_then_atomic": {"present": False},
                "model_overlap_granted": True,
                "overlap_implementable": True,
                "matmuls": [
                    _cube_matmul(0, 0, 1, 2, -1, False),
                    _cube_matmul(1, 2, 3, 4, 0, True),
                ],
            }
        ],
    }


def test_partition_diagram_uses_color_without_cluster_boxes(vector_problem, vector_solution):
    dot = visualize.build_solution_dot(vector_problem, vector_solution)

    assert "subgraph cluster" not in dot
    assert "KernelLegend" in dot
    assert "splines=ortho" in dot
    assert "outputorder=edgesfirst" in dot
    assert "T0:s -> Op0:n" in dot
    assert "Op0:s -> T1:n" in dot


def test_vector_algorithm_shows_phase_local_pipeline_and_liveness(vector_problem, vector_solution):
    dot = visualize.build_algorithm_dot(vector_problem, vector_solution, 0)

    assert "VectorStreamPlan: softmax_flash" in dot
    assert "statistics init · serial" in dot
    assert "load chunk k+1 overlaps statistics(k)" in dot
    assert "Op 0 · row_sum is supplied by finalized online statistics" in dot
    assert "release T0 after its last topological use" in dot
    assert "UB → GM" in dot


def test_cube_algorithm_shows_tile_flow_and_recursive_lifetime(cube_problem, cube_solution):
    dot = visualize.build_algorithm_dot(cube_problem, cube_solution, 0)

    assert "CubeSchedulePlan: uniform" in dot
    assert "MATMUL REQUEST 0 · Op 0" in dot
    assert "MATMUL REQUEST 1 · Op 1" in dot
    assert "OUTPUT-TILE LOOP" in dot
    assert "one iteration shown: C [32×32]" in dot
    assert "Panels available to this C tile" in dot
    assert "K0 · LHS + RHS K panel: GM → L1" in dot
    assert "K0: L1 → L0 → Matrix" in dot
    assert "K1: LHS + RHS K panel: GM → L1" in dot
    assert "repeat ×2" in dot
    assert "no next prefetch" in dot
    assert "C tile in L0C" in dot
    assert "Σ K0…K2" in dot
    assert "FIXPIPE" in dot
    assert "C tile → L1 Mat" in dot
    assert "C tile → GM" in dot
    assert "LHS: L1 result from request 0 · release after load" in dot
    assert "R0_Fill:s -> R0_Fill_C:n" not in dot
    assert "R0_First:s -> R0_First_C:n" in dot
    assert "· COMPUTE</B>" not in dot
    assert dot.index("MATMUL REQUEST 0 · Op 0") < dot.index("MATMUL REQUEST 1 · Op 1")


def test_cube_split_merge_is_shown_as_ordered_aic_phases(cube_problem, cube_solution):
    cube_solution["cube_schedule"][0]["split_merge_policy"] = "first_partial_then_atomic"
    cube_solution["cube_schedule"][0]["first_partial_then_atomic"] = {
        "present": True,
        "first_work_units": 4,
        "atomic_work_units": 16,
        "synchronization_cycles": 0,
    }

    dot = visualize.build_algorithm_dot(cube_problem, cube_solution, 0)

    assert "ordered split-K merge" in dot
    assert "phase 1 · 4 AIC tasks · share 0 → normal store" in dot
    assert "dependency boundary · 0 modeled cycles" in dot
    assert "phase 2 · 16 AIC tasks · remaining shares → atomic add" in dot
    assert "AIV" not in dot


def test_serial_cube_k_loop_does_not_claim_pipeline_overlap(cube_problem, cube_solution):
    for matmul in cube_solution["cube_schedule"][0]["matmuls"]:
        matmul["k_loop"] = {
            "l1_window_k": 64,
            "chunk": 64,
            "full_chunks": 1,
            "tail": 0,
            "pipeline_stages": 1,
        }

    dot = visualize.build_algorithm_dot(cube_problem, cube_solution, 0)

    assert "K0 · LHS + RHS K panel: GM → L1" in dot
    assert "after feed completes" in dot
    assert "R0_Load0:s -> R0_Load0_C:n" not in dot
    assert "R0_Work0:s -> R0_Work0_C:n" in dot
    assert "K1: LHS + RHS K panel: GM → L1" not in dot


def test_cube_retained_panel_is_shown_as_one_preload(cube_problem, cube_solution):
    cube_solution["cube_schedule"][0]["matmuls"][0]["retained_panels"] = {
        "lhs": True,
        "rhs": False,
        "lhs_bytes": 16384,
        "rhs_bytes": 0,
    }

    dot = visualize.build_algorithm_dot(cube_problem, cube_solution, 0)

    assert "LHS: GM T0 [64×64] → retained L1 once (16384 B)" in dot
    assert "K0 · RHS K panel: GM → L1" in dot
    assert "K1: RHS K panel: GM → L1" in dot
    assert "K1: LHS + RHS K panel: GM → L1" not in dot


def test_algorithm_requires_a_current_plan_descriptor(vector_problem, vector_solution):
    vector_solution["vector_stream"] = [None]

    with pytest.raises(ValueError, match="regenerate the solution"):
        visualize.build_algorithm_dot(vector_problem, vector_solution, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
