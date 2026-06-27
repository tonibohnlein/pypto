# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: per-rank DFX artifact isolation.

A rank-pinned chip dispatch (``self.add_one(x[r], y[r], device=r)``) runs with a
:class:`~pypto.runtime.RunConfig` that enables the runtime diagnostics
(``enable_dep_gen`` / ``enable_scope_stats``). Each chip collects its own DFX
buffers and the L3 driver namespaces them per rank, so the artifacts land under
``<output_dir>/dfx_outputs/rank{r}/`` with no cross-rank collisions:

    dfx_outputs/
      rank0/deps.json
      rank0/scope_stats/scope_stats.jsonl
      rank1/deps.json
      rank1/scope_stats/scope_stats.jsonl

This exercises the driver wiring (``_make_call_config`` setting the DFX flags +
base ``output_prefix``) and the codegen wiring (``_submit_chip`` appending the
``/rank{worker}`` suffix per dispatch).

``enable_l2_swimlane`` is also covered: on L3 it co-enables dep_gen and emits
``rank{r}/l2_swimlane_records.json`` + ``rank{r}/deps.json`` per chip; onboard it
is additionally converted to ``rank{r}/merged_swimlane_*.json`` (conversion is
skipped on the simulator, which does not ship the converter's task metadata).
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import RunConfig

ROWS = 16
COLS = 32


def _build_per_rank_program():
    """Build a minimal per-rank ``+ 1`` dispatch program at call time."""

    @pl.program
    class PerRankAddOne:
        @pl.function(type=pl.FunctionType.InCore)
        def add_one(
            self,
            x: pl.Tensor[[ROWS, COLS], pl.FP32],
            y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            for row in pl.parallel(ROWS):
                x_row = pl.slice(x, [1, COLS], [row, 0])
                y_row = pl.add(x_row, 1.0)
                y = pl.assemble(y, y_row, [row, 0])
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def child(
            self,
            x: pl.Tensor[[ROWS, COLS], pl.FP32],
            y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            y = self.add_one(x, y)
            return y

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            x: pl.Tensor[[pl.dynamic("NR"), ROWS, COLS], pl.FP32],
            y: pl.Out[pl.Tensor[[pl.dynamic("NR"), ROWS, COLS], pl.FP32]],
        ):
            # One rank-pinned child dispatch per rank — the ``device=r`` path the
            # codegen routes through ``_submit_chip`` for per-rank DFX dirs.
            for r in pl.range(pld.world_size()):
                self.child(x[r], y[r], device=r)

    return PerRankAddOne


class TestL3DfxPerRank:
    """L3 distributed runtime: DFX artifacts are isolated per rank."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_dfx_artifacts_isolated_per_rank(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"per-rank DFX P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_per_rank_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = torch.randn((n_ranks, ROWS, COLS), dtype=torch.float32)
        outputs = torch.zeros((n_ranks, ROWS, COLS), dtype=torch.float32)

        run_config = RunConfig(
            platform=test_config.platform,
            enable_dep_gen=True,
            enable_scope_stats=True,
        )
        compiled(inputs, outputs, config=run_config)

        # Compute is still correct with DFX enabled.
        assert torch.allclose(outputs, inputs + 1.0), (
            f"per-rank DFX P={n_ranks} mismatch: max diff = {(outputs - inputs).abs().max().item()}"
        )

        # Each rank owns a distinct DFX subdir with non-empty diagnostic artifacts.
        dfx_base = compiled.output_dir / "dfx_outputs"
        assert dfx_base.is_dir(), f"missing DFX base dir: {dfx_base}"
        for r in range(n_ranks):
            rank_dir = dfx_base / f"rank{r}"
            assert rank_dir.is_dir(), f"missing per-rank DFX dir: {rank_dir}"

            deps = rank_dir / "deps.json"
            assert deps.is_file() and deps.stat().st_size > 0, f"empty/missing deps.json for rank {r}: {deps}"

            scope_stats = rank_dir / "scope_stats" / "scope_stats.jsonl"
            assert scope_stats.is_file() and scope_stats.stat().st_size > 0, (
                f"empty/missing scope_stats.jsonl for rank {r}: {scope_stats}"
            )

    def test_swimlane_per_rank(self, test_config, device_ids):
        """Swimlane on L3 co-enables dep_gen and produces per-rank records.

        On a real device the records are additionally converted to
        ``rank{r}/merged_swimlane_*.json``; on the simulator conversion is
        skipped (no task metadata), so only the raw records are asserted.
        """
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"swimlane P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_per_rank_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = torch.randn((n_ranks, ROWS, COLS), dtype=torch.float32)
        outputs = torch.zeros((n_ranks, ROWS, COLS), dtype=torch.float32)

        # User asks for swimlane only; dep_gen is co-enabled by the driver.
        run_config = RunConfig(platform=test_config.platform, enable_l2_swimlane=True)
        compiled(inputs, outputs, config=run_config)

        assert torch.allclose(outputs, inputs + 1.0)

        is_sim = test_config.platform.endswith("sim")
        dfx_base = compiled.output_dir / "dfx_outputs"
        for r in range(n_ranks):
            rank_dir = dfx_base / f"rank{r}"
            records = rank_dir / "l2_swimlane_records.json"
            assert records.is_file() and records.stat().st_size > 0, (
                f"empty/missing l2_swimlane_records.json for rank {r}: {records}"
            )
            # dep_gen is co-enabled so the converter has a task graph.
            deps = rank_dir / "deps.json"
            assert deps.is_file() and deps.stat().st_size > 0, f"empty/missing deps.json for rank {r}: {deps}"

            if not is_sim:
                merged = list(rank_dir.glob("merged_swimlane_*.json"))
                assert merged and merged[0].stat().st_size > 0, (
                    f"missing merged_swimlane_*.json for rank {r} in {rank_dir}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
