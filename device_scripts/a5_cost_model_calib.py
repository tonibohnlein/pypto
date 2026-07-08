# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""a5 (Ascend950) L0 cost-model calibration probe.

Two things:
  1. DIFFERENTIAL PICK AUDIT (fully working here) — runs the compiled chooser with the
     a5 L0 CAPACITIES and two constant sets (A2A3_BASELINE = what a5 inherits today, vs
     A5_FITTED = your a5-sim fit), over a shape grid, and prints every tile pick that
     changes. This is the sim-free validation of the calibration's EFFECT on a5 picks.
  2. SIM-SWEEP RECIPE (documented below) — how to obtain the A5_FITTED numbers from an
     a5-sim forced-tile sweep; mirrors the a2a3 playbook.

Run:  PYTHONPATH=<pypto>/python python a5_cost_model_calib.py

Workflow: run the a5-sim sweep (recipe below) -> fill A5_FITTED -> re-run this ->
inspect the pick diff -> paste A5_FITTED into Backend950::GetL0CostModel().
"""

from pypto.pypto_core.passes import l0_tile_chooser as L
from pypto.pypto_core.passes.l0_tile_chooser import Stationarity

# --- a5 (Ascend950) L0 capacities — from pto-isa buffer_limits.hpp (PTO_NPU_ARCH_A5),
#     matches pypto Backend950. These are CORRECT; do not calibrate them.
A5_L0A = A5_L0B = 64 * 1024
A5_L0C = 256 * 1024  # 2x a2a3 — the reason a5 already tiles differently

# --- Constant set a5 INHERITS today (the a2a3 device-calibrated default). ---
A2A3_BASELINE = dict(
    bw_a=129.7,
    bw_b=85.4,
    bw_drain=118.0,
    drain_fixed_cycles=164.0,
    drain_row_cycles=4.45,
    drain_penalty_cycles=2.6,
    drain_c0_bytes=32,
    mad_head=21,
    mad_k_fractal_bytes=32,
    mad_fp32_passes=2,
)

# --- a5-fitted constants. ---
# MEASURED on a5-sim: mad_fp32_passes=8 (fp32 cube ~4x a2a3/fractal), mad_head=25 (fp32+bf16
# k-sweep intercept; a2a3=21). bf16 confirmed cpr=1. bw / drain NOT yet fit (a5-sim too slow
# + LOAD_2Dv2 extractor gap) -> left at baseline, so the diff below isolates the CUBE fix.
A5_FITTED = dict(A2A3_BASELINE, mad_fp32_passes=8, mad_head=25)

STAT = {Stationarity.OutputStationary: "OS", Stationarity.AStationary: "A", Stationarity.BStationary: "B"}


def _cfg(M, K, N, bytes_ab, c):
    x = L.L0TileConfig()
    x.M, x.N, x.K = M, N, K
    x.l0a_bytes, x.l0b_bytes, x.l0c_bytes = A5_L0A, A5_L0B, A5_L0C
    x.bytes_a = x.bytes_b = bytes_ab
    x.bytes_c = 4
    x.min_m = x.min_n = x.min_k = 16
    x.align_m = x.align_n = x.align_k = 16
    x.allow_a_stationary = x.allow_b_stationary = True
    x.allow_double_buffer_c = False  # dbc=0 default-planner picks
    x.c_read = False
    x.bw_a, x.bw_b, x.bw_drain = c["bw_a"], c["bw_b"], c["bw_drain"]
    x.drain_fixed_cycles = c["drain_fixed_cycles"]
    x.drain_row_cycles = c["drain_row_cycles"]
    x.drain_penalty_cycles = c["drain_penalty_cycles"]
    x.drain_c0_bytes = c["drain_c0_bytes"]
    x.mad_head = c["mad_head"]
    x.mad_k_fractal_bytes = c["mad_k_fractal_bytes"]
    x.mad_fp32_passes = c["mad_fp32_passes"]
    x.allow_padding = x.allow_k_boundary = True
    return x


def pick(M, K, N, ba, c):
    r = L.choose_l0_tile(_cfg(M, K, N, ba, c))
    hold = ("A" if r.os_holds_a else "B") if r.stationarity == Stationarity.OutputStationary else ""
    return f"({r.m},{r.n},{r.k}){STAT[r.stationarity]}{hold}"


# a5-relevant grid: bigger L0C fits larger tiles, so include large M/N + misaligned dims.
Ms = [256, 320, 512, 544, 768, 1024]
Ks = [64, 128, 512]
Ns = [128, 160, 256, 320, 512]
dtypes = [("bf16", 2), ("fp32", 4)]

flips = 0
total = 0
print(f"a5 differential pick audit (L0C={A5_L0C // 1024}KB): A2A3_BASELINE -> A5_FITTED\n")
for dn, ba in dtypes:
    for M in Ms:
        for K in Ks:
            for N in Ns:
                total += 1
                old = pick(M, K, N, ba, A2A3_BASELINE)
                new = pick(M, K, N, ba, A5_FITTED)
                if old != new:
                    flips += 1
                    print(f"  {dn} {M}x{K}x{N:>4}: {old:>16} -> {new:>16}")
print(
    f"\n{flips}/{total} picks change under A5_FITTED "
    f"({'no-op — fill A5_FITTED and re-run' if flips == 0 else 'inspect + device-check the changes'})."
)
