# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Replay the scalar control and explicit synchronization of generated prefill kernels."""

import json
import re
import subprocess
from collections import Counter

PIPES = ["MTE2", "MTE1", "M", "FIX", "V", "MTE3", "S"]
PIPE = {p: i for i, p in enumerate(PIPES)}
VOPS = {"TADD", "TADDS", "TCVT", "TEXP", "TMUL", "TNEG", "TRECIP"}


def trace(path, args, width, temp):
    _, body = re.split(r"\bAICORE void \w+\(", path.read_text())
    params, body = body.split(") {", 1)
    scalars = re.findall(r"\b((?:u?int)(?:32|64)_t v\d+)", params)
    assert len(scalars) == len(args) + 2
    lines = ["#include <cstdint>", "#include <cstdio>", "void run(" + ",".join(scalars) + "){"]
    static = Counter()
    for line in map(str.strip, body.splitlines()):
        s = line
        if (
            not s
            or s == ";"
            or s.startswith(
                (
                    "//",
                    "using ",
                    "Tile<",
                    "GlobalTensor<",
                    "pto::Shape<",
                    "pto::Stride<",
                    "__gm__ ",
                    "TASSIGN(",
                    "TRESHAPE(",
                    "RoundMode ",
                    "SaturationMode ",
                    "set_mask_norm(",
                    "set_vector_mask(",
                )
            )
        ):
            continue
        if s.startswith("#"):
            assert re.fullmatch(r"#if defined\(__DAV_(?:CUBE|VEC)__\)|#endif // __DAV_(?:VEC|CUBE)__", s), s
            continue
        flag = re.fullmatch(r"(set|wait)_flag\(PIPE_(\w+), PIPE_(\w+), EVENT_ID(\d+)\);", s)
        barrier = re.fullmatch(r"pipe_barrier\(PIPE_(\w+)\);", s)
        op = re.fullmatch(r"(T[A-Z_]+)(?:<[^;]*>)?\([^;]*\);", s)
        if flag:
            event = " ".join(flag.groups())
        elif barrier:
            event = "bar " + barrier[1]
        elif s == "ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);":
            event = "bar ALL"
        elif op:
            name = op[1]
            assert name in VOPS | {"TSTORE", "TLOAD", "TEXTRACT", "TMATMUL", "TMATMUL_ACC"}, s
            event = "op " + name
        elif s.startswith("if ("):
            assert s.endswith(") {"), s
            static["if"] += 1
            lines.extend(['puts("if");', s])
            continue
        elif re.match(
            r"(?:const )?(?:u?int(?:64|32|16)_t|unsigned|bool|float) v\d+\b|v\d+ =|for \(", s
        ) or s in ("} else {", "else {", "}", "};", "return;"):
            lines.append(s)
            continue
        else:
            raise ValueError((path, s))
        static[" ".join(event.split()[:2]) if event.startswith("bar") else event.split()[0]] += 1
        lines.append("puts(" + json.dumps(event) + ");")
    values = ",".join(map(str, [*args, "worker", width]))
    lines.append(
        "int main(){for(int worker=0;worker<"
        + str(width)
        + ';++worker){puts("WORKER");run('
        + values
        + ");}}"
    )
    cpp = temp / "trace.cpp"
    cpp.write_text("\n".join(lines))
    binary = temp / "trace"
    subprocess.run(["g++", "-w", "-O0", "-std=c++17", str(cpp), "-o", str(binary)], check=True)
    output = subprocess.check_output([str(binary)], text=True)
    workers = [chunk.splitlines() for chunk in output.split("WORKER\n")[1:]]
    assert len(workers) == width
    return dict(static), workers


def profile(events, cube):
    # Entry j of each vector is the last source completion on pipe j ordered before the current issue.
    gates = [[0] * len(PIPE) for _ in PIPE]
    latest = [0] * len(PIPE)
    rows, payloads, live = [], [], {}
    counts = Counter()

    def completed(p):
        v = gates[p].copy()
        v[p] = latest[p]
        return v

    def join(vs):
        return list(map(max, zip(*vs)))

    for e in events:
        kind, *rest = e.split()
        counts[" ".join([kind, *rest]) if kind == "bar" else kind] += 1
        if kind == "if":
            continue
        if kind == "op":
            (op,) = rest
            p = PIPE[
                {
                    "TSTORE": "FIX" if cube else "MTE3",
                    "TLOAD": "MTE2",
                    "TEXTRACT": "MTE1",
                    "TMATMUL": "M",
                    "TMATMUL_ACC": "M",
                }.get(op, "V")
            ]
            rows.append(tuple(gates[p]))
            payloads.append(op)
            latest[p] += 1
        elif kind == "bar":
            affected = list(PIPE.values()) if rest == ["ALL"] else [PIPE[rest[0]]]
            v = join([completed(p) for p in affected])
            for p in affected:
                gates[p] = v.copy()
        else:
            src, dst, slot = rest
            key = (src, dst, slot)
            p, q = PIPE[src], PIPE[dst]
            if kind == "set":
                assert key not in live, key
                live[key] = completed(p)
            else:
                assert kind == "wait" and key in live, key
                gates[q] = join([gates[q], live.pop(key)])
    assert not live, live
    return payloads, rows, dict(counts)
