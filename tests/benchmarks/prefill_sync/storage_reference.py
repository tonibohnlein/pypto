# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bounded original-PTO storage reference for these full-tile A2/A3 cube leaves."""

import re
from pathlib import Path

from control_trace import PIPE


def names(text: str) -> list[str]:
    return re.findall(r"%(\w+)", text)


def scalar_value(rhs, env):
    op = rhs.split()[0][6:]
    v = [env[n] for n in names(rhs)]
    if op == "constant":
        n = int(rhs.split()[1])
    elif op == "index_cast":
        n = v[0]
        assert -(2**31) <= n < 2**31
    elif op == "addi":
        n = v[0] + v[1]
    elif op == "muli":
        n = v[0] * v[1]
    elif op == "andi":
        n = v[0] & v[1]
    elif op == "remsi":
        n = v[0] % v[1] if v[0] >= 0 else -((-v[0]) % v[1])
    elif op == "select":
        n = v[1] if v[0] else v[2]
    elif op == "cmpi":
        cmp = rhs.split()[1]
        assert cmp in ("slt,", "eq,")
        n = v[0] < v[1] if cmp == "slt," else v[0] == v[1]
    else:
        raise ValueError(rhs)
    return n


def source_trace(path: Path, args: list[int], width: int, worker: int):
    inputs = path.read_text().splitlines()
    header = inputs[1]
    nptr = header.count("!pto.ptr")
    env = {"__pypto_spmd_block_idx": worker, "__pypto_spmd_block_num": width}
    env.update({f"arg{i}": (f"arg{i}" if i < nptr else args[i - nptr]) for i in range(nptr + len(args))})
    rects, allocations, ops = {}, {}, []

    def full(tile):
        return rects[tile]

    def interp(line):
        var, sep, rhs = line.partition(" = ")
        if not sep:
            var, rhs = None, line
        if rhs.startswith("arith."):
            env[var[1:]] = scalar_value(rhs, env)
        elif rhs.startswith("pto.alloc_tile"):
            info = re.search(r"loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+), v_row=(\d+), v_col=(\d+)", rhs)
            space, dtype, m, n, vm, vn = info.groups()
            assert (m, n) == (vm, vn)
            address = env[names(rhs)[0]]
            ident = var[1:]
            size = int(m) * int(n) * {"bf16": 2, "f32": 4}[dtype]
            for bank, base, length in allocations.values():
                assert bank != space or address + size <= base or base + length <= address, (
                    ident,
                    space,
                    address,
                    size,
                    bank,
                    base,
                    length,
                )
            allocations[ident] = (space, address, size)
            rects[ident] = (ident, 0, 0, int(m), int(n))
        elif rhs.startswith("pto.make_tensor_view"):
            base, m, n, s0, s1 = names(rhs)
            assert env[s0] == env[n] and env[s1] == 1
            rects[var[1:]] = (env[base], 0, 0, env[m], env[n])
        elif rhs.startswith("pto.partition_view"):
            view, i, j, m, n = names(rhs)
            base, r, c, h, w = rects[view]
            x, y, rows, cols = [env[k] for k in (i, j, m, n)]
            assert 0 <= x <= h - rows and 0 <= y <= w - cols
            rects[var[1:]] = (base, r + x, c + y, rows, cols)
        else:
            match = re.fullmatch(
                r"pto\.(tload|tstore|textract|tmatmul(?:\.acc)?) ins\(([^:]*):.*\) outs\(([^:]*):.*\)", rhs
            )
            assert match, rhs
            op = match[1]
            ins = names(match[2])
            (out,) = names(match[3])
            writes = [full(out)]
            if op == "textract":
                src, i, j = ins
                base, r, c, h, w = full(src)
                _, _, _, rows, cols = full(out)
                x, y = env[i], env[j]
                assert 0 <= x <= h - rows and 0 <= y <= w - cols
                reads = [(base, r + x, c + y, rows, cols)]
            else:
                reads = [full(i) for i in ins]
            if op.startswith("tmatmul"):
                lhs, rhs_tile = reads[-2:]
                dest = writes[0]
                assert allocations[dest[0]][0] == "acc" and (*lhs[3:], *rhs_tile[3:], *dest[3:]) == (
                    128,
                    32,
                    32,
                    256,
                    128,
                    256,
                )
                if op == "tmatmul.acc":
                    assert reads[0] == dest
            pipe = {"tload": "MTE2", "tstore": "FIX", "textract": "MTE1", "tmatmul": "M", "tmatmul.acc": "M"}[
                op
            ]
            ops.append((op.upper().replace(".", "_"), PIPE[pipe], reads, writes))

    # Replay branch and loop control, interpreting each instruction against the original storage.
    code = []
    depth = 0
    for raw in inputs[2:]:
        s = raw.strip().split(' loc("')[0]
        if s == "return":
            break
        if s == "}":
            depth -= 1
            continue
        if s == "} else {":
            code.append("    " * (depth - 1) + "else:")
            continue
        if s.startswith("scf.for "):
            i, start, end, step = re.findall(r"%(\w+)", s)
            code.append(
                "    " * depth + f"for env[{i!r}] in range(env[{start!r}],env[{end!r}],env[{step!r}]):"
            )
            depth += 1
        elif s.startswith("scf.if "):
            (key,) = re.findall(r"%(\w+)", s)
            code.append("    " * depth + f"if env[{key!r}]:")
            depth += 1
        elif s:
            code.append("    " * depth + f"interp({s!r})")
    assert depth == 0
    exec("\n".join(code), {"env": env, "interp": interp})
    return ops


def intersects(x, y):
    b, r, c, h, w = x
    a, i, j, m, n = y
    return b == a and r < i + m and i < r + h and c < j + n and j < c + w


def reference(ops):
    # Enumerate all RAW/WAR/WAW relations. Native issue and completion order close them on each pipe.
    rows = []
    completed = []
    latest = {}
    epoch = 0
    epochs = []
    for i, (op, pipe, reads, writes) in enumerate(ops):
        if op == "TMATMUL":
            epoch += 1
        epochs.append(epoch)
        start = rows[latest[pipe]].copy() if pipe in latest else [0] * len(PIPE)
        for j, (prior, p, rd, wr) in enumerate(ops[:i]):
            if p == pipe == PIPE["M"] and op == "TMATMUL_ACC" and epochs[j] == epoch:
                continue  # Same qualified accumulator chain, with no intervening FIX access.
            if any(
                intersects(a, b)
                for left, right in ((rd, writes), (wr, reads + writes))
                for a in left
                for b in right
            ):
                start = list(map(max, start, completed[j]))
        finish = start.copy()
        if pipe in latest:
            finish = list(map(max, finish, completed[latest[pipe]]))
        finish[pipe] += 1
        rows.append(start)
        completed.append(finish)
        latest[pipe] = i
        if op == "TSTORE":
            epoch += 1
    return rows
