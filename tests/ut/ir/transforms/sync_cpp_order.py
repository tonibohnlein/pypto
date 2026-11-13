# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Strict bounded C++ control trace and independent literal-command graph.

Payload footprints come from the original-IR oracle, never compiler demand
records. The supported macro sequence must agree with that IR. This adapter
assumes those known payload macros preserve their lowering contract; it does
not emulate native instructions or prove general C++ payload equivalence.
"""

import json
import re
import subprocess
from pathlib import Path

from pypto import ir

from .sync_matmul_oracle import hardware_reference, interpret
from .sync_order_metrics import compare_order
from .test_finite_sync_analysis import closure


def cpp_trace(source: Path, output: Path, scalar_values: list[int]):
    """Retain scalar control and reject any unsupported executable statement."""
    text = source.read_text()
    header = re.search(r"AICORE void \w+\(([^\n]*)\) \{", text)
    assert header, "unsupported C++ function header"
    params = re.findall(r"\b(int64_t|int32_t|uint64_t|uint32_t) (v\d+)", header[1])
    assert len(params) == len(scalar_values), "unqualified scalar ABI"
    for (dtype, _), supplied in zip(params, scalar_values, strict=True):
        bits = 64 if "64" in dtype else 32
        signed = not dtype.startswith("u")
        lo = -(1 << (bits - 1)) if signed else 0
        hi = (1 << (bits - int(signed))) - 1
        assert lo <= supplied <= hi, "scalar argument outside its original ABI type"
    start = header.end()
    end, depth = start, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    body = []
    for line in text[start : end - 1].splitlines():
        s = line.strip()
        if s.startswith("#"):
            assert s in ("#if defined(__DAV_CUBE__)", "#endif // __DAV_CUBE__"), s
            continue
        if not s or s.startswith(("//", "using ")) or s == ";":
            continue
        flag = re.fullmatch(r"(set_flag|wait_flag)\((PIPE_\w+), (PIPE_\w+), EVENT_ID(\d+)\);", s)
        barrier = re.fullmatch(r"pipe_barrier\((PIPE_\w+)\);", s)
        payload = re.fullmatch(
            r"(TASSIGN|TLOAD|TMOV|TEXTRACT|TMATMUL|TMATMUL_ACC|TSTORE)(?:<[^;]*>)?\([^;]*\);", s
        )
        event = None
        if flag:
            event = " ".join(flag.groups())
        elif barrier:
            event = "barrier " + barrier[1]
        elif s.startswith("ptoas_auto_sync_tail("):
            assert s == "ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);", s
            event = "return_barrier PIPE_ALL"
        elif payload:
            if payload[1] == "TASSIGN":
                continue
            event = "payload " + payload[1]
        if event:
            body.append("std::cout << " + json.dumps(event + "\n") + ";")
        elif (
            re.match(
                r"(?:const )?(?:int64_t|uint64_t|int32_t|uint32_t|uint16_t|int16_t|unsigned|bool) "
                r"v\d+(?: =|;)",
                s,
            )
            or re.match(r"v\d+ =", s)
            or s.startswith(("if (", "for (", "} else", "else {"))
            or s in ("}", "};", "return;")
        ):
            body.append(s)
        elif s.startswith(("Tile<", "GlobalTensor<", "pto::Shape<", "pto::Stride<", "__gm__ ")):
            continue
        else:
            raise AssertionError(("unsupported C++ statement", s))
    source_text = (
        "#include <cstdint>\n#include <iostream>\nvoid run("
        + ",".join(dtype + " " + name for dtype, name in params)
        + "){\n"
    )
    source_text += (
        "\n".join(body) + "\n}\nint main(){run(" + ",".join(str(n) for n in scalar_values) + ");}\n"
    )
    cpp = output.with_suffix(".cpp")
    cpp.write_text(source_text)
    subprocess.run(["g++", "-std=c++17", "-O0", str(cpp), "-o", str(output)], check=True)
    return [line.split() for line in subprocess.check_output([str(output)], text=True).splitlines()]


def command_graph(calls, events, eligible):
    """Use native start/completion chains and actual consuming command events.

    Initial SETs are allowed: PTOAS primes cyclic protocols before any payload.
    ALL completes preceding payloads and command events; it never clears flags.
    Reuse edges are CHECKED against closure, never inserted to make reuse pass.
    """
    vertices: set[tuple[int | str, int]] = {(i, kind) for i in range(len(calls)) for kind in (False, True)}
    edges, live, consumed, reuses = set(), {}, {}, []
    latest, gate, commands = {}, {}, {}
    payload_index = 0
    tail_count = 0
    macro_ops = {
        macro: ir.get_op(op).name
        for macro, op in [
            ("TLOAD", "buffer.load"),
            ("TMOV", "buffer.copy"),
            ("TEXTRACT", "buffer.extract"),
            ("TMATMUL", "buffer.matmul"),
            ("TMATMUL_ACC", "buffer.matmul_acc"),
            ("TSTORE", "buffer.store"),
        ]
    }
    pipe_names = {"PIPE_" + name: pipe for name, pipe in ir.PipeType.__members__.items()}
    for ordinal, event in enumerate(events):
        kind = event[0]
        if kind == "payload":
            assert payload_index < len(calls), "extra lowered payload"
            call, _, pipe, _, _ = calls[payload_index]
            assert call.op.name == macro_ops[event[1]], "changed payload macro sequence"
            start, completion = (payload_index, False), (payload_index, True)
            edges.add((start, completion))
            if pipe in latest:
                edges.update([((latest[pipe], False), start), ((latest[pipe], True), completion)])
            if pipe in gate:
                edges.add((gate[pipe], start))
            latest[pipe] = payload_index
            payload_index += 1
            continue
        node = ("command", ordinal)
        vertices.add(node)
        if kind in ("barrier", "return_barrier"):
            tail_count += kind == "return_barrier"
            affected = set(pipe_names.values()) if event[1] == "PIPE_ALL" else {pipe_names[event[1]]}
            for pipe in affected:
                if pipe in latest:
                    edges.add(((latest[pipe], True), node))
                edges.update((previous, node) for previous in commands.get(pipe, []))
                gate[pipe] = node
                commands.setdefault(pipe, []).append(node)
            continue
        assert kind in ("set_flag", "wait_flag"), event
        source, target, identifier = pipe_names[event[1]], pipe_names[event[2]], int(event[3])
        assert source != target and identifier in eligible
        key = source, target, identifier
        pipe = source if kind == "set_flag" else target
        if pipe in latest:
            edges.add(((latest[pipe], False), node))
        if pipe in gate:
            edges.add((gate[pipe], node))
        if kind == "set_flag":
            assert key not in live, "overwrite of outstanding notification"
            if pipe in latest:
                edges.add(((latest[pipe], True), node))
            edges.update((previous, node) for previous in commands.get(pipe, []))
            live[key] = node
            if key in consumed:
                reuses.append((consumed[key], node))
        else:
            assert key in live, "wait without publication"
            edges.add((live.pop(key), node))
            consumed[key] = node
            gate[pipe] = node
        commands.setdefault(pipe, []).append(node)
    assert payload_index == len(calls), "missing lowered payload"
    assert not live, "outstanding flags at exit"
    order = closure(vertices, edges)
    assert not any(a == b for a, b in order), "cyclic command graph"
    assert all(pair in order for pair in reuses), "unproved consumption before republication"
    payload_vertices = {(i, kind) for i in range(len(calls)) for kind in (False, True)}
    projected = {(a, b) for a, b in order if a in payload_vertices and b in payload_vertices}
    return projected, {"reuses": len(reuses), "return_barriers": tail_count, "events": len(events)}


def compare_cpp_order(original, before_events, after_events, parameters=None, block=None):
    _, calls = interpret(original, parameters or {}, block_index=block)
    reference = hardware_reference(original, calls)[2]
    before, old = command_graph(calls, before_events, range(8))
    after, new = command_graph(calls, after_events, range(6))
    result = compare_order([call[2] for call in calls], reference, before, after)
    result.update({"ptoas": old, "pypto": new})
    return result
