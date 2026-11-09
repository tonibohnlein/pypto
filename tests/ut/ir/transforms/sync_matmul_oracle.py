# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bounded oracle interpreting original IR and physical element addresses.

Uses no compiler cells, storage generators, rank rows or quotient distances.
It shares the declared instruction/layout model, not native qualification.
"""

from itertools import combinations
from typing import Any

from pypto import ir

from .test_finite_sync_analysis import PIPES, closure


def integer(expr, env):
    if isinstance(expr, ir.ConstInt):
        return expr.value
    if isinstance(expr, ir.Var):
        return env[expr.name_hint]
    if isinstance(expr, ir.Add):
        return integer(expr.left, env) + integer(expr.right, env)
    if isinstance(expr, ir.Mul):
        return integer(expr.left, env) * integer(expr.right, env)
    if isinstance(expr, ir.Eq):
        return integer(expr.left, env) == integer(expr.right, env)
    raise AssertionError(f"Oracle does not interpret {type(expr)}")


def address_set(storage, row, col, rows, cols):
    """Independent ND/NZ/ZN element-address map (FP32 corpus)."""
    space, base, height, width, layout, scalar_bytes, *pitch = storage
    points = set()
    for r in range(row, row + rows):
        for c in range(col, col + cols):
            assert 0 <= r < height and 0 <= c < width
            if layout is None or layout.slayout == ir.TileLayout.none_box:
                offset = r * (pitch[0] // scalar_bytes if pitch else width) + c
            else:
                h, w = 16, layout.fractal // scalar_bytes // 16
                outer = (
                    (r // h) * (width // w) + c // w
                    if layout.blayout == ir.TileLayout.row_major
                    else (c // w) * (height // h) + r // h
                )
                inner = (
                    (r % h) * w + c % w if layout.slayout == ir.TileLayout.row_major else (c % w) * h + r % h
                )
                offset = outer * h * w + inner
            # All admitted corpus accesses use identical byte-aligned FP32.
            # One address per element is sufficient for intersection checks.
            if space == ir.MemorySpace.Vec:
                # Vec aliases can reinterpret element widths. Compare every byte,
                # independently of the compiler's interval partition.
                points.update((space, base + offset * scalar_bytes + b) for b in range(scalar_bytes))
            else:
                points.add((space, base + offset * scalar_bytes))
    return points


def interpret(function):
    storage = {}
    for param in function.params:
        t = param.type
        shape = [integer(x, {}) for x in t.shape]
        storage[param.name_hint] = (param.name_hint, 0, *shape, None, t.dtype.get_byte())
    calls = []

    def access(arg, rect=None):
        s = storage[arg.name_hint]
        return address_set(s, *(rect or (0, 0, s[2], s[3])))

    def define_storage(stmt, env):
        assert isinstance(stmt.value, ir.Call)
        name = stmt.value.op.name
        t = stmt.var.type
        if name in {ir.get_op("buffer.subview").name, ir.get_op("buffer.reshape").name}:
            source_arg = stmt.value.args[0]
            assert isinstance(source_arg, ir.Var)
            source = storage[source_arg.name_hint]
            space, base, height, width, layout, scalar_bytes, *inherited = source
            pitch = inherited[0] if inherited else width * scalar_bytes
            if name == ir.get_op("buffer.subview").name:
                offsets = stmt.value.args[1]
                assert isinstance(offsets, ir.MakeTuple)
                row, col = [integer(e, env) for e in offsets.elements]
                base += row * pitch + col * scalar_bytes
            else:
                assert pitch == width * scalar_bytes
                pitch = t.shape[1] * t.dtype.get_byte()
            storage[stmt.var.name_hint] = (space, base, *t.shape, t, t.dtype.get_byte(), pitch)
            return
        assert name == ir.get_op("buffer.alloc").name
        assert isinstance(t, ir.BufferType)
        storage[stmt.var.name_hint] = (
            t.memory_space,
            integer(stmt.value.args[1], env),
            *t.shape,
            t,
            t.dtype.get_byte(),
        )

    def execute(stmt, env):
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                execute(s, env)
        elif isinstance(stmt, ir.AssignStmt):
            define_storage(stmt, env)
        elif isinstance(stmt, ir.ForStmt):
            for value in range(integer(stmt.start, env), integer(stmt.stop, env), integer(stmt.step, env)):
                execute(stmt.body, {**env, stmt.loop_var.name_hint: value})
        elif isinstance(stmt, ir.IfStmt):
            execute(stmt.then_body if integer(stmt.condition, env) else stmt.else_body, env)
        elif isinstance(stmt, (ir.YieldStmt, ir.ReturnStmt)):
            return
        elif isinstance(stmt, ir.EvalStmt):
            call = stmt.expr
            assert isinstance(call, ir.Call)
            name, args = call.op.name, call.args
            if name == ir.get_op("buffer.load").name:
                assert isinstance(args[1], ir.MakeTuple) and isinstance(args[2], ir.MakeTuple)
                rect = [integer(e, env) for e in [*args[1].elements, *args[2].elements]]
                reads, writes, pipe = access(args[0], rect), access(args[3]), ir.PipeType.MTE2
            elif name == ir.get_op("buffer.store").name:
                assert isinstance(args[1], ir.MakeTuple) and isinstance(args[2], ir.MakeTuple)
                rect = [integer(e, env) for e in [*args[1].elements, *args[2].elements]]
                reads, writes = access(args[0]), access(args[3], rect)
                assert isinstance(args[0].type, ir.BufferType)
                pipe = (
                    ir.PipeType.MTE3 if args[0].type.memory_space == ir.MemorySpace.Vec else ir.PipeType.FIX
                )
            elif name in {ir.get_op("buffer.copy").name, ir.get_op("buffer.extract").name}:
                dest = args[-1]
                assert isinstance(dest.type, ir.BufferType)
                rect = (
                    [integer(args[1], env), integer(args[2], env), *dest.type.shape]
                    if name == ir.get_op("buffer.extract").name
                    else None
                )
                reads, writes, pipe = access(args[0], rect), access(dest), ir.PipeType.MTE1
            elif name in {
                ir.get_op("buffer.add").name,
                ir.get_op("buffer.mul").name,
                ir.get_op("buffer.sub").name,
                ir.get_op("buffer.abs").name,
                ir.get_op("buffer.adds").name,
                ir.get_op("buffer.recip").name,
            }:
                reads = set().union(*(access(a) for a in args[:-1] if isinstance(a.type, ir.BufferType)))
                writes, pipe = access(args[-1]), ir.PipeType.V
            else:
                assert name in {ir.get_op("buffer.matmul").name, ir.get_op("buffer.matmul_acc").name}
                reads = access(args[0]) | access(args[1])
                writes = access(args[2])
                if name == ir.get_op("buffer.matmul_acc").name:
                    reads |= writes
                pipe = ir.PipeType.M
            calls.append((call, env.copy(), pipe, reads, writes))
        else:
            raise AssertionError(f"Unsupported oracle statement {type(stmt)}")

    execute(function.body, {})
    return storage, calls


def validate(function, analysis, query_all=True):
    description = analysis.describe()
    storage, calls = interpret(function)
    occurrences, effects = check_imported_accesses(description, storage, calls)
    vertices = [(o, c) for o in occurrences for c in (False, True)]
    base = {((o, False), (o, True)) for o in occurrences}
    for pipe in PIPES:
        same = [o for o, effect in zip(occurrences, effects) if effect[0] == pipe]
        for a, b in zip(same, same[1:]):
            base.update([((a, False), (b, False)), ((a, True), (b, True))])
    required = set(base)
    for a, b in combinations(range(len(occurrences)), 2):
        _, ar, aw = effects[a]
        _, br, bw = effects[b]
        if not aw.isdisjoint(br) or not aw.isdisjoint(bw) or not ar.isdisjoint(bw):
            required.add(((occurrences[a], True), (occurrences[b], False)))
    order = closure(vertices, required)
    assert not any(a == b for a, b in order)
    covers = {
        (a[0], b[0])
        for a, b in order - base
        if not any((a, v) in order and (v, b) in order for v in vertices)
    }
    selected = set(description["crossing_demands"])
    for r, region in enumerate(description["regions"]):
        for source, target, distance in region["demands"]:
            for visit in range(region["count"] - distance):
                selected.add(((r, visit, source), (r, visit + distance, target)))
    assert selected == covers, ("covers", selected - covers, covers - selected)
    assert closure(vertices, base | {((a, True), (b, False)) for a, b in selected}) == order
    if query_all:
        for a in vertices:
            for b in vertices:
                assert analysis.precedes(a[0], a[1], b[0], b[1]) == ((a, b) in order), (a, b)
    checks = validate_boundaries(description, storage, effects, occurrences, analysis)
    return {
        "payloads": len(occurrences),
        "covers": len(covers),
        "query_checks": len(vertices) ** 2 if query_all else 0,
        "boundary_checks": checks,
        "access_checks": len(calls),
    }


def domain_addresses(domain, storage, row, col, rows, cols):
    if domain.get("byte_coordinates", False):
        assert row == 0 and rows == 1
        return {(domain["space"], byte) for byte in range(col, col + cols)}
    return address_set(storage[domain["name"]], row, col, rows, cols)


def validate_boundaries(description, storage, effects, occurrences, analysis):
    # Boundary points include interior/exterior and corners of each extracted
    # rectangle. Point membership is looked up in the independent physical map.
    checks = 0
    for sid, domain in enumerate(description["storage"]):
        points = {(0, 0), (domain["rows"] - 1, domain["columns"] - 1)}
        for r in description["regions"]:
            for site in r["sites"]:
                for aid, _, row, col, dr, dc, rows, cols, _ in site["accesses"]:
                    if aid == sid:
                        for i in range(r["count"]):
                            points.update(
                                {
                                    (row + i * dr, col + i * dc),
                                    (row + i * dr + rows - 1, col + i * dc + cols - 1),
                                }
                            )
        for row, col in points:
            point = domain_addresses(domain, storage, row, col, 1, 1)
            writers = [i for i, (_, _, w) in enumerate(effects) if not w.isdisjoint(point)]
            expected: dict[str, Any] = {
                "first_writer": occurrences[writers[0]] if writers else None,
                "last_writer": occurrences[writers[-1]] if writers else None,
                "first_readers": [None] * 7,
                "last_readers": [None] * 7,
            }
            for pipe in PIPES:
                readers = [
                    i
                    for i, (p, read, write) in enumerate(effects)
                    if p == pipe and not read.isdisjoint(point) and write.isdisjoint(point)
                ]
                first = [i for i in readers if not writers or i < writers[0]]
                last = [i for i in readers if not writers or i > writers[-1]]
                expected["first_readers"][int(pipe)] = occurrences[first[0]] if first else None
                expected["last_readers"][int(pipe)] = occurrences[last[-1]] if last else None
            assert analysis.boundary(sid, row, col) == expected
            checks += 1
        assert analysis.boundary(sid, domain["rows"], 0) is None
    return checks


def check_imported_accesses(description, storage, calls):
    mapping = {}
    for r, region in enumerate(description["regions"]):
        for s, site in enumerate(region["sites"]):
            mapping[site["original"]] = (r, s)
            if site["continuation"] is not None:
                mapping[site["continuation"]] = (r, s)
    occurrences, effects = [], []
    for call, env, pipe, reads, writes in calls:
        r, s = mapping[call]
        region = description["regions"][r]
        loop = region["loop"]
        visit = (
            0
            if loop is None
            else (env[loop.loop_var.name_hint] - integer(loop.start, {})) // integer(loop.step, {})
        )
        occurrences.append((r, visit, s))
        effects.append((pipe, reads, writes))
        modeled = [set(), set()]
        for sid, write, row, col, dr, dc, rows, cols, after_first in region["sites"][s]["accesses"]:
            if after_first and visit == 0:
                continue
            domain = description["storage"][sid]
            modeled[write].update(
                domain_addresses(domain, storage, row + visit * dr, col + visit * dc, rows, cols)
            )
        assert modeled == [reads, writes], (r, visit, s, "access mismatch")
    return occurrences, effects
