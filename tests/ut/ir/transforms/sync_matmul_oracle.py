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

from itertools import combinations, product
from typing import Any

from pypto import ir

from .test_finite_sync_analysis import PIPES, closure, reference


def sequence_stmts(stmt: ir.Stmt) -> list[ir.Stmt]:
    """Require a sequence without flattening or changing occurrence structure."""
    assert isinstance(stmt, ir.SeqStmts)
    return stmt.stmts


def cast_integer(value, dtype):
    """Machine-value cast, independently of compiler range evidence."""
    if dtype == ir.DataType.INDEX:
        bits, signed = 64, True
    else:
        assert dtype.is_signed_int() or dtype.is_unsigned_int()
        bits, signed = dtype.get_bit(), dtype.is_signed_int()
    result = value % (1 << bits)
    return result - (1 << bits) if signed and result >= (1 << (bits - 1)) else result


def integer(expr, env):
    if isinstance(expr, (ir.ConstInt, ir.ConstBool)):
        return expr.value
    if isinstance(expr, ir.Var):
        return env[expr] if expr in env else env[expr.name_hint]
    if isinstance(expr, ir.Cast):
        assert isinstance(expr.type, ir.ScalarType)
        return cast_integer(integer(expr.operand, env), expr.type.dtype)
    if isinstance(expr, ir.Add):
        return integer(expr.left, env) + integer(expr.right, env)
    if isinstance(expr, ir.Sub):
        return integer(expr.left, env) - integer(expr.right, env)
    if isinstance(expr, ir.Mul):
        return integer(expr.left, env) * integer(expr.right, env)
    if isinstance(expr, ir.FloorMod):
        return integer(expr.left, env) % integer(expr.right, env)
    if isinstance(expr, ir.FloorDiv):
        return integer(expr.left, env) // integer(expr.right, env)
    if isinstance(expr, ir.And):
        return bool(integer(expr.left, env)) and bool(integer(expr.right, env))
    if isinstance(expr, ir.Or):
        return bool(integer(expr.left, env)) or bool(integer(expr.right, env))
    if isinstance(expr, ir.Call) and expr.op.name == ir.get_op("buffer.get_block_idx").name:
        return env["__block_index__"]
    if isinstance(expr, ir.Eq):
        return integer(expr.left, env) == integer(expr.right, env)
    for kind, compare in (
        (ir.Lt, lambda a, b: a < b),
        (ir.Le, lambda a, b: a <= b),
        (ir.Gt, lambda a, b: a > b),
        (ir.Ge, lambda a, b: a >= b),
        (ir.Ne, lambda a, b: a != b),
    ):
        if isinstance(expr, kind):
            return compare(integer(expr.left, env), integer(expr.right, env))
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


def define_slot_storage(stmt, env, storage):
    """Interpret original allocation and selection before any compiler normalization."""
    name, t = stmt.value.op.name, stmt.var.type
    if name == ir.get_op("buffer.alloc_multi").name:
        assert isinstance(t, ir.MultiBufferType)
        element = t.element_type
        storage[stmt.var.name_hint] = (
            element.memory_space,
            integer(stmt.value.args[1], env),
            *element.shape,
            element,
            element.dtype.get_byte(),
        )
        return True
    if name == ir.get_op("buffer.slot").name:
        parent, selector = stmt.value.args
        assert isinstance(parent.type, ir.MultiBufferType)
        slot = integer(selector, env)
        assert 0 <= slot < parent.type.slot_count
        space, base, height, width, layout, scalar_bytes = storage[parent.name_hint]
        # Admitted whole-slot fixtures have no physical padding. Derive
        # the pitch from original IR, never from the compiler's slot map.
        storage[stmt.var.name_hint] = (
            space,
            base + slot * height * width * scalar_bytes,
            height,
            width,
            layout,
            scalar_bytes,
        )
        return True
    return False


def operand_signature(expr, env, storage):
    if isinstance(expr, ir.MakeTuple):
        return tuple(operand_signature(x, env, storage) for x in expr.elements)
    if isinstance(expr.type, ir.ScalarType):
        if isinstance(expr, ir.ConstFloat):
            return (expr.type, float(expr.value).hex())
        return (expr.type, integer(expr, env))
    assert isinstance(expr, ir.Var)
    # Keep each operand separately: unioned effects alone would miss a
    # swapped matrix operand or a changed scalar computation.
    descriptor = storage[expr.name_hint]
    return tuple(descriptor)


def _vector_effects(name, args, access):
    """Elementwise operations read inputs and write their typed destination."""
    # TCVT can change byte widths; partial native footprints are not qualified.
    assert name != ir.get_op("buffer.cast").name or all(a.type.valid_shape == a.type.shape for a in args)
    reads = set().union(*(access(a) for a in args[:-1] if isinstance(a.type, ir.BufferType)))
    return reads, access(args[-1]), ir.PipeType.V


def interpret(
    function,
    parameters=None,
    *,
    block_index=None,
    shape_parameters=None,
    operand_signatures=None,
    enumerate_effects=True,
):
    parameters = parameters or {}
    environment = {function.params[i].name_hint: value for i, value in parameters.items()}
    # Execution inputs, independently supplied by the test's caller contract.
    # They are never read from the compiler's retained proof or phase tree.
    if shape_parameters:
        assert not environment.keys() & shape_parameters.keys()
        environment.update(shape_parameters)
    if block_index is not None:
        assert block_index >= 0
        environment["__block_index__"] = block_index
    storage = {}
    for param in function.params:
        t = param.type
        if isinstance(t, ir.ScalarType):
            assert param.name_hint in environment, "missing invocation coordinate"
            continue
        shape = [integer(x, environment) for x in t.shape]
        storage[param.name_hint] = (param.name_hint, 0, *shape, None, t.dtype.get_byte())
    calls = []

    def access(arg, rect=None):
        s = storage[arg.name_hint]
        return address_set(s, *(rect or (0, 0, s[2], s[3]))) if enumerate_effects else set()

    def define_storage(stmt, env):
        assert isinstance(stmt.value, ir.Call)
        name = stmt.value.op.name
        t = stmt.var.type
        if define_slot_storage(stmt, env, storage):
            return
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

    def payload_effects(call, env):
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
            pipe = ir.PipeType.MTE3 if args[0].type.memory_space == ir.MemorySpace.Vec else ir.PipeType.FIX
        elif name in {ir.get_op("buffer.copy").name, ir.get_op("buffer.extract").name}:
            dest = args[-1]
            assert isinstance(dest.type, ir.BufferType)
            rect = (
                [integer(args[1], env), integer(args[2], env), *dest.type.shape]
                if name == ir.get_op("buffer.extract").name
                else None
            )
            reads, writes, pipe = access(args[0], rect), access(dest), ir.PipeType.MTE1
        elif name in {ir.get_op("buffer.gather").name, ir.get_op("buffer.scatter").name}:
            # Independently construct the declared whole-buffer model; actual
            # indexed reads are checked separately with concrete index values.
            reads = access(args[0]) | access(args[1])
            writes = access(args[-1])
            if name == ir.get_op("buffer.gather").name:
                writes |= access(args[2])
            pipe = ir.PipeType.V
        elif name in {ir.get_op("buffer.row_max").name, ir.get_op("buffer.row_sum").name}:
            # Independently interpret the declared workspace effect. This
            # is an enclosure, not a reconstruction of native scratch use.
            reads = access(args[0])
            writes, pipe = access(args[1]) | access(args[2]), ir.PipeType.V
        elif name in {
            ir.get_op("buffer.add").name,
            ir.get_op("buffer.mul").name,
            ir.get_op("buffer.sub").name,
            ir.get_op("buffer.abs").name,
            ir.get_op("buffer.neg").name,
            ir.get_op("buffer.cast").name,
            ir.get_op("buffer.adds").name,
            ir.get_op("buffer.recip").name,
            ir.get_op("buffer.exp").name,
            ir.get_op("buffer.row_expand_sub").name,
            ir.get_op("buffer.row_expand_div").name,
        }:
            reads, writes, pipe = _vector_effects(name, args, access)
        else:
            assert name in {ir.get_op("buffer.matmul").name, ir.get_op("buffer.matmul_acc").name}
            reads = access(args[0]) | access(args[1])
            writes = access(args[2])
            if name == ir.get_op("buffer.matmul_acc").name:
                reads |= writes
            pipe = ir.PipeType.M
        return pipe, reads, writes

    def execute(stmt, env):
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                execute(s, env)
        elif isinstance(stmt, ir.AssignStmt):
            if isinstance(stmt.var.type, ir.ScalarType):
                env[stmt.var] = integer(stmt.value, env)
            else:
                define_storage(stmt, env)
        elif isinstance(stmt, ir.ForStmt):
            for value in range(integer(stmt.start, env), integer(stmt.stop, env), integer(stmt.step, env)):
                execute(stmt.body, {**env, stmt.loop_var: value})
        elif isinstance(stmt, ir.IfStmt):
            execute(stmt.then_body if integer(stmt.condition, env) else stmt.else_body, dict(env))
        elif stmt is None or isinstance(stmt, (ir.YieldStmt, ir.ReturnStmt)):
            return
        elif isinstance(stmt, ir.EvalStmt):
            call = stmt.expr
            assert isinstance(call, ir.Call)
            pipe, reads, writes = payload_effects(call, env)
            calls.append((call, env.copy(), pipe, reads, writes))
            if operand_signatures is not None:
                operand_signatures.append(
                    (call.op.name, call.kwargs, tuple(operand_signature(a, env, storage) for a in call.args))
                )
        else:
            raise AssertionError(f"Unsupported oracle statement {type(stmt)}")

    execute(function.body, environment)
    return storage, calls


def protected_accumulator_pairs(function, calls, target="a2a3"):
    """Independent bounded hardware-contract check over actual byte accesses.

    Matrix hazard protection supplies no event edge. All intermediate accesses
    are inspected here; production uses sparse occurrence counters instead.
    """
    if target != "a2a3":
        return set()
    excluded = set()

    class BranchScope(ir.IRVisitor):
        def visit_for_stmt(self, op):
            body = op.body.stmts if isinstance(op.body, ir.SeqStmts) else [op.body]
            seen_matrix = False
            for stmt in body:
                if isinstance(stmt, ir.IfStmt):

                    def calls_in(arm):
                        seq = arm.stmts if isinstance(arm, ir.SeqStmts) else [arm]
                        return [s.expr for s in seq if isinstance(s, ir.EvalStmt)]

                    arms = calls_in(stmt.then_body) + calls_in(stmt.else_body)
                    if seen_matrix:
                        excluded.update(arms)
                    seen_matrix = True
                elif isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
                    seen_matrix |= stmt.expr.op.name in {
                        ir.get_op("buffer.matmul").name,
                        ir.get_op("buffer.matmul_acc").name,
                    }

    BranchScope().visit_function(function)

    def qualifies(entry):
        call, _, pipe, _, _ = entry
        if pipe != ir.PipeType.M or call in excluded:
            return False
        lhs, rhs, dst = [arg.type for arg in call.args]
        return (
            all(isinstance(t, ir.BufferType) for t in [lhs, rhs, dst])
            and all(0 < d <= 4095 for d in [lhs.shape[0], lhs.shape[1], rhs.shape[1]])
            and lhs.shape[1] == rhs.shape[0]
            and list(dst.shape) == [lhs.shape[0], rhs.shape[1]]
            and (lhs.shape[0] // 16) * (rhs.shape[1] // 16) >= 10
        )

    result = set()
    for a, b in combinations(range(len(calls)), 2):
        source, consumer = calls[a], calls[b]
        if not qualifies(source) or not qualifies(consumer):
            continue
        if consumer[0].op.name != ir.get_op("buffer.matmul_acc").name or source[4] != consumer[4]:
            continue
        footprint = source[4]
        if any(
            (entry[2] == ir.PipeType.M and (not qualifies(entry) or entry[4] != footprint))
            or (
                entry[2] != ir.PipeType.M
                and (not entry[3].isdisjoint(footprint) or not entry[4].isdisjoint(footprint))
            )
            for entry in calls[a + 1 : b]
        ):
            continue
        result.add((a, b))
    return result


def hardware_reference(function, calls, target="a2a3"):
    effects = [(pipe, True, reads, writes) for _, _, pipe, reads, writes in calls]
    return reference(effects, protected=protected_accumulator_pairs(function, calls, target))


def validate_slot_descriptors(function, description):
    """Check even untouched slots against original uniform allocation declarations."""
    groups = {}

    class Groups(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.var.type, ir.MultiBufferType):
                groups[op.var.name_hint] = op.var.type
            super().visit_assign_stmt(op)

    Groups().visit_function(function)
    for domain in description["storage"]:
        if domain["name"] in groups:
            group = groups[domain["name"]]
            element = group.element_type
            pitch = element.shape[0] * element.shape[1] * element.dtype.get_byte()
            assert domain["slot_count"] == group.slot_count
            assert domain["slot_bytes"] == pitch
            assert domain["bytes"] == pitch * group.slot_count


def translate_access_description(description, parameters, block_index):
    # Translate the compiler's CLAIMED relative footprints for comparison. The
    # independent interpreter below evaluates original accesses from the IR.
    for region in description["regions"]:
        for site in region["sites"]:
            accesses = []
            for access in site["accesses"]:
                aid, write, row, col, dr, dc, rows, cols, after_first = access
                domain = description["storage"][aid]
                row += sum(parameters[p] * c for p, c in domain.get("row_parameters", {}).items())
                col += sum(parameters[p] * c for p, c in domain.get("column_parameters", {}).items())
                if domain.get("row_block") or domain.get("column_block"):
                    assert block_index is not None
                    row += domain["row_block"] * (block_index - domain.get("block_origin", 0))
                    col += domain["column_block"] * (block_index - domain.get("block_origin", 0))
                accesses.append((aid, write, row, col, dr, dc, rows, cols, after_first))
            site["accesses"] = accesses


def validate(function, analysis, query_all=True, parameters=None, *, block_index=None):
    parameters = parameters or {}
    description = analysis.describe()
    validate_slot_descriptors(function, description)
    translate_access_description(description, parameters, block_index)
    storage, calls = interpret(function, parameters, block_index=block_index)
    occurrences, effects = check_imported_accesses(description, storage, calls)
    vertices = [(o, c) for o in occurrences for c in (False, True)]
    base = {((o, False), (o, True)) for o in occurrences}
    for pipe in PIPES:
        same = [o for o, effect in zip(occurrences, effects) if effect[0] == pipe]
        for a, b in zip(same, same[1:]):
            base.update([((a, False), (b, False)), ((a, True), (b, True))])
    protected = protected_accumulator_pairs(function, calls, description["hardware_target"])
    required = set(base)
    for a, b in combinations(range(len(occurrences)), 2):
        _, ar, aw = effects[a]
        _, br, bw = effects[b]
        if (not aw.isdisjoint(br) or not aw.isdisjoint(bw) or not ar.isdisjoint(bw)) and (
            a,
            b,
        ) not in protected:
            required.add(((occurrences[a], True), (occurrences[b], False)))
    order = closure(vertices, required)
    assert not any(a == b for a, b in order)
    covers = {
        (a[0], b[0])
        for a, b in order - base
        if not any((a, v) in order and (v, b) in order for v in vertices)
    }
    if description.get("hierarchical"):
        selected = set(analysis.expand_retained_demands(len(calls)))
    else:
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
    checks = validate_boundaries(
        description, storage, effects, occurrences, analysis, parameters, block_index
    )
    return {
        "payloads": len(occurrences),
        "covers": len(covers),
        "query_checks": len(vertices) ** 2 if query_all else 0,
        "boundary_checks": checks,
        "access_checks": len(calls),
    }


def domain_addresses(domain, storage, row, col, rows, cols, slot=0):
    if domain.get("byte_coordinates", False):
        assert row == 0 and rows == 1
        return {(domain["space"], byte) for byte in range(col, col + cols)}
    descriptor = list(storage[domain["name"]])
    descriptor[1] += slot * domain.get("slot_bytes", 0)
    return address_set(descriptor, row, col, rows, cols)


def validate_boundaries(
    description, storage, effects, occurrences, analysis, parameters=None, block_index=None
):
    # Boundary points include interior/exterior and corners of each extracted
    # rectangle. Point membership is looked up in the independent physical map.
    checks = 0
    for sid, domain in enumerate(description["storage"]):
        points = {(0, 0), (domain["rows"] - 1, domain["columns"] - 1)}
        for r in description["regions"]:
            for site in r["sites"]:
                for access_index, (aid, _, row, col, dr, dc, rows, cols, _) in enumerate(site["accesses"]):
                    if aid == sid:
                        for visit in range(r["count"]):
                            i = visit % r["reset_period"] if r.get("reset_period") else visit
                            frame = site.get("access_frames", [None] * len(site["accesses"]))[access_index]
                            if frame is not None and r.get("nested_frames"):
                                remaining = visit
                                for f, count, _, _ in reversed(r["nested_frames"]):
                                    digit = remaining % count
                                    remaining //= count
                                    if f.same_as(frame):
                                        i = digit
                                        break
                            points.update(
                                {
                                    (row + i * dr, col + i * dc),
                                    (row + i * dr + rows - 1, col + i * dc + cols - 1),
                                }
                            )
        for slot, (row, col) in product(range(domain.get("slot_count", 1)), points):
            if domain.get("row_extent_parameter") is not None:
                actual_rows = storage[domain["name"]][2]
                if row >= actual_rows:
                    assert analysis.boundary(sid, row, col, parameters or {}, block_index) is None
                    checks += 1
                    continue
            point = domain_addresses(domain, storage, row, col, 1, 1, slot)
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
            if domain.get("slot_bytes", 0):
                assert analysis.boundary_slot(sid, slot, row, col, parameters or {}, block_index) == expected
            else:
                assert analysis.boundary(sid, row, col, parameters or {}, block_index) == expected
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
            else (integer(loop.loop_var, env) - integer(loop.start, env)) // integer(loop.step, env)
        )
        if region.get("nested_frames"):
            visit = 0
            for frame, count, start, step in region["nested_frames"]:
                digit = (integer(frame.loop_var, env) - integer(frame.start, env)) // step
                assert 0 <= digit < count
                visit = visit * count + digit
        occurrences.append((r, visit, s))
        effects.append((pipe, reads, writes))
        modeled = [set(), set()]
        local_visit = visit % region["reset_period"] if region.get("reset_period") else visit
        site = region["sites"][s]
        initializing = local_visit == 0
        if site.get("initialization_scope") is not None:
            scoped = False
            initializing = True
            for frame, _, start, _ in region["nested_frames"]:
                scoped |= frame.same_as(site["initialization_scope"])
                if scoped:
                    initializing &= integer(frame.loop_var, env) == start
        selections = site.get("slot_selections", [(0, 0)] * len(site["accesses"]))
        for access_index, (access, selection) in enumerate(zip(site["accesses"], selections, strict=True)):
            sid, write, row, col, dr, dc, rows, cols, after_first = access
            if after_first and initializing:
                continue
            domain = description["storage"][sid]
            access_visit = local_visit
            frame = site.get("access_frames", [None] * len(site["accesses"]))[access_index]
            if frame is not None:
                access_visit = (integer(frame.loop_var, env) - integer(frame.start, env)) // integer(
                    frame.step, env
                )
            slot = (selection[0] + local_visit * selection[1]) % domain.get("slot_count", 1)
            modeled[write].update(
                domain_addresses(
                    domain, storage, row + access_visit * dr, col + access_visit * dc, rows, cols, slot
                )
            )
        assert modeled == [reads, writes], (
            r,
            visit,
            s,
            "access mismatch",
            [
                (len(m), len(a), sorted(m - a, key=repr)[:3], sorted(a - m, key=repr)[:3])
                for m, a in zip(modeled, [reads, writes], strict=True)
            ],
        )
    return occurrences, effects
