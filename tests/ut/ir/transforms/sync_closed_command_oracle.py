# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent bounded command execution, including empty completion prefixes.

This oracle has no payload-per-record or payload-per-wait requirement. It reads
actual logical tokens or physical flags and never inserts resource-reuse edges.
Only the fixture load/copy/vector/Cube and local scalar instruction model is supported; physical
bytes and native accumulator exemptions come from the existing independent
hardware oracle, never compiler effects, token generators or injected edges.
"""

from collections import defaultdict
from collections.abc import Sequence

from pypto import ir

from .sync_logical_oracle import Token, value
from .sync_matmul_oracle import address_set, protected_accumulator_pairs
from .test_finite_sync_analysis import closure


def opname(name):
    return ir.get_op(name).name


_BARRIERS = {
    opname("system.bar_mte2"): ir.PipeType.MTE2,
    opname("system.bar_mte3"): ir.PipeType.MTE3,
    opname("system.bar_v"): ir.PipeType.V,
    opname("system.bar_mte1"): ir.PipeType.MTE1,
    opname("system.bar_m"): ir.PipeType.M,
    opname("system.bar_fix"): ir.PipeType.FIX,
}


class ClosedCommands:
    def __init__(
        self,
        function,
        parameters,
        eligible,
        tensor_bases=None,
        *,
        shape_parameters=None,
        block_index=None,
        scalar_memory=None,
    ):
        self.function = function
        self.env = {function.params[i]: x for i, x in parameters.items()}
        self.env.update(shape_parameters or {})
        if block_index is not None:
            self.env["__block_index__"] = block_index
        self.storage = {}
        self.layouts = {}
        self.footprints = {}
        self.calls = []
        self.vertices, self.edges = set(), set()
        self.last_issue, self.last_payload, self.gates = {}, {}, {}
        self.commands = defaultdict(list)
        self.live, self.last_consumption = {}, {}
        self.reuses, self.publications, self.acquisitions = [], [], []
        self.used_tokens = set()
        self.record_origins = {}
        self.logical_records = False
        self.trace, self.payloads, self.descriptors = [], [], []
        self.ids = defaultdict(set)
        self.eligible = set(eligible)
        self.instructions = 0
        self.barriers = 0
        self.while_tests = 0
        self.returns = []
        self.last_all = None
        self.scalar_dependencies = {}
        self.control_gate = None
        self.control_evaluations = 0
        self.result_dependencies = ()
        # Optional concrete byte-addressed values for scalar-control fixtures.
        # These come from fixture inputs, independently of compiler effects.
        self.scalar_memory = None if scalar_memory is None else dict(scalar_memory)
        for i, parameter in enumerate(function.params):
            if isinstance(parameter.type, ir.TensorType):
                shape = tuple(value(x, self.env) for x in parameter.type.shape)
                assert len(shape) in (1, 2)
                if len(shape) == 1:
                    shape = (1, shape[0])
                self.storage[parameter] = (
                    "gm",
                    (tensor_bases or {}).get(i, i * 4096),
                    *shape,
                    parameter.type.dtype.get_byte(),
                )

    def edge(self, source, destination):
        if source is not None:
            self.edges.add((source, destination))

    def issue(self, pipe, node):
        self.vertices.add(node)
        self.edge(self.last_issue.get(pipe), node)
        self.edge(self.gates.get(pipe), node)
        self.edge(self.control_gate, node)
        # The native scalar exemption establishes completion before the next
        # scalar instruction. It is not a completion fence for other pipes.
        if pipe == ir.PipeType.S:
            self.edge(self.last_payload.get(pipe), node)
        self.last_issue[pipe] = node

    def command_issue(self, pipe):
        node = ("command_issue", self.instructions)
        self.instructions += 1
        self.issue(pipe, node)
        return node

    def publish(self, pipe):
        issue = self.command_issue(pipe)
        node = ("publish", len(self.publications))
        self.vertices.add(node)
        self.edge(issue, node)
        self.edge(self.last_payload.get(pipe), node)
        # A publication includes earlier pipe commands, including an acquired
        # empty prefix. It does not block later payload issue on the same pipe.
        for previous in self.commands[pipe]:
            self.edge(previous, node)
        self.commands[pipe].append(node)
        self.publications.append(node)
        self.trace.append(("publish", pipe))
        return node

    def acquire(self, source, destination, publication):
        issue = self.command_issue(destination)
        node = ("acquire", len(self.acquisitions))
        self.vertices.add(node)
        self.edge(issue, node)
        self.edge(publication, node)
        self.gates[destination] = node
        self.commands[destination].append(node)
        self.acquisitions.append(node)
        self.trace.append(("acquire", source, destination))
        return node

    def barrier(self, pipe):
        issue = self.command_issue(pipe)
        node = ("barrier", self.barriers)
        self.barriers += 1
        self.vertices.add(node)
        self.edge(issue, node)
        self.edge(self.last_payload.get(pipe), node)
        for previous in self.commands[pipe]:
            self.edge(previous, node)
        self.commands[pipe].append(node)
        self.gates[pipe] = node

    def complete_all(self):
        # Qualified native ALL completes every preceding pipe prefix and
        # command. It does not acquire or consume an outstanding event flag.
        node = ("all_barrier", self.barriers)
        self.barriers += 1
        self.vertices.add(node)
        self.edge(self.control_gate, node)
        for previous in self.last_payload.values():
            self.edge(previous, node)
        for previous in self.last_issue.values():
            self.edge(previous, node)
        for commands in self.commands.values():
            for previous in commands:
                self.edge(previous, node)
        for pipe in (
            ir.PipeType.S,
            ir.PipeType.MTE2,
            ir.PipeType.MTE3,
            ir.PipeType.V,
            ir.PipeType.MTE1,
            ir.PipeType.M,
            ir.PipeType.FIX,
        ):
            self.gates[pipe] = node
            self.commands[pipe].append(node)
            self.last_issue[pipe] = node
        self.last_all = node

    def physical(self, call):
        expected = "aiv" if self.function.func_type == ir.FunctionType.AIV else "aic"
        assert call.kwargs["core_type"] == expected, "wrong command core"
        if call.op.name == opname("system.bar_all"):
            self.complete_all()
            return
        if call.op.name in _BARRIERS:
            self.barrier(_BARRIERS[call.op.name])
            return
        assert not call.args, "oracle expects literal physical IDs"
        source = ir.PipeType(call.kwargs["src_pipe"])
        destination = ir.PipeType(call.kwargs["dst_pipe"])
        key = source, destination, call.kwargs["event_id"]
        assert source != destination and key[2] in self.eligible, "ineligible directional ID"
        self.ids[source, destination].add(key[2])
        if call.op.name == opname("system.set_flag"):
            assert key not in self.live, "publication overwrites an unconsumed flag"
            node = self.publish(source)
            self.live[key] = node
            if key in self.last_consumption:
                self.reuses.append((self.last_consumption[key], node))
        else:
            assert call.op.name == opname("system.wait_flag")
            assert key in self.live, "wait without a matching publication"
            publication = self.live.pop(key)
            self.last_consumption[key] = self.acquire(source, destination, publication)

    def signature(self, operand):
        layout = self.layouts.get(operand)
        geometry = None if layout is None else (layout.blayout, layout.slayout, layout.fractal)
        return *self.storage[operand], geometry

    def bytes(self, operand, rect=None):
        space, base, height, width, item_bytes = self.storage[operand]
        rect = rect or (0, 0, height, width)
        key = self.signature(operand), rect
        if key not in self.footprints:
            descriptor = (space, base, height, width, self.layouts.get(operand), item_bytes)
            points = address_set(descriptor, *rect)
            # The shared geometry oracle already expands Vec to bytes. Its
            # other spaces supply element addresses; expand those here too.
            self.footprints[key] = (
                frozenset(points)
                if space == ir.MemorySpace.Vec
                else frozenset(
                    (memory, address + byte) for memory, address in points for byte in range(item_bytes)
                )
            )
        return self.footprints[key]

    def memory_footprint(self, call, env):
        name, args = call.op.name, call.args
        rect = None
        if name in (opname("buffer.load"), opname("buffer.store")):
            rect = tuple(value(x, env) for group in args[1:3] for x in group.elements)
            if name == opname("buffer.load") and len(args[0].type.shape) == 1:
                # Original rank-one indexing supplies the concrete GM point.
                # The selected nonaligned MTE2 load can write its UB dummy tail
                # within the private 32-byte carrier; no adjacent GM bytes are
                # read to obtain that alignment. These are different domains.
                assert len(rect) == 2 and rect[1] == 1
                assert list(args[3].type.shape) == [1, 8] and args[3].type.dtype == ir.DataType.INT32
                rect = (0, rect[0], 1, 1)
                pipe, reads, writes = (
                    ir.PipeType.MTE2,
                    self.bytes(args[0], rect),
                    self.bytes(args[3], (0, 0, 1, 8)),
                )
            elif name == opname("buffer.load"):
                pipe, reads, writes = ir.PipeType.MTE2, self.bytes(args[0], rect), self.bytes(args[3])
            else:
                pipe = (
                    ir.PipeType.MTE3 if args[0].type.memory_space == ir.MemorySpace.Vec else ir.PipeType.FIX
                )
                reads, writes = self.bytes(args[0]), self.bytes(args[3], rect)
                if call.kwargs.get("atomic", 0):
                    reads |= writes
            signature = (
                name,
                self.signature(args[0]),
                rect,
                self.signature(args[3]),
                call.kwargs.get("atomic", 0),
            )
        elif name in (opname("buffer.read"), opname("buffer.write")):
            coordinates = tuple(value(x, env) for x in args[1].elements)
            cells = self.bytes(args[0], (*coordinates, 1, 1))
            pipe = ir.PipeType.S
            reads, writes = (cells, frozenset()) if name == opname("buffer.read") else (frozenset(), cells)
            signature = name, self.signature(args[0]), coordinates
            if name == opname("buffer.write"):
                signature += (value(args[2], env),)
        elif name in (opname("buffer.copy"), opname("buffer.extract")):
            target = 3 if name == opname("buffer.extract") else 1
            rect = None
            if target == 3:
                rect = (
                    value(args[1], env),
                    value(args[2], env),
                    self.storage[args[target]][2],
                    self.storage[args[target]][3],
                )
            pipe, reads, writes = ir.PipeType.MTE1, self.bytes(args[0], rect), self.bytes(args[target])
            signature = name, self.signature(args[0]), rect, self.signature(args[target])
        elif name == opname("buffer.full"):
            # The initialization writes its current active destination; its
            # scalar fill value is not a storage read or a completion fact.
            pipe, reads, writes = ir.PipeType.V, frozenset(), self.bytes(args[-1])
            signature = name, value(args[0], env), self.signature(args[-1])
        else:
            return None
        return pipe, reads, writes, signature, rect

    def compute_footprint(self, call, env):
        name, args = call.op.name, call.args
        if name in (opname("buffer.add"), opname("buffer.mul")):
            pipe = ir.PipeType.V
            reads, writes = self.bytes(args[0]) | self.bytes(args[1]), self.bytes(args[2])
            signature = name, *(self.signature(arg) for arg in args)
        elif name in (opname("buffer.adds"), opname("buffer.muls")):
            pipe = ir.PipeType.V
            reads, writes = self.bytes(args[0]), self.bytes(args[2])
            signature = name, self.signature(args[0]), value(args[1], env), self.signature(args[2])
        elif name == opname("buffer.cast"):
            pipe = ir.PipeType.V
            reads, writes = self.bytes(args[0]), self.bytes(args[1])
            signature = name, self.signature(args[0]), self.signature(args[1]), tuple(call.kwargs.items())
        elif name in (opname("buffer.row_sum"), opname("buffer.row_max")):
            # The focused runtime-axis fixtures reduce <=64 full columns, so
            # the audited one-repeat native path does not touch its workspace.
            source_type = args[0].type
            assert isinstance(source_type, ir.BufferType) and source_type.shape[1] <= 64
            pipe = ir.PipeType.V
            reads, writes = self.bytes(args[0]), self.bytes(args[2])
            signature = name, *(self.signature(arg) for arg in args)
        elif name == opname("buffer.row_expand_mul"):
            pipe = ir.PipeType.V
            source_type, carrier_type = args[0].type, args[1].type
            assert isinstance(source_type, ir.BufferType) and isinstance(carrier_type, ir.BufferType)
            valid = getattr(self, "valid", {}).get(args[0], tuple(source_type.valid_shape))
            rows = valid[0]
            assert 0 < rows <= 255 and carrier_type.shape[0] % 8 == 0
            rounded = ((rows + 7) // 8) * 8
            carrier = self.bytes(args[1], (0, 0, rounded, 1))
            # Vbrcb writes 32 bytes per rounded carrier row; both native modes
            # consume the staged values from the same fixed workspace.
            temporary = frozenset((ir.MemorySpace.Vec, 184 * 1024 + i) for i in range(rounded * 32))
            reads = self.bytes(args[0]) | carrier | temporary
            writes = self.bytes(args[2]) | temporary
            signature = name, *(self.signature(arg) for arg in args)
        elif name == opname("buffer.col_expand_mul"):
            # A row carrier is read repeatedly; the footprint is the explicit
            # matrix plus carrier, with no row-expand implicit scratch.
            pipe = ir.PipeType.V
            reads, writes = self.bytes(args[0]) | self.bytes(args[1]), self.bytes(args[2])
            signature = name, *(self.signature(arg) for arg in args)
        elif name in (opname("buffer.matmul"), opname("buffer.matmul_acc")):
            pipe = ir.PipeType.M
            reads, writes = self.bytes(args[0]) | self.bytes(args[1]), self.bytes(args[2])
            if name == opname("buffer.matmul_acc"):
                reads |= writes
            signature = name, *(self.signature(arg) for arg in args)
        else:
            assert name in (opname("buffer.abs"), opname("buffer.rsqrt")), (
                f"unsupported fixture payload {name}"
            )
            pipe, reads, writes = ir.PipeType.V, self.bytes(args[0]), self.bytes(args[1])
            signature = name, self.signature(args[0]), self.signature(args[1])
        return pipe, reads, writes, signature, None

    def payload(self, call, env):
        name, args = call.op.name, call.args
        footprint = self.memory_footprint(call, env) or self.compute_footprint(call, env)
        pipe, reads, writes, signature, rect = footprint
        index = len(self.payloads)
        start, completion = ("payload", index, "I"), ("payload", index, "C")
        self.issue(pipe, start)
        self.vertices.add(completion)
        self.edge(start, completion)
        self.edge(self.last_payload.get(pipe), completion)
        self.last_payload[pipe] = completion
        self.payloads.append((signature, pipe, reads, writes))
        self.calls.append((call, dict(env), pipe, reads, writes))
        for dependency in self.dependencies(call.args):
            self.edge(dependency, start)
        if self.scalar_memory is not None:
            if name in (opname("buffer.load"), opname("buffer.store")):
                assert rect is not None
                if name == opname("buffer.load"):
                    destination = self.bytes(args[3], (0, 0, rect[2], rect[3]))
                    source = reads
                    if len(args[0].type.shape) == 1:
                        # Deliberately poison inactive/dummy lanes: readiness
                        # and values must rely only on the original GM point.
                        for cell in writes - destination:
                            self.scalar_memory[cell] = -987654321
                else:
                    source = self.bytes(args[0], (0, 0, rect[2], rect[3]))
                    destination = writes
                # Byte ordering here is only for the focused packed ND scalar
                # fixtures; unsupported layouts remain outside this value model.
                for src, dst in zip(
                    sorted(source, key=lambda x: x[1]), sorted(destination, key=lambda x: x[1]), strict=True
                ):
                    if src in self.scalar_memory:
                        self.scalar_memory[dst] = self.scalar_memory[src]
            elif name == opname("buffer.write"):
                for cell in writes:
                    self.scalar_memory[cell] = value(args[2], env)

    def assign(self, statement, env):
        call = statement.value
        if isinstance(call, ir.Call) and call.op.name == opname("buffer.alloc"):
            kind = statement.var.type
            assert isinstance(kind, ir.BufferType)
            self.storage[statement.var] = (
                kind.memory_space,
                value(call.args[1], env),
                *kind.shape,
                kind.dtype.get_byte(),
            )
            self.layouts[statement.var] = kind
            self.descriptors.append(self.signature(statement.var))
        elif isinstance(call, ir.Call) and call.op.name == opname("buffer.read"):
            self.payload(call, env)
            assert self.scalar_memory is not None, "scalar read requires independent fixture memory"
            _, _, reads, _ = self.payloads[-1]
            values = {self.scalar_memory[cell] for cell in reads}
            assert len(values) == 1, "fixture scalar bytes disagree"
            env[statement.var] = values.pop()
            self.scalar_dependencies[statement.var] = frozenset({("payload", len(self.payloads) - 1, "C")})
        elif isinstance(call, ir.Call) and call.op.name == opname("sync.record"):
            self.logical_records = True
            pipe = ir.PipeType(call.kwargs["pipe"])
            assert isinstance(statement.var.type, ir.SyncTokenType) and statement.var.type.pipe == pipe
            publication = self.publish(pipe)
            self.record_origins[publication] = statement.var.name_hint
            env[statement.var] = pipe, publication
        else:
            env[statement.var] = value(statement.value, env)
            self.scalar_dependencies[statement.var] = self.dependencies([statement.value])

    def evaluate(self, statement, env):
        call = statement.expr
        assert isinstance(call, ir.Call)
        if call.op.name == opname("sync.wait"):
            fact = env[call.args[0]]
            source, publication = (fact.pipe, fact.node) if isinstance(fact, Token) else fact
            # A completed fact has no publication and imposes no ordering.
            # Carried direct-token recipes may use it on the first/empty visit.
            if publication is not None:
                self.used_tokens.add(publication)
                self.acquire(source, ir.PipeType(call.kwargs["dst_pipe"]), publication)
        elif call.op.name in _BARRIERS or call.op.name in (
            opname("system.set_flag"),
            opname("system.wait_flag"),
            opname("system.bar_all"),
        ):
            self.physical(call)
        else:
            self.payload(call, env)

    def dependencies(self, expressions):
        """Completion facts required by actual scalar SSA operands, not memory effects."""
        dependencies = set()
        seen = set()
        pending = list(expressions)
        while pending:
            expression = pending.pop()
            if expression in seen:
                continue
            seen.add(expression)
            if isinstance(expression, (ir.Var, ir.IterArg)):
                dependencies.update(self.scalar_dependencies.get(expression, ()))
            elif isinstance(expression, ir.BinaryExpr):
                pending.extend([expression.left, expression.right])
            elif isinstance(expression, ir.UnaryExpr):
                pending.append(expression.operand)
            elif isinstance(expression, ir.MakeTuple):
                pending.extend(expression.elements)
            elif isinstance(expression, ir.TupleGetItemExpr):
                pending.append(expression.tuple)
        return frozenset(dependencies)

    def control(self, expressions):
        # A dynamic decision must obtain its actual operand values before its
        # selected body or continuation can issue. An unused scalar read and a
        # scalar write create no such decision and impose no cross-pipe fence.
        dependencies = self.dependencies(expressions)
        if not dependencies:
            return
        node = ("control_evaluation", self.control_evaluations)
        self.control_evaluations += 1
        self.vertices.add(node)
        self.edge(self.control_gate, node)
        for dependency in dependencies:
            self.edge(dependency, node)
        self.control_gate = node

    def conditional(self, statement, env):
        self.control([statement.condition])
        condition_dependencies = self.dependencies([statement.condition])
        self.result_dependencies = ()
        selected = statement.then_body if value(statement.condition, env) else statement.else_body
        result = self.execute(selected, dict(env)) if selected is not None else ()
        if result is None:
            result = ()
            self.result_dependencies = ()
        assert len(result) == len(statement.return_vars)
        env.update(zip(statement.return_vars, result, strict=True))
        for variable, dependencies in zip(statement.return_vars, self.result_dependencies, strict=True):
            self.scalar_dependencies[variable] = dependencies | condition_dependencies

    def loop(self, statement, env):
        carried = tuple(value(arg.initValue, env) for arg in statement.iter_args)
        carried_dependencies = tuple(self.dependencies([arg.initValue]) for arg in statement.iter_args)
        if isinstance(statement, ir.ForStmt):
            assert statement.kind == ir.ForKind.Sequential
            self.control([statement.start, statement.stop, statement.step])
            start, stop, step = (value(x, env) for x in (statement.start, statement.stop, statement.step))
            assert isinstance(start, int) and isinstance(stop, int) and isinstance(step, int)
            assert step > 0
            indices = range(start, stop, step)
            assert len(indices) <= 32, "bounded oracle trip limit"
        else:
            indices = range(33)
        for index in indices:
            local = dict(env)
            local.update(zip(statement.iter_args, carried, strict=True))
            for argument, dependencies in zip(statement.iter_args, carried_dependencies, strict=True):
                self.scalar_dependencies[argument] = dependencies
            if isinstance(statement, ir.ForStmt):
                local[statement.loop_var] = index
            else:
                self.while_tests += 1
                self.control([statement.condition])
                if not value(statement.condition, local):
                    break
                assert index < 32, "bounded oracle trip limit"
            self.result_dependencies = ()
            result = self.execute(statement.body, local)
            carried = () if result is None else result
            carried_dependencies = () if result is None else self.result_dependencies
            assert len(carried) == len(statement.iter_args)
        assert len(carried) == len(statement.return_vars)
        env.update(zip(statement.return_vars, carried, strict=True))
        for variable, dependencies in zip(statement.return_vars, carried_dependencies, strict=True):
            self.scalar_dependencies[variable] = dependencies

    def execute(self, statement, env):
        if isinstance(statement, ir.SeqStmts):
            for child in statement.stmts:
                result = self.execute(child, env)
                if result is not None:
                    return result
        elif isinstance(statement, ir.AssignStmt):
            self.assign(statement, env)
        elif isinstance(statement, ir.EvalStmt):
            self.evaluate(statement, env)
        elif isinstance(statement, ir.IfStmt):
            self.conditional(statement, env)
        elif isinstance(statement, (ir.ForStmt, ir.WhileStmt)):
            self.loop(statement, env)
        elif isinstance(statement, ir.YieldStmt):
            self.result_dependencies = tuple(self.dependencies([x]) for x in statement.value)
            return tuple(value(x, env) for x in statement.value)
        else:
            assert isinstance(statement, ir.ReturnStmt)
            node = ("return", len(self.returns))
            self.vertices.add(node)
            self.returns.append(node)
            self.edge(self.last_all, node)
            self.edge(self.control_gate, node)
            for dependency in self.dependencies(statement.value):
                self.edge(dependency, node)
            self.result_dependencies = ()
            return ()
        return None

    def run(self):
        self.execute(self.function.body, dict(self.env))
        assert not self.live, "outstanding flags at invocation exit"
        # Immutable SSA completion facts may be unused or have several uses.
        # They are not consuming hardware notifications. Physical closure is
        # checked independently by the strict live-flag inventory above.
        order = closure(self.vertices, self.edges)
        self.order = order
        assert all(a != b for a, b in order), "cyclic command graph"
        assert all(pair in order for pair in self.reuses), "consumption does not precede republication"
        return order


def check_closed_commands(
    logical,
    physical,
    parameters,
    eligible: Sequence[int] = range(8),
    *,
    tensor_bases=None,
    compare_commands=True,
    check_hazards=True,
    hardware_source=None,
):
    before = ClosedCommands(logical, parameters, eligible, tensor_bases)
    after = ClosedCommands(physical, parameters, eligible, tensor_bases)
    expected, actual = before.run(), after.run()
    assert before.payloads == after.payloads, "changed payload operands or physical bytes"
    assert before.descriptors == after.descriptors, "changed original allocations"
    assert before.while_tests == after.while_tests, "changed while conditions/final false visit"
    common = {node for node in before.vertices if node[0] == "payload"}
    if compare_commands:
        assert before.trace == after.trace, "changed logical command directions/order"
        common.update(node for node in before.vertices if node[0] in ("publish", "acquire"))
    assert {(a, b) for a, b in expected if a in common and b in common} == {
        (a, b) for a, b in actual if a in common and b in common
    }, "changed actual-token event order"
    protected = set()
    if any(pipe == ir.PipeType.M for _, pipe, _, _ in before.payloads):
        assert hardware_source is not None, "Cube protection requires original source IR"
        original = ClosedCommands(hardware_source, parameters, eligible, tensor_bases)
        original.run()
        assert original.payloads == before.payloads, "changed original Cube payload geometry"
        protected = protected_accumulator_pairs(hardware_source, original.calls)
    if check_hazards:
        for i, (_, _, reads, writes) in enumerate(before.payloads):
            for j in range(i):
                _, _, earlier_reads, earlier_writes = before.payloads[j]
                if (earlier_writes & (reads | writes) or earlier_reads & writes) and (j, i) not in protected:
                    assert (("payload", j, "C"), ("payload", i, "I")) in actual, (
                        "unordered physical-byte conflict",
                        j,
                        i,
                    )
    return before, after
