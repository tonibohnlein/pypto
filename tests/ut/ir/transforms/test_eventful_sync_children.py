# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Whole local direct-token children inside irregular conservative control."""

import pytest
from pypto import ir, passes, testing

from .sync_closed_command_oracle import check_closed_commands
from .sync_matmul_oracle import sequence_stmts, validate
from .sync_shenggan_cases import capture_shenggan
from .test_autosync_switch import clone
from .test_closed_prefix_sync import Services
from .test_conservative_sync_children import resident_child_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
BASES = {0: 0, 1: 1 << 20, 2: 2 << 20}


def eventful_fixture(count=3, alias=False):
    original, old_child = resident_child_fixture(count, 1)
    span, index = original.span, ir.DataType.INDEX
    allocations: list[ir.AssignStmt] = []
    for statement in list(sequence_stmts(original.body))[:5]:
        assert isinstance(statement, ir.AssignStmt)
        allocations.append(statement)
    a, b, left, right, _ = [assignment.var for assignment in allocations]
    repeated_left = ir.Var("same_physical_left", left.type, span) if alias else left

    def payload(name, operands):
        return ir.EvalStmt(ir.Call(ir.get_op(name), operands, ir.VoidType(), span), span)

    child = ir.ForStmt(
        old_child.loop_var,
        old_child.start,
        old_child.stop,
        old_child.step,
        [],
        ir.SeqStmts(
            [
                payload("buffer.copy", [a, repeated_left]),
                payload("buffer.copy", [b, right]),
                *sequence_stmts(old_child.body),
            ],
            span,
        ),
        [],
        span,
    )
    visits = ir.Var("visits", ir.ScalarType(index), span)
    execute = ir.Var("execute_child", ir.ScalarType(ir.DataType.BOOL), span)

    # Rebuild the known original wrapper explicitly. Python mutator callbacks
    # need not redispatch a nested same-kind For through the trampoline.
    root = list(sequence_stmts(original.body))
    outer = root[-2]
    assert isinstance(outer, ir.ForStmt)
    outer_body = [
        ir.IfStmt(execute, child, None, [], span) if member is old_child else member
        for member in sequence_stmts(outer.body)
    ]
    wrapper = ir.ForStmt(
        outer.loop_var, outer.start, visits, outer.step, [], ir.SeqStmts(outer_body, span), [], span
    )
    body = ir.SeqStmts([*root[:-2], wrapper, root[-1]], span)
    if alias:
        statements = list(sequence_stmts(body))
        # Distinct descriptors intentionally share the same physical Left bytes.
        statements.insert(5, ir.AssignStmt(repeated_left, allocations[2].value, span))
        body = ir.SeqStmts(statements, span)
    function = ir.Function(
        original.name,
        [*zip(original.params, original.param_directions), visits, execute],
        original.return_types,
        body,
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    return function, child


def construct(function, scope):
    with passes.PassContext([]):
        child = passes.prepare_eventful_sync_children(function, [scope])[0]
    program = ir.Program([function], "EventfulChild", function.span)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync(eventful_children=[child])(program)
        optimized = passes.optimize_sync()(logical_program)
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(optimized)
    logical = next(iter(logical_program.functions.values()))
    physical = next(iter(physical_program.functions.values()))
    return child, logical, physical


@pytest.mark.parametrize("count", [0, 1, 3])
def test_eventful_local_child_inside_symbolic_parent(count):
    function, scope = eventful_fixture(count)
    before = ir.serialize(function)
    child, logical, physical = construct(function, scope)
    assert child.owner is function and child.scope is scope
    assert ir.serialize(function) == before
    # A single retained symbolic recipe is exercised at all admitted visits,
    # including skipped invocations and inner-counter resets.
    for visits in [0, 1, 3]:
        for execute in [False, True]:
            _, checked = check_closed_commands(
                logical,
                physical,
                {3: visits, 4: execute},
                range(6),
                tensor_bases=BASES,
                compare_commands=False,
                hardware_source=function,
            )
            assert checked.payloads is not None


def test_eventful_child_context_cannot_be_transplanted():
    function, scope = eventful_fixture(2)
    with passes.PassContext([]):
        child = passes.prepare_eventful_sync_children(function, [scope])[0]
    foreign = clone(function)
    before = ir.serialize(foreign)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        with pytest.raises(ValueError, match="unique original owner"):
            passes.insert_sync(eventful_children=[child])(ir.Program([foreign], "Foreign", foreign.span))
    assert ir.serialize(foreign) == before


@pytest.mark.parametrize("alias", [False, True])
def test_child_aliases_and_serialized_actual_tokens_keep_the_same_contract(alias):
    function, scope = eventful_fixture(2, alias=alias)
    _, logical, physical = construct(function, scope)
    restored = ir.deserialize(ir.serialize(logical))
    assert isinstance(restored, ir.Function)
    with passes.PassContext([]):
        program = passes.lower_sync()(ir.Program([restored], "Restored", restored.span))
    reparsed = next(iter(program.functions.values()))
    for candidate in [physical, reparsed]:
        check_closed_commands(
            logical,
            candidate,
            {3: 3, 4: True},
            range(6),
            tensor_bases=BASES,
            compare_commands=False,
            hardware_source=function,
        )


def test_missing_child_entry_rearm_or_capacity_is_an_explicit_refusal():
    function, scope = eventful_fixture(2)
    _, logical, _ = construct(function, scope)
    before = ir.serialize(logical)
    with pytest.raises(ValueError):
        testing.lower_sync_schedule(logical, eligible_ids=[0, 1])
    assert ir.serialize(logical) == before

    removed = False

    def remove_entry(statement):
        nonlocal removed
        if isinstance(statement, ir.SeqStmts):
            statements = [remove_entry(member) for member in statement.stmts]
            for i, member in enumerate(statements):
                if isinstance(member, ir.ForStmt) and member.attrs.get("pypto.closed_sync_child"):
                    kept = []
                    position = 0
                    while position < i:
                        publication = statements[position]
                        acquisition = statements[position + 1] if position + 1 < i else None
                        acknowledgment = statements[position + 2] if position + 2 < i else None
                        consumption = statements[position + 3] if position + 3 < i else None
                        if (
                            isinstance(publication, ir.AssignStmt)
                            and isinstance(publication.value, ir.Call)
                            and publication.value.op.name == ir.get_op("sync.record").name
                            and publication.value.kwargs["pipe"] == int(ir.PipeType.MTE1)
                            and isinstance(acquisition, ir.EvalStmt)
                            and isinstance(acquisition.expr, ir.Call)
                            and acquisition.expr.op.name == ir.get_op("sync.wait").name
                            and acquisition.expr.kwargs["dst_pipe"] == int(ir.PipeType.M)
                            and acquisition.expr.args[0] is publication.var
                            and isinstance(acknowledgment, ir.AssignStmt)
                            and isinstance(acknowledgment.value, ir.Call)
                            and acknowledgment.value.op.name == ir.get_op("sync.record").name
                            and acknowledgment.value.kwargs["pipe"] == int(ir.PipeType.M)
                            and isinstance(consumption, ir.EvalStmt)
                            and isinstance(consumption.expr, ir.Call)
                            and consumption.expr.op.name == ir.get_op("sync.wait").name
                            and consumption.expr.kwargs["dst_pipe"] == int(ir.PipeType.MTE1)
                            and consumption.expr.args[0] is acknowledgment.var
                        ):
                            removed = True
                            position += 4
                        elif (
                            isinstance(publication, ir.AssignStmt)
                            and isinstance(publication.value, ir.Call)
                            and publication.value.op.name == ir.get_op("sync.record").name
                            and publication.value.kwargs["pipe"] in (int(ir.PipeType.MTE1), int(ir.PipeType.M))
                            and isinstance(acquisition, ir.EvalStmt)
                            and isinstance(acquisition.expr, ir.Call)
                            and acquisition.expr.op.name == ir.get_op("sync.wait").name
                            and acquisition.expr.kwargs["dst_pipe"] in (int(ir.PipeType.MTE1), int(ir.PipeType.M))
                            and acquisition.expr.kwargs["dst_pipe"] != publication.value.kwargs["pipe"]
                            and acquisition.expr.args[0] is publication.var
                        ):
                            removed = True
                            position += 2
                        else:
                            kept.append(publication)
                            position += 1
                    return ir.SeqStmts([*kept, *statements[i:]], statement.span)
            return ir.SeqStmts(statements, statement.span)
        if isinstance(statement, ir.IfStmt):
            return ir.IfStmt(
                statement.condition,
                remove_entry(statement.then_body),
                remove_entry(statement.else_body) if statement.else_body is not None else None,
                statement.return_vars,
                statement.span,
            )
        if isinstance(statement, ir.ForStmt):
            return ir.ForStmt(
                statement.loop_var,
                statement.start,
                statement.stop,
                statement.step,
                statement.iter_args,
                remove_entry(statement.body),
                statement.return_vars,
                statement.span,
                statement.kind,
                statement.attrs,
                statement.hardware_loop_policy,
            )
        return statement

    missing = clone(logical, body=remove_entry(logical.body))
    assert removed
    with passes.PassContext([]):
        with pytest.raises(ValueError, match="acknowledgment"):
            passes.lower_sync()(ir.Program([missing], "MissingEntry", missing.span))


def test_enclosing_coordinate_is_not_replaced_by_a_representative():
    function, scope = eventful_fixture(2)
    outer = next(s for s in sequence_stmts(function.body) if isinstance(s, ir.ForStmt))
    changed = ir.ForStmt(
        scope.loop_var, outer.loop_var, scope.stop, scope.step, [], scope.body, [], scope.span
    )

    root = list(sequence_stmts(function.body))
    assert isinstance(outer.body, ir.SeqStmts)
    changed_body = [
        ir.IfStmt(member.condition, changed, None, [], member.span)
        if isinstance(member, ir.IfStmt) and member.then_body is scope
        else member
        for member in outer.body.stmts
    ]
    changed_outer = ir.ForStmt(
        outer.loop_var,
        outer.start,
        outer.stop,
        outer.step,
        [],
        ir.SeqStmts(changed_body, outer.span),
        [],
        outer.span,
    )
    unsupported = clone(function, body=ir.SeqStmts([*root[:-2], changed_outer, root[-1]], function.span))
    before = ir.serialize(unsupported)
    with passes.PassContext([]):
        with pytest.raises(ValueError, match="unavailable enclosing coordinate"):
            passes.prepare_eventful_sync_children(unsupported, [changed])
    assert ir.serialize(unsupported) == before


def test_immutable_unused_record_is_not_an_unconsumed_physical_flag():
    span = ir.Span.unknown()
    fact = ir.Var("unused_completion", ir.SyncTokenType(ir.PipeType.MTE1), span)
    record = ir.Call(ir.get_op("sync.record"), [], {"pipe": int(ir.PipeType.MTE1)}, fact.type, span)
    logical = ir.Function(
        "unused_fact",
        [],
        [],
        ir.SeqStmts([ir.AssignStmt(fact, record, span), ir.ReturnStmt([], span)], span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(ir.Program([logical], "UnusedFact", span))
    physical = next(iter(physical_program.functions.values()))
    _, checked = check_closed_commands(logical, physical, {}, range(6), compare_commands=False)
    assert not checked.live and not checked.publications


def test_sibling_children_share_parent_naming_and_rearm_private_palettes():
    function, scope = eventful_fixture(2)
    coordinate = ir.Var("second_k", scope.loop_var.type, scope.span)
    sibling_body = [
        ir.EvalStmt(member.expr, member.span) if isinstance(member, ir.EvalStmt) else member
        for member in sequence_stmts(scope.body)
    ]
    sibling = ir.ForStmt(
        coordinate,
        scope.start,
        scope.stop,
        scope.step,
        [],
        ir.SeqStmts(sibling_body, scope.span),
        [],
        scope.span,
    )
    root = list(sequence_stmts(function.body))
    outer = root[-2]
    assert isinstance(outer, ir.ForStmt)
    body = []
    for member in sequence_stmts(outer.body):
        if isinstance(member, ir.IfStmt) and member.then_body is scope:
            member = ir.IfStmt(
                member.condition, ir.SeqStmts([scope, sibling], scope.span), None, [], member.span
            )
        body.append(member)
    wrapper = ir.ForStmt(
        outer.loop_var, outer.start, outer.stop, outer.step, [], ir.SeqStmts(body, outer.span), [], outer.span
    )
    original = clone(function, body=ir.SeqStmts([*root[:-2], wrapper, root[-1]], function.span))
    with passes.PassContext([]):
        children = passes.prepare_eventful_sync_children(original, [scope, sibling])
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync(eventful_children=children)(
            ir.Program([original], "Siblings", original.span)
        )
    logical = next(iter(logical_program.functions.values()))
    names = []
    pending = [logical.body]
    while pending:
        statement = pending.pop()
        variables = []
        if isinstance(statement, ir.SeqStmts):
            pending.extend(statement.stmts)
        elif isinstance(statement, ir.AssignStmt):
            variables = [statement.var]
        elif isinstance(statement, ir.ForStmt):
            variables = [*statement.iter_args, *statement.return_vars]
            pending.append(statement.body)
        elif isinstance(statement, ir.IfStmt):
            variables = statement.return_vars
            pending.append(statement.then_body)
            if statement.else_body is not None:
                pending.append(statement.else_body)
        names.extend(
            variable.name_hint for variable in variables if isinstance(variable.type, ir.SyncTokenType)
        )
    assert len(names) == len(set(names)), "completion names must come from one parent owner"
    restored = ir.deserialize(ir.serialize(logical_program))
    assert isinstance(restored, ir.Program)
    with passes.PassContext([]):
        physical = next(iter(passes.lower_sync()(restored).functions.values()))
    check_closed_commands(
        logical,
        physical,
        {3: 3, 4: True},
        range(6),
        tensor_bases=BASES,
        compare_commands=False,
        hardware_source=original,
    )


def count_statements(statement):
    count = 0
    pending = [statement]
    while pending:
        current = pending.pop()
        count += 1
        if isinstance(current, ir.SeqStmts):
            pending.extend(current.stmts)
        elif isinstance(current, ir.ForStmt):
            pending.append(current.body)
        elif isinstance(current, ir.IfStmt):
            pending.append(current.then_body)
            if current.else_body is not None:
                pending.append(current.else_body)
    return count


@pytest.mark.parametrize("children", [1, 2, 4, 8])
def test_shared_original_context_index_tracks_static_input_once(children):
    function, first = eventful_fixture(2)
    scopes = [first]
    for child in range(1, children):
        coordinate = ir.Var(f"child_k_{child}", first.loop_var.type, first.span)
        body = [
            ir.EvalStmt(member.expr, member.span) if isinstance(member, ir.EvalStmt) else member
            for member in sequence_stmts(first.body)
        ]
        scopes.append(
            ir.ForStmt(
                coordinate,
                first.start,
                first.stop,
                first.step,
                [],
                ir.SeqStmts(body, first.span),
                [],
                first.span,
            )
        )
    root = list(sequence_stmts(function.body))
    outer = root[-2]
    assert isinstance(outer, ir.ForStmt)
    body = []
    scope_statements: list[ir.Stmt] = [scope for scope in scopes]
    for member in sequence_stmts(outer.body):
        if isinstance(member, ir.IfStmt) and member.then_body is first:
            member = ir.IfStmt(
                member.condition, ir.SeqStmts(scope_statements, first.span), None, [], member.span
            )
        body.append(member)
    wrapper = ir.ForStmt(
        outer.loop_var, outer.start, outer.stop, outer.step, [], ir.SeqStmts(body, outer.span), [], outer.span
    )
    original = clone(function, body=ir.SeqStmts([*root[:-2], wrapper, root[-1]], function.span))
    with passes.PassContext([]):
        capabilities = passes.prepare_eventful_sync_children(original, scopes)
    costs = [capability.describe() for capability in capabilities]
    # This is one shared build cost, deliberately NOT summed per child. The
    # per-region scope and descriptor work is output-sensitive and additive.
    assert {cost["shared_index_source_nodes"] for cost in costs} == {count_statements(original.body)}
    assert sum(cost["query_scope_nodes"] for cost in costs) == sum(
        count_statements(scope) for scope in scopes
    )
    assert sum(cost["binding_definitions"] for cost in costs) == 5 * children


def test_child_count_growth_keeps_retained_mathematics_compact():
    sizes = []
    for count in [2, 17, 1023, 1000000]:
        function, scope = eventful_fixture(count)
        with passes.PassContext([]):
            capability = passes.prepare_eventful_sync_children(function, [scope])[0]
        cost = capability.describe()
        sizes.append(
            (
                cost["analyzed_occurrences"],
                cost["retained_cells"],
                cost["retained_regions"],
                cost["query_scope_nodes"],
                cost["binding_definitions"],
            )
        )
    assert all(size == sizes[0] for size in sizes)


def test_command_only_child_pipes_require_their_own_entry_evidence():
    """A routing marker cannot hide command pipes behind an empty payload set."""
    span = ir.Span.unknown()
    coordinate = ir.Var("command_visit", ir.ScalarType(ir.DataType.INDEX), span)

    def ci(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    child = ir.ForStmt(
        coordinate,
        ci(0),
        ci(3),
        ci(1),
        [],
        ir.SeqStmts([*Services().closed(ir.PipeType.MTE1, ir.PipeType.FIX), ir.YieldStmt([], span)], span),
        [],
        span,
        attrs={"pypto.closed_sync_child": 1},
    )
    logical = ir.Function(
        "command_only_child",
        [],
        [],
        ir.SeqStmts([child, ir.ReturnStmt([], span)], span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    before = ir.serialize(logical)
    with passes.PassContext([]):
        with pytest.raises(ValueError, match="acknowledgment"):
            passes.lower_sync()(ir.Program([logical], "CommandOnlyChild", span))
    assert ir.serialize(logical) == before


def readonly_external_fixture(count=3, *, local_offset=False):
    function, previous = eventful_fixture(count)
    span = function.span
    root = list(sequence_stmts(function.body))
    outer = root[-2]
    assert isinstance(outer, ir.ForStmt)
    allocations = root[:5]
    assert all(isinstance(statement, ir.AssignStmt) for statement in allocations)
    handles = [statement.var for statement in allocations if isinstance(statement, ir.AssignStmt)]
    address = ir.Var("opaque_address", ir.ScalarType(ir.DataType.INDEX), span)

    def ci(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    def payload(name, args):
        return ir.EvalStmt(ir.Call(ir.get_op(name), args, ir.VoidType(), span), span)

    def rectangle(values):
        return ir.MakeTuple(values, span)

    body: list[ir.Stmt] = [
        ir.AssignStmt(address, ir.Mul(outer.loop_var, ci(0), ir.DataType.INDEX, span), span),
        payload(
            "buffer.load",
            [function.params[0], rectangle([address, ci(0)]), rectangle([ci(64), ci(32)]), handles[0]],
        ),
        payload(
            "buffer.load",
            [function.params[1], rectangle([ci(0), ci(0)]), rectangle([ci(32), ci(64)]), handles[1]],
        ),
        *sequence_stmts(previous.body),
    ]
    if local_offset:
        body[3] = payload("buffer.extract", [handles[0], address, ci(0), handles[2]])
    child = ir.ForStmt(
        previous.loop_var, previous.start, previous.stop, previous.step, [], ir.SeqStmts(body, span), [], span
    )
    enclosing = []
    for member in sequence_stmts(outer.body):
        if isinstance(member, ir.IfStmt) and member.then_body is previous:
            # This later guard does not establish a range at address's original
            # defining cut, and is never imported as an unstated caller premise.
            member = ir.IfStmt(member.condition, child, None, [], member.span)
        enclosing.append(member)
    wrapper = ir.ForStmt(
        outer.loop_var, outer.start, outer.stop, outer.step, [], ir.SeqStmts(enclosing, span), [], span
    )
    original = clone(function, body=ir.SeqStmts([*root[:-2], wrapper, root[-1]], span))
    oracle = ir.Function(
        "readonly_child_oracle",
        [*zip(original.params, original.param_directions), outer.loop_var],
        [],
        ir.SeqStmts([*allocations, child, ir.ReturnStmt([], span)], span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return original, child, oracle


@pytest.mark.parametrize("count", [0, 1, 3])
def test_readonly_external_projection_retains_exact_local_math_and_unavailable_gm(count):
    original, scope, oracle = readonly_external_fixture(count)
    child, logical, physical = construct(original, scope)
    assert child.describe()["unavailable_storage_domains"] == 1
    retained = testing.inspect_eventful_sync_child(child)
    description = retained.describe()
    unavailable = [
        index for index, domain in enumerate(description["storage"]) if domain["space"] == ir.MemorySpace.DDR
    ]
    assert len(unavailable) == 2
    assert all(not description["storage"][index]["boundary_available"] for index in unavailable)
    # This is the actual retained result, with every original load still an
    # event. No diagnostic reanalysis or specialization produces an easier plan.
    checked = validate(
        oracle,
        retained,
        parameters={3: 1, 4: True, 5: 0},
        unavailable_boundaries=unavailable,
        readonly_external_projection=True,
    )
    assert checked["payloads"] == count * 5
    for index in unavailable:
        assert retained.boundary(index, 0, 0) is None
        assert retained.boundary_slot(index, 0, 0, 0) is None
    for visits in [0, 1, 3]:
        for execute in [False, True]:
            for bases in [BASES, {0: 0, 1: 0, 2: 2 << 20}, {0: 0, 1: 1 << 20, 2: 0}]:
                check_closed_commands(
                    logical,
                    physical,
                    {3: visits, 4: execute},
                    range(6),
                    tensor_bases=bases,
                    compare_commands=False,
                    hardware_source=original,
                )


def test_opaque_external_coordinate_cannot_supply_a_local_numeric_proof():
    original, scope, _ = readonly_external_fixture(3, local_offset=True)
    before = ir.serialize(original)
    with passes.PassContext([]):
        with pytest.raises(ValueError, match="caller range|synchronization access|affine"):
            passes.prepare_eventful_sync_children(original, [scope])
    assert ir.serialize(original) == before


def test_readonly_projection_cannot_capture_a_future_enclosing_coordinate():
    original, scope, _ = readonly_external_fixture(3)
    root = list(sequence_stmts(original.body))
    child_body = list(sequence_stmts(scope.body))
    address = child_body.pop(0)
    assert isinstance(address, ir.AssignStmt)
    changed = ir.ForStmt(
        scope.loop_var,
        scope.start,
        scope.stop,
        scope.step,
        [],
        ir.SeqStmts(child_body, scope.span),
        [],
        scope.span,
    )
    outer = root[-2]
    assert isinstance(outer, ir.ForStmt)
    outer_body = [
        ir.IfStmt(member.condition, changed, None, [], member.span)
        if isinstance(member, ir.IfStmt) and member.then_body is scope
        else member
        for member in sequence_stmts(outer.body)
    ]
    wrapper = ir.ForStmt(
        outer.loop_var,
        outer.start,
        outer.stop,
        outer.step,
        [],
        ir.SeqStmts(outer_body, outer.span),
        [],
        outer.span,
    )
    # The address's dependence on this IV predates entry to the IV's scope.
    invalid = clone(original, body=ir.SeqStmts([*root[:-2], address, wrapper, root[-1]], original.span))
    before = ir.serialize(invalid)
    with passes.PassContext([]):
        with pytest.raises(ValueError, match="original definition/use cut"):
            passes.prepare_eventful_sync_children(invalid, [changed])
    assert ir.serialize(invalid) == before


def physical_shape_fixture(*, foreign=False, valid_shape_only=False):
    original, old_scope, _ = readonly_external_fixture(3)
    old_parameter = original.params[0]
    old_type = old_parameter.type
    assert isinstance(old_type, ir.TensorType)
    extent = ir.Var("physical_rows", ir.ScalarType(ir.DataType.INDEX), original.span)
    shape: list[ir.Expr] = [extent, ir.ConstInt(32, ir.DataType.INDEX, original.span)]
    view = None
    if valid_shape_only:
        shape = [
            ir.ConstInt(64, ir.DataType.INDEX, original.span),
            ir.ConstInt(32, ir.DataType.INDEX, original.span),
        ]
        view = ir.TensorView([32, 1], ir.TensorLayout.ND, valid_shape=[extent, 32])
    parameter = ir.Var(
        old_parameter.name_hint,
        ir.TensorType(shape, old_type.dtype, memref=None, tensor_view=view),
        original.span,
    )
    body = ir.substitute_stmt(original.body, [(old_parameter, parameter)])
    updated = ir.Function(
        original.name,
        [
            (parameter if item is old_parameter else item, direction)
            for item, direction in zip(original.params, original.param_directions, strict=True)
        ],
        original.return_types,
        body,
        original.span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    root = list(sequence_stmts(updated.body))
    outer = root[-2]
    assert isinstance(outer, ir.ForStmt)
    branch = next(member for member in sequence_stmts(outer.body) if isinstance(member, ir.IfStmt))
    scope = branch.then_body
    assert isinstance(scope, ir.ForStmt) and scope.loop_var is old_scope.loop_var
    if foreign or valid_shape_only:
        foreign_extent = ir.Var(extent.name_hint, extent.type, extent.span) if foreign else extent
        child_body = list(sequence_stmts(scope.body))
        first = child_body[0]
        assert isinstance(first, ir.AssignStmt)
        child_body[0] = ir.AssignStmt(first.var, foreign_extent, first.span)
        replacement = ir.ForStmt(
            scope.loop_var,
            scope.start,
            scope.stop,
            scope.step,
            [],
            ir.SeqStmts(child_body, scope.span),
            [],
            scope.span,
        )
        outer_body = [
            ir.IfStmt(member.condition, replacement, None, [], member.span)
            if isinstance(member, ir.IfStmt) and member is branch
            else member
            for member in sequence_stmts(outer.body)
        ]
        wrapper = ir.ForStmt(
            outer.loop_var,
            outer.start,
            outer.stop,
            outer.step,
            [],
            ir.SeqStmts(outer_body, outer.span),
            [],
            outer.span,
        )
        updated = clone(updated, body=ir.SeqStmts([*root[:-2], wrapper, root[-1]], updated.span))
        scope = replacement
    return updated, scope


def test_original_physical_shape_metadata_is_ready_but_not_numerically_bounded():
    original, scope = physical_shape_fixture()
    with passes.PassContext([]):
        child = passes.prepare_eventful_sync_children(original, [scope])[0]
    retained = testing.inspect_eventful_sync_child(child)
    assert child.describe()["unavailable_storage_domains"] == 1
    assert any(
        domain["space"] != ir.MemorySpace.DDR and domain["boundary_available"]
        for domain in retained.describe()["storage"]
    )
    exported = testing.export_native_sync_child(retained, scope)
    assert exported["status"] == "access_premises" and exported["child"] is None
    assert "external-domain contract" in exported["reason"]


@pytest.mark.parametrize("kind", ["foreign", "valid_shape_only"])
def test_opaque_shape_readiness_requires_original_physical_abi_identity(kind):
    original, scope = physical_shape_fixture(
        foreign=kind == "foreign", valid_shape_only=kind == "valid_shape_only"
    )
    before = ir.serialize(original)
    with passes.PassContext([]):
        with pytest.raises(ValueError, match="unavailable enclosing coordinate|ND descriptor"):
            passes.prepare_eventful_sync_children(original, [scope])
    assert ir.serialize(original) == before


def test_eventful_child_preserves_preexisting_debt_across_partial_completion():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    vec_t = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)
    src = ir.Var("src", gm_t, span)
    out = ir.Var("out", gm_t, span)
    flag = ir.Var("flag", ir.ScalarType(ir.DataType.BOOL), span)
    x = ir.Var("x", vec_t, span)
    y = ir.Var("y", vec_t, span)
    k = ir.Var("k", ir.ScalarType(idx), span)

    def ci(val):
        return ir.ConstInt(val, idx, span)

    def tup(vals):
        return ir.MakeTuple([ci(v) for v in vals], span)

    child = ir.ForStmt(
        k,
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [x, y], ir.VoidType(), span), span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    fn = ir.Function(
        "partial_child_debt",
        [src, (out, ir.ParamDirection.InOut), flag],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(x, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span),
                ir.AssignStmt(y, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], vec_t, span), span),
                # Prior MTE2 write on y (must remain live for MTE3 store even across child).
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), y], ir.VoidType(), span), span),
                ir.IfStmt(flag, child, None, [], span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.store"), [y, tup([0, 0]), tup([2, 8]), out], ir.VoidType(), span), span),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([]):
        ev = passes.prepare_eventful_sync_children(fn, [child])[0]
    program = ir.Program([fn], "PartialChildDebt", span)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical = passes.insert_sync(eventful_children=[ev])(program)
        optimized = passes.optimize_sync()(logical)
    with passes.PassContext([]):
        physical = passes.lower_sync()(optimized)
    log_fn = next(iter(optimized.functions.values()))
    phys_fn = next(iter(physical.functions.values()))
    for cond in (False, True):
        check_closed_commands(
            log_fn,
            phys_fn,
            {2: cond},
            range(6),
            tensor_bases={0: 0, 1: 1 << 20},
            compare_commands=False,
            hardware_source=fn,
        )


def test_eventful_child_does_not_claim_post_record_write_complete():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    vec_t = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)
    src = ir.Var("src", gm_t, span)
    out = ir.Var("out", gm_t, span)
    a = ir.Var("a", vec_t, span)
    b = ir.Var("b", vec_t, span)
    c = ir.Var("c", vec_t, span)
    z = ir.Var("z", vec_t, span)
    k = ir.Var("k", ir.ScalarType(idx), span)

    def ci(val):
        return ir.ConstInt(val, idx, span)

    def tup(vals):
        return ir.MakeTuple([ci(v) for v in vals], span)

    # Inside child: load(a) -> record(MTE2); load(c) on MTE2 AFTER record;
    # wait(MTE2 -> V) on a's earlier record; abs(a, b) on V.
    # Because load(c) executed after a's sync.record(MTE2), exiting child does
    # NOT guarantee post_write_safe[MTE2] covers V for parent's subsequent abs(c, z).
    child = ir.ForStmt(
        k,
        ci(0),
        ci(1),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), a], ir.VoidType(), span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), c], ir.VoidType(), span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [a, b], ir.VoidType(), span), span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    fn = ir.Function(
        "post_record_write_child",
        [src, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(a, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span),
                ir.AssignStmt(b, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], vec_t, span), span),
                ir.AssignStmt(c, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(256)], vec_t, span), span),
                ir.AssignStmt(z, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(384)], vec_t, span), span),
                child,
                ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [c, z], ir.VoidType(), span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.store"), [z, tup([0, 0]), tup([2, 8]), out], ir.VoidType(), span), span),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([]):
        ev = passes.prepare_eventful_sync_children(fn, [child])[0]
    program = ir.Program([fn], "PostRecordWriteChild", span)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical = passes.insert_sync(eventful_children=[ev])(program)
        optimized = passes.optimize_sync()(logical)
    with passes.PassContext([]):
        physical = passes.lower_sync()(optimized)
    phys_fn = next(iter(physical.functions.values()))
    check_closed_commands(
        next(iter(optimized.functions.values())),
        phys_fn,
        {},
        range(4),
        tensor_bases={0: 0, 1: 1 << 20},
        compare_commands=False,
        hardware_source=fn,
    )


def test_eventful_child_preserves_unrelated_self_pipe_debt():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    vec_t = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)
    src = ir.Var("src", gm_t, span)
    out = ir.Var("out", gm_t, span)
    a = ir.Var("a", vec_t, span)
    x = ir.Var("x", vec_t, span)
    y = ir.Var("y", vec_t, span)
    z = ir.Var("z", vec_t, span)
    k = ir.Var("k", ir.ScalarType(idx), span)

    def ci(val):
        return ir.ConstInt(val, idx, span)

    def tup(vals):
        return ir.MakeTuple([ci(v) for v in vals], span)

    child = ir.ForStmt(
        k,
        ci(0),
        ci(1),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    fn = ir.Function(
        "self_pipe_debt_across_child",
        [src, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(a, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span),
                ir.AssignStmt(x, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], vec_t, span), span),
                ir.AssignStmt(y, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(256)], vec_t, span), span),
                ir.AssignStmt(z, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(384)], vec_t, span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), a], ir.VoidType(), span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [a, y], ir.VoidType(), span), span),
                child,
                ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [y, z], ir.VoidType(), span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.store"), [z, tup([0, 0]), tup([2, 8]), out], ir.VoidType(), span), span),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([]):
        ev = passes.prepare_eventful_sync_children(fn, [child])[0]
    program = ir.Program([fn], "SelfPipeDebtAcrossChild", span)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical = passes.insert_sync(eventful_children=[ev])(program)
        optimized = passes.optimize_sync()(logical)
    with passes.PassContext([]):
        physical = passes.lower_sync()(optimized)
    phys_fn = next(iter(physical.functions.values()))
    check_closed_commands(
        next(iter(optimized.functions.values())),
        phys_fn,
        {},
        range(4),
        tensor_bases={0: 0, 1: 1 << 20},
        compare_commands=False,
        hardware_source=fn,
    )
    assert "pl.system.barrier(pipe=pl.PipeType.V" in str(phys_fn)


def test_readonly_child_binds_if_return_var_scalars():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX

    def ci(val):
        return ir.ConstInt(val, idx, span)

    def tup(vals):
        return ir.MakeTuple([ci(v) for v in vals], span)

    mat_t = ir.BufferType(
        [128, 256],
        ir.DataType.FP16,
        ir.MemorySpace.Mat,
        slayout=ir.TileLayout.row_major,
        blayout=ir.TileLayout.col_major,
        fractal=512,
    )
    mat_multi_t = ir.MultiBufferType(mat_t, 2)
    vec_t = ir.BufferType([16, 64], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([512, 64], ir.DataType.FP32)

    a = ir.Var("a", gm_t, span)
    out = ir.Var("out", gm_t, span)
    cond = ir.Var("cond", ir.ScalarType(ir.DataType.BOOL), span)
    row_phi = ir.Var("row_phi", ir.ScalarType(idx), span)
    m0 = ir.Var("m0", ir.ScalarType(idx), span)
    k = ir.Var("k", ir.ScalarType(idx), span)
    x_buf = ir.Var("x_buf", vec_t, span)
    y = ir.Var("y", vec_t, span)
    a_pool = ir.Var("a_pool", mat_multi_t, span)
    a_slot = ir.Var("a_slot", mat_t, span)

    branch = ir.IfStmt(
        cond,
        ir.SeqStmts([ir.YieldStmt([ci(0)], span)], span),
        ir.SeqStmts([ir.YieldStmt([ci(1)], span)], span),
        [row_phi],
        span,
    )
    child = ir.ForStmt(
        k,
        ci(0),
        ci(4),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(
                    ir.Call(
                        ir.get_op("buffer.load"),
                        [a, ir.MakeTuple([m0, ci(0)], span), tup([16, 64]), x_buf],
                        ir.VoidType(),
                        span,
                    ),
                    span,
                ),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [x_buf, y], ir.VoidType(), span), span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    fn = ir.Function(
        "if_return_var_readonly_child",
        [a, (out, ir.ParamDirection.InOut), cond],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    x_buf, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span
                ),
                ir.AssignStmt(
                    y, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(8192)], vec_t, span), span
                ),
                branch,
                ir.AssignStmt(m0, ir.Mul(row_phi, ci(16), idx, span), span),
                child,
                ir.EvalStmt(
                    ir.Call(
                        ir.get_op("buffer.store"),
                        [y, tup([0, 0]), tup([16, 64]), out],
                        ir.VoidType(),
                        span,
                    ),
                    span,
                ),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )

    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        ev = passes.prepare_eventful_sync_children(fn, [child])[0]
    program = ir.Program([fn], "IfReturnVarReadonlyChild", span)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_policy="conservative_prefix"):
        p_realize = passes.realize_sync()(program)
        logical = passes.insert_sync(eventful_children=[ev])(program)
        optimized = passes.optimize_sync()(logical)
    with passes.PassContext([]):
        p_lower = passes.lower_sync()(optimized)

    for p_fn in (next(iter(p_realize.functions.values())), next(iter(p_lower.functions.values()))):
        assert p_fn.attrs.get("pypto_autosync") is True
        assert p_fn.attrs.get("autosync_retained_child_count") == 1
        check_closed_commands(
            next(iter(optimized.functions.values())),
            p_fn,
            {2: True},
            range(8),
            tensor_bases={0: 0, 1: 1 << 20},
            compare_commands=False,
            hardware_source=fn,
        )

    # Negative check: using an IfStmt-defined scalar for local on-chip slot indexing must be refused.
    gm_aic_t = ir.TensorType([512, 512], ir.DataType.FP16)
    gm_aic_out_t = ir.TensorType([512, 512], ir.DataType.FP32)
    a_aic = ir.Var("a_aic", gm_aic_t, span)
    out_aic = ir.Var("out_aic", gm_aic_out_t, span)
    bad_child = ir.ForStmt(
        k,
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    a_slot, ir.Call(ir.get_op("buffer.slot"), [a_pool, row_phi], mat_t, span), span
                ),
                ir.EvalStmt(
                    ir.Call(
                        ir.get_op("buffer.load"),
                        [a_aic, tup([0, 0]), tup([128, 256]), a_slot],
                        ir.VoidType(),
                        span,
                    ),
                    span,
                ),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    bad_fn = ir.Function(
        "if_return_var_local_slot_refused",
        [a_aic, (out_aic, ir.ParamDirection.InOut), cond],
        [out_aic.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    a_pool,
                    ir.Call(ir.get_op("buffer.alloc_multi"), [tup([]), ci(0)], mat_multi_t, span),
                    span,
                ),
                branch,
                bad_child,
                ir.ReturnStmt([out_aic], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        with pytest.raises(ValueError, match="Invocation coordinate has no established caller range"):
            passes.prepare_eventful_sync_children(bad_fn, [bad_child])


def test_shenggan_swizzle_promotes_kb_in_realize_and_three_pass(tmp_path):
    program = capture_shenggan(tmp_path / "capture", m=256, n=512, k=512, workers=4, swizzle=True)
    for disjoint_gm in (False, True):
        with passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=True,
            autosync_policy="conservative_prefix",
            autosync_disjoint_gm=disjoint_gm,
        ):
            realized = passes.realize_sync()(program)
            lowered = passes.lower_sync()(passes.optimize_sync()(passes.insert_sync()(program)))
        for prog in (realized, lowered):
            fn = next(f for f in prog.functions.values() if "hpgemm_hpgemm" in f.name)
            assert fn.attrs.get("pypto_autosync") is True
            refusals = str(fn.attrs.get("autosync_child_refusals", ""))
            assert "kb__idx_v0" not in refusals
            assert "n_col" not in refusals
            if disjoint_gm:
                assert fn.attrs.get("autosync_retained_child_count") is None
                assert "autosync_analysis_policy" not in fn.attrs
            else:
                assert fn.attrs.get("autosync_retained_child_count") == 1


def test_shenggan_4k_swizzle_exact_and_double_buffer_child_ids(tmp_path):
    program = capture_shenggan(tmp_path / "capture4k", m=2048, n=4096, k=4096, workers=24, swizzle=True)
    for policy in ("exact", "conservative_prefix"):
        with passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=True,
            autosync_policy=policy,
            autosync_disjoint_gm=True,
        ):
            realized = passes.realize_sync()(program)
        fn = next(f for f in realized.functions.values() if "hpgemm_hpgemm" in f.name)
        assert fn.attrs.get("pypto_autosync") is True
        assert fn.attrs.get("autosync_retained_child_count") is None
        assert "autosync_fallback_reason" not in fn.attrs

    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=False,
    ):
        realized_m1 = passes.realize_sync()(program)
    fn_m1 = next(f for f in realized_m1.functions.values() if "hpgemm_hpgemm" in f.name)
    text_m1 = str(fn_m1)
    assert fn_m1.attrs.get("autosync_retained_child_count") == 1
    assert "kb__idx_v0 % 15" not in text_m1
    assert "MTE2, dst_pipe=pl.PipeType.MTE1, event_id=5" not in text_m1
    assert "MTE2, dst_pipe=pl.PipeType.MTE1, event_id=6" not in text_m1


def test_qwen3_jit_pipeline_unrolled_acc_children_promote():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX

    def ci(val):
        return ir.ConstInt(val, idx, span)

    def tup(vals):
        return ir.MakeTuple([ci(v) for v in vals], span)

    mat_a_t = ir.BufferType(
        [16, 128],
        ir.DataType.BF16,
        ir.MemorySpace.Mat,
        slayout=ir.TileLayout.row_major,
        blayout=ir.TileLayout.col_major,
        fractal=512,
    )
    mat_b_t = ir.BufferType(
        [128, 256],
        ir.DataType.BF16,
        ir.MemorySpace.Mat,
        slayout=ir.TileLayout.row_major,
        blayout=ir.TileLayout.col_major,
        fractal=512,
    )
    left_t = ir.BufferType(
        [16, 32],
        ir.DataType.BF16,
        ir.MemorySpace.Left,
        slayout=ir.TileLayout.row_major,
        blayout=ir.TileLayout.row_major,
        fractal=512,
    )
    right_t = ir.BufferType(
        [32, 256],
        ir.DataType.BF16,
        ir.MemorySpace.Right,
        slayout=ir.TileLayout.col_major,
        blayout=ir.TileLayout.row_major,
        fractal=512,
    )
    acc_t = ir.BufferType(
        [16, 256],
        ir.DataType.FP32,
        ir.MemorySpace.Acc,
        slayout=ir.TileLayout.row_major,
        blayout=ir.TileLayout.col_major,
        fractal=1024,
    )
    gm_a_t = ir.TensorType([16, 1024], ir.DataType.BF16)
    gm_b_t = ir.TensorType([1024, 256], ir.DataType.BF16)
    gm_c_t = ir.TensorType([16, 256], ir.DataType.FP32)

    a = ir.Var("a", gm_a_t, span)
    b = ir.Var("b", gm_b_t, span)
    c = ir.Var("c", gm_c_t, span)
    acc = ir.Var("acc", acc_t, span)
    m4 = ir.Var("m4", mat_a_t, span)
    m5 = ir.Var("m5", mat_b_t, span)
    m6 = ir.Var("m6", mat_a_t, span)
    m7 = ir.Var("m7", mat_b_t, span)
    l9 = ir.Var("l9", left_t, span)
    r10 = ir.Var("r10", right_t, span)
    l21 = ir.Var("l21", left_t, span)
    r22 = ir.Var("r22", right_t, span)

    kb = ir.Var("kb", ir.ScalarType(idx), span)
    k0 = ir.Var("k0", ir.ScalarType(idx), span)
    k1 = ir.Var("k1", ir.ScalarType(idx), span)
    ko = ir.Var("ko", ir.ScalarType(idx), span)
    ko1 = ir.Var("ko1", ir.ScalarType(idx), span)
    ko2 = ir.Var("ko2", ir.ScalarType(idx), span)

    def acc_loop(loop_var, ma, mb, la, rb, init_first):
        mm_stmt = ir.EvalStmt(ir.Call(ir.get_op("buffer.matmul"), [la, rb, acc], ir.VoidType(), span), span)
        mm_acc_stmt = ir.EvalStmt(
            ir.Call(ir.get_op("buffer.matmul_acc"), [la, rb, acc], ir.VoidType(), span), span
        )
        mm_call = (
            ir.IfStmt(
                ir.Eq(loop_var, ci(0), ir.DataType.BOOL, span),
                ir.SeqStmts([mm_stmt], span),
                ir.SeqStmts([mm_acc_stmt], span),
                [],
                span,
            )
            if init_first
            else ir.EvalStmt(
                ir.Call(ir.get_op("buffer.matmul_acc"), [la, rb, acc], ir.VoidType(), span), span
            )
        )
        return ir.ForStmt(
            loop_var,
            ci(0),
            ci(128),
            ci(64),
            [],
            ir.SeqStmts(
                [
                    ir.EvalStmt(
                        ir.Call(ir.get_op("buffer.extract"), [ma, ci(0), loop_var, la], ir.VoidType(), span),
                        span,
                    ),
                    ir.EvalStmt(
                        ir.Call(ir.get_op("buffer.extract"), [mb, loop_var, ci(0), rb], ir.VoidType(), span),
                        span,
                    ),
                    mm_call,
                    ir.YieldStmt([], span),
                ],
                span,
            ),
            [],
            span,
        )

    def load_mat(src, row_off, col_off, shape, dst):
        return ir.EvalStmt(
            ir.Call(
                ir.get_op("buffer.load"),
                [src, ir.MakeTuple([row_off, col_off], span), tup(shape), dst],
                ir.VoidType(),
                span,
            ),
            span,
        )

    kb_loop = ir.ForStmt(
        kb,
        ci(0),
        ci(6),
        ci(2),
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(k0, ir.Mul(kb, ci(128), idx, span), span),
                ir.AssignStmt(k1, ir.Add(k0, ci(128), idx, span), span),
                load_mat(a, ci(0), k0, [16, 128], m4),
                load_mat(b, k0, ci(0), [128, 256], m5),
                load_mat(a, ci(0), k1, [16, 128], m6),
                load_mat(b, k1, ci(0), [128, 256], m7),
                ir.IfStmt(
                    ir.Eq(k0, ci(0), ir.DataType.BOOL, span),
                    ir.SeqStmts([acc_loop(ko, m4, m5, l9, r10, True), ir.YieldStmt([], span)], span),
                    ir.SeqStmts([acc_loop(ko1, m4, m5, l9, r10, False), ir.YieldStmt([], span)], span),
                    [],
                    span,
                ),
                acc_loop(ko2, m6, m7, l21, r22, False),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )

    def alloc_buf(offset, buf_t):
        return ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(offset)], buf_t, span)

    fn = ir.Function(
        "qwen3_jit_unrolled_pipeline_proj",
        [a, b, (c, ir.ParamDirection.InOut)],
        [c.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(acc, alloc_buf(0, acc_t), span),
                ir.AssignStmt(m4, alloc_buf(0, mat_a_t), span),
                ir.AssignStmt(m5, alloc_buf(4096, mat_b_t), span),
                ir.AssignStmt(m6, alloc_buf(69632, mat_a_t), span),
                ir.AssignStmt(m7, alloc_buf(73728, mat_b_t), span),
                ir.AssignStmt(l9, alloc_buf(0, left_t), span),
                ir.AssignStmt(r10, alloc_buf(0, right_t), span),
                ir.AssignStmt(l21, alloc_buf(1024, left_t), span),
                ir.AssignStmt(r22, alloc_buf(16384, right_t), span),
                kb_loop,
                ir.EvalStmt(
                    ir.Call(
                        ir.get_op("buffer.store"), [acc, tup([0, 0]), tup([16, 256]), c], ir.VoidType(), span
                    ),
                    span,
                ),
                ir.ReturnStmt([c], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    program = ir.Program([fn], "Qwen3JitUnrolledPipeline", span)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        realized = passes.realize_sync()(program)
        lowered = passes.lower_sync()(passes.optimize_sync()(passes.insert_sync()(program)))
    for prog in (realized, lowered):
        out_fn = next(iter(prog.functions.values()))
        assert out_fn.attrs.get("pypto_autosync") is True
        # Outer multi-stage unrolled kb loop stays unpromoted due to IfStmt over distinct nested ForStmts,
        # while all 3 inner acc loops promote cleanly as EventfulSyncChildren.
        assert out_fn.attrs.get("autosync_retained_child_count") == 3
        assert "Only first-visit initialization/continuation selection is qualified" in str(
            out_fn.attrs.get("autosync_child_refusals", "")
        )
