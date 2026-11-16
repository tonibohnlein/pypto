# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Original-root GM workspace allocation/offset agreement and capability refusals."""

import gc
import re

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _build_group_mapping, _ungrouped_workspace_context


@pytest.fixture
def gm_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    yield
    backend.reset_for_testing()


def _workspace():
    span = ir.Span.unknown()
    return ir.Var("__gm_pipe_buffer", ir.TensorType([1], DataType.FP32), span)


def _setup(pipe_id=0, slot_bytes=512):
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, DataType.INT32, span)
    return ir.EvalStmt(
        ir.create_op_call(
            "system.aic_initialize_pipe",
            [zero, zero],
            {"id": pipe_id, "dir_mask": 3, "slot_size": slot_bytes, "slot_num": 2},
            span,
        ),
        span,
    )


def _function(name, params, stmts, kind=ir.FunctionType.AIC):
    span = ir.Span.unknown()
    return ir.Function(name, params, [], ir.SeqStmts([*stmts, ir.ReturnStmt(span)], span), span, type=kind)


def _program(functions):
    return ir.Program(functions, "workspace_context", ir.Span.unknown())


def _edge(callee, actual, submit=False):
    span = ir.Span.unknown()
    if submit:
        value = ir.Submit(ir.GlobalVar(callee), [actual], [], ir.ScalarType(DataType.TASK_ID), span)
    else:
        value = ir.Call(ir.GlobalVar(callee), [actual], ir.VoidType(), span)
    return ir.EvalStmt(value, span)


@pytest.mark.parametrize("submit", [False, True])
def test_context_checks_actual_forwarding_and_memoizes_refusal(submit):
    leaf_param = _workspace()
    leaf = _function("leaf", [leaf_param], [_setup()])
    root_param = _workspace()
    good = _function("good", [root_param], [_edge("leaf", root_param, submit)], ir.FunctionType.Group)
    different = ir.Var("other", root_param.type, ir.Span.unknown())
    bad = _function("bad", [root_param, different], [_edge("leaf", different, submit)], ir.FunctionType.Group)
    owner = codegen.GMPipeLayoutEvidence(_program([leaf, good, bad]))
    context = owner.bind_root(good)
    assert context.bytes == 2048 and context.offsets == {(0, 3): 0}
    with pytest.raises(ValueError, match="exact original workspace formal"):
        owner.bind_root(bad)
    work = owner.work()
    for _ in range(16):
        assert owner.bind_root(good) is context
        with pytest.raises(ValueError, match="exact original workspace formal"):
            owner.bind_root(bad)
    after = owner.work()
    assert after["forwarding_edges"] == work["forwarding_edges"]
    assert after["setup_validations"] == work["setup_validations"]
    assert after["bindings_rejected"] == work["bindings_rejected"]
    assert after["binding_cache_hits"] - work["binding_cache_hits"] == 32


def test_context_rejects_missing_or_ambiguous_workspace_and_setup():
    span = ir.Span.unknown()
    no_formal = _function("none", [], [_setup()])
    duplicate = _function("duplicate", [_workspace(), _workspace()], [_setup()])
    cut = _setup()
    repeated = _function("repeated", [_workspace()], [cut, cut])
    owner = codegen.GMPipeLayoutEvidence(_program([no_formal, duplicate, repeated]))
    for function in (no_formal, duplicate):
        with pytest.raises(ValueError, match="one original injected formal"):
            owner.bind_root(function)
    with pytest.raises(ValueError, match="unique original statement occurrence"):
        owner.bind_root(repeated)
    foreign = ir.Function(no_formal.name, [], [], no_formal.body, span)
    with pytest.raises(ValueError, match="original Program"):
        owner.bind_root(foreign)


@pl.program
class IndependentWorkspaces:
    @pl.function(type=pl.FunctionType.AIC)
    def cube_a(self):
        first = pl.reserve_buffer(name="a_first", size=1024, base=pl.AUTO)
        second = pl.reserve_buffer(name="a_second", size=2048, base=pl.AUTO)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), first, dir_mask=2, slot_size=512, slot_num=2, id=3)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), second, dir_mask=2, slot_size=1024, slot_num=2, id=4)

    @pl.function(type=pl.FunctionType.AIV)
    def vector_a(self):
        first = pl.import_peer_buffer(name="a_first", peer_func="cube_a")
        second = pl.import_peer_buffer(name="a_second", peer_func="cube_a")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), first, dir_mask=2, slot_size=512, slot_num=2, id=3)
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), second, dir_mask=2, slot_size=1024, slot_num=2, id=4)

    @pl.function(type=pl.FunctionType.Group)
    def group_a(self):
        self.cube_a()
        self.vector_a()

    @pl.function(type=pl.FunctionType.AIC)
    def cube_b(self):
        slot = pl.reserve_buffer(name="b_slot", size=4096, base=pl.AUTO)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), slot, dir_mask=2, slot_size=2048, slot_num=2, id=0)

    @pl.function(type=pl.FunctionType.AIV)
    def vector_b(self):
        slot = pl.import_peer_buffer(name="b_slot", peer_func="cube_b")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), slot, dir_mask=2, slot_size=2048, slot_num=2, id=0)

    @pl.function(type=pl.FunctionType.Group)
    def group_b(self):
        self.cube_b()
        self.vector_b()

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self):
        self.group_a()
        self.group_b()


def _transformed():
    return ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(IndependentWorkspaces)


def _get(program, name):
    function = program.get_function(name)
    assert function is not None
    return function


def test_actual_allocation_and_offsets_share_original_group_context(gm_backend):
    program = _transformed()
    owner = codegen.GMPipeLayoutEvidence(program)
    groups, _ = _build_group_mapping(program)
    a = owner.bind_root(_get(program, "group_a"))
    b = owner.bind_root(_get(program, "group_b"))
    assert (a.bytes, b.bytes) == (3072, 4096)
    assert a.offsets == {(3, 2): 0, (4, 2): 1024}
    assert b.offsets == {(0, 2): 0}
    orchestration = codegen.generate_orchestration(program, _get(program, "main"), layout=owner).code
    sizes = re.findall(r"gm_pipe_buffer_\d+_ci_shapes\[1\]\s*=\s*\{(\d+)\};", orchestration)
    assert sizes == ["768", "1024"]
    generator = codegen.PTOCodegen()
    modules = {name: _program(functions) for name, functions in groups.items()}
    for name, context in (("group_a", a), ("group_b", b), ("group_a", a)):
        text = generator.generate(modules[name], emit_source_loc=False, workspace=context)
        bundle = generator.generate_bundle(modules[name], emit_source_loc=False, workspace=context)
        assert text == bundle.caller_source
        if name == "group_a":
            assert "arith.constant 256 : index" in text  # 1024 bytes / FP32 element
            assert text.count("pto.addptr") == 2
        else:
            assert "pto.addptr" not in text
    all_members = _program([*groups["group_a"], *groups["group_b"]])
    with pytest.raises(ValueError, match="incompatible GM workspace layouts"):
        generator.generate(all_members)
    with pytest.raises(ValueError, match="does not cover"):
        generator.generate(all_members, workspace=a)
    # Even same names and identical source text do not transfer source identity.
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    restored_groups, _ = _build_group_mapping(restored)
    with pytest.raises(ValueError, match="does not own"):
        generator.generate(_program(restored_groups["group_a"]), workspace=a)
    fresh = codegen.GMPipeLayoutEvidence(restored).bind_root(_get(restored, "group_a"))
    assert generator.generate(_program(restored_groups["group_a"]), workspace=fresh)
    # The context remains valid after its memo owner and original wrapper die.
    module_a = modules["group_a"]
    del owner, program
    gc.collect()
    assert generator.generate(module_a, workspace=a)


def test_unmanaged_direct_spelling_has_no_managed_pointer_or_offsets(gm_backend):
    first = _function("first", [], [_setup(3, 512)])
    second = _function("second", [], [_setup(0, 1024)])
    program = _program([first, second])
    generator = codegen.PTOCodegen()
    text = generator.generate(program, emit_source_loc=False)
    assert text.count("pto.aic_initialize_pipe") == 2
    assert "gm_slot_buffer" not in text
    assert "pto.addptr" not in text
    # Each body is identical to its independently emitted legacy form. Other
    # roots' geometry must not introduce a pointer, offset or new instruction.
    for function in (first, second):
        single = generator.generate(_program([function]), emit_source_loc=False)
        marker = f"func.func @{function.name}"
        body = single[single.index(marker) :].rsplit("}", 1)[0].strip()
        assert body in text
    owner = codegen.GMPipeLayoutEvidence(program)
    with pytest.raises(ValueError, match="one original injected formal"):
        owner.bind_root(first)
    # Legacy spelling still preserves the original numeric layout refusal.
    conflict = _function("conflict", [], [_setup(3, 1024)])
    with pytest.raises(ValueError, match="inconsistent slot_size"):
        generator.generate(_program([first, conflict]))
    # A mixed module cannot use the unmanaged compatibility path.
    managed = _function("managed", [_workspace()], [_setup(0, 1024)])
    with pytest.raises(ValueError, match="one original injected formal"):
        generator.generate(_program([first, managed]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_unrelated_peer_definition_does_not_acquire_workspace_ownership(gm_backend):
    """Bundling a peer definition cannot manufacture a shared launch allocation."""
    root = _function("root", [_workspace()], [_setup(pipe_id=0)])
    peer = _function("peer", [_workspace()], [_setup(pipe_id=1)])
    original = _program([root, peer])
    context = codegen.GMPipeLayoutEvidence(original).bind_root(root)
    with pytest.raises(ValueError, match="does not cover an emitted original setup"):
        codegen.PTOCodegen().generate(original, workspace=context)


def test_equivalent_peer_definitions_have_independent_codegen_bindings(gm_backend):
    root = _function("root", [_workspace()], [_setup(pipe_id=0)])
    peer = _function("peer", [_workspace()], [_setup(pipe_id=0)])
    program = _program([root, peer])
    evidence = codegen.GMPipeLayoutEvidence(program)
    with pytest.raises(ValueError, match="does not cover"):
        codegen.PTOCodegen().generate(program, workspace=evidence.bind_root(root))
    context = evidence.infer_codegen_context()
    assert context.bytes == evidence.bind_root(root).bytes
    assert codegen.PTOCodegen().generate(program, workspace=context).count("pto.aic_initialize_pipe") == 2


def test_equal_workspace_extent_does_not_prove_equal_native_ring_geometry(gm_backend):
    root = _function("root", [_workspace()], [_setup(slot_bytes=512)])
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, DataType.INT32, span)
    different_geometry = ir.EvalStmt(
        ir.create_op_call(
            "system.aic_initialize_pipe",
            [zero, zero],
            {"id": 0, "dir_mask": 3, "slot_size": 256, "slot_num": 4},
            span,
        ),
        span,
    )
    peer = _function("peer", [_workspace()], [different_geometry])
    evidence = codegen.GMPipeLayoutEvidence(_program([root, peer]))
    assert evidence.bind_root(root).bytes == evidence.bind_root(peer).bytes
    with pytest.raises(ValueError, match="inconsistent slot counts"):
        evidence.infer_codegen_context()


def test_peer_inventory_cannot_shrink_the_original_root_layout(gm_backend):
    root_workspace = _workspace()
    helper = _function("helper", [_workspace()], [_setup(pipe_id=0)])
    root = _function("root", [root_workspace], [_setup(pipe_id=1), _edge("helper", root_workspace)])
    peer = _function("peer", [_workspace()], [_setup(pipe_id=1)])
    evidence = codegen.GMPipeLayoutEvidence(_program([root, helper, peer]))
    assert evidence.bind_root(root).offsets[(1, 3)] == 2048
    # A narrowed inventory could independently place both visible setups at
    # offset zero. It must not override the actual root's larger allocation.
    narrowed = _program([root, peer])
    with pytest.raises(ValueError, match="differs from the original launch root"):
        _ungrouped_workspace_context(evidence, root, narrowed, True)


@pl.program
class ReusedReciprocalGroups:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, weight: pl.Tensor[[16, 16], pl.FP16]):
        local = pl.reserve_buffer(name="to_cube", size=4096, base=pl.AUTO)
        peer = pl.import_peer_buffer(name="to_vector", peer_func="vector")
        pl.aic_initialize_pipe(peer, local, dir_mask=3, slot_size=1024, slot_num=2)
        value: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
        matrix = pl.load(weight, [0, 0], [16, 16])
        result = pl.matmul(value, matrix)
        pl.tfree_to_aiv(value)
        pl.tpush_to_aiv(result, split=0)

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, source: pl.Tensor[[16, 16], pl.FP16], output: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
        local = pl.reserve_buffer(name="to_vector", size=4096, base=pl.AUTO)
        peer = pl.import_peer_buffer(name="to_cube", peer_func="cube")
        pl.aiv_initialize_pipe(local, peer, dir_mask=3, slot_size=1024, slot_num=2)
        value = pl.load(source, [0, 0], [16, 16])
        pl.tpush_to_aic(value, split=0)
        result: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        output = pl.store(result, [0, 0], output)
        pl.tfree_to_aic(result)
        return output

    @pl.function(type=pl.FunctionType.Group)
    def first(
        self,
        source: pl.Tensor[[16, 16], pl.FP16],
        weight: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        self.cube(weight)
        return self.vector(source, output)

    @pl.function(type=pl.FunctionType.Group)
    def second(
        self,
        source: pl.Tensor[[16, 16], pl.FP16],
        weight: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        self.cube(weight)
        return self.vector(source, output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        source: pl.Tensor[[16, 16], pl.FP16],
        weight: pl.Tensor[[16, 16], pl.FP16],
        first_output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        second_output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        first_output = self.first(source, weight, first_output)
        second_output = self.second(source, weight, second_output)
        return first_output, second_output


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
def test_default_codegen_reused_reciprocal_groups_preserves_original_entries(gm_backend, tmp_path, planner):
    """Peer definitions in original public units use equivalent layouts, not the Group's allocation."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is required for the public backend integration regression")
    with passes.PassContext([], memory_planner=planner, enable_pypto_autosync=False):
        transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(
            ReusedReciprocalGroups
        )
        groups, ungrouped = _build_group_mapping(transformed)
        assert {"first", "second"} <= groups.keys()
        assert {"cube", "vector"} <= {function.name for function in ungrouped}
        for members in groups.values():
            assert all(function.name not in {"cube", "vector"} for function in members)
        ir.compile(ReusedReciprocalGroups, output_dir=str(tmp_path), dump_passes=False, skip_ptoas=False)
    emitted = {path.stem for path in (tmp_path / "ptoas").glob("*.pto")}
    assert {"first", "second", "cube", "vector"} <= emitted
    assert list((tmp_path / "kernels").rglob("*.cpp"))
