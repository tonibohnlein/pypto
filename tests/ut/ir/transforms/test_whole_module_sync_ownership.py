# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit module ownership includes empty/native-only plans without fake flags."""

import json
import subprocess
from pathlib import Path

import pytest
from pypto import codegen, ir, passes
from pypto._function_attrs import AUTOSYNC_WHOLE_MODULE_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.backend import pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend._sync_ownership import (
    OWNED_PTO_MARKER,
    make_sync_ownership,
    owned_rebuild_flags,
    validate_sync_ownership,
)
from pypto.compile_profiling import StageRecord
from pypto.ir.op import sync
from pypto.pypto_core import ir as core_ir
from pypto.runtime.debug import pto_rebuild

from .sync_matmul_cases import capture
from .test_runtime_valid_broadcast import metadata_fixture
from .test_token_sync_lowering import _check_actual_tokens

pytestmark = pytest.mark.usefixtures("ascend_backend")
S = ir.Span.unknown()


def empty_function(name="empty", kind=ir.FunctionType.AIV):
    return ir.Function(
        name, [], [], ir.SeqStmts([ir.ReturnStmt([], S)], S), S, kind, ir_stage=ir.FunctionIRStage.Buffer
    )


def apply(program, *, realize=False):
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        return (passes.realize_sync if realize else passes.lower_sync)(require_whole_module=True)(program)


def calls(function):
    class Collect(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.names = []

        def visit_call(self, op):
            self.names.append(op.op.name)
            super().visit_call(op)

    result = Collect()
    result.visit_function(function)
    return result.names


@pytest.mark.parametrize("realize", [False, True])
def test_explicit_empty_module_is_owned_without_commands_and_roundtrips(realize):
    original = empty_function()
    program = ir.Program([original], "Empty", S)
    with passes.PassContext([]):
        ordinary = passes.lower_sync()(program).get_function("empty")
    assert ordinary is not None and not dict(ordinary.attrs).get(PYPTO_AUTOSYNC_ATTR, False)
    result = apply(program, realize=realize).get_function("empty")
    assert result is not None
    assert dict(result.attrs)[PYPTO_AUTOSYNC_ATTR] and dict(result.attrs)[AUTOSYNC_WHOLE_MODULE_ATTR]
    assert calls(result) == []
    ir.assert_structural_equal(result, ir.deserialize(ir.serialize(result)))
    ir.assert_structural_equal(original.body, result.body)
    report = json.loads(
        pto_backend._autosync_report(ir.Program([result], "Owned", S))["report/autosync.json"]
    )
    assert report["downstream_autosync_policy"] == "per-unit-manifest"
    assert report["kernels"][0]["reason"] == "complete module invocation contract"


@pytest.mark.parametrize("realize", [False, True])
def test_whole_module_respects_kernel_memory_planner_override(realize):
    leaf = empty_function()
    delegated = ir.Function(
        leaf.name,
        leaf.params,
        leaf.return_types,
        leaf.body,
        S,
        leaf.func_type,
        ir_stage=leaf.ir_stage,
        attrs={"ptoas_plan_memory": True},
    )
    program = ir.Program([delegated], "Delegated", S)
    with pytest.raises(ValueError, match="requires PyPTO-placed storage"):
        apply(program, realize=realize)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        result = passes.realize_sync()(program).get_function("empty")
    assert result is not None and not result.attrs[PYPTO_AUTOSYNC_ATTR]


def test_explicit_request_does_not_reprocess_an_owned_empty_module():
    owned = apply(ir.Program([empty_function()], "Owned", S))
    before = ir.serialize(owned)
    with pytest.raises(ValueError, match="cannot reprocess an owned physical plan"):
        apply(owned)
    assert ir.serialize(owned) == before


@pytest.mark.parametrize(
    "kind,reason", [("mismatch", "matching runtime rows"), ("unguarded", "positive-row guard")]
)
def test_whole_module_ownership_requires_native_runtime_metadata_proof(kind, reason):
    program = ir.Program([metadata_fixture(kind)], "UnprovedMetadata", S)
    before = ir.serialize(program)
    with pytest.raises(ValueError, match=reason):
        apply(program)
    assert ir.serialize(program) == before


def barrier_program(*, manual_before_fill=False):
    def ci(v):
        return ir.ConstInt(v, ir.DataType.INDEX, S)

    def op(name, args, typ=None):
        if typ is not None:
            return core_ir._create_internal_op_call(name, args, {}, typ, S)
        return core_ir._create_internal_op_call(name, args, {}, S)

    ty = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    buffer = ir.Var("v", ty, S)
    allocation = ir.AssignStmt(buffer, op("buffer.alloc", [ir.MakeTuple([], S), ci(0)], ty), S)
    full = ir.EvalStmt(op("buffer.full", [ir.ConstFloat(1.0, ir.DataType.FP32, S), buffer]), S)
    absolute = ir.EvalStmt(op("buffer.abs", [buffer, buffer]), S)
    body: list[ir.Stmt] = [allocation]
    if manual_before_fill:
        # Wait for work on an independent buffer before starting the fill.
        # The storage hazard on the original buffer occurs later, before abs.
        prior = ir.Var("prior", ty, S)
        token = ir.Var("ready", ir.SyncTokenType(ir.PipeType.V), S)
        body.extend(
            [
                ir.AssignStmt(prior, op("buffer.alloc", [ir.MakeTuple([], S), ci(4096)], ty), S),
                ir.EvalStmt(op("buffer.full", [ir.ConstFloat(2.0, ir.DataType.FP32, S), prior]), S),
                ir.AssignStmt(token, sync.record(ir.PipeType.V), S),
                ir.EvalStmt(sync.wait(token, ir.PipeType.V), S),
            ]
        )
    original = ir.Function(
        "barrier_only",
        [],
        [],
        ir.SeqStmts([*body, full, absolute, ir.ReturnStmt([], S)], S),
        S,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return original, ir.Program([original], "Barrier", S)


def test_hybrid_lower_preserves_manual_barrier_before_independent_fill():
    original, program = barrier_program(manual_before_fill=True)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        lowered = passes.lower_sync()(program).get_function(original.name)
    assert lowered is not None
    assert dict(lowered.attrs)[PYPTO_AUTOSYNC_ATTR] is False
    full, barrier, absolute = (ir.get_op(op).name for op in ("buffer.full", "system.bar_v", "buffer.abs"))
    assert [op for op in calls(lowered) if op in {full, barrier, absolute}] == [full, barrier, full, absolute]


@pytest.mark.ptoas_compile
def test_ptoas_preserves_manual_barrier_and_orders_later_storage_hazard(tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    _, program = barrier_program(manual_before_fill=True)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        lowered = passes.lower_sync()(program)

    source = tmp_path / "barriers.pto"
    source.write_text(codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False))
    output = tmp_path / "barriers.cpp"
    pto_backend._run_ptoas(
        str(source), str(output), ["--pto-arch=a2", "--pto-level=level3", "--enable-insert-sync=true"]
    )
    _, between_fills, after_fills = output.read_text().split("TEXPANDS(")
    assert "pipe_barrier(PIPE_V)" in between_fills
    assert after_fills.index("pipe_barrier(PIPE_V)") < after_fills.index("TABS(")


def test_barrier_only_module_keeps_actual_token_order_and_exit_completion():
    original, program = barrier_program()
    with passes.PassContext([], autosync_disjoint_gm=True):
        logical_program = passes.insert_sync()(program)
    logical = logical_program.get_function(original.name)
    physical = apply(logical_program).get_function(original.name)
    assert logical is not None and physical is not None
    names = calls(physical)
    assert ir.get_op("system.set_flag").name not in names
    assert ir.get_op("system.wait_flag").name not in names
    assert ir.get_op("system.bar_v").name in names
    assert names[-1] == ir.get_op("system.bar_all").name
    _check_actual_tokens(original, logical, physical)
    assert "pto.barrier" in codegen.PTOCodegen().generate(ir.Program([physical], "Physical", S))

    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        delegated_lower = passes.lower_sync(require_whole_module=False)(logical_program).get_function(
            original.name
        )
        delegated_realize = passes.realize_sync(require_whole_module=False)(program).get_function(
            original.name
        )
    assert delegated_lower is not None and delegated_realize is not None
    assert dict(delegated_lower.attrs)[PYPTO_AUTOSYNC_ATTR] is False
    assert dict(delegated_realize.attrs)[PYPTO_AUTOSYNC_ATTR] is False
    assert ir.get_op("system.bar_v").name in calls(delegated_lower)
    assert "SyncToken" not in str(delegated_lower)
    ir.assert_structural_equal(original.body, delegated_realize.body)


@pytest.mark.parametrize("realize", [False, True])
def test_explicit_flag_module_completes_last_store_and_preserves_default_plan(realize):
    original = capture("matmul_64")
    source = ir.Program([original], "FinalStore", S)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        logical_program = passes.insert_sync()(source)
        default = passes.lower_sync()(logical_program).get_function(original.name)
    result = apply(source if realize else logical_program, realize=realize).get_function(original.name)
    assert result is not None and default is not None
    ir.assert_structural_equal(result.body, default.body)
    assert dict(result.attrs)[AUTOSYNC_WHOLE_MODULE_ATTR]
    assert not dict(default.attrs).get(AUTOSYNC_WHOLE_MODULE_ATTR, False)
    assert calls(result)[-1] == ir.get_op("system.bar_all").name


@pytest.mark.parametrize("kind", [ir.FunctionType.SIMT, ir.FunctionType.Opaque, ir.FunctionType.InCore])
def test_unqualified_helper_refuses_entire_module_without_mutation(kind):
    original = ir.Program([empty_function(), empty_function("helper", kind)], "Mixed", S)
    before = ir.serialize(original)
    with pytest.raises(ValueError, match="Whole-module synchronization ownership"):
        apply(original)
    assert ir.serialize(original) == before


def unit(function, source="module {}", *, domains=None, requested=True):
    return pto_backend._CodegenUnit(
        function.name,
        source,
        [function],
        False,
        StageRecord(name="test", start=0),
        sync_domains=domains if domains is not None else [(function.name, True)],
        sync_whole_requested=requested,
    )


def test_unit_inventory_refuses_mixed_helpers_before_running_compiler(tmp_path, monkeypatch):
    function = apply(ir.Program([empty_function()], "Owned", S)).get_function("empty")
    assert function is not None
    inventory = pto_backend._sync_domain_inventory(
        ir.Program([function, empty_function("helper", ir.FunctionType.SIMT)], "Mixed", S)
    )
    assert dict(inventory) == {"empty": True, "helper": False}
    monkeypatch.setattr(pto_backend, "_run_ptoas", lambda *a, **kw: pytest.fail("must not invoke compiler"))
    result = pto_backend._emit_unit(unit(function, domains=inventory), str(tmp_path), False)
    assert isinstance(result.error, ValueError)
    assert not list(tmp_path.glob("ptoas/*.pto"))


def test_owned_unit_manifest_controls_compiler_and_raw_artifact(tmp_path, monkeypatch):
    observed = []

    def downstream(input_path, output_path, *, ptoas_flags):
        observed.append(ptoas_flags)
        assert Path(input_path).read_text().startswith(OWNED_PTO_MARKER)
        Path(output_path).write_text("AICORE void empty() {}")

    monkeypatch.setattr(pto_backend, "_run_ptoas", downstream)
    result = pto_backend._emit_unit(unit(empty_function()), str(tmp_path), False)
    assert result.error is None
    assert observed and "--enable-insert-sync=false" in observed[0]
    source = (tmp_path / "ptoas/empty.pto").read_text()
    manifest = json.loads(result.files["ptoas/empty.sync.json"])
    assert validate_sync_ownership(source, manifest)
    raw = pto_backend._emit_unit(unit(empty_function()), str(tmp_path), True)
    assert raw.error is None
    assert "kernels/aiv/empty.sync.json" in raw.files
    assert raw.files["kernels/aiv/empty.pto"].startswith(OWNED_PTO_MARKER)
    assert owned_rebuild_flags(tmp_path / "ptoas/empty.pto", source) == observed[0]


def test_owned_unit_rejects_unqualified_native_target_options(tmp_path, monkeypatch):
    class OtherTarget:
        def get_extra_ptoas_flags(self):
            return ["--pto-arch", "a5"]

    monkeypatch.setattr(pto_backend._backend_core, "get_handler", lambda: OtherTarget())
    result = pto_backend._emit_unit(unit(empty_function()), str(tmp_path), True)
    assert isinstance(result.error, ValueError)
    assert "qualified A2/A3" in str(result.error)
    assert not result.files


def test_owned_unit_cannot_replan_the_certified_physical_storage(tmp_path):
    result = pto_backend._emit_unit(unit(empty_function()), str(tmp_path), True, passes.MemoryPlanner.PTOAS)
    assert isinstance(result.error, ValueError)
    assert "preserved physical placement" in str(result.error)
    assert not result.files


def test_grouped_raw_artifact_has_matching_manifest_path(tmp_path):
    grouped = unit(empty_function())
    grouped.is_group = True
    result = pto_backend._emit_unit(grouped, str(tmp_path), True)
    assert result.error is None
    assert "kernels/empty.pto" in result.files and "kernels/empty.sync.json" in result.files


def test_owned_debug_rebuild_preserves_options_and_refuses_changed_source(tmp_path, monkeypatch):
    source = OWNED_PTO_MARKER + "module {}"
    pto = tmp_path / "owned.pto"
    pto.write_text(source)
    manifest = make_sync_ownership(source, [("owned", True)], [])
    manifest["compile_flags"] = ["--enable-insert-sync=false", "--pto-level=level3", "--pto-arch=a2"]
    pto.with_suffix(".sync.json").write_text(json.dumps(manifest))
    observed = []
    monkeypatch.setattr(pto_rebuild, "_check_ptoas_version", lambda path: None)

    def run(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(pto_rebuild.subprocess, "run", run)
    pto_rebuild._run_ptoas("ptoas", pto, tmp_path / "owned.cpp")
    assert "--enable-insert-sync=false" in observed[0]
    assert "--pto-level=level3" in observed[0] and "--pto-arch=a2" in observed[0]
    pto.write_text(source + "\n// edited")
    with pytest.raises(ValueError, match="Owned PTO was edited"):
        pto_rebuild._run_ptoas("ptoas", pto, tmp_path / "owned.cpp")
    assert len(observed) == 1
    pto.with_suffix(".sync.json").unlink()
    with pytest.raises(ValueError, match="paired synchronization"):
        owned_rebuild_flags(pto, pto.read_text())


def test_owned_marker_cannot_be_rebuilt_as_an_unowned_artifact(tmp_path):
    source = OWNED_PTO_MARKER + "module {}"
    pto = tmp_path / "owned.pto"
    pto.write_text(source)
    pto.with_suffix(".sync.json").write_text(json.dumps(make_sync_ownership(source, [("owned", False)], [])))
    with pytest.raises(ValueError, match="complete synchronization ownership"):
        owned_rebuild_flags(pto, source)
