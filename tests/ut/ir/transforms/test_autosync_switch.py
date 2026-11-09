# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compilation ownership, cache isolation and independent physical checks."""

import json
import re
from pathlib import Path

import pytest
import torch
from pypto import codegen, ir, passes, testing
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType, pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _get_ptoas_flags, _run_ptoas
from pypto.ir.pass_manager import PassManager
from pypto.jit.cache import make_cache_key
from pypto.runtime import RunConfig

from .sync_matmul_cases import CASES, capture, load_case, staged_fixture
from .sync_physical_oracle import run_physical

pytestmark = pytest.mark.usefixtures("ascend_backend")


def context(instruments=()):
    return passes.PassContext(list(instruments), enable_pypto_autosync=True, autosync_disjoint_gm=True)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("dump", [False, True])
def test_compile_owns_sync_after_context_exit(name, dump, tmp_path, monkeypatch):
    observed = []
    transformed = []

    def after_pass(pass_obj, program):
        if pass_obj.get_name() == "RealizeSync":
            transformed.extend(program.functions.values())

    def downstream(input_path, output_path, *, ptoas_flags=None):
        observed.append(ptoas_flags)
        assert "pto.set_flag" in Path(input_path).read_text()
        Path(output_path).write_text("// Host unit-test compiler stub\n")

    monkeypatch.setattr(pto_backend, "_run_ptoas", downstream)
    m, k, n = CASES[name][1]
    original = capture(name)
    with context([passes.CallbackInstrument(after_pass=after_pass)]):
        compiled = load_case(name).compile(
            torch.empty(m, k),
            torch.empty(k, n),
            torch.empty(m, n),
            config=RunConfig(
                arch=BackendType.Ascend910,
                codegen_only=True,
                save_kernels_dir=str(tmp_path),
                dump_passes=dump,
                dump_ptoas_passes=dump,
                compile_profiling=True,
            ),
        )
    assert observed and all("--enable-insert-sync" in flags for flags in observed)
    assert all(("--mlir-print-ir-after-all" in flags) == dump for flags in observed)
    assert all("--mlir-print-ir-module-scope" not in flags for flags in observed)
    assert compiled.program is not None
    functions = [f for f in transformed if f.func_type == ir.FunctionType.AIC]
    assert len(functions) == 1
    function = functions[0]
    assert dict(function.attrs)[PYPTO_AUTOSYNC_ATTR] is True
    run_physical(original, function, range(6))
    report = json.loads((tmp_path / "report/autosync.json").read_text())
    assert report["downstream_autosync_enabled"]
    assert report["kernels"] == [
        {"function": function.name, "selected_route": "pypto", "reason": "complete literal flag plan"}
    ]
    assert "--enable-insert-sync" in _get_ptoas_flags()
    path = tmp_path / "owned.msgpack"
    ir.serialize_to_file(function, str(path))
    restored = ir.deserialize_from_file(str(path))
    assert isinstance(restored, ir.Function)
    assert dict(restored.attrs)[PYPTO_AUTOSYNC_ATTR] is True


def test_context_contract_and_pipeline_identity():
    assert not passes.PassContext([]).get_enable_pypto_autosync()
    with context():
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        assert manager.pass_names[-2:] == ["LowerTileToBuffer", "RealizeSync"]
    with pytest.raises(RuntimeError, match="autosync contract changed"):
        manager._check_planner_consistency()
    assert "RealizeSync" not in PassManager.get_strategy(ir.OptimizationStrategy.Default).pass_names


def test_cache_keys_distinguish_owner_and_contract():
    baseline = make_cache_key("same", [], {}, {}, set())
    contract = make_cache_key("same", [], {}, {}, set(), autosync_disjoint_gm=True)
    manual = make_cache_key("same", [], {}, {}, set(), enable_pypto_autosync=True, autosync_disjoint_gm=True)
    assert len({baseline, contract, manual}) == 3


def test_empty_candidate_delegates_original_body():
    original = staged_fixture(0)
    with context():
        result = passes.realize_sync()(ir.Program([original], "empty", original.span))
    function = next(iter(result.functions.values()))
    run_physical(original, function, range(6))
    assert not dict(function.attrs)[PYPTO_AUTOSYNC_ATTR]
    assert "no SET/WAIT marker" in dict(function.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(original.body, function.body)


def clone(function, name=None, body=None, kind=None):
    return ir.Function(
        name or function.name,
        list(zip(function.params, function.param_directions)),
        function.return_types,
        body or function.body,
        function.span,
        type=kind or function.func_type,
        level=function.level,
        role=function.role,
        attrs=dict(function.attrs),
        ir_stage=function.ir_stage,
    )


def apply_hybrid(functions, **kwargs):
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True, **kwargs):
        program = passes.realize_sync()(ir.Program(functions, "hybrid", functions[0].span))
    return {f.name: f for f in program.functions.values()}


def test_missing_contract_and_placement_delegate_without_insertion():
    original = capture("matmul_64")
    with passes.PassContext([], enable_pypto_autosync=True):
        result = passes.realize_sync()(ir.Program([original], "no_contract", original.span))
    unchanged = next(iter(result.functions.values()))
    assert not dict(unchanged.attrs)[PYPTO_AUTOSYNC_ATTR]
    assert "contract was not supplied" in dict(unchanged.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(original.body, unchanged.body)
    unchanged = apply_hybrid([original], memory_planner=passes.MemoryPlanner.PTOAS)[original.name]
    assert "physically placed" in dict(unchanged.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(original.body, unchanged.body)


def test_empty_aiv_and_tile_ir_delegate_without_insertion():
    span = ir.Span.unknown()
    vector = ir.Function(
        "vector",
        [],
        [],
        ir.ReturnStmt([], span),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    result = apply_hybrid([vector])[vector.name]
    assert "no SET/WAIT marker" in dict(result.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(vector.body, result.body)
    with passes.PassContext([], enable_buffer_ir=False):
        program = load_case("matmul_64").lower(
            torch.empty(64, 64),
            torch.empty(64, 64),
            torch.empty(64, 64),
            config=RunConfig(arch=BackendType.Ascend910),
        )
    tile = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    assert tile.ir_stage == ir.FunctionIRStage.Functional
    result = apply_hybrid([tile])[tile.name]
    assert "physically placed" in dict(result.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(tile.body, result.body)


def mixed_functions():
    owned = clone(capture("matmul_64"), "handled")
    # Existing barriers alone do NOT cause PTOAS's explicit-sync skip. The
    # prototype refuses this input, leaving PTOAS to cover its cross-pipe edges.
    barrier = ir.EvalStmt(ir.Call(ir.get_op("system.bar_m"), [], ir.VoidType(), owned.span), owned.span)
    assert isinstance(owned.body, ir.SeqStmts)
    fallback = clone(owned, "delegated", ir.SeqStmts([barrier, *owned.body.stmts], owned.span))
    return owned, fallback


def test_mixed_module_preserves_complete_child_and_refused_body(tmp_path, monkeypatch):
    original, fallback = mixed_functions()
    results = apply_hybrid([original, fallback])
    assert dict(results["handled"].attrs)[PYPTO_AUTOSYNC_ATTR]
    assert not dict(results["delegated"].attrs)[PYPTO_AUTOSYNC_ATTR]
    assert dict(results["delegated"].attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(fallback.body, results["delegated"].body)
    run_physical(original, results["handled"], range(6))
    calls = []

    def downstream(input_path, output_path, *, ptoas_flags=None):
        calls.append(ptoas_flags)
        Path(output_path).write_text("// Host unit-test compiler stub\n")

    monkeypatch.setattr(pto_backend, "_run_ptoas", downstream)
    files = {}
    pto_backend._emit_group_output(files, "mixed", list(results.values()), "module {}", str(tmp_path), False)
    assert len(calls) == 1 and "--enable-insert-sync" in calls[0]
    assert "--enable-insert-sync=false" not in calls[0]


def cpp_body(text, name):
    match = re.search(r"AICORE void " + name + r"\(", text)
    assert match is not None
    begin = text.index("{", match.end())
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin:end]


@pytest.mark.ptoas_compile
def test_real_ptoas_skips_owned_function_only(tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    original, fallback = mixed_functions()
    results = apply_hybrid([original, fallback])

    def compile_functions(functions, label, enabled):
        program = ir.Program(functions, "hybrid", original.span)
        source, output = tmp_path / f"{label}.pto", tmp_path / f"{label}.cpp"
        source.write_text(codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False))
        _run_ptoas(
            str(source),
            str(output),
            ["--pto-level=level3", "--pto-arch=a2", f"--enable-insert-sync={'true' if enabled else 'false'}"],
        )
        return output.read_text()

    combined = compile_functions(list(results.values()), "mixed", True)
    manual = compile_functions([results["handled"]], "manual", False)
    delegated = compile_functions([fallback], "delegated", True)
    unsynchronized = compile_functions([fallback], "unsynchronized", False)
    assert cpp_body(combined, "handled") == cpp_body(manual, "handled")
    assert cpp_body(combined, "delegated") == cpp_body(delegated, "delegated")
    assert cpp_body(delegated, "delegated") != cpp_body(unsynchronized, "delegated")
    assert "set_flag(" in cpp_body(delegated, "delegated")
    assert "wait_flag(" in cpp_body(delegated, "delegated")


def test_late_realization_refusal_keeps_original_body():
    # Three M operations: first/last overwrite one accumulator while the
    # middle operation uses another. A local barrier would add payload order,
    # so mathematics succeeds but the physical realization must refuse.
    class Nonadjacent(ir.IRMutator):
        def visit_eval_stmt(self, op):
            call = op.expr
            if not isinstance(call, ir.Call) or call.op.name != ir.get_op("buffer.matmul").name:
                return op
            accumulator = ir.Var("independent_acc", call.args[-1].type, op.span)
            address = ir.ConstInt(32768, ir.DataType.INDEX, op.span)
            allocation = ir.AssignStmt(
                accumulator,
                ir.Call(
                    ir.get_op("buffer.alloc"), [ir.MakeTuple([], op.span), address], accumulator.type, op.span
                ),
                op.span,
            )
            independent = ir.EvalStmt(
                ir.Call(
                    call.op, [*call.args[:-1], accumulator], dict(call.kwargs), None, call.type, call.span
                ),
                op.span,
            )
            overwrite = ir.EvalStmt(
                ir.Call(call.op, call.args, dict(call.kwargs), None, call.type, call.span), op.span
            )
            return ir.SeqStmts([allocation, op, independent, overwrite], op.span)

    original = Nonadjacent().visit_function(capture("matmul_64"))
    retained = testing.analyze_placed_sync(original, True)
    before = retained.describe()
    with pytest.raises(ValueError, match="order-equal same-pipe barrier"):
        retained.realize_physical("a2a3")
    assert retained.describe() == before
    result = apply_hybrid([original])[original.name]
    assert not dict(result.attrs)[PYPTO_AUTOSYNC_ATTR]
    assert "order-equal same-pipe barrier" in dict(result.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(original.body, result.body)


@pytest.mark.parametrize("attribute", ["external_source", "builtin_template_dir"])
def test_external_ownership_is_not_reported_as_ptoas(attribute):
    original = capture("matmul_64")
    external = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        original.body,
        original.span,
        type=original.func_type,
        attrs={attribute: "externally-owned"},
        ir_stage=original.ir_stage,
    )
    result = apply_hybrid([external])[external.name]
    assert PYPTO_AUTOSYNC_ATTR not in dict(result.attrs)
    ir.assert_structural_equal(external, result)


def test_owned_plan_reprocessing_is_not_silently_delegated():
    original = capture("matmul_64")
    result = apply_hybrid([original])[original.name]
    with pytest.raises(ValueError, match="reprocess"):
        apply_hybrid([result])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
