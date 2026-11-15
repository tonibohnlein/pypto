# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit logical-producer policy survives contexts and compiled-artifact identity."""

import importlib
from contextlib import nullcontext
from typing import Any

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.compile_profiling import CompileProfiler
from pypto.ir.pass_manager import PassManager
from pypto.jit.cache import make_cache_key
from pypto.runtime.debug.pto_rebuild import _ptoas_flags

from .sync_closed_command_oracle import check_closed_commands
from .test_autosync_switch import clone
from .test_closed_prefix_sync import PIPES, SPAN, Services, command_fixture, seq
from .test_conservative_sync_dataflow import conservative_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
POLICY = "conservative_prefix"


def context(policy=POLICY, instruments=()):
    return passes.PassContext(list(instruments), enable_pypto_autosync=True, autosync_policy=policy)


def program(function):
    return ir.Program([function], "Policy", function.span)


def only_function(value):
    return next(iter(value.functions.values()))


def active_context():
    current = passes.PassContext.current()
    assert current is not None
    return current


def test_policy_defaults_validation_and_nested_restore():
    default = passes.PassContext([])
    assert default.get_autosync_policy() == "exact"
    assert not default.get_enable_pypto_autosync()
    with context():
        assert active_context().get_autosync_policy() == POLICY
        with passes.PassContext([]):
            assert active_context().get_autosync_policy() == "exact"
            assert not active_context().get_enable_pypto_autosync()
        assert active_context().get_autosync_policy() == POLICY
    for unknown in ("", "fallback", "EXACT", "conservative"):
        with pytest.raises(ValueError, match="autosync_policy must be"):
            passes.PassContext([], autosync_policy=unknown)


@pytest.mark.parametrize("first,second", [("exact", POLICY), (POLICY, "exact")])
def test_manager_refuses_policy_change_after_construction(first, second):
    with context(first):
        manager = PassManager(ir.OptimizationStrategy.Default)
    with context(second), pytest.raises(RuntimeError, match="autosync contract changed"):
        manager.run_passes(program(conservative_fixture()))


@pytest.mark.parametrize("mode", ["plain", "dump", "profile", "dump_profile"])
def test_dump_and_profiling_contexts_preserve_policy(mode, tmp_path):
    seen = []

    def observe(_pass, _program):
        seen.append(active_context().get_autosync_policy())

    instrument = passes.CallbackInstrument(before_pass=observe, name="PolicyObserver")
    with context(instruments=[instrument]):
        manager = PassManager(ir.OptimizationStrategy.Default)
        # Exercise the real context-cloning paths around one ordinary pass.
        # Whole-pipeline transformations are covered by separate production tests.
        manager._pipeline = passes.PassPipeline()
        manager._pipeline.add_pass(passes.optimize_sync())
        profiler = CompileProfiler() if "profile" in mode else nullcontext()
        with profiler:
            manager.run_passes(
                program(conservative_fixture()), dump_ir="dump" in mode, output_dir=str(tmp_path / mode)
            )
        assert active_context().get_autosync_policy() == POLICY
    assert seen == [POLICY]


def test_compile_pipeline_context_clone_preserves_policy(monkeypatch):
    compile_module = importlib.import_module("pypto.ir.compile")
    seen = []

    class ObservePipeline:
        def run_passes(self, source, **_kwargs):
            active = active_context()
            seen.append(
                (
                    active.get_enable_pypto_autosync(),
                    active.get_autosync_policy(),
                    active.get_autosync_disjoint_gm(),
                )
            )
            return source

    monkeypatch.setattr(
        compile_module.PassManager, "get_strategy", lambda *_args, **_kwargs: ObservePipeline()
    )
    with context():
        compile_module._run_pass_pipeline(program(conservative_fixture()), operation="compile")
    assert seen == [(True, POLICY, False)]


def _policy_kernel(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        y = pl.slice(x, [32, 128], [0, 0])
    return y


def test_jit_policy_separates_artifacts_and_reuses_matching_policy(monkeypatch):
    compile_module = importlib.import_module("pypto.ir.compile")
    policies = []

    def compile_program(_source, **_kwargs):
        policies.append(active_context().get_autosync_policy())
        return object()

    monkeypatch.setattr(compile_module, "compile", compile_program)
    kernel = pl.jit(_policy_kernel)
    with context("exact"):
        exact = kernel.compile()
        assert kernel.compile() is exact
    with context():
        conservative = kernel.compile()
        assert kernel.compile() is conservative
    with context("exact"):
        assert kernel.compile() is exact
    assert conservative is not exact and policies == ["exact", POLICY]
    # Intentionally heterogeneous keyword bundle shared by these two calls.
    kwargs: dict[str, Any] = dict(
        source_hash="source",
        param_names=[],
        tensor_shapes={},
        tensor_dtypes={},
        dynamic_dims=set(),
        enable_pypto_autosync=True,
    )
    assert make_cache_key(**kwargs) != make_cache_key(**kwargs, autosync_policy=POLICY)


@pytest.mark.parametrize("kind", ["for", "while"])
def test_explicit_policy_produces_without_disjoint_gm_and_split_lowering_is_policy_independent(kind):
    original = conservative_fixture(kind)
    before = ir.serialize(original)
    with context():
        inserted = passes.insert_sync()(program(original))
    logical = only_function(ir.deserialize(ir.serialize(inserted)))
    with context("exact"):
        physical = only_function(passes.lower_sync()(program(logical)))
    assert ir.serialize(original) == before
    assert dict(physical.attrs)["pypto_autosync"]
    check_closed_commands(
        logical, physical, {3: 3, 4: 2, 5: 0}, range(8), tensor_bases={2: 4096}, compare_commands=False
    )


@pytest.mark.parametrize("kind", ["for", "while"])
def test_realize_policy_is_explicit_and_default_body_is_preserved(kind):
    original = conservative_fixture(kind)
    with context("exact"):
        exact = only_function(passes.realize_sync()(program(original)))
    assert not dict(exact.attrs)["pypto_autosync"]
    assert "Disjoint GM" in dict(exact.attrs)["autosync_fallback_reason"]
    ir.assert_structural_equal(exact.body, original.body)
    with context():
        logical = only_function(passes.insert_sync()(program(original)))
        physical = only_function(passes.realize_sync()(program(original)))
    assert dict(physical.attrs)["pypto_autosync"]
    check_closed_commands(logical, physical, {3: 2, 4: 2, 5: 1}, range(8), compare_commands=False)


def test_local_only_commands_do_not_fabricate_manual_ownership():
    logical = clone(
        command_fixture(), body=seq(*Services().closed(PIPES[0], PIPES[0]), ir.ReturnStmt([], SPAN))
    )
    with context():
        physical = only_function(passes.lower_sync()(program(logical)))
    assert not dict(physical.attrs).get("pypto_autosync", False)
    assert "system.set_flag" not in str(physical) and "system.wait_flag" not in str(physical)


def test_debug_rebuild_keeps_downstream_sync_for_mixed_ownership():
    manual = "pto.set_flag [<PIPE_MTE2>, <PIPE_V>], 2"
    delegated = "// delegated function has no explicit flag marker"
    with context():
        assert "--enable-insert-sync" in _ptoas_flags(manual + "\n" + delegated)
    with context("exact"):
        assert _ptoas_flags(manual + "\n" + delegated) == ["--enable-insert-sync", "--pto-level=level2"]
