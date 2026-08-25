# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "export_pypto_lib_dsa_corpus.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_export_pypto_lib_dsa_corpus", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


def _problem(*, instance: str, members: list[str], penalties: int = 1) -> dict:
    return {
        "instance": instance,
        "metadata": {"reuse_penalty_recognizer": "quadratic_route_frontier_v3"},
        "problem": {
            "pools": [{"id": 1, "name": "Vec", "capacity": 1024}],
            "buffers": [{"id": 0, "size": 32, "alignment": 32, "allowed_pools": [1]}],
            "constraints": {},
            "cost_model": {
                "reuse_penalties": [{"first": 0, "second": 1, "weight": 1} for _ in range(penalties)]
            },
            "pypto_structure": {"alias_classes": [{"members": members}]},
        },
    }


def test_semantic_fingerprint_ignores_only_alias_member_spelling() -> None:
    first = _problem(instance="kernel", members=["%inline1", "%inline2"])
    renamed = _problem(instance="kernel", members=["%inline8", "%inline9"])
    resized = _problem(instance="kernel", members=["%inline1", "%inline2"])
    resized["problem"]["buffers"][0]["size"] = 64

    assert exporter.semantic_problem_fingerprint(first) == exporter.semantic_problem_fingerprint(renamed)
    assert exporter.semantic_problem_fingerprint(first) != exporter.semantic_problem_fingerprint(resized)


def test_manifest_selects_only_successful_exports(tmp_path: Path) -> None:
    manifest = tmp_path / "status.tsv"
    manifest.write_text(
        "models/a.py\tEXPORTED\tsim\t0\t3\n"
        "models/b.py\tRESOURCE_BLOCKED\tsim\t200\t0\n"
        "models/c.py\tEXPORTED\tsim\t0\t2\n",
        encoding="utf-8",
    )
    assert exporter.read_export_plan(manifest, "a2a3sim") == [
        ("models/a.py", "a2a3sim"),
        ("models/c.py", "a2a3sim"),
    ]


def test_export_plan_preserves_real_platform_launch_mode(tmp_path: Path) -> None:
    manifest = tmp_path / "status.tsv"
    manifest.write_text(
        "models/sim.py\tEXPORTED\tfinal-sim\t0\t3\n"
        "models/a2a3.py\tEXPORTED\tfinal-a2a3\t0\t2\n"
        "models/a5.py\tEXPORTED\tfinal-a5\t0\t1\n",
        encoding="utf-8",
    )
    assert exporter.read_export_plan(manifest, "a2a3sim") == [
        ("models/sim.py", "a2a3sim"),
        ("models/a2a3.py", "a2a3"),
        ("models/a5.py", "a5"),
    ]


def test_export_status_launch_mode_is_reusable(tmp_path: Path) -> None:
    manifest = tmp_path / "export-status.tsv"
    manifest.write_text(
        "script\tstatus\tlaunch_mode\treturncode\tproblems\nmodels/a2a3.py\tEXPORTED\tfinal-a2a3\t0\t2\n",
        encoding="utf-8",
    )
    assert exporter.read_export_plan(manifest, "a2a3sim") == [("models/a2a3.py", "a2a3")]


def test_partial_export_launch_mode_is_reusable(tmp_path: Path) -> None:
    manifest = tmp_path / "export-status.tsv"
    manifest.write_text(
        "script\tstatus\tlaunch_mode\treturncode\tproblems\nmodels/a2a3.py\tPARTIAL\tfinal-a2a3\t1\t2\n",
        encoding="utf-8",
    )
    assert exporter.read_export_plan(manifest, "a2a3sim") == [("models/a2a3.py", "a2a3")]


def test_successful_entry_point_without_dsa_is_not_a_compile_failure() -> None:
    assert exporter._classify_export_status("EXPORTED", 0) == "NO_DSA"
    assert exporter._classify_export_status("EXPORTED", 2) == "EXPORTED"
    assert exporter._classify_export_status("FAILED", 0) == "FAILED"
    assert exporter._classify_export_status("FAILED", 2) == "PARTIAL"
    assert exporter._classify_export_status("TIMEOUT", 2) == "PARTIAL"


def test_golden_patch_injects_compile_only_dsa_configuration(tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_run(*_args, **kwargs):
        calls.append(kwargs)
        return "ok"

    golden = SimpleNamespace(run=fake_run, run_jit=fake_run)
    passes = SimpleNamespace(
        MemoryPlanner=SimpleNamespace(DSA="dsa"),
        DsaReusePenaltyRecognizer=SimpleNamespace(QUADRATIC="quadratic"),
    )
    exporter._patch_golden(golden, passes, tmp_path)

    assert golden.run(compile_cfg={"dump_passes": True}) == "ok"
    assert golden.run_jit() == "ok"
    direct, jit = calls
    assert direct["compile_only"] is True
    assert direct["save_data"] is False
    assert direct["compile_cfg"]["memory_planner"] == "dsa"
    assert direct["compile_cfg"]["dsa_reuse_penalty_recognizer"] == "quadratic"
    assert direct["compile_cfg"]["skip_ptoas"] is True
    assert direct["compile_cfg"]["dump_passes"] is False
    assert jit["compile_cfg"]["codegen_only"] is True
    assert "skip_ptoas" not in jit["compile_cfg"]
    assert (tmp_path / "call-001").is_dir()
    assert (tmp_path / "call-002").is_dir()


def test_compile_for_test_patch_is_optional_for_current_jit_api(tmp_path: Path) -> None:
    jit_decorator = SimpleNamespace(JITFunction=type("JITFunction", (), {}))
    passes = SimpleNamespace()

    assert exporter._patch_compile_for_test(jit_decorator, passes, tmp_path) is False


def test_driver_structure_counts_logical_submit_sites(tmp_path: Path) -> None:
    (tmp_path / "orchestration").mkdir()
    (tmp_path / "kernels" / "aiv").mkdir(parents=True)
    (tmp_path / "orchestration" / "driver.cpp").write_text(
        "rt_submit_aiv_task(0, first);\nrt_submit_mix_task(1, 2, second);\n",
        encoding="utf-8",
    )
    (tmp_path / "kernels" / "aiv" / "first.pto").write_text("func.func @first()\n", encoding="utf-8")

    assert exporter.inspect_driver_structure(tmp_path) == {
        "orchestration_files": 1,
        "submit_sites": 2,
        "emitted_kernels": 1,
        "driver_mode": "MULTI_SUBMIT_PARENT",
    }


def test_status_checkpoint_is_replaced_with_complete_rows(tmp_path: Path) -> None:
    path = tmp_path / "export-status.tsv"
    first = {"script": "models/a.py", "status": "EXPORTED"}
    second = {"script": "models/b.py", "status": "PARTIAL"}

    exporter._write_status_rows(path, [first])
    exporter._write_status_rows(path, [first, second])

    with path.open(encoding="utf-8", newline="") as source:
        assert list(csv.DictReader(source, delimiter="\t")) == [first, second]
    assert b"\r" not in path.read_bytes()
    assert not path.with_suffix(".tsv.pending").exists()


def test_inventory_deduplicates_semantic_exports(tmp_path: Path) -> None:
    first_root = tmp_path / "captures" / "first"
    second_root = tmp_path / "captures" / "second"
    first_root.mkdir(parents=True)
    second_root.mkdir(parents=True)
    first = _problem(instance="kernel", members=["%inline1"], penalties=1)
    second = _problem(instance="kernel", members=["%inline7"], penalties=1)
    (first_root / "first.dsa.json").write_text(json.dumps(first), encoding="utf-8")
    (second_root / "second.dsa.json").write_text(json.dumps(second), encoding="utf-8")

    invocations, unique, penalties = exporter.build_inventory(
        tmp_path,
        {first_root: "models/a.py", second_root: "models/b.py"},
    )

    assert (invocations, unique, penalties) == (2, 1, 1)
    assert len(list((tmp_path / "corpus" / "penalty-bearing").glob("*.dsa.json"))) == 1
    rows = (tmp_path / "invocations.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert b"\r" not in (tmp_path / "invocations.tsv").read_bytes()
    assert b"\r" not in (tmp_path / "unique-problems.tsv").read_bytes()


def test_missing_stale_source_is_terminal_without_aborting_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "pypto-lib"
    source = library / "models" / "present.py"
    source.parent.mkdir(parents=True)
    source.write_text("# present\n", encoding="utf-8")
    pypto_python = tmp_path / "pypto-python"
    pypto_python.mkdir()
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "models/missing.py\tEXPORTED\tfinal-a2a3sim\t0\t1\n"
        "models/present.py\tEXPORTED\tfinal-a2a3sim\t0\t1\n",
        encoding="utf-8",
    )

    def fake_export(command, **_kwargs):
        root = Path(command[command.index("--export-root") + 1])
        root.mkdir(parents=True)
        (root / "kernel.dsa.json").write_text(
            json.dumps(_problem(instance="kernel", members=["%inline1"])),
            encoding="utf-8",
        )
        return "EXPORTED", 0

    monkeypatch.setattr(exporter, "_run_export_subprocess", fake_export)
    arguments = SimpleNamespace(
        output_root=tmp_path / "output",
        pypto_lib_root=library,
        pypto_python=pypto_python,
        manifest=manifest,
        python=Path("/usr/bin/python3"),
        platform="a2a3sim",
        scripts=None,
        limit=None,
        timeout=30,
        prune_builds=True,
    )

    assert exporter.export_corpus(arguments) == 0
    with (tmp_path / "output" / "export-status.tsv").open(encoding="utf-8", newline="") as source_file:
        rows = list(csv.DictReader(source_file, delimiter="\t"))

    assert [row["status"] for row in rows] == ["SOURCE_MISSING", "EXPORTED"]
    assert rows[0]["driver_mode"] == "NO_SOURCE"
    assert len((tmp_path / "output" / "invocations.tsv").read_text().splitlines()) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
