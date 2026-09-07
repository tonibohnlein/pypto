# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Publish portable five-arm inputs from frozen structural artifacts, without timing.

Every baseline is retained unchanged. New latency maps are published only when
every child scores; incomplete parents remain explicit terminal rows. No NPU,
compiler, timing table, source checkout or large build tree is required.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from pypto.tools.dsa_latency_planner import problem_fingerprint
from pypto.tools.dsa_pto_isa_duration import PtoIsaDurationProvider
from pypto.tools.dsa_schedule_model import DurationModel

ARMS = ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_structural", "dsa_rp_latency")
PTOAS_REVISION = "062d4b16f27f7a6baef91b5d6cfdcf6fe5f2f26f"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(root: Path) -> dict[str, str]:
    """Reject file-list drift, changed bytes and all symlinks before reading maps."""
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("packet symlink refused")
    hashes = load(root / "MANIFEST.json")
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if files != set(hashes) | {"MANIFEST.json"}:
        raise ValueError("manifest coverage mismatch")
    for name, expected in hashes.items():
        path = root / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError(f"manifest hash/symlink failure: {name}")
    return hashes


def verify_packet(root: Path) -> dict[str, Any]:
    """Verify exact file coverage, hashes, all 95 slots and published map legality."""
    hashes = verify_manifest(root)
    release = load(root / "release.json")
    if digest(root / release["model"]) != release["model_sha256"]:
        raise ValueError("model digest mismatch")
    status = load(root / "host-status.json")
    if len(status["workloads"]) != 19 or len(status["slots"]) != 95:
        raise ValueError("incomplete 19 x 5 status matrix")
    keys = {(s["workload_id"], s["arm"]) for s in status["slots"]}
    expected_keys = {(r["workload_id"], a) for r in status["workloads"] for a in ARMS}
    if keys != expected_keys:
        raise ValueError("duplicate or missing logical slot")
    if len({r["workload_id"] for r in status["workloads"]}) != 19:
        raise ValueError("duplicate workload identity")
    ready_count = sum(s["arm"] == "dsa_rp_latency" and s["status"] == "HOST_VALID" for s in status["slots"])
    if ready_count != release["five_arm_host_valid"]:
        raise ValueError("five-arm coverage count mismatch")
    verified = 0
    for slot in status["slots"]:
        if slot["status"] != "HOST_VALID":
            if slot.get("map"):
                raise ValueError("blocked slot has a published map")
            continue
        directory = root / slot["map"]
        if map_digest(directory) != slot["map_digest"]:
            raise ValueError("complete map digest mismatch")
        row = next(r for r in status["workloads"] if r["workload_id"] == slot["workload_id"])
        solutions = list(directory.glob("*.solution.json"))
        if len(solutions) != row["child_instances"]:
            raise ValueError("missing child map")
        for path in solutions:
            solution = load(path)
            base = root / "inputs" / slot["workload_id"] / solution["instance"]
            native, derived = load(base / "native-problem.json"), load(base / "search-problem.json")
            if solution["problem_fingerprint"] != problem_fingerprint(native):
                raise ValueError("native replay identity mismatch")
            validate(native, solution)
            validate(derived, solution)
            negative_control(derived, solution)
            verified += 1
    return {
        "manifest_files": len(hashes),
        "logical_slots": len(keys),
        "validated_child_solutions": verified,
        "five_arm_host_valid": release["five_arm_host_valid"],
    }


def seal_packet(root: Path, archive: Path) -> dict[str, Any]:
    """Stage an explicit allowlist, cap size, archive, and verify fresh extraction."""
    stage = archive.parent / f"{archive.name}.stage"
    stage.mkdir(parents=True, exist_ok=False)
    allowed_top = {"primary-manifest.json", "model.json", "host-status.json", "release.json"}
    allowed_suffixes = {".json", ".jsonl", ".txt", ".pto", ".py", ".bundle"}
    total = 0
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("packet symlink refused")
        if not p.is_file():
            continue
        relative = p.relative_to(root)
        if "oracle" in relative.parts:
            continue
        if (
            relative.parts[0] not in {"inputs", "maps", "harness", "provenance"}
            and str(relative) not in allowed_top
        ):
            raise ValueError(f"unlisted payload class: {relative}")
        if p.suffix not in allowed_suffixes or p.stat().st_size > 25 * 1024**2:
            raise ValueError(f"forbidden/oversized packet file: {relative}")
        total += p.stat().st_size
        if total > 128 * 1024**2:
            raise ValueError("packet exceeds 128 MiB apparent limit")
        dest = stage / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    hashes = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    write(stage / "MANIFEST.json", hashes)
    before = verify_packet(stage)
    with tarfile.open(archive, "x:gz") as tar:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(stage)), recursive=False)
    archive.with_name(archive.name + ".sha256").write_text(f"{digest(archive)}  {archive.name}\n")
    extraction = archive.parent / f"{archive.name}.verified"
    extraction.mkdir(exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe archive member")
            dest = extraction / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("archive member unreadable")
            with stream, dest.open("wb") as out:
                shutil.copyfileobj(stream, out)
    if verify_packet(extraction) != before:
        raise ValueError("fresh extraction changed verification")
    return {
        **before,
        "sha256": digest(archive),
        "archive_bytes": archive.stat().st_size,
        "apparent_bytes": total,
    }


def attach_evidence(root: Path, analysis: Path, ptoas_repo: Path) -> None:
    """Attach committed-source proof and reproduce every dispatchable child graph."""
    prefix = ["git", "-C", str(ptoas_repo)]
    directories = [
        "include/PTO/Transforms/KernelScheduling",
        "lib/PTO/Transforms/KernelScheduling",
        "tools/pto-test-opt",
        "include/PTO/Transforms/Passes.td",
    ]
    paths = subprocess.check_output(
        [*prefix, "ls-tree", "-r", "--name-only", PTOAS_REVISION, *directories], text=True
    ).splitlines()
    source_proof = []
    for name in paths:
        committed = subprocess.check_output([*prefix, "show", f"{PTOAS_REVISION}:{name}"])
        path = analysis / "ptoas-clean-src" / name
        if not path.is_file() or path.read_bytes() != committed:
            raise ValueError(f"research graph source differs from pinned commit: {name}")
        source_proof.append({"file": name, "sha256": digest(path), "matches_commit": PTOAS_REVISION})
    binary = analysis / "ptoas-clean-build/tools/pto-test-opt/pto-test-opt"
    graph_proof = []
    status = load(root / "host-status.json")
    for row in status["workloads"]:
        if not all(child["status"] == "COMPLETE" for child in row["children"]):
            continue
        for child in row["children"]:
            directory = root / child["input"]
            command = [
                str(binary),
                str(directory / "pre-insert-sync.pto"),
                "-mlir-disable-threading",
                "-o",
                "/dev/null",
                "-pto-print-kernel-schedule-graph=format=text",
            ]
            result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
            if result.returncode or result.stdout != (directory / "research-graph.txt").read_text():
                raise ValueError(f"frozen graph does not reproduce: {child['input']}")
            graph_proof.append(
                {
                    "input": child["input"],
                    "graph_sha256": digest(directory / "research-graph.txt"),
                    "regenerated_byte_identical": True,
                }
            )
    proof = root / "provenance"
    proof.mkdir(exist_ok=True)
    write(
        proof / "research-exporter.json",
        {
            "revision": PTOAS_REVISION,
            "graph_source_files": source_proof,
            "graphs": graph_proof,
            "binary_sha256": digest(binary),
            "build_caveat": "Existing clean-source build has unrelated CanonicalSync warning suppression; "
            "only the listed graph-source paths are claimed byte-identical to the commit.",
            "research_tool_role": "analysis-only; never used for device code generation",
        },
    )
    bundle = proof / "research-ptoas.bundle"
    if bundle.exists():
        raise ValueError("refusing to replace provenance bundle")
    if subprocess.check_output([*prefix, "rev-parse", "HEAD"], text=True).strip() != PTOAS_REVISION:
        raise ValueError("PTOAS HEAD differs from the requested committed bundle source")
    subprocess.run(
        [
            *prefix,
            "bundle",
            "create",
            str(bundle.resolve()),
            "HEAD",
            "^9d72b90ff49f67749ede982b21b30d98508028fb",
        ],
        check=True,
    )
    subprocess.run([*prefix, "bundle", "verify", str(bundle.resolve())], check=True, capture_output=True)
    harness = root / "harness"
    harness.mkdir(exist_ok=True)
    shutil.copyfile(__file__, harness / "prepare_dsa_five_arm_release.py")
    release = load(root / "release.json")
    release["packet_tooling_pypto"] = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
    ).strip()
    release["research_ptoas_bundle"] = {
        "file": "provenance/research-ptoas.bundle",
        "sha256": digest(bundle),
        "prerequisite": "9d72b90ff49f67749ede982b21b30d98508028fb",
        "included_head": PTOAS_REVISION,
    }
    write(root / "release.json", release)


def placement_digest(solution: dict[str, Any]) -> str:
    """Match the historical complete-map identity, not solver metadata or timing."""
    items = sorted((int(p["pool"]), str(p["buffer"]), int(p["offset"])) for p in solution["placements"])
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


def map_digest(directory: Path) -> str:
    children = {}
    for path in sorted(directory.glob("*.solution.json")):
        solution = load(path)
        if solution["instance"] in children:
            raise ValueError("duplicate map child")
        children[solution["instance"]] = placement_digest(solution)
    if not children:
        raise ValueError("empty complete map")
    return hashlib.sha256(json.dumps(children, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def validate(problem: dict[str, Any], solution: dict[str, Any]) -> None:
    """Independent pair-scan validator for this corpus's schema-v1 geometry.

    Does not call the new planner's legality implementation or a solver. Rejects
    unsupported hard fields rather than treating them as absent.
    """
    for key in ("schema_version", "profile", "instance"):
        if problem[key] != solution[key]:
            raise ValueError(f"solution envelope mismatch: {key}")
    constraints = problem["problem"]["constraints"]
    supported = {"separations", "no_partial_overlaps"}
    if any(value for key, value in constraints.items() if key not in supported):
        raise ValueError("independent validator does not support these additional hard constraints")
    pools = {p["id"]: p for p in problem["problem"]["pools"]}
    buffers = {b["id"]: b for b in problem["problem"]["buffers"]}
    placements = {p["buffer"]: p for p in solution["placements"]}
    if len(placements) != len(solution["placements"]) or set(placements) != set(buffers):
        raise ValueError("duplicate/missing/extra placed buffer")
    for bid, p in placements.items():
        b = buffers[bid]
        offset = p["offset"]
        if type(offset) is not int or offset < 0 or offset % b["alignment"]:
            raise ValueError("invalid aligned offset")
        if type(p["pool"]) is not int or p["pool"] not in b["allowed_pools"] or p["pool"] not in pools:
            raise ValueError("invalid pool")
        end = offset + b["size"]
        if end > pools[p["pool"]]["capacity"] or end >= 2**64:
            raise ValueError("capacity/address overflow")
        if any(offset < r["end"] and r["begin"] < end for r in pools[p["pool"]].get("reserved_ranges", [])):
            raise ValueError("reserved range overlap")
    sep = {tuple(sorted((p["first"], p["second"]))) for p in constraints.get("separations", [])}
    partial = {tuple(sorted((p["first"], p["second"]))) for p in constraints.get("no_partial_overlaps", [])}
    ids = sorted(buffers)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            x, y = placements[a], placements[b]
            ax, by = buffers[a], buffers[b]
            if x["pool"] != y["pool"]:
                continue
            if x["offset"] >= y["offset"] + by["size"] or y["offset"] >= x["offset"] + ax["size"]:
                continue
            live = any(
                u["lower"] < v["upper"] and v["lower"] < u["upper"]
                for u in ax["live_intervals"]
                for v in by["live_intervals"]
            )
            if live or (a, b) in sep:
                raise ValueError("hard lifetime/separation violation")
            if (a, b) in partial and (x["offset"], ax["size"]) != (y["offset"], by["size"]):
                raise ValueError("no-partial-overlap violation")


def negative_control(problem: dict[str, Any], solution: dict[str, Any]) -> None:
    bad = copy.deepcopy(solution)
    b = bad["placements"][0]
    pool = next(p for p in problem["problem"]["pools"] if p["id"] == b["pool"])
    b["offset"] = pool["capacity"]
    try:
        validate(problem, bad)
    except ValueError:
        return
    raise ValueError("validator accepted injected capacity overflow")


def search_command(child: str, objective: str, evaluations: int) -> list[str]:
    """Packet-root-relative invocation: no origin-host absolute paths."""
    return [
        sys.executable,
        "-m",
        "pypto.tools.dsa_latency_planner",
        "--problem",
        f"{child}/search-problem.json",
        "--seed-solution",
        f"{child}/search-seed.json",
        "--objective",
        objective,
        "--schedule",
        f"{child}/schedule.jsonl",
        "--graph",
        f"{child}/research-graph.txt",
        "--model",
        "model.json",
        "--output-root",
        f"{child}/{objective}-search",
        "--max-evaluations",
        str(evaluations),
        "--max-candidates",
        "10000",
    ]


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))


def run_search(root: Path, child: str, objective: str, evaluations: int) -> dict[str, Any]:
    command = search_command(child, objective, evaluations)
    write(root / child / f"{objective}-command.json", ["python", *command[1:]])
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "python")
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            preexec_fn=_limit_child,
        )
    except subprocess.TimeoutExpired:
        return {"status": "SEARCH_RESOURCE_BLOCKED", "reason": "120 second per-child time limit"}
    log = result.stderr[-5000:].replace(str(root), "$PACKET")
    (root / child / f"{objective}-stderr.txt").write_text(log)
    if result.returncode:
        return {"status": "MODEL_OR_SEARCH_BLOCKED", "reason": log, "returncode": result.returncode}
    report = load(root / child / f"{objective}-search/search.json")
    return {"status": "COMPLETE", "report": report}


def prepare_child(
    root: Path,
    source: Path,
    row: dict[str, Any],
    wid: str,
    function: str,
    native: dict[str, Any],
    seed: dict[str, Any],
) -> dict[str, Any]:
    if not function or Path(function).name != function or function.startswith("."):
        raise ValueError("unsafe function filename")
    child = f"inputs/{wid}/{function}"
    directory = root / child
    derived = copy.deepcopy(native)
    if function == row["target"]:
        for pool in derived["problem"]["pools"]:
            if pool["id"] == row["tightened_pool_id"]:
                if row["capacity_bytes"] > pool["capacity"]:
                    raise ValueError("selected capacity exceeds native")
                pool["capacity"] = row["capacity_bytes"]
    search_seed = copy.deepcopy(seed)
    search_seed["problem_fingerprint"] = problem_fingerprint(derived)
    write(directory / "native-problem.json", native)
    write(directory / "search-problem.json", derived)
    write(directory / "search-seed.json", search_seed)
    validate(derived, search_seed)
    base = source / "complete-placement-analysis" / row["geometry_ff_map"] / function
    required = {"official-v057-schedule.jsonl": "schedule.jsonl", "research-graph.txt": "research-graph.txt"}
    missing = [name for name in required if not (base / name).is_file() or not (base / name).stat().st_size]
    if missing:
        return {"function": function, "status": "MODEL_INPUT_BLOCKED", "reason": f"missing {missing}"}
    for old, new in required.items():
        shutil.copyfile(base / old, directory / new)
    pto = source / "endpoint-pto" / row["geometry_ff_map"] / "pre-insert-sync"
    matches = [p for p in pto.glob("*.pto") if f"func.func @{function}(" in p.read_text()]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen PTO for {wid}/{function}")
    shutil.copyfile(matches[0], directory / "pre-insert-sync.pto")
    if (base / "result.json").is_file():
        old = load(base / "result.json")
        write(
            directory / "origin-provenance.json",
            {
                k: old[k]
                for k in (
                    "pto_sha256",
                    "research_graph_sha256",
                    "official_v057_schedule_rc",
                    "research_official_access_op_lowering_join",
                )
                if k in old
            },
        )
    outcomes = {
        name: run_search(root, child, name, row["search_evaluations"]) for name in ("structural", "latency")
    }
    result = {"function": function, "input": child, "searches": outcomes}
    if outcomes["latency"]["status"] != "COMPLETE":
        return {**result, "status": outcomes["latency"]["status"]}
    solution = load(directory / "latency-search/solution.json")
    validate(derived, solution)
    native_solution = copy.deepcopy(solution)
    native_solution["problem_fingerprint"] = seed["problem_fingerprint"]
    validate(native, native_solution)
    negative_control(derived, solution)
    write(directory / "native-replay.solution.json", native_solution)
    return {**result, "status": "COMPLETE", "native_replay": f"{child}/native-replay.solution.json"}


def prepare(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = args.analysis_root.resolve()
    manifest = load(args.manifest)
    if len(manifest["rows"]) != 19:
        raise ValueError("fixed panel must contain exactly 19 workloads")
    write(root / "primary-manifest.json", manifest)
    model = DurationModel(
        sync_latency_cycles=16,
        pto_isa_provider=PtoIsaDurationProvider.from_checkout(
            args.pto_isa_root, expected_revision=manifest["pins"]["pto_isa"]
        ),
    )
    write(root / "model.json", model.to_json())
    with (source / "current-export-all/invocations.tsv").open(newline="") as f:
        documents = {
            (r["script"], r["instance"]): source / "current-export-all" / r["document"]
            for r in csv.DictReader(f, delimiter="\t")
        }
    rows, slots = [], []
    for i, original in enumerate(manifest["rows"]):
        row = {**original, "workload_id": f"w{i:02d}", "search_evaluations": args.max_evaluations}
        wid = row["workload_id"]
        geometry = source / "reconstructed-maps" / row["geometry_ff_map"]
        seeds = {load(p)["instance"]: load(p) for p in geometry.glob("*.solution.json")}
        if len(seeds) != row["child_instances"]:
            raise ValueError("frozen child inventory mismatch")
        native = {fn: load(documents[row["script"], fn]) for fn in seeds}
        row["capacity_vectors"] = {
            fn: {
                str(p["id"]): (
                    row["capacity_bytes"]
                    if fn == row["target"] and p["id"] == row["tightened_pool_id"]
                    else p["capacity"]
                )
                for p in document["problem"]["pools"]
            }
            for fn, document in native.items()
        }
        for arm in ARMS[:-1]:
            old_arm = "dsa_rp_cg" if arm == "dsa_rp_structural" else arm
            old = source / "reconstructed-maps" / row[f"{old_arm}_map"]
            if map_digest(old) != row[f"{old_arm}_map"]:
                raise ValueError("historical map identity mismatch")
            target = root / "maps" / wid / arm
            target.mkdir(parents=True)
            child_names = set()
            for p in old.glob("*.solution.json"):
                sol = load(p)
                fn = sol["instance"]
                child_names.add(fn)
                if sol["problem_fingerprint"] != problem_fingerprint(native[fn]):
                    raise ValueError(f"native fingerprint mismatch: {wid}/{arm}/{fn}")
                validate(native[fn], sol)
                reduced = copy.deepcopy(native[fn])
                for pool in reduced["problem"]["pools"]:
                    pool["capacity"] = row["capacity_vectors"][fn][str(pool["id"])]
                validate(reduced, sol)
                negative_control(reduced, sol)
                shutil.copyfile(p, target / p.name)
            if child_names != set(seeds):
                raise ValueError("baseline complete-map child mismatch")
            slots.append(
                {
                    "workload_id": wid,
                    "arm": arm,
                    "status": "HOST_VALID",
                    "map": str(target.relative_to(root)),
                    "map_digest": map_digest(target),
                }
            )
        row["children"] = [
            prepare_child(root, source, row, wid, fn, native[fn], seed) for fn, seed in sorted(seeds.items())
        ]
        ready = all(child["status"] == "COMPLETE" for child in row["children"])
        slot = {
            "workload_id": wid,
            "arm": "dsa_rp_latency",
            "status": "HOST_VALID" if ready else "MODEL_OR_SEARCH_BLOCKED",
        }
        if ready:
            target = root / "maps" / wid / "dsa_rp_latency"
            target.mkdir(parents=True)
            for child in row["children"]:
                shutil.copyfile(
                    root / child["native_replay"], target / f"pypto_{child['function']}.dsa.solution.json"
                )
            slot.update(map=str(target.relative_to(root)), map_digest=map_digest(target))
        slots.append(slot)
        rows.append(row)
        write(root / "host-status.json", {"workloads": rows, "slots": slots})
        print(wid, row["artifact_id"], slot["status"], flush=True)
    release = {
        "schema_version": 1,
        "status": "HOST_PACKET_COMPLETE",
        "workloads": 19,
        "logical_slots": 95,
        "five_arm_host_valid": sum(
            s["arm"] == "dsa_rp_latency" and s["status"] == "HOST_VALID" for s in slots
        ),
        "product_pins": manifest["pins"],
        "research_ptoas": PTOAS_REVISION,
        "planner_pypto": "ae16f3a931a7cd47972e659c5c777c5f7b025420",
        "model": "model.json",
        "model_sha256": digest(root / "model.json"),
        "global_sync_cycles": 16,
        "calibration": "DEVELOPMENT_FIXED_NOT_FITTED",
        "seed": "geometry_ff_complete_map",
        "search_evaluations": args.max_evaluations,
        "parent_policy": "independent_child_search_no_parent_latency_prediction",
        "fallback_children": False,
        "device_status": "NOT_RUN",
        "graph_delivery": "frozen_graphs_included_no_device_exporter_build_required_for_scoring",
        "graph_regeneration_argv": [
            "pto-test-opt",
            "pre-insert-sync.pto",
            "-mlir-disable-threading",
            "-o",
            "/dev/null",
            "-pto-print-kernel-schedule-graph=format=text",
        ],
    }
    write(root / "release.json", release)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"verify", "seal"}:
        command_parser = argparse.ArgumentParser()
        command_parser.add_argument("command", choices=("verify", "seal"))
        command_parser.add_argument("packet", type=Path)
        command_parser.add_argument("archive", type=Path, nargs="?")
        command_args = command_parser.parse_args()
        if command_args.command == "verify":
            print(json.dumps(verify_packet(command_args.packet), indent=2))
        else:
            if command_args.archive is None:
                command_parser.error("seal requires an archive path")
            print(json.dumps(seal_packet(command_args.packet, command_args.archive), indent=2))
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pto-isa-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=128)
    prepare(parser.parse_args())
