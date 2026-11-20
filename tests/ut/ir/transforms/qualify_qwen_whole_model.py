# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-only current-library whole-Qwen compilation and fail-closed route inventory."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from pypto import backend, ir, passes, testing
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _get_ptoas_flags, _run_ptoas
from pypto.runtime import RunConfig

from .sync_physical_oracle import run_physical


def digest(path: Path) -> str:
    """Hash a source or tool artifact without executing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualification_context(
    policy: str,
    *,
    gm_alias_policy: str = "may-alias",
    gm_contract: Path | None = None,
    scalar_gm_dma: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build independent policy, alias and native-implementation settings.

    A supplied contract records a caller promise; this host qualifier cannot
    establish the ABI or infer disjointness from distinct SSA/pointer names.
    Read its bytes once so the retained promise and its hash describe the same
    input, including when the original file later changes.
    """
    if policy not in {"exact", "conservative_prefix"}:
        raise ValueError("Unknown autosync policy")
    if gm_alias_policy not in {"may-alias", "disjoint"}:
        raise ValueError("Unknown GM alias policy")
    provenance = None
    if gm_alias_policy == "disjoint":
        if gm_contract is None:
            raise ValueError("Disjoint GM qualification requires --gm-contract caller/ABI provenance")
        try:
            contract_path = gm_contract.resolve(strict=True)
            contents = contract_path.read_bytes()
            promise = contents.decode("utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError("GM contract must be a readable UTF-8 caller/ABI provenance file") from error
        if not promise.strip():
            raise ValueError("GM contract caller/ABI provenance must not be empty")
        provenance = {
            "kind": "explicit_caller_promise",
            "path": str(contract_path),
            "sha256": hashlib.sha256(contents).hexdigest(),
            "promise": promise,
        }
    elif gm_contract is not None:
        raise ValueError("--gm-contract requires --gm-alias-policy=disjoint")
    disjoint = gm_alias_policy == "disjoint"
    options: dict[str, Any] = {
        "autosync_policy": policy,
        "autosync_disjoint_gm": disjoint,
        "enable_scalar_gm_dma": scalar_gm_dma,
    }
    access_contract = {
        "gm_alias_policy": gm_alias_policy,
        "autosync_disjoint_gm": disjoint,
        "provenance": provenance,
        "scope": (
            "Supplied disjointness of distinct original external GM roots only; "
            "same-root overlaps, canonical aliases/views and local physical reuse remain dependencies. "
            "Unresolved roots require the importer's conservative coverage or refusal."
            if disjoint
            else "No distinct-GM-root disjointness supplied; retain may-alias coverage or refusal."
        ),
        "qualification": "Caller promise is not an analysis proof and is never inferred from SSA names.",
        "access_validity": "Original source/caller access-validity preconditions remain required.",
    }
    return options, access_contract


def static_control_inventory(function: ir.Function) -> dict[str, Any]:
    """Count original syntactic sites without unfolding loops or choosing arms."""
    counts: Counter[str] = Counter()

    class Calls(ir.IRVisitor):
        def visit_call(self, op):
            counts[f"call:{op.op.name}"] += 1
            super().visit_call(op)

    # Walk structural statements explicitly. Python override recursion through
    # a native visitor trampoline must not hide a nested same-kind loop.
    pending = [function.body]
    while pending:
        statement = pending.pop()
        if isinstance(statement, ir.SeqStmts):
            pending.extend(reversed(statement.stmts))
        elif isinstance(statement, ir.ForStmt):
            counts[f"for:{statement.kind}"] += 1
            pending.append(statement.body)
        elif isinstance(statement, ir.WhileStmt):
            counts["while"] += 1
            pending.append(statement.body)
        elif isinstance(statement, ir.IfStmt):
            counts["if"] += 1
            pending.append(statement.then_body)
            if statement.else_body is not None:
                pending.append(statement.else_body)
        elif isinstance(statement, (ir.ScopeStmt, ir.SimtScopeStmt)):
            counts[f"scope:{type(statement).__name__}"] += 1
            pending.append(statement.body)
    Calls().visit_function(function)
    return dict(sorted(counts.items()))


def domain_ownership(receipt, kernels, cpp_names, execution_domains=()):
    """Join observed domains, retaining absent and delegated owners explicitly."""
    before = {f["name"]: f for f in receipt["functions_before_sync"]}
    selected = {f["name"]: f for f in receipt["functions_after_buffer_selection"]}
    after = {f["name"]: f for f in receipt["functions_after_sync"]}
    reports = {k["function"]: k for k in kernels}
    emitted_domains = {row["function"]: row for row in execution_domains}
    result = []
    for name in sorted(before.keys() | selected.keys() | after.keys()):
        source = before.get(name, selected.get(name, after.get(name)))
        assert source is not None, "observed domain has no original inventory entry"
        emitted = after.get(name)
        owner = emitted.get("synchronization_owner") if emitted else None
        report = reports.get(name, {})
        generated = name in cpp_names
        verified = (
            receipt["compile_success"]
            and owner is True
            and generated
            and report.get("selected_route") == "pypto"
        )
        task = emitted_domains.get(name, {})
        task_owned = (
            receipt["compile_success"]
            and task.get("status") == "task_control_owner_observed"
            and task.get("owner") == "runtime_task_codegen"
            and task.get("scope") == "task_control"
            and task.get("function_type") == source.get("type")
            and source.get("type")
            in {
                str(kind)
                for kind in (
                    ir.FunctionType.Orchestration,
                    ir.FunctionType.Graph,
                    ir.FunctionType.Spmd,
                    ir.FunctionType.Group,
                )
            }
        )
        result.append(
            {
                "function": name,
                "original_domain": source,
                "requested_policy": receipt["policy"],
                "observed_after_sync": emitted is not None,
                "synchronization_owner": owner,
                "route_report": report,
                "generated_cpp": generated,
                "verified_pypto_leaf": verified,
                "task_control": task,
                "task_control_owner_observed": task_owned,
                "status": "verified_pypto_leaf"
                if verified
                else (
                    "task_control_owner_observed"
                    if task_owned
                    else "delegated"
                    if owner is False
                    else "unverified_domain"
                ),
            }
        )
    return result


def expected_ownership_failures(domains, expected):
    """Named qualification is independent of the whole-model ownership claim."""
    observed = {row["function"]: row for row in domains}
    return [
        {"function": name, "status": observed[name]["status"] if name in observed else "not_observed"}
        for name in sorted(set(expected))
        if name not in observed or not observed[name]["verified_pypto_leaf"]
    ]


def function_inventory(program: ir.Program) -> list[dict[str, Any]]:
    """Keep selector obligations separate from synchronization and ownership.

    Every function/domain is included, even scalar or externally bound work.
    An absent selector result means not evaluated, never successful admission.
    """
    result = []
    for function in program.functions.values():
        attrs = dict(function.attrs)
        selected = attrs.get("buffer_ir_selected")
        admission = {"status": "not_evaluated" if selected is None else "selected" if selected else "refused"}
        if "buffer_admission_version" in attrs:
            for key in ("version", "property", "stage", "operation", "location"):
                admission[key] = attrs[f"buffer_admission_{key}"]
        result.append(
            {
                "name": function.name,
                "type": str(function.func_type),
                "stage": str(function.ir_stage),
                "requires_runtime_binding": function.requires_runtime_binding,
                "synchronization_owner": attrs.get("pypto_autosync"),
                "static_control": static_control_inventory(function),
                "attrs": {str(k): str(v) for k, v in attrs.items()},
                "buffer_admission": admission,
            }
        )
    return result


def model_arguments(entry: str, layers: int, batch: int, sequence: int | None):
    """Bind original model entry points to shape-only inputs, including stacked layer storage."""
    module = importlib.import_module(f"{entry}_fwd")
    if entry == "prefill":
        assert sequence is not None
        specs = module.build_tensor_specs(batch=batch, max_seq=sequence, num_layers=layers, use_max_seq=True)
        return (
            module.prefill_fwd,
            [spec.name for spec in specs],
            [torch.empty(spec.shape, dtype=spec.dtype, device="meta") for spec in specs],
        )
    # decode_fwd's serving smoke declares one layer. Its model validation stacks
    # exactly the following weights and KV pools along axis zero for N layers.
    # The public batch, rotary tables, sampling buffers and LM head are shared.
    entry_point = module.decode_fwd
    names = entry_point._param_names()
    stacked = {
        "input_rms_weight",
        "wq",
        "wk",
        "wv",
        "q_norm_weight",
        "k_norm_weight",
        "k_cache",
        "v_cache",
        "wo",
        "w_gate",
        "w_up",
        "w_down",
        "post_rms_weight",
    }
    if not stacked.issubset(names):
        raise ValueError("Decode smoke and stacked model ABI disagree")
    with torch.device("meta"):
        tensors = [
            value.repeat(layers, 1) if name in stacked else value
            for name, value in zip(names, module._decode_smoke_inputs(batch=batch), strict=True)
        ]
    module.__dict__["_FWD_NLAYERS"] = layers
    return entry_point, names, tensors


def audit_ptoas_insertion(output: Path, leaves: set[str]) -> list[dict[str, Any]]:
    """Recompile every hash-matched PTO unit with PTOAS synchronization disabled."""
    directory = output / "compiled" / "ptoas"
    rebuilt = output / "without-ptoas-insertion"
    rebuilt.mkdir()
    domains: set[str] = set()
    result = []
    for source in sorted(directory.glob("*.pto")):
        manifest = json.loads(source.with_suffix(".sync.json").read_text())
        if digest(source) != manifest["source_sha256"]:
            raise ValueError(f"Stale synchronization manifest for {source.name}")
        names = [d["function"] for d in manifest["domains"]]
        if not names or domains.intersection(names) or len(set(names)) != len(names):
            raise ValueError(f"Missing or duplicate function in {source.name}")
        domains.update(names)
        candidate = rebuilt / source.with_suffix(".cpp").name
        _run_ptoas(str(source), str(candidate), _get_ptoas_flags(sync_owned=True))
        ordinary = digest(source.with_suffix(".cpp"))
        disabled = digest(candidate)
        result.append(
            {
                "unit": source.stem,
                "functions": names,
                "whole_module_domains": [d["function"] for d in manifest["domains"] if d["owned"] is True],
                "helpers": manifest["helpers"],
                "module_requested_insertion": manifest["ptoas_insert_sync"],
                "normal_cpp_sha256": ordinary,
                "insertion_disabled_cpp_sha256": disabled,
                "identical": ordinary == disabled,
            }
        )
    if not result or domains != leaves:
        raise ValueError(f"Emitted PTO domain inventory differs from model leaves: {domains ^ leaves}")
    return result


def apply_ptoas_audit(domains, units):
    """Reconcile unit-level insertion evidence with the per-leaf ownership claims."""
    by_name = {row["function"]: row for row in domains}
    errors = []
    for unit in units:
        rows = [by_name[name] for name in unit["functions"]]
        # The manifest's whole-module opt-in is stronger than per-leaf ownership.
        # A PyPTO-owned leaf can also rely on PTOAS recognizing its event markers.
        for name in unit["whole_module_domains"]:
            if by_name[name]["synchronization_owner"] is not True:
                errors.append(f"{unit['unit']}: whole-module claim lacks a PyPTO owner for {name}")
        if unit["identical"]:
            continue
        # A changed mixed unit can owe its difference to a delegated peer or
        # helper. Unit hashes cannot localize that change to a particular leaf.
        fully_owned = not unit["helpers"] and all(row["synchronization_owner"] is True for row in rows)
        if fully_owned:
            errors.append(f"{unit['unit']}: disabling PTOAS insertion changed a fully PyPTO-owned unit")
        for row in rows:
            if row["verified_pypto_leaf"]:
                row["verified_pypto_leaf"] = False
                row["status"] = "unverified_domain"
                row["ptoas_audit"] = "owned_unit_changed" if fully_owned else "mixed_unit_changed"
    return errors


def parse_arguments() -> tuple[argparse.Namespace, dict[str, Any], dict[str, Any]]:
    """Parse whole-model workload and qualification options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--sequence", type=int, help="Prefill prompt length (default 128)")
    parser.add_argument("--mode", choices=("pypto", "ptoas"), default="pypto")
    parser.add_argument("--policy", choices=("exact", "conservative_prefix"), default="exact")
    parser.add_argument(
        "--gm-alias-policy",
        choices=("may-alias", "disjoint"),
        default="may-alias",
        help="Independent GM alias contract; disjoint requires explicit caller/ABI provenance",
    )
    parser.add_argument("--gm-contract", type=Path, help="Pinned UTF-8 caller/ABI disjointness promise")
    parser.add_argument(
        "--audit-ptoas-insertion",
        action="store_true",
        help="Compare native C++ with downstream automatic insertion disabled for each unit",
    )
    parser.add_argument("--require-pypto", action="store_true")
    parser.add_argument("--scalar-gm-dma", action="store_true", help="Explicit DMA scalar implementation")
    parser.add_argument(
        "--require-domain-owners",
        action="store_true",
        help="Require an observed control/local owner for every original domain; delegation is separate",
    )
    parser.add_argument(
        "--require-function",
        action="append",
        default=[],
        help="Require verified PyPTO ownership for this named generated domain (repeatable)",
    )
    parser.add_argument("--tool-artifact", type=Path, action="append", default=[])
    parser.add_argument("--check-function", action="append", default=[])
    args = parser.parse_args()
    if args.layers < 1 or args.batch < 1:
        parser.error("layers and batch must be positive")
    if args.entry == "decode" and args.sequence is not None:
        parser.error("decode uses the model's paged-cache ABI; --sequence applies to prefill")
    try:
        context_options, access_contract = qualification_context(
            args.policy,
            gm_alias_policy=args.gm_alias_policy,
            gm_contract=args.gm_contract,
            scalar_gm_dma=args.scalar_gm_dma,
        )
    except ValueError as error:
        parser.error(str(error))
    return args, context_options, access_contract


def main() -> None:
    """Compile original model bodies with meta tensors, retaining refusal evidence."""
    args, context_options, access_contract = parse_arguments()
    configuration: dict[str, Any] = {"entry": args.entry, "layers": args.layers, "batch": args.batch}
    if args.entry == "prefill":
        configuration["sequence"] = 128 if args.sequence is None else args.sequence
    library, output = args.library.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit("output must be a new directory; stale artifacts cannot establish coverage")
    output.mkdir(parents=True)
    model = library / "models/qwen3_14b"
    root = Path.cwd()
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], text=True).splitlines()
    added = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    sources = sorted(
        p for p in set(changed + added) if p.startswith(("src/", "include/", "python/", "tests/"))
    )
    tool = find_ptoas_binary()
    sys.path[:0] = [str(model), str(library)]
    receipt: dict[str, Any] = {
        "compiler_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "library_revision": subprocess.check_output(
            ["git", "-C", str(library), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_sources": {p.name: digest(p) for p in sorted(model.glob("*.py"))},
        "configuration": configuration,
        "mode": args.mode,
        "policy": args.policy,
        "gm_alias_policy": args.gm_alias_policy,
        "autosync_disjoint_gm": context_options["autosync_disjoint_gm"],
        "enable_scalar_gm_dma": args.scalar_gm_dma,
        "command": sys.argv,
        "python": sys.version,
        "compiler_dirty_sources": {p: digest(root / p) if (root / p).is_file() else None for p in sources},
        "extension_hashes": {str(p): digest(p) for p in (root / "python/pypto").glob("pypto_core*.so")},
        "ptoas": {
            "path": tool,
            "hash": digest(Path(tool)) if tool else None,
            "version": subprocess.check_output([tool, "--version"], text=True).strip() if tool else None,
            "additional_artifacts": {str(p.resolve()): digest(p) for p in args.tool_artifact},
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("PYTHONPATH", "PTOAS_ROOT", "OMP_NUM_THREADS", "PYPTO_CODEGEN_MAX_WORKERS")
        },
        "access_contract": access_contract,
        "device": "not run",
        "passes": [],
        "functions_before_sync": [],
        "functions_after_sync": [],
        "expected_pypto_functions": sorted(set(args.require_function)),
        "functions_after_buffer_selection": [],
        "compile_success": False,
        "all_generated_leaves_pypto": False,
    }
    started = time.perf_counter()

    def before(pass_obj, program):
        name = pass_obj.get_name()
        receipt["passes"].append(name)
        if name == "RealizeSync":
            (output / "before-sync.py").write_text(str(program))
            ir.serialize_to_file(program, str(output / "before-sync.msgpack"))
            receipt["functions_before_sync"] = function_inventory(program)

    def after(pass_obj, program):
        if pass_obj.get_name() == "SelectBufferIRFunctions":
            receipt["functions_after_buffer_selection"] = function_inventory(program)
            ir.serialize_to_file(program, str(output / "after-buffer-selection.msgpack"))
        if pass_obj.get_name() == "RealizeSync":
            receipt["functions_after_sync"] = function_inventory(program)
            ir.serialize_to_file(program, str(output / "after-sync.msgpack"))

    try:
        entry_point, names, tensors = model_arguments(
            args.entry, args.layers, args.batch, configuration.get("sequence")
        )
        receipt["arguments"] = [
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in zip(names, tensors, strict=True)
        ]
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        with passes.PassContext(
            [passes.CallbackInstrument(before_pass=before, after_pass=after)],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=args.mode == "pypto",
            **context_options,
        ):
            entry_point.compile(
                *tensors,
                config=RunConfig(
                    arch=BackendType.Ascend910,
                    codegen_only=True,
                    save_kernels_dir=str(output / "compiled"),
                ),
            )
        receipt["compile_success"] = True
    except Exception:
        receipt["error"] = traceback.format_exc()
    finally:
        receipt["compile_seconds_inclusive"] = time.perf_counter() - started
        report_path = output / "compiled/report/autosync.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        receipt["autosync"] = report
        cpp = sorted((output / "compiled/kernels").rglob("*.cpp"))
        receipt["cpp"] = {str(p.relative_to(output)): digest(p) for p in cpp}
        kernels = report.get("kernels", [])
        receipt["domains"] = domain_ownership(
            receipt, kernels, {p.stem for p in cpp}, report.get("execution_domains", ())
        )
        # Source census and route census must agree; unknown/external owners are
        # unresolved here, never silently counted as successful PyPTO leaves.
        leaves = {
            f["name"]
            for f in receipt["functions_before_sync"]
            if f["type"] in ("FunctionType.AIC", "FunctionType.AIV")
        }
        units = []
        if receipt["compile_success"] and (args.audit_ptoas_insertion or args.require_pypto):
            try:
                units = audit_ptoas_insertion(output, leaves)
                receipt["ptoas_insertion_audit"] = units
                contradictions = apply_ptoas_audit(receipt["domains"], units)
                if contradictions:
                    receipt["ptoas_audit_error"] = "; ".join(contradictions)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                receipt["ptoas_audit_error"] = str(error)
                for row in receipt["domains"]:
                    if row["verified_pypto_leaf"]:
                        row.update(
                            verified_pypto_leaf=False, status="unverified_domain", ptoas_audit="incomplete"
                        )
        receipt["domain_owner_counts"] = dict(Counter(row["status"] for row in receipt["domains"]))
        receipt["unverified_domain_owners"] = [
            row["function"] for row in receipt["domains"] if row["status"] == "unverified_domain"
        ]
        receipt["expected_ownership_failures"] = expected_ownership_failures(
            receipt["domains"], args.require_function
        )
        receipt["all_generated_leaves_pypto"] = (
            receipt["compile_success"]
            and bool(leaves)
            and len(kernels) == len(cpp)
            and {k["function"] for k in kernels} == {p.stem for p in cpp} == leaves
            and not expected_ownership_failures(receipt["domains"], leaves)
            and "ptoas_audit_error" not in receipt
            and all(not unit["helpers"] for unit in units)
        )
        if args.check_function and receipt["compile_success"]:
            original = ir.deserialize_from_file(str(output / "before-sync.msgpack"))
            physical = ir.deserialize_from_file(str(output / "after-sync.msgpack"))
            assert isinstance(original, ir.Program)
            assert isinstance(physical, ir.Program)
            evidence = testing.sync_invocation_evidence(original)
            originals = {f.name: f for f in original.functions.values()}
            emitted = {f.name: f for f in physical.functions.values()}
            receipt["independent_physical_checks"] = []
            try:
                for name in args.check_function:
                    assert any(k["function"] == name and k["selected_route"] == "pypto" for k in kernels)
                    assert not evidence["parameters"].get(name), (
                        "parameter valuation needs a separate contract"
                    )
                    first, last = evidence["block_indices"][name]
                    assert last - first < 64, "exhaustive launch oracle is intentionally bounded"
                    for block in range(first, last + 1):
                        run_physical(originals[name], emitted[name], range(6), block_index=block)
                    receipt["independent_physical_checks"].append(
                        {"function": name, "logical_blocks": [first, last], "passed": True}
                    )
            except Exception:
                receipt["oracle_error"] = traceback.format_exc()
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if (
        not receipt["compile_success"]
        or "ptoas_audit_error" in receipt
        or "oracle_error" in receipt
        or receipt["expected_ownership_failures"]
        or (args.require_domain_owners and receipt["unverified_domain_owners"])
        or (args.require_pypto and not receipt["all_generated_leaves_pypto"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
