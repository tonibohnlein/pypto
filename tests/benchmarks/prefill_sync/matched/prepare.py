# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Materialize wrapped whole-model leaves and one-band launch programs."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
for label in ("current", "before", "stock"):
    parser.add_argument(f"--{label}", type=Path, required=True)
args = parser.parse_args()
HERE = args.output.resolve()
arms = {label: getattr(args, label).resolve() for label in ("current", "before", "stock")}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


manifest = {
    "library": "182990263ee87ef29c835ead20a716e3f4aa3d19",
    "compiler_production_revisions": {"current": "1056de181", "before": "03f912b1a", "stock": "03f912b1a"},
    "prefill": {},
    "persistent": {},
}
unit_hashes = {}
for arm, root in arms.items():
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["compile_success"] and receipt["library_revision"] == manifest["library"]
    assert receipt["configuration"] == dict(entry="prefill", layers=40, batch=1, sequence=128)
    source = root / "compiled"
    manifest["prefill"][arm] = {}
    unit_hashes[arm] = {f.name: digest(f) for f in (source / "ptoas").glob("*.cpp")}
    out = HERE / "integrated" / arm
    # Save the actual runnable configurations and compiler inputs, omitting IR dumps.
    shutil.copytree(
        source, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.o", "*.so")
    )
    shutil.copy(root / "receipt.json", out / "receipt.json")
    model_orch = next((source / "orchestration").glob("*.cpp")).read_text()
    config = (source / "kernel_config.py").read_text()
    for name in ("gate_up_proj", "down_proj"):
        gate = name == "gate_up_proj"
        blocks = 24 if gate else 20
        shapes = (
            [(256, 4352), (128, 5120), (5120, 17408), (5120, 17408)]
            if gate
            else [(128, 5120), (128, 4352), (17408, 5120)]
        )
        params = [
            dict(
                name=f"tensor{i}",
                direction="InOut" if i == 0 else "In",
                shape=list(shape),
                dtype="fp32" if i == 0 else "bfloat16",
                layout="ND",
            )
            for i, shape in enumerate(shapes)
        ]
        match = re.search(
            rf"// Spmd {name}_spmd: {name}\s+CoreTaskArgs (\w+);(.*?)rt_submit_aic_task\((\d+), \1\);",
            model_orch,
            re.S,
        )
        assert match and f"set_block_num({blocks})" in match[2]
        expected_kinds = ["inout"] + ["input"] * (len(shapes) - 1) + ["scalar"] * (3 if gate else 2)
        assert re.findall(rf"{match[1]}\.add_(\w+)\(", match[2]) == expected_kinds
        dirs = ["_D.INOUT"] + ["_D.IN"] * (len(shapes) - 1)
        kernel_line = next(line for line in config.splitlines() if f'"name": "{name}"' in line)
        entry = re.sub(r'"func_id": \d+', '"func_id": 0', kernel_line)
        for band in range(4):
            target = HERE / "isolated" / arm / name / f"band{band}"
            kernel = target / "kernels/aic" / f"{name}.cpp"
            kernel.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source / "kernels/aic" / kernel.name, kernel)
            assert digest(kernel) == receipt["cpp"]["compiled/kernels/aic/" + kernel.name]
            orch = target / "orchestration" / "launch.cpp"
            orch.parent.mkdir(exist_ok=True)
            prefix = model_orch.split("// External tensors", 1)[0]
            prefix = re.sub(r"expected_arg_count = \d+", f"expected_arg_count = {len(params)}", prefix)
            scalar = [6 * band, 17 * band, 0] if gate else [0, 4352 * band]
            body = ["    SIMPLER_SCOPE() {", "        CoreTaskArgs params;"]
            for i in range(len(shapes)):
                body.append(f"        params.add_{expected_kinds[i]}(orch_args.tensor({i}).ref());")
            for value in scalar:
                body.append(f"        params.add_scalar(static_cast<int64_t>({value}));")
            body.extend(
                [
                    f"        params.launch_spec.set_block_num({blocks});",
                    "        rt_submit_aic_task(0, params);",
                    "    }",
                    "}",
                    "}",
                ]
            )
            orch.write_text(prefix.rstrip() + "\n" + "\n".join(body) + "\n")
            cfg = (
                config.split("ORCHESTRATION =", 1)[0]
                + "ORCHESTRATION = {\n"
                + '    "source": str(_ROOT_DIR / "orchestration/launch.cpp"),\n'
                + '    "function_name": "aicpu_orchestration_entry",\n    "signature": ['
                + ", ".join(dirs)
                + "],\n}\nKERNELS = [\n"
                + entry
                + "\n]\n"
            )
            (target / "kernel_config.py").write_text(cfg)
            write_json(
                target / "compiled_meta.json",
                dict(
                    schema=3,
                    supported_execution_modes=["program"],
                    params=params,
                    lowered_params=params,
                    num_return_types=1,
                    return_aliases=[0],
                    platform="a2a3",
                    backend_type="Ascend910",
                ),
            )
        manifest["prefill"][arm][name] = dict(
            blocks=blocks,
            shapes=shapes,
            wrapper_sha256=digest(source / "kernels/aic" / kernel.name),
            ptoas_cpp_sha256=digest(source / "ptoas" / kernel.name),
        )
changed = [n for n in unit_hashes["current"] if unit_hashes["current"][n] != unit_hashes["before"][n]]
assert unit_hashes["current"].keys() == unit_hashes["before"].keys() and changed == ["gate_up_proj.cpp"]
for arm in ("stock", "pypto", "disjoint"):
    root = HERE / "persistent" / arm
    info = json.loads((root / "compiled_meta.json").read_text())
    assert [p["shape"] for p in info["params"]] == [[8192, 8192], [8192, 1024], [8192, 1024]]
    report = (
        json.loads((root / "report/autosync.json").read_text())
        if (root / "report/autosync.json").exists()
        else {}
    )
    selected = [k["selected_route"] for k in report.get("kernels", [])]
    assert ("pypto" in selected) == (arm != "stock")
    manifest["persistent"][arm] = dict(
        wrapper_sha256=digest(root / "kernels/aic/tl_persistent_gemm.cpp"), selected_routes=selected
    )
assert len({v["wrapper_sha256"] for v in manifest["persistent"].values()}) >= 2
manifest["changed_units_before_vs_current"] = changed
write_json(HERE / "manifest.json", manifest)

expected = json.loads(Path(__file__).with_name("expected.json").read_text())
if json.loads((HERE / "manifest.json").read_text()) != expected:
    raise SystemExit(
        "Generated fingerprints differ from the reviewed benchmark; compare manifest.json to expected.json"
    )
