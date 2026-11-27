# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare the slot-reader × merge comparison from two whole-model qualification builds."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
READER = ("kernels/aic/qk_pv_partial_plan_aic.cpp", "kernels/aiv/qk_pv_partial_plan_aiv.cpp")
MERGE = ("kernels/aiv/qk_pv_partial_merge.cpp",)


def inputs(root: Path, mode: str, sequence: int) -> dict[str, bytes]:
    receipt = json.loads((root / "receipt.json").read_text())
    geometry = {"entry": "prefill", "batch": 1, "layers": 2, "sequence": sequence}
    if not receipt["compile_success"] or receipt["configuration"] != geometry or receipt["mode"] != mode:
        raise ValueError(f"Expected a successful {mode} build for {geometry} in {root}")
    if mode == "pypto":
        audit = receipt["ptoas_insertion_audit"]
        if not (
            receipt["all_generated_leaves_pypto"]
            and not receipt["unverified_domain_owners"]
            and len(audit) == 29
            and all(unit["identical"] for unit in audit)
        ):
            raise ValueError(f"Incomplete PyPTO ownership or downstream insertion audit in {root}")
    compiled = root / "compiled"
    paths = [compiled / "compiled_meta.json", compiled / "kernel_config.py"]
    paths.extend(sorted((compiled / "orchestration").glob("*.cpp")))
    paths.extend(sorted((compiled / "kernels").rglob("*.cpp")))
    return {path.relative_to(compiled).as_posix(): path.read_bytes() for path in paths}


def fingerprint(files: dict[str, bytes]) -> str:
    # Include every relative path so missing, additional and swapped units are detectable.
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def prepare(candidate: Path, stock: Path, output: Path, sequence: int) -> None:
    programs = {"D": inputs(candidate, "pypto", sequence), "R": inputs(stock, "ptoas", sequence)}
    expected = json.loads((HERE / "fingerprints.json").read_text())[str(sequence)]
    if output.exists():
        raise FileExistsError(f"Choose a fresh preparation directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "programs"
        # The old and new reader use the same payloads. Both mixed wrappers embed
        # the cube source, so substitute the schedule in both copies of the unit.
        for arm, edits in (("A", ("reader", "merge")), ("B", ("merge",)), ("M", ("reader",))):
            programs[arm] = programs["D"].copy()
            for edit in edits:
                for name in READER if edit == "reader" else MERGE:
                    source = Path(temp) / "candidate.cpp"
                    source.write_bytes(programs["D"][name])
                    subprocess.run(
                        [
                            "patch",
                            "--batch",
                            "--forward",
                            "--fuzz=0",
                            "-i",
                            HERE / f"{edit}-control.patch",
                            source,
                        ],
                        check=True,
                    )
                    programs[arm][name] = source.read_bytes()
        for arm, files in programs.items():
            actual = fingerprint(files)
            if actual != expected[arm]:
                raise ValueError(
                    f"{arm} program fingerprint mismatch: expected {expected[arm]}, got {actual}"
                )
            for name, data in files.items():
                path = stage / arm / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--stock", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=int, choices=(128, 256))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.candidate, args.stock, args.output, args.sequence)
