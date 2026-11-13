# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Census archived Qwen C++ control; this is not a device or dependency simulator.

Erase only typed storage declarations and replace payload/flag calls by counters.
Retain original integer arithmetic, loops and guards and compile them on the host.
Unknown statements and unresolved data-dependent scalar values fail explicitly.
Counts describe pre-optimizer source execution, not optimized instructions or
latency. No source kernel or synchronization protocol is modified.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.archive
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, bands, workers in [("output", [0], 20), ("gate_up", range(4), 24)]:
        for band in bands:
            for mode in ["pypto", "ptoas"]:
                function = "out_proj_aic" if kind == "output" else "gate_up_proj"
                source = root / f"results/t-b1-{kind}-h5120-l1-b{band}-{mode}/kernels/aic/{function}.cpp"
                text = source.read_text()
                header = re.search("static __aicore__ void " + function + "\\(([^\\n]+)\\) \\{", text)
                assert header, source
                params = re.findall("int32_t (v\\d+)", header[1])
                assert len(params) == 2, params
                start = header.end()
                depth = 1
                end = start
                while depth:
                    depth += (text[end] == "{") - (text[end] == "}")
                    end += 1
                lines = []
                for line in text[start : end - 1].splitlines():
                    s = line.strip()
                    if s.startswith("#"):
                        # This census selects the archived cube body; do not
                        # erase an unknown conditional compilation branch.
                        assert s in ("#if defined(__DAV_CUBE__)", "#endif // __DAV_CUBE__"), s
                        continue
                    if not s or s.startswith(("//", "using ")) or s == ";":
                        continue
                    if s.startswith(("set_flag(", "wait_flag(", "pipe_barrier(")):
                        assert re.fullmatch(r"(?:set_flag|wait_flag|pipe_barrier)\([^;]*\);", s), s
                        label = s.split("(", 1)[0]
                        lines.append(f'++counts["{label}"];')
                        continue
                    if s.startswith("ptoas_auto_sync_tail("):
                        # Its other switch arm contains a flag pair, but the
                        # archived call selects only the ALL barrier arm.
                        assert s == "ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);", s
                        lines.append('++counts["pipe_barrier"];')
                        continue
                    payload = re.fullmatch(
                        r"(TASSIGN|TLOAD|TEXTRACT|TMATMUL|TMATMUL_ACC|TSTORE)(?:<[^;]*>)?\([^;]*\);", s
                    )
                    if payload:
                        lines.append('++counts["' + payload[1] + '"];')
                        continue
                    if (
                        re.match(
                            "(?:const )?(?:int64_t|uint64_t|int32_t|uint32_t|uint16_t|int16_t|bool) "
                            "v\\d+(?: =|;)",
                            s,
                        )
                        or re.match("v\\d+ =", s)
                        or s.startswith(("if (", "for (", "} else", "else {"))
                        or (s in ("}", "};", "return;"))
                    ):
                        lines.append(re.sub("^if \\((.*)\\) \\{$", "if (predicate(\\1)) {", s))
                        continue
                    if s.startswith(("Tile<", "GlobalTensor<", "pto::Shape<", "pto::Stride<", "__gm__ ")):
                        # Scalar dependence on an erased storage value will
                        # fail host compilation instead of inventing a value.
                        continue
                    raise AssertionError((source, s))
                cpp = (
                    "#include <cstdint>\n#include <iostream>\n#include <map>\n#include <string>\n"
                    "std::map<std::string,int64_t> counts;\n"
                    'bool predicate(bool value){++counts["predicate"];return value;}\nvoid run(int32_t '
                    + params[0]
                    + ",int32_t "
                    + params[1]
                    + ") {\n"
                    + "\n".join(lines)
                    + "\n}\n"
                )
                cpp += (
                    "int main(){for(int b=0;b<"
                    + str(workers)
                    + ";++b){counts.clear();run(b,"
                    + str(workers)
                    + '); std::cout << b;for(auto &p:counts)std::cout<<" "<<p.first<<"="<<p.second;'
                    'std::cout<<"\\n";}}\n'
                )
                name = f"{kind}-b{band}-{mode}"
                path = out / (name + ".cpp")
                path.write_text(cpp)
                subprocess.run(["g++", "-std=c++17", "-O0", str(path), "-o", str(out / name)], check=True)
                data = subprocess.check_output([str(out / name)], text=True)
                parsed = []
                for line in data.splitlines():
                    b, *terms = line.split()
                    parsed.append({"block": int(b), **{k: int(v) for k, v in (t.split("=") for t in terms)}})
                rows.append(
                    {
                        "case": name,
                        "source": str(source.relative_to(root)),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "control_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "blocks": parsed,
                    }
                )
                print(
                    name,
                    "sets",
                    sorted(set(x.get("set_flag", 0) for x in parsed)),
                    "barriers",
                    sorted(set(x.get("pipe_barrier", 0) for x in parsed)),
                    flush=True,
                )
    # Sanity-check the projection against identical per-block payload counts.
    # This is not an access/dependency or native macro-semantics oracle.
    for i in range(0, len(rows), 2):
        a, b = rows[i : i + 2]
        for x, y in zip(a["blocks"], b["blocks"], strict=True):
            assert {k: v for k, v in x.items() if k.startswith("T") and k != "TASSIGN"} == {
                k: v for k, v in y.items() if k.startswith("T") and k != "TASSIGN"
            }, (a["case"], x, y)
    (out / "census.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
