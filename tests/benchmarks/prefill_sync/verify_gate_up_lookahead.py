# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check emitted Qwen lookahead code, including operand addresses and flag lifetimes."""

import argparse
import itertools
import json
import re
import subprocess
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

from control_trace import PIPE, profile
from storage_reference import intersects, reference
from verify import check_flag_lifetimes, order_delta

OP_PIPE = {"TLOAD": "MTE2", "TEXTRACT": "MTE1", "TMATMUL": "M", "TMATMUL_ACC": "M", "TSTORE": "FIX"}
TILE_ARGS = {
    "TLOAD": (0,),
    "TEXTRACT": (0, 1),
    "TMATMUL": (0, 1, 2),
    "TMATMUL_ACC": (0, 1, 2, 3),
    "TSTORE": (1,),
}
DTYPE_BYTES = {"half": 2, "bfloat16_t": 2, "float": 4}


def trace_slots(path, width, temp):  # noqa: PLR0912 - explicit generated-C++ grammar
    _, body = re.split(r"\bAICORE void \w+\(", path.read_text())
    params, body = body.split(") {", 1)
    params = params.split(", ")
    gm = [re.fullmatch(r"__gm__ (?:float|half|bfloat16_t)\* (v\d+)", param) for param in params[:]]
    scalars = [param for param, matched in zip(params, gm, strict=True) if not matched]
    assert len(scalars) == 2
    assert all(re.fullmatch(r"(?:u?int)(?:32|64)_t v\d+", param) for param in scalars)
    lines = [
        """
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
using half = unsigned short;
using bfloat16_t = unsigned short;
using event_t = int;
namespace pto {
    template <int64_t... Ns> struct Shape {
        std::array<int64_t, sizeof...(Ns)> dims{Ns...};
        Shape() { for (auto n : dims) if (n < 0) std::abort(); }
        template <typename... Int> Shape(Int... vals) : dims{static_cast<int64_t>(vals)...} {
            static_assert(sizeof...(vals) == sizeof...(Ns));
        }
    };
    template <int64_t... Ns> using Stride = Shape<Ns...>;
    enum class Layout { ND, NZ, DN };
}
template <typename T, typename Shape, typename Stride, pto::Layout Layout> struct GlobalTensor {
    int64_t addr;
    Shape shape;
    Stride stride;
    GlobalTensor(int64_t addr, Shape shape, Stride stride) : addr(addr), shape(shape), stride(stride) {}
};
#define PTOAS__GLOBAL_TENSOR_DATA(x) ((x).addr)
""",
        "void run(" + ",".join(scalars) + "){",
    ]
    lines.extend(
        f"constexpr int64_t {match[1]} = {(index + 1) * 10**12}LL;" for index, match in enumerate(gm) if match
    )
    tile_codes = []
    tile_declarations = {}
    for s in map(str.strip, body.splitlines()):
        if not s or s == ";" or s.startswith(("//", "using ")):
            continue
        if s.startswith("#"):
            assert re.fullmatch(r"#if defined\(__DAV_CUBE__\)|#endif // __DAV_CUBE__", s), s
            continue
        flag = re.fullmatch(r"(set|wait)_flag\(PIPE_(\w+), PIPE_(\w+), (EVENT_ID\d+|v\d+)\);", s)
        barrier = re.fullmatch(r"pipe_barrier\(PIPE_(\w+)\);", s)
        tile = re.fullmatch(r"Tile<TileType::(\w+), (\w+), (\d+), (\d+), (.+)> (v\d+);", s)
        assign = re.fullmatch(r"TASSIGN\((v\d+), ([^;]+)\);", s)
        op = re.fullmatch(r"(T[A-Z_]+)(?:<[^;]*>)?\(([^;]*)\);", s)
        if flag:
            kind, source, dest, ident = flag.groups()
            if ident.startswith("EVENT_ID"):
                ident = ident.removeprefix("EVENT_ID")
            lines.append(f'printf("{kind} {source} {dest} %lld\\n", (long long) ({ident}));')
        elif barrier:
            lines.append(f'puts("bar {barrier[1]}");')
        elif s == "ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);":
            lines.append('puts("bar ALL");')
        elif tile:
            space, dtype, rows, cols, layout, value = tile.groups()
            assert dtype in DTYPE_BYTES and space in {"Mat", "Left", "Right", "Acc"}
            desc = (space, dtype, int(rows), int(cols), layout)
            if desc not in tile_codes:
                tile_codes.append(desc)
            tile_declarations[value] = tile_codes.index(desc)
            lines.append(f"int64_t {value} = -1;")
        elif assign:
            lines.append(f"{assign[1]} = {assign[2]};")
        elif s.startswith(("GlobalTensor<", "pto::Shape<", "pto::Stride<")):
            lines.append(s)
        elif gm_ptr := re.fullmatch(r"__gm__ (?:float|half|bfloat16_t)\* (v\d+) = (.*);", s):
            lines.append(f"int64_t {gm_ptr[1]} = {gm_ptr[2]};")
        elif op:
            name = op[1]
            operands = op[2].split(", ")
            members = [operands[i] for i in TILE_ARGS[name]]
            body = " ".join(f"{tile_declarations[v]}:%lld" for v in members)
            args_expr = ", ".join(f"(long long) {v}" for v in members)
            if name in ("TLOAD", "TSTORE"):
                operand = operands[1 if name == "TLOAD" else 0]
                gm_fields = [
                    f"{operand}.addr",
                    *(f"{operand}.shape.dims[{i}]" for i in range(5)),
                    *(f"{operand}.stride.dims[{i}]" for i in range(5)),
                ]
                body += " %lld" * len(gm_fields)
                args_expr += ", " + ", ".join(f"(long long) {v}" for v in gm_fields)
            elif name == "TEXTRACT":
                body += " %lld %lld"
                args_expr += ", " + ", ".join(f"(long long) {v}" for v in operands[2:])
            lines.append(f'printf("op {name} // {body}\\n", {args_expr});')
        elif s.startswith("if ("):
            lines.extend(['puts("if");', s])
        elif re.match(
            r"(?:const )?(?:(?:u?int)(?:64|32|16)_t|unsigned|bool|float|event_t) v\d+\b|v\d+ =|for \(", s
        ) or s in {"} else {", "else {", "}", "};", "return;"}:
            lines.append(s)
        else:
            raise ValueError((path, s))
    argc = ",".join(map(str, ["worker", width]))
    lines.append(
        f'int main(){{ for (int worker = 0; worker < {width}; ++worker) {{ puts("WORKER"); run({argc}); }} }}'
    )
    cpp = temp / "slots.cpp"
    cpp.write_text("\n".join(lines))
    exe = temp / "slots"
    subprocess.run(["g++", "-O0", "-std=c++17", "-w", str(cpp), "-o", str(exe)], check=True)
    output = subprocess.check_output([str(exe)], text=True)
    workers = output.split("WORKER\n")[1:]
    assert len(workers) == width
    return [parse_worker(worker.splitlines(), tile_codes, path) for worker in workers]


def parse_worker(raw, tile_codes, path):
    events = []
    payloads = []
    for line in raw:
        control, _, metadata = line.partition(" // ")
        events.append(control)
        if metadata:
            encoded = metadata.split()
            op_name = control.removeprefix("op ")
            slots = []
            for field in encoded[: len(TILE_ARGS[op_name])]:
                code, addr = map(int, field.split(":"))
                assert addr >= 0
                slots.append((*tile_codes[code], addr))
            payloads.append((op_name, tuple(slots), tuple(map(int, encoded[len(slots) :]))))
    op_names, rows, counters = profile(events, True)
    assert op_names == [op[0] for op in payloads]
    windows = set(slot for _, slots, _ in payloads for slot in slots)

    def end(slot):
        return slot[-1] + slot[2] * slot[3] * DTYPE_BYTES[slot[1]]

    for a, b in itertools.combinations(windows, 2):
        if a[0] != b[0]:
            continue
        assert end(a) <= b[-1] or end(b) <= a[-1], (path, a, b)
    event_ids = {}
    for command in events:
        if command.startswith("set "):
            _, src, dst, slot = command.split()
            event_ids.setdefault(f"{src}->{dst}", set()).add(int(slot))
    ids = {direction: sorted(slots) for direction, slots in event_ids.items()}
    return payloads, rows, counters, check_flag_lifetimes(events, True, event_slots=8), ids


def window(slot):
    space, _, rows, cols, _, addr = slot
    return ((space, addr), 0, 0, rows, cols)


def effects(entry):
    name, slots, operands = entry
    tile = [window(slot) for slot in slots]
    if name in ("TLOAD", "TSTORE"):
        address, *shape_and_stride = operands
        shape = shape_and_stride[:5]
        stride = shape_and_stride[5:]
        assert shape[:3] == [1, 1, 1] and stride[-1] == 1
        root, offset = divmod(address, 10**12)
        row, col = divmod(offset, stride[3])
        assert 1 <= root <= 4 and col + shape[4] <= stride[3]
        gm = (("GM", root), row, col, *shape[3:])
        assert gm[-2:] == tile[0][-2:]
        reads, writes = ((gm,), tuple(tile)) if name == "TLOAD" else (tuple(tile), (gm,))
    elif name == "TEXTRACT":
        view, _, _, rows, cols = tile[1]
        y, x = operands
        height, width = tile[0][-2:]
        assert 0 <= y <= rows - height and 0 <= x <= cols - width
        reads, writes = ((view, y, x, height, width),), (tile[0],)
    else:
        assert name in ("TMATMUL", "TMATMUL_ACC")
        if name == "TMATMUL_ACC":
            assert slots[0] == slots[1], "Accumulator input and output must bind the same physical tile"
        cube_tiles = [slots[0], *slots[-2:]]
        assert [slot[0] for slot in cube_tiles] == ["Acc", "Left", "Right"]
        assert [slot[2:4] for slot in cube_tiles] == [(128, 256), (128, 32), (32, 256)]
        assert [slot[1] for slot in cube_tiles] == ["float", "bfloat16_t", "bfloat16_t"]
        reads = tuple(tile[1:])
        writes = (tile[0],)
    return name, PIPE[OP_PIPE[name]], reads, writes


def check_expected_worker(payloads, band, worker):
    rotated = (worker + 6 * band) % 24
    planned = [
        (row, tile, root) for row, tile, root in ((0, rotated, 3), (128, (rotated + 17) % 24, 4)) if tile < 17
    ]
    ops = list(map(effects, payloads))
    assert len(ops) == 521 * len(planned)
    for projection, (output_row, tile, weight_root) in enumerate(planned):
        batch = ops[521 * projection : 521 * (projection + 1)]
        loads = [r[0] for name, _, r, _ in batch if name == "TLOAD"]
        expected = []
        for k in range(20):
            expected.extend(
                [
                    (("GM", 2), 0, 256 * k, 128, 256),
                    (("GM", weight_root), 256 * k, 4352 * band + 256 * tile, 256, 256),
                ]
            )
        assert loads == expected, (band, worker, projection)
        assert batch[-1][3] == ((("GM", 1), output_row, 256 * tile, 128, 256),)
    return [(row, 256 * tile) for row, tile, _ in planned]


def lookahead_order(payloads):
    order = list(range(len(payloads)))
    selected = []
    stores = [i for i, payload in enumerate(payloads) if payload[0] == "TSTORE"]
    for projection, store in enumerate(stores):
        start = projection * 521
        assert store - start == 520
        for group in range(9):
            base = start + group * 52
            early = [base + i for i in (52, 53, 56, 57, 58, 59)]
            assert [payloads[i][0] for i in early] == ["TLOAD"] * 2 + ["TEXTRACT"] * 4
            order = [i for i in order if i not in early]
            dest = order.index(base + 28)
            order[dest:dest] = early
            selected.extend(early[2:])
    return order, selected


def check_moved_pairs(original, order):
    effects_by_site = [effects(payload) for payload in original]
    position = {old: new for new, old in enumerate(order)}
    for j, (_, _, new_reads, new_writes) in enumerate(effects_by_site):
        # Every moved instruction stays within two neighboring K groups.
        for i in range(max(0, j - 52), j):
            if position[i] <= position[j]:
                continue
            _, _, old_reads, old_writes = effects_by_site[i]
            for lhs, rhs in ((old_reads, new_writes), (old_writes, new_reads + new_writes)):
                assert not any(intersects(x, y) for x in lhs for y in rhs), (i, j)


@lru_cache(maxsize=8)
def reference_with_fixed_columns(payloads):
    """Deduplicate workers with the same projection role and disjoint GM rows."""
    # Each worker reads its own weight columns and writes exactly one output
    # tile per executed projection to separate rows. Normalize the common
    # output-column placement after the caller validates it is disjoint.
    return reference(payloads)


def normalize_gm(ops):
    def norm(rect):
        ident, row, col, height, width = rect
        if ident[0] == "GM":
            col = 0 if ident[1] != 2 else col
        return ident, row, col, height, width

    return tuple((op, pipe, tuple(map(norm, r)), tuple(map(norm, w))) for op, pipe, r, w in ops)


def check_pair(a, b, *, lookahead):
    a_payload, a_rows, a_counts, a_reuses, a_ids = a
    b_payload, b_rows, b_counts, b_reuses, b_ids = b
    assert len(a_payload) == len(b_payload) in (521, 1042)
    order, selected = lookahead_order(a_payload) if lookahead else (list(range(len(a_payload))), [])
    assert [a_payload[i] for i in order] == b_payload
    check_moved_pairs(a_payload, order)
    a_effects, b_effects = (list(map(effects, p)) for p in (a_payload, b_payload))
    for program in (a_effects, b_effects):
        output = [w[0] for op, _, _, w in program if op == "TSTORE"]
        assert [rect[1] for rect in output] in ([0], [128], [0, 128])
        assert all(rect[0] == ("GM", 1) for rect in output)
        assert all(r[0][0][0] == "GM" and 2 <= r[0][0][1] <= 4 for op, _, r, _ in program if op == "TLOAD")
    a_required, b_required = (reference_with_fixed_columns(normalize_gm(e)) for e in (a_effects, b_effects))
    a_delta, b_delta = order_delta(a_rows, a_required), order_delta(b_rows, b_required)
    assert a_delta == b_delta == {"extra": 0, "missing": 0}, (a_delta, b_delta)
    new_index = {old: new for new, old in enumerate(order)}
    m = PIPE["M"]
    advance = Counter(a_rows[i][m] - b_rows[new_index[i]][m] for i in selected)
    if lookahead:
        assert set(advance) == {6, 7} and advance[6] == advance[7] == len(selected) // 2, advance
    return {
        "commands": [a_counts, b_counts],
        "rearms": [a_reuses, b_reuses],
        "event_ids": [a_ids, b_ids],
        "advance_M_completion_cut": advance,
        "order_delta": [a_delta, b_delta],
    }


def main():
    if not __debug__:
        raise RuntimeError("Run the benchmark with Python assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("builds", type=Path, help="parent of b0-baseline, b0-lookahead, etc.")
    options = parser.parse_args()
    result = {}
    with tempfile.TemporaryDirectory(dir=options.builds) as temporary:
        for band in range(4):
            compiled = [
                options.builds / f"b{band}-{variant}" for variant in ("baseline", "peeled", "lookahead")
            ]
            for path in compiled:
                report = json.loads((path / "report/autosync.json").read_text())
                leaf = next(row for row in report["kernels"] if row["function"] == "gate_up_proj")
                assert leaf["selected_route"] == "pypto", leaf
            inputs = [trace_slots(path / "ptoas/gate_up_proj.cpp", 24, Path(temporary)) for path in compiled]
            groups = {}
            coverage = Counter()
            for worker, (a, peeled, b) in enumerate(zip(*inputs, strict=True)):
                coverage.update(check_expected_worker(a[0], band, worker))
                record = {
                    "baseline_vs_peeled": check_pair(a, peeled, lookahead=False),
                    "peeled_vs_lookahead": check_pair(peeled, b, lookahead=True),
                }
                rotated = (worker + 6 * band) % 24
                role = "both" if 7 <= rotated < 17 else "gate" if rotated < 7 else "up"
                key = json.dumps([role, record], sort_keys=True)
                group = groups.setdefault(key, {"role": role, "workers": [], **record})
                group["workers"].append(worker)
            assert coverage == Counter({(row, 256 * tile): 1 for row in (0, 128) for tile in range(17)})
            result[band] = list(groups.values())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
