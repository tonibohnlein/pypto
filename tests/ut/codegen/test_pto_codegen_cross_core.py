# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
910B PTO Backend: Cross-Core Communication (TPUSH/TPOP) Codegen Test.

This test validates code generation for the complete TPUSH/TPOP cross-core
communication protocol. Elementwise ops (add, exp) run on Vector cores, while
matmul runs on Cube cores — matching the hardware architecture.

Protocol under test (V2C unidirectional):
  1. Vector (producer): load + add, tpush_to_aic → Cube
  2. Cube (consumer):   tpop_from_aiv, matmul, store

Bidirectional:
  1. Vector → Cube (V2C): Vector preprocesses (add), pushes to Cube
  2. Cube → Vector (C2V): Cube does matmul, pushes result back to Vector for post-processing

Multiple explicit pipes:
  1. Vector → Cube (V2C): Vector pushes one tile on pipe id 0
  2. Vector → Cube (V2C): Vector pushes another tile on pipe id 1
"""

import json
import re
from pathlib import Path
from textwrap import indent

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes, testing
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _build_group_mapping
from pypto.runtime import RunConfig


def _extract_func_section(mlir_code: str, func_name: str) -> str:
    """Return the MLIR slice for a single func.func body."""
    start_token = f"func.func @{func_name}"
    start = mlir_code.find(start_token)
    assert start != -1, f"Expected function {func_name!r} in MLIR:\n{mlir_code}"
    next_start = mlir_code.find("\n  func.func @", start + len(start_token))
    if next_start == -1:
        return mlir_code[start:]
    return mlir_code[start:next_start]


def _paired_consumer(
    local_depth: int,
    *,
    read_after_wrap: bool = True,
    free_before_wrap: bool = False,
    intervening_read: bool = False,
    authored_waits: bool = False,
) -> ir.Program:
    first_read = "combined = pl.add(first, first)"
    first_free = "pl.tfree_to_aic(first)"
    early = "" if read_after_wrap else first_read
    prior = "first" if read_after_wrap else "combined"
    intervening = "doubled = pl.add(second, second)" if intervening_read else ""
    second_arg = "doubled" if intervening_read else "second"
    free_early = first_free if free_before_wrap else ""
    free_late = "" if free_before_wrap else first_free

    def sync(src: str, dst: str, event: int) -> str:
        args = f"pl.PipeType.{src}, pl.PipeType.{dst}, {event}"
        return f"pl.system.set_flag({args})\npl.system.wait_flag({args})" if authored_waits else ""

    return _paired_receive_program(
        local_depth,
        """
values = pl.load(x, [0, 0], [16, 16])
other = pl.load(x, [16, 0], [16, 16])
first = pl.matmul(values, values)
second = pl.matmul(other, other)
pl.tpush_to_aiv(first, split=0)
pl.tpush_to_aiv(second, split=0)
""",
        f"""
first: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
{sync("MTE2", "V", 0)}
{early}
{free_early}
{sync("V", "MTE2", 0)}
second: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
{sync("MTE2", "V", 1)}
{intervening}
combined = pl.add({prior}, {second_arg})
{free_late}
pl.tfree_to_aic(second)
{sync("V", "MTE3", 0)}
return pl.store(combined, [0, 0], out)
""",
    )


def _paired_receive_program(
    local_depth: int, producer: str, consumer: str, *, slot_bytes: int = 1024, ring_depth: int = 2
) -> ir.Program:
    return pl.parse_program(f"""
@pl.program
class PairedConsumer:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, x: pl.Tensor[[32, 16], pl.FP16]):
        peer = pl.import_peer_buffer(name="peer_c2v", peer_func="vector")
        pl.aic_initialize_pipe(peer, pl.const(0, pl.INT32), dir_mask=1, slot_size={slot_bytes},
                               slot_num={ring_depth}, local_slot_num={local_depth})
{indent(producer.strip(), "        ")}

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        fifo = pl.reserve_buffer(name="peer_c2v", size={slot_bytes * local_depth}, base=4096)
        pl.aiv_initialize_pipe(fifo, pl.const(0, pl.INT32), dir_mask=1, slot_size={slot_bytes},
                               slot_num={ring_depth}, local_slot_num={local_depth})
{indent(consumer.strip(), "        ")}

    @pl.function(type=pl.FunctionType.Group)
    def paired(self, x: pl.Tensor[[32, 16], pl.FP16],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        self.cube(x)
        return self.vector(out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[32, 16], pl.FP16],
             out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        return self.paired(x, out)
""")


def _mat_receive_epilogue(width, dtype, ring_depth, *, sends=4, loop=None, two_readers=False):
    """Each new pop overwrites the Mat slot read two transfers earlier."""
    stride = 16 * width * 4

    def visit_ops(visit, row):
        rhs = f"right{visit}" if two_readers else "right"
        extra = [f"{rhs} = pl.move(r{visit}, target_memory=pl.Mem.Right)"] if two_readers else []
        return "\n".join(
            [
                f"r{visit}: pl.Tile[[16, {width}], pl.{dtype}, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)",
                f"l{visit} = pl.move(r{visit}, target_memory=pl.Mem.Left)",
                *extra,
                f"pl.tfree_to_aiv(r{visit})",
                f"product{visit} = pl.matmul(l{visit}, {rhs})",
                f"out = pl.store(product{visit}, [{row}, 0], out)",
            ]
        )

    if loop is None:
        pops = [visit_ops(i, 16 * i) for i in range(4)]
    else:
        trips, per_trip = loop
        body = "\n".join(visit_ops(i, 16 * i) for i in range(per_trip))
        pops = [f"for i in pl.range({trips}):\n" + indent(body, "    ")]
        pops.extend(visit_ops(f"tail{i}", 16 * (i + 2)) for i in range(2))
        sends = trips * per_trip + 2
    return pl.parse_program(f"""
@pl.program
class MatReceiver:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, rhs: pl.Tensor[[{width}, 16], pl.{dtype}],
             out: pl.Out[pl.Tensor[[64, 16], pl.FP32]]) -> pl.Tensor[[64, 16], pl.FP32]:
        fifo = pl.reserve_buffer(name="inbox", size={2 * stride}, base=0)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), fifo, dir_mask=2, slot_size={stride},
                               slot_num={ring_depth}, local_slot_num=2)
        matrix = pl.load(rhs, [0, 0], [{width}, 16], target_memory=pl.Mem.Mat)
        right = pl.move(matrix, target_memory=pl.Mem.Right)
{indent(chr(10).join(pops), "        ")}
        return out

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, x: pl.Tensor[[16, {width}], pl.{dtype}]):
        peer = pl.import_peer_buffer(name="inbox", peer_func="cube")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size={stride},
                               slot_num={ring_depth}, local_slot_num=2)
        data = pl.load(x, [0, 0], [16, {width}], target_memory=pl.Mem.Vec)
{indent(chr(10).join(["pl.tpush_to_aic(data, split=0)"] * sends), "        ")}

    @pl.function(type=pl.FunctionType.Group)
    def paired(self, rhs: pl.Tensor[[{width}, 16], pl.{dtype}], x: pl.Tensor[[16, {width}], pl.{dtype}],
               out: pl.Out[pl.Tensor[[64, 16], pl.FP32]]) -> pl.Tensor[[64, 16], pl.FP32]:
        self.vector(x)
        return self.cube(rhs, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, rhs: pl.Tensor[[{width}, 16], pl.{dtype}], x: pl.Tensor[[16, {width}], pl.{dtype}],
             out: pl.Out[pl.Tensor[[64, 16], pl.FP32]]) -> pl.Tensor[[64, 16], pl.FP32]:
        return self.paired(rhs, x, out)
""")


def _check_mat_receive_schedule(body):
    # Replay explicit SET/WAIT prefix transfer in the emitted cube function.
    # Recording the last reader's completion on MTE2 is essential at wrap.
    known = {}
    pending = {}
    received = {}
    slots: list[int | None] = [None, None]
    reads = 0
    for cut, line in enumerate(body.splitlines()):
        flag = re.search(r"(set|wait)_flag\(PIPE_(\w+), PIPE_(\w+), EVENT_ID(\d+)\)", line)
        if flag:
            kind, src, dst, idx = flag.groups()
            key = (src, dst, idx)
            if kind == "set":
                assert key not in pending
                pending[key] = set(known.get(src, ()))
            else:
                known.setdefault(dst, set()).update(pending.pop(key))
        elif "TPOP<TPipe<" in line:
            target = re.search(r", (v\d+)\);$", line)
            assert target is not None, line
            name = target[1]
            slot = len(received) % 2
            if slots[slot] is not None:
                assert slots[slot] in known["MTE2"], "receive slot reused before MTE1 read completed"
            known.setdefault("MTE2", set()).add(cut)
            received[name] = (cut, slot)
        elif move := re.search(r"TMOV\(v\d+, (v\d+)\);", line):
            if move[1] in received:
                pop, slot = received[move[1]]
                assert pop in known["MTE1"], "Mat read issued before TPOP completed"
                known["MTE1"].add(cut)
                slots[slot] = cut
                reads += 1
    assert len(received) == reads == 4
    assert not pending


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize(
    "mat_geometry",
    [None, (16, "FP16", 4), (128, "BF16", 2)],
    ids=["vec-receiver", "mat-slot-epilogue", "prefill-mat-slot-geometry"],
)
def test_complete_group_owns_both_native_endpoints(tmp_path, mat_geometry):
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    program = _mat_receive_epilogue(*mat_geometry) if mat_geometry else _paired_consumer(2)
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=True,
    ):
        ir.compile(program, **cfg.compile_kwargs())
    info = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    assert info["owned"] and not info["ptoas_insert_sync"], info
    assert "--enable-insert-sync=false" in info["compile_flags"]
    assert {d["function"] for d in info["domains"] if d["owned"]} == {"cube", "vector"}
    if mat_geometry:
        native = (tmp_path / "ptoas" / "paired.cpp").read_text().split("AICORE void cube(", 1)[1]
        cube = native.split("AICORE void ", 1)[0]
        _check_mat_receive_schedule(cube)
        without_releases = re.sub(r"^.*(?:set|wait)_flag\(PIPE_MTE1, PIPE_MTE2,.*$", "", cube, flags=re.M)
        with pytest.raises(AssertionError, match="receive slot reused"):
            _check_mat_receive_schedule(without_releases)


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
def test_single_core_autosync_preserves_an_unbalanced_authored_protocol(tmp_path):
    # Compile only. The extra authored send makes the communication protocol
    # unbalanced. All four local Mat receives still require valid slot ordering.
    snapshots = []

    def capture(transform, program):
        if transform.get_name() == "RealizeSync":
            snapshots.append(program)

    program = _mat_receive_epilogue(128, "BF16", 2, sends=5)
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext(
        [passes.CallbackInstrument(before_pass=capture, after_pass=capture)],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=False,
    ):
        ir.compile(program, **cfg.compile_kwargs())
    before, after = snapshots
    assert "stream counts" in testing.inspect_peer_fifo_protocols(before)["paired"]

    class Payloads(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.original_calls = []

        def visit_call(self, op):
            if not (
                op.op.name.startswith("system.bar_")
                or op.op.name in {ir.get_op("system.set_flag").name, ir.get_op("system.wait_flag").name}
            ):
                self.original_calls.append(op)
            super().visit_call(op)

    for name in ("cube", "vector"):
        old, new = Payloads(), Payloads()
        old.visit_function(before.get_function(name))
        new.visit_function(after.get_function(name))
        assert old.original_calls and new.original_calls == old.original_calls

    manifest = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    assert manifest["owned"] and not manifest["ptoas_insert_sync"]
    native = (tmp_path / "ptoas" / "paired.cpp").read_text()
    cube = native.split("AICORE void cube(", 1)[1].split("AICORE void ", 1)[0]
    _check_mat_receive_schedule(cube)
    # Both active and empty Vec lanes keep the five original sends.
    assert native.count("TPUSH<") == 10
    assert cube.count("TPOP<") == 4


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("count", [3, 9])
def test_peer_event_pressure_is_atomic(tmp_path, count):
    # Distinct live values force a fresh MTE2 -> V completion per receive.
    # Their GM stores come later, so no reverse service permits early ID reuse.
    producer = "\n".join(
        ["a = pl.load(x, [0, 0], [16, 16])", "m = pl.matmul(a, a)"] + ["pl.tpush_to_aiv(m, split=0)"] * count
    )
    receiver = []
    for i in range(count):
        receiver.extend(
            [
                f"v{i}: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)",
                f"t{i} = pl.add(v{i}, v{i})",
                f"pl.tfree_to_aic(v{i})",
            ]
        )
    receiver.append("s0 = t0")
    receiver.extend(f"s{i} = pl.add(s{i - 1}, t{i})" for i in range(1, count))
    receiver.append(f"return pl.store(s{count - 1}, [0, 0], out)")
    program = _paired_receive_program(16, producer, "\n".join(receiver), ring_depth=16)
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=True,
    ):
        ir.compile(program, **cfg.compile_kwargs())
    info = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    assert info["owned"] == (count == 3)
    assert {d["owned"] for d in info["domains"]} == {count == 3}
    if count == 9:
        reasons = json.loads((tmp_path / "report" / "autosync.json").read_text())["kernels"]
        per_core = {k["function"]: k["reason"] for k in reasons}
        assert per_core["cube"].startswith("local plan verified; shared PTOAS invocation")
        assert per_core["vector"].startswith("local planning for vector")
        assert all("required_at_least=9 eligible=8" in reason for reason in per_core.values())
        source = (tmp_path / "ptoas" / "paired.pto").read_text()
        assert "pto.set_flag" not in source and "pto.wait_flag" not in source


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("free_before_wrap", [False, True], ids=["retained", "freed-with-gap"])
def test_explicit_peer_rejects_native_read_after_slot_wrap(tmp_path: Path, free_before_wrap: bool):
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Native peer pop overwrites a consumer slot"):
        prog = _paired_consumer(1, free_before_wrap=free_before_wrap, intervening_read=free_before_wrap)
        ir.compile(prog, **cfg.compile_kwargs())


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("enable_autosync", [False, True], ids=["stock", "conservative-prefix"])
@pytest.mark.parametrize(
    "depth,read_after_wrap,free_before_wrap",
    [(2, True, False), (2, False, False), (1, False, False), (1, False, True)],
    ids=["two-slot-late", "two-slot-early", "one-slot-retained", "one-slot-freed"],
)
def test_explicit_peer_orders_physical_slot_reuse(
    tmp_path: Path, depth: int, read_after_wrap: bool, free_before_wrap: bool, enable_autosync: bool
):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        enable_pypto_autosync=enable_autosync,
        autosync_policy="conservative_prefix",
    ):
        prog = _paired_consumer(depth, read_after_wrap=read_after_wrap, free_before_wrap=free_before_wrap)
        ir.compile(prog, **cfg.compile_kwargs())
    kernels = list((tmp_path / "kernels").rglob("*.cpp"))
    assert len(kernels) == 2
    body = (tmp_path / "kernels" / "aiv" / "vector.cpp").read_text()
    # Each output includes both peer definitions. The AIV definition retains
    # two pop/free epochs in both the active and zero-valid handshake lanes.
    assert body.count("TPUSH<") == 2
    assert body.count("TPOP<") == 4
    assert body.count("TFREE<") == 4
    native = (tmp_path / "ptoas" / "paired.cpp").read_text()
    first_pop, second_pop, replay_first, _ = (match.start() for match in re.finditer(r"\bTPOP<", native))
    between_pops = native[first_pop:second_pop]
    barrier = "pipe_barrier(PIPE_ALL);"
    owned = enable_autosync and (depth > 1 or free_before_wrap)
    if depth == 1 and owned:
        read_suffix = between_pops.split("TADD(", 1)[1]
        release = re.search(r"set_flag\(PIPE_V, PIPE_MTE2, (EVENT_ID\d+)\);", read_suffix)
        assert release
        assert f"wait_flag(PIPE_V, PIPE_MTE2, {release[1]});" in read_suffix[release.end() :]
        assert barrier not in between_pops
    elif depth == 1:
        assert between_pops.index("TADD(") < between_pops.index(barrier)
    else:
        assert barrier not in between_pops
    if not owned:
        assert barrier not in native[replay_first:]
    else:
        # Each local return drains its own lane, including the handshake lane.
        assert barrier not in native[replay_first : native.index("TPOP<", replay_first + 1)]
    assert "wait_flag(PIPE_MTE2, PIPE_V," in native[second_pop:replay_first]
    ownership = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    assert ownership["ptoas_insert_sync"] == (not owned)
    assert {domain["owned"] for domain in ownership["domains"]} == {owned}


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
def test_explicit_peer_preserves_authored_handoff(tmp_path: Path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    prog = _paired_consumer(1, read_after_wrap=False, authored_waits=True)
    ir.compile(prog, **cfg.compile_kwargs())
    native = (tmp_path / "ptoas" / "paired.cpp").read_text()
    first_pop, second_pop, *_ = (match.start() for match in re.finditer(r"\bTPOP<", native))
    between_pops = native[first_pop:second_pop]
    assert "pipe_barrier(PIPE_ALL);" not in between_pops
    assert between_pops.index("TADD(") < between_pops.index("wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);")


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("depth", [1, 2])
def test_explicit_peer_slot_stride_is_independent_of_tile_extent(tmp_path: Path, depth: int):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    prog = _paired_receive_program(
        depth,
        """
values = pl.load(x, [0, 0], [16, 16])
wide = pl.load(x, [0, 0], [32, 16])
first = pl.matmul(wide, values)
second = pl.matmul(values, values)
pl.tpush_to_aiv(first, split=0)
pl.tpush_to_aiv(second, split=0)
""",
        """
first: pl.Tile[[32, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
doubled = pl.add(first, first)
prefix = pl.slice(doubled, [16, 16], [0, 0])
out = pl.store(prefix, [0, 0], out)
second: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
out = pl.store(second, [0, 0], out)
pl.tfree_to_aic(first)
pl.tfree_to_aic(second)
return out
""",
        slot_bytes=2048,
    )
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    ir.compile(prog, **cfg.compile_kwargs())
    native = (tmp_path / "ptoas" / "paired.cpp").read_text()
    first_pop, second_pop, *_ = list(re.finditer(r"\bTPOP<.*", native))
    assert "Vec, float, 32, 16," in first_pop[0]
    assert "Vec, float, 16, 16," in second_pop[0]
    # The shorter receive overwrites the prior payload's prefix in one slot.
    # A second slot starts at the declared byte stride, even for unequal tiles.
    between_pops = native[first_pop.end() : second_pop.start()]
    assert ("pipe_barrier(PIPE_ALL);" in between_pops) == (depth == 1)


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize(
    "depth,two_sites", [(1, False), (1, True), (2, True)], ids=["same-site", "distinct-sites", "two-slot"]
)
def test_explicit_peer_loop_orders_slot_reuse(tmp_path: Path, depth: int, two_sites: bool):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    work = """
tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
doubled = pl.add(tile, tile)
pl.tfree_to_aic(tile)
out = pl.store(doubled, [0, 0], out)
""".strip()
    receive = [work]
    if two_sites:
        receive.append(work.replace("tile", "other").replace("doubled", "second_doubled"))
    producer = [
        "values = pl.load(x, [0, 0], [16, 16])",
        "first = pl.matmul(values, values)",
        "for _ in pl.range(4):",
        indent("\n".join(["pl.tpush_to_aiv(first, split=0)"] * len(receive)), "    "),
    ]
    consumer = ["for _ in pl.range(4):", indent("\n".join(receive), "    "), "return out"]
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    prog = _paired_receive_program(depth, "\n".join(producer), "\n".join(consumer))
    ir.compile(prog, **cfg.compile_kwargs())

    native = (tmp_path / "ptoas" / "paired.cpp").read_text()
    loop_start = native.rfind("for (", 0, native.index("TPOP<"))
    first_pop = native.index("TPOP<")
    read = native.index("TADD(", first_pop)
    if two_sites and depth == 1:
        second_pop = native.index("TPOP<", first_pop + 1)
        # Both different receive sites target the single slot, including on
        # the next iteration's backedge.
        assert "pipe_barrier(PIPE_ALL);" in native[loop_start:first_pop]
        assert "pipe_barrier(PIPE_ALL);" in native[read:second_pop]
    else:
        assert "pipe_barrier(PIPE_ALL);" not in native[loop_start:]
        before_pop = native[loop_start:first_pop]
        wait = re.search(r"wait_flag\(PIPE_V, PIPE_MTE2, (EVENT_ID\d+)\);", before_pop)
        assert wait is not None
        release = native[read : native.index("TFREE<", read)]
        assert f"set_flag(PIPE_V, PIPE_MTE2, {wait[1]});" in release


@pytest.mark.usefixtures("ascend_backend")
def test_peer_fifo_counts_independent_ids_and_refuses_early_exit():
    def original(early_exit: bool) -> ir.Program:
        exit_early = "if index == 0:\n                continue" if early_exit else ""
        return pl.parse_program(f"""
@pl.program
class IndependentFifos:
    @pl.function(type=pl.FunctionType.AIV)
    def vector(self):
        a = pl.reserve_buffer(name="a", size=1024, base=4096)
        b = pl.reserve_buffer(name="b", size=1024, base=8192)
        pl.aiv_initialize_pipe(a, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=1, id=0)
        pl.aiv_initialize_pipe(b, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=1, id=1)
        for index in pl.range(4096):
            first: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0, id=0)
            second: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0, id=1)
            {exit_early}
            pl.tfree_to_aic(second, id=1)
            pl.tfree_to_aic(first, id=0)
""")

    for early_exit in (False, True):
        source = original(early_exit)
        with passes.PassContext([], enable_pypto_autosync=True, autosync_policy="conservative_prefix"):
            result = passes.realize_sync()(source)
        before, after = source.get_function("vector"), result.get_function("vector")
        assert before is not None and after is not None
        assert not after.attrs["pypto_autosync"]
        why = after.attrs["autosync_fallback_reason"]
        if early_exit:
            assert "early-exit control does not prove local FIFO order" in why
        else:
            for pipe_id, base in enumerate((4096, 8192)):
                assert (
                    f"pipe id {pipe_id} has peak marker depth 1, declared local capacity 1, "
                    f"1024-byte slot stride in Vec via MTE2, reserve bytes [{base}, {base + 1024})"
                ) in why
        ir.assert_structural_equal(before.body, after.body)


def _peer_transport_effects(program: ir.Program, name: str):
    function = program.get_function(name)
    assert function is not None, name
    return testing.analyze_peer_transport_effects(function)


class _SharePopExpression(ir.IRMutator):
    pop = None

    def visit_call(self, op):
        if op.op.name != ir.get_op("tile.tpop_from_aiv").name:
            return super().visit_call(op)
        if self.pop is None:
            self.pop = op
        return self.pop


@pytest.mark.usefixtures("ascend_backend")
def test_peer_fifo_counts_each_statement_with_a_shared_pop_expression():
    source = pl.parse_program("""
@pl.program
class RepeatedPopExpression:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self):
        fifo = pl.reserve_buffer(name="fifo", size=1024, base=0)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), fifo, dir_mask=2, slot_size=512, slot_num=2)
        first: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
        second: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
        pl.tfree_to_aiv(first)
        pl.tfree_to_aiv(second)
""")

    # A loop unroller can reuse a nullary TPOP expression while allocating a
    # distinct SSA result for each cloned statement. Both borrows are live here.
    source = _SharePopExpression().visit_program(source)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_policy="conservative_prefix"):
        result = passes.realize_sync()(source)
    cube = result.get_function("cube")
    assert cube is not None
    assert not cube.attrs["pypto_autosync"]
    why = cube.attrs["autosync_fallback_reason"]
    assert (
        "pipe id 0 has peak marker depth 2, declared local capacity 2, "
        "512-byte slot stride in Mat via MTE2, reserve bytes [0, 1024)"
    ) in why
    sites = _peer_transport_effects(source, "cube")["sites"]
    assert [site[0] for site in sites] == ["pop", "pop", "free", "free"]


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("split,slot_bytes", [(0, 1024), (1, 1024), (0, 128)])
def test_native_peer_transport_effects(split: int, slot_bytes: int):
    source = pl.parse_program(f"""
@pl.program
class Exchange:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, acc: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc]):
        mat = pl.reserve_buffer(name="mat", size=2048, base=8192)
        peer = pl.import_peer_buffer(name="vec", peer_func="vector")
        pl.aic_initialize_pipe(peer, mat, dir_mask=3, slot_size={slot_bytes},
                               slot_num=6, local_slot_num=2, id=7)
        pl.tpush_to_aiv(acc, split={split}, id=7)
        x: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split={split}, id=7)
        pl.tfree_to_aiv(x, id=7)

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, source: pl.Tile[[16, 16], pl.FP16, pl.Mem.Vec]):
        vec = pl.reserve_buffer(name="vec", size=2048, base=4096)
        peer = pl.import_peer_buffer(name="mat", peer_func="cube")
        pl.aiv_initialize_pipe(vec, peer, dir_mask=3, slot_size={slot_bytes},
                               slot_num=6, local_slot_num=2, id=7)
        x: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split={split}, id=7)
        pl.tfree_to_aic(x, id=7)
        pl.tpush_to_aic(source, split={split}, id=7)
""")
    cube, vector = (_peer_transport_effects(source, s) for s in ("cube", "vector"))
    if split or slot_bytes < 1024:
        for side in (cube, vector):
            assert side["unavailable"] and not side["sites"] and not side["pipes"]
        return
    assert cube["unavailable"] == vector["unavailable"] == ""
    assert cube["pipes"] == vector["pipes"] == {7: (3, 6, 1024, 3, 2)}
    # Relative group flags depend on direction, even with a nonzero logical
    # pipe id. A free marker names the same borrowed storage as its pop.
    assert cube["sites"] == [
        ("push", 7, "Acc", "FIX", 1024, 0, 1),
        ("pop", 7, "Mat", "MTE2", 512, 2, 3),
        ("free", 7, "Mat", "MTE2", 512, 2, 3),
    ]
    assert vector["sites"] == [
        ("pop", 7, "Vec", "MTE2", 1024, 0, 1),
        ("free", 7, "Vec", "MTE2", 1024, 0, 1),
        ("push", 7, "Vec", "MTE3", 512, 2, 3),
    ]


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("early_free,output_base", [(False, 2048), (True, 2048), (False, 128)])
def test_native_receive_borrow_reader_cuts(early_free: bool, output_base: int):
    release = "pl.tfree_to_aic(received)"
    source = pl.parse_program(f"""
@pl.program
class ReadBorrow:
    @pl.function(type=pl.FunctionType.AIV)
    def vector(self):
        reserve = pl.reserve_buffer(name="ring", size=2048, base=0)
        pl.aiv_initialize_pipe(reserve, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)
        allocation: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        for i in pl.range(8):
            received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
            alias: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.set_validshape(received, 16, 16)
            first: pl.Tile[[16, 16], pl.FP32, pl.MemRef(allocation, {output_base}, 1024), pl.Mem.Vec] = (
                pl.mul(alias, 2.0)
            )
            {release if early_free else ""}
            last: pl.Tile[[16, 16], pl.FP32, pl.MemRef(allocation, {output_base}, 1024), pl.Mem.Vec] = (
                pl.mul(alias, 3.0)
            )
            {"" if early_free else release}
""")
    evidence = _peer_transport_effects(source, "vector")
    assert evidence["unavailable"] == ""
    if early_free:
        assert "after logical release" in evidence["borrow_unavailable"]
    elif output_base < 2048:
        assert "overlaps a native receive reserve" in evidence["borrow_unavailable"]
    else:
        assert evidence["borrow_unavailable"] == ""
        [(pipe_id, distance, empty, readers)] = evidence["borrows"]
        assert (pipe_id, distance, empty) == (0, 2, False)
        assert list(readers) == ["V"]
        assert [cut.var.name_hint for cut in readers["V"]] == ["first", "last"]


@pytest.mark.usefixtures("ascend_backend")
def test_native_receive_matrix_move_and_empty_vector_lane():
    source = pl.parse_program("""
@pl.program
class PeerUses:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self):
        ring = pl.reserve_buffer(name="mat", size=8192, base=0)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), ring, dir_mask=2, slot_size=4096, slot_num=2)
        for i in pl.range(8):
            mat: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
            left: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.move(mat, target_memory=pl.Mem.Left)
            pl.tfree_to_aiv(mat)

    @pl.function(type=pl.FunctionType.AIV)
    def shadow(self):
        ring = pl.reserve_buffer(name="vec", size=16384, base=0)
        pl.aiv_initialize_pipe(ring, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, slot_num=2)
        for i in pl.range(8):
            empty: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = (
                pl.tpop_from_aic(split=0)
            )
            pl.tfree_to_aic(empty)
""")
    cube = _peer_transport_effects(source, "cube")
    shadow = _peer_transport_effects(source, "shadow")
    assert cube["borrow_unavailable"] == shadow["borrow_unavailable"] == ""
    [(pipe_id, distance, empty, readers)] = cube["borrows"]
    assert (pipe_id, distance, empty) == (0, 2, False)
    assert list(readers) == ["MTE1"]
    assert [cut.var.name_hint for cut in readers["MTE1"]] == ["left", "left"]
    assert shadow["borrows"] == [(0, 2, True, {})]
    assert [site[0] for site in shadow["sites"]] == ["pop", "free"]


# ============================================================================
# Test Program: Vector Producer + Cube Consumer (V2C unidirectional)
# ============================================================================


@pl.program
class CrossCoreTpushTpopProgram:
    """V2C unidirectional cross-core program with orchestration wrapper.

    Vector producer: loads tiles a and b, computes add and sub, pushes both to Cube.
    Cube consumer: pops tiles, performs matmul, stores result.
    """

    @pl.function(type=pl.FunctionType.AIV)
    def vector_producer(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
        pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=v2c_peer)

        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP16] = pl.load(b, [0, 0], [16, 16])
        result_add: pl.Tile[[16, 16], pl.FP16] = pl.add(tile_a, tile_b)
        result_sub: pl.Tile[[16, 16], pl.FP16] = pl.sub(tile_a, tile_b)

        pl.tpush_to_aic(result_add, split=1)
        pl.tpush_to_aic(result_sub, split=1)

    @pl.function(type=pl.FunctionType.AIC)
    def cube_consumer(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
        pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf)

        received_add: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=1)
        received_sub: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=1)
        received_add_left = pl.move(received_add, target_memory=pl.Mem.Left)
        received_sub_right = pl.move(received_sub, target_memory=pl.Mem.Right)

        mm_result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(received_add_left, received_sub_right)

        pl.tfree_to_aiv(received_add)
        pl.tfree_to_aiv(received_sub)

        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(mm_result, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        updated = self.cube_consumer(a, b, output)
        self.vector_producer(a, b, output)
        return updated

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        out = self.group_func(a, b, output)
        return out


# ============================================================================
# Bidirectional Test Program
# ============================================================================


@pl.program
class BidirectionalCrossCorProgram:
    """Bidirectional cross-core: Vector preprocesses → Cube matmul → Vector post-processes.

    vector_bidir: Loads + adds (V2C push), receives matmul result (C2V pop), applies exp, stores.
    cube_bidir: Receives preprocessed data (V2C pop), does matmul, pushes result back (C2V push).
    """

    @pl.function(type=pl.FunctionType.InCore)
    def vector_bidir(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        # C2V consumer: reserve buffer for incoming data from Cube (explicit base)
        c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=2048, base=0x2000)
        # V2C producer: import cube's reserved buffer
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_bidir")
        # Bidirectional init with consumer buffer addresses
        pl.aiv_initialize_pipe(dir_mask=3, slot_size=512, c2v_consumer_buf=c2v_buf, v2c_consumer_buf=v2c_peer)

        # Preprocess: elementwise add (Vector op)
        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP16] = pl.load(b, [0, 0], [16, 16])
        sum_tile: pl.Tile[[16, 16], pl.FP16] = pl.add(tile_a, tile_b)

        # Push preprocessed data to Cube for matmul (V2C direction)
        pl.tpush_to_aic(sum_tile, split=0)

        # Receive matmul result back from Cube (C2V direction)
        mm_result: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)

        # Post-process: apply exp (Vector op)
        processed: pl.Tile[[16, 16], pl.FP32] = pl.exp(mm_result)

        # Release C2V slot
        pl.tfree_to_aic(mm_result)

        # Store final result
        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(processed, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.InCore)
    def cube_bidir(
        self,
        weight: pl.Tensor[[16, 16], pl.FP16],
    ):
        # V2C consumer: reserve buffer for incoming data from Vector (explicit base)
        v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=2048, base=0x1000)
        # C2V producer: import vector's reserved buffer
        c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_bidir")
        # Bidirectional init with explicit consumer buffer addresses
        pl.aic_initialize_pipe(dir_mask=3, slot_size=512, c2v_consumer_buf=c2v_peer, v2c_consumer_buf=v2c_buf)

        # Receive preprocessed tile from Vector (V2C direction)
        received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)

        # Matmul (Cube op)
        w_tile: pl.Tile[[16, 16], pl.FP16] = pl.load(weight, [0, 0], [16, 16])
        mm_result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(received, w_tile)

        # Release V2C slot
        pl.tfree_to_aiv(received)

        # Push matmul result back to Vector for post-processing (C2V direction)
        pl.tpush_to_aiv(mm_result, split=0)


# ============================================================================
# Multiple-Pipe Cross-Core Test Program
# ============================================================================


@pl.program
class MultiPipeSameDirectionCrossCoreProgram:
    """Two explicit V2C frontend pipes from Vector to Cube."""

    @pl.function(type=pl.FunctionType.InCore)
    def vector_multi_pipe(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
    ):
        v2c_peer_0 = pl.import_peer_buffer(name="v2c_slot_buffer_0", peer_func="cube_multi_pipe")
        v2c_peer_1 = pl.import_peer_buffer(name="v2c_slot_buffer_1", peer_func="cube_multi_pipe")
        pl.aiv_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_peer_0,
            dir_mask=2,
            slot_size=512,
            id=0,
        )
        pl.aiv_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_peer_1,
            dir_mask=2,
            slot_size=512,
            id=1,
        )

        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP16] = pl.load(b, [0, 0], [16, 16])
        sum_tile: pl.Tile[[16, 16], pl.FP16] = pl.add(tile_a, tile_b)
        diff_tile: pl.Tile[[16, 16], pl.FP16] = pl.sub(tile_a, tile_b)
        pl.tpush_to_aic(sum_tile, split=0, id=0)
        pl.tpush_to_aic(diff_tile, split=0, id=1)

    @pl.function(type=pl.FunctionType.InCore)
    def cube_multi_pipe(self):
        v2c_buf_0 = pl.reserve_buffer(name="v2c_slot_buffer_0", size=4096, base=0x1000)
        v2c_buf_1 = pl.reserve_buffer(name="v2c_slot_buffer_1", size=4096, base=0x2000)
        pl.aic_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_buf_0,
            dir_mask=2,
            slot_size=512,
            id=0,
        )
        pl.aic_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_buf_1,
            dir_mask=2,
            slot_size=512,
            id=1,
        )

        sum_tile: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0, id=0)
        diff_tile: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0, id=1)
        sum_left = pl.move(sum_tile, target_memory=pl.Mem.Left)
        diff_right = pl.move(diff_tile, target_memory=pl.Mem.Right)
        _ = pl.matmul(sum_left, diff_right)
        pl.tfree_to_aiv(sum_tile)
        pl.tfree_to_aiv(diff_tile)


# ============================================================================
# Explicit slot_num Test Program (V2C unidirectional)
# ============================================================================


@pl.program
class CrossCoreExplicitSlotNumProgram:
    """V2C unidirectional program that pins explicit slot_num and local_slot_num.

    Mirrors CrossCoreTpushTpopProgram but passes slot_num=16 and local_slot_num=4
    to both init pipes (a3 sizes the buffer slot_size * local_slot_num).
    """

    @pl.function(type=pl.FunctionType.AIV)
    def vector_producer(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
        pl.aiv_initialize_pipe(
            dir_mask=2, slot_size=512, slot_num=16, local_slot_num=4, v2c_consumer_buf=v2c_peer
        )

        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP16] = pl.load(b, [0, 0], [16, 16])
        result_add: pl.Tile[[16, 16], pl.FP16] = pl.add(tile_a, tile_b)

        pl.tpush_to_aic(result_add, split=1)

    @pl.function(type=pl.FunctionType.AIC)
    def cube_consumer(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=2048, base=0x1000)
        pl.aic_initialize_pipe(
            dir_mask=2, slot_size=512, slot_num=16, local_slot_num=4, v2c_consumer_buf=pipe_buf
        )

        received_add: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=1)
        received_add_left = pl.move(received_add, target_memory=pl.Mem.Left)

        mm_result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(received_add_left, received_add_left)

        pl.tfree_to_aiv(received_add)

        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(mm_result, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        updated = self.cube_consumer(a, b, output)
        self.vector_producer(a, b, output)
        return updated

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        out = self.group_func(a, b, output)
        return out


# ============================================================================
# Test Suite
# ============================================================================


class TestCrossCoreTpushTpopCodegen:
    """Tests for cross-core TPUSH/TPOP PTO code generation."""

    @staticmethod
    def _compile_and_generate(program) -> dict[str, str]:
        """Compile program and return dict of {func_name: mlir_code}.

        Uses a custom pipeline matching the Default strategy but without
        ExpandMixedKernel, since these programs already have manual cross-core
        setup with explicit AIC/AIV tpush/tpop ops.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        # backend.set_backend_type(BackendType.Ascend950)

        pipeline = passes.PassPipeline()
        for factory in [
            passes.unroll_loops,
            passes.convert_to_ssa,
            passes.flatten_call_expr,
            passes.outline_incore_scopes,
            passes.outline_cluster_scopes,
            passes.convert_tensor_to_tile_ops,
            passes.flatten_tile_nd_to_2d,
            passes.infer_tile_memory_space,
            passes.resolve_backend_op_layouts,
            passes.init_mem_ref,
            passes.memory_reuse,
            passes.allocate_memory_addr,
            passes.stamp_tfree_split,
        ]:
            pipeline.add_pass(factory())
        optimized = pipeline.run(program)

        result = {}
        codegen_instance = codegen.PTOCodegen()
        groups, ungrouped = _build_group_mapping(optimized)

        # Grouped: one module per group
        for group_name, members in groups.items():
            grouped_program = ir.Program(members, group_name, optimized.span)
            mlir_code = codegen_instance.generate(grouped_program)
            result[group_name] = mlir_code
            for func in members:
                result[func.name] = mlir_code

        # Ungrouped: one module per function (existing behavior)
        for func in ungrouped:
            single = ir.Program([func], func.name, optimized.span)
            mlir_code = codegen_instance.generate(single)
            result[func.name] = mlir_code
        return result

    def test_unidirectional_v2c_vector_producer(self):
        """Test Vector producer generates correct V2C cross-core PTO ops."""
        codes = self._compile_and_generate(CrossCoreTpushTpopProgram)
        vector_code = codes["vector_producer"]

        assert vector_code, "Vector producer MLIR should not be empty"
        assert "pto.import_reserved_buffer" in vector_code, "Should contain pto.import_reserved_buffer"
        assert "peer_func = @cube_consumer" in vector_code, (
            "Should reference cube_consumer with MLIR symbol syntax"
        )
        assert "-> i32" in vector_code, "import_reserved_buffer should return i32"
        assert "pto.aiv_initialize_pipe" in vector_code, "Should contain pto.aiv_initialize_pipe"
        assert "dir_mask = 2" in vector_code, "Should have dir_mask = 2 (V2C)"
        assert "v2c_consumer_buf = " in vector_code, "Should have v2c_consumer_buf as SSA reference"
        assert "c2v_consumer_buf = " in vector_code, "Should have c2v_consumer_buf as SSA reference"
        assert "pto.tpush_to_aic" in vector_code, "Should contain pto.tpush_to_aic"
        assert "pto.tadd" in vector_code, "Should contain elementwise add (Vector op)"

    def test_unidirectional_v2c_cube_consumer(self):
        """Test Cube consumer generates correct V2C cross-core PTO ops."""
        codes = self._compile_and_generate(CrossCoreTpushTpopProgram)
        cube_code = codes["cube_consumer"]

        assert cube_code, "Cube consumer MLIR should not be empty"
        assert "pto.reserve_buffer" in cube_code, "Should contain pto.reserve_buffer"
        assert 'name = "v2c_slot_buffer"' in cube_code, "Should reference v2c_slot_buffer"
        assert "auto = false" in cube_code, "Should have auto = false for explicit base address"
        assert "base = 4096" in cube_code, "Should have explicit base address (0x1000 = 4096)"
        assert "location = #pto.address_space<" in cube_code, "Should have location attribute"
        assert "-> i32" in cube_code, "reserve_buffer should return i32"
        assert "pto.aic_initialize_pipe" in cube_code, "Should contain pto.aic_initialize_pipe"
        assert "dir_mask = 2" in cube_code, "Should have dir_mask = 2 (V2C)"
        assert "v2c_consumer_buf = " in cube_code, "Should have v2c_consumer_buf as SSA reference"
        assert "c2v_consumer_buf = " in cube_code, "Should have c2v_consumer_buf as SSA reference"
        assert "arith.constant 0 : i32" in cube_code, "Should emit i32 constant for default consumer buf"
        assert "pto.tpop_from_aiv" in cube_code, "Should contain pto.tpop_from_aiv"
        assert "= pto.tpop_from_aiv" in cube_code, "tpop should produce SSA result"
        assert "-> !pto.tile_buf<" in cube_code, "tpop should use -> result type syntax"
        assert "pto.tfree_from_aiv" in cube_code, "Should contain pto.tfree_from_aiv"
        tfree_line = next(line for line in cube_code.splitlines() if "pto.tfree_from_aiv" in line)
        assert "{split = " in tfree_line, f"tfree should have split attribute: {tfree_line}"
        assert "pto.tmatmul" in cube_code, "Should contain matmul (Cube op)"

    def test_explicit_slot_num_emitted(self):
        """Explicit slot_num / local_slot_num flow into both AIV and AIC init pipe PTO ops."""
        codes = self._compile_and_generate(CrossCoreExplicitSlotNumProgram)

        vector_code = codes["vector_producer"]
        assert "pto.aiv_initialize_pipe" in vector_code, "Should contain pto.aiv_initialize_pipe"
        assert "slot_num = 16" in vector_code, f"AIV init pipe should carry slot_num = 16:\n{vector_code}"
        assert "local_slot_num = 4" in vector_code, (
            f"AIV init pipe should carry local_slot_num = 4:\n{vector_code}"
        )

        cube_code = codes["cube_consumer"]
        assert "pto.aic_initialize_pipe" in cube_code, "Should contain pto.aic_initialize_pipe"
        assert "slot_num = 16" in cube_code, f"AIC init pipe should carry slot_num = 16:\n{cube_code}"
        assert "local_slot_num = 4" in cube_code, (
            f"AIC init pipe should carry local_slot_num = 4:\n{cube_code}"
        )

    def test_slot_num_omitted_by_default(self):
        """A hand-written init pipe without the knobs emits no slot_num/local_slot_num.

        Only hand-authored setup reaches PTOAS's own dir_mask-derived depth: pipes built by
        ExpandMixedKernel always carry an explicit slot_num.
        """
        codes = self._compile_and_generate(CrossCoreTpushTpopProgram)
        assert "slot_num" not in codes["vector_producer"], "Default path must not emit slot_num"
        assert "slot_num" not in codes["cube_consumer"], "Default path must not emit slot_num"

    def test_tpop_dynamic_valid_shape_operands(self):
        """Dynamic tpop result valid_shape should emit PTOAS frontend operands."""
        span = ir.Span.unknown()
        valid_row = ir.Var("valid_row", ir.ScalarType(pl.INDEX), span)
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)
        tile_view = ir.TileView(valid_shape=[valid_row, valid_col])
        # MemRef-less: a tpop result owns no general-pool buffer (InitMemRef leaves
        # it MemRef-less); codegen must emit no alloc_tile from the op name alone.
        tile_type = ir.TileType([16, 64], pl.FP32, None, tile_view, ir.MemorySpace.Vec)
        recv_tile = ir.Var("recv_tile", tile_type, span)
        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 2}, tile_type, span)
        body = ir.SeqStmts([ir.AssignStmt(recv_tile, tpop_call, span)], span)
        func = ir.Function(
            "dynamic_tpop",
            [(valid_row, ir.ParamDirection.In), (valid_col, ir.ParamDirection.In)],
            [],
            body,
            span,
            ir.FunctionType.AIV,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "dynamic_tpop_program", span))
        tpop_line = next(line.strip() for line in mlir_code.splitlines() if "pto.tpop_from_aic" in line)

        assert "pto.tpop_from_aic(%arg0, %arg1) {split = 2}" in tpop_line
        assert "v_row=?" in tpop_line
        assert "v_col=?" in tpop_line
        assert "pto.alloc_tile" not in mlir_code

    def test_tpop_no_split_acc_to_vec_preserves_dynamic_valid_shape(self):
        """A dynamic C2V valid shape stays on TPOP because treshape has no dynamic operands."""
        span = ir.Span.unknown()
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)
        tile_view = ir.TileView(valid_shape=[ir.ConstInt(8, pl.INT64, span), valid_col])
        # MemRef-less: a tpop result owns no general-pool buffer (InitMemRef leaves
        # it MemRef-less); codegen must emit no alloc_tile from the op name alone.
        tile_type = ir.TileType([16, 64], pl.FP32, None, tile_view, ir.MemorySpace.Vec)
        recv_tile = ir.Var("recv_tile", tile_type, span)
        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, tile_type, span)
        body = ir.SeqStmts([ir.AssignStmt(recv_tile, tpop_call, span)], span)
        func = ir.Function(
            "dynamic_tpop_static_row",
            [(valid_col, ir.ParamDirection.In)],
            [],
            body,
            span,
            ir.FunctionType.AIV,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "dynamic_tpop_static_row_program", span))
        lines = [line.strip() for line in mlir_code.splitlines()]
        tpop_line = next(line for line in lines if "pto.tpop_from_aic" in line)

        assert "pto.tpop_from_aic(%c8_index, %arg0) {split = 0}" in tpop_line
        assert "v_row=?" in tpop_line
        assert "v_col=?" in tpop_line
        assert "pto.treshape" not in mlir_code
        assert "pto.subview" not in mlir_code
        assert "pto.set_validshape" not in mlir_code

    def test_tpop_static_non_full_valid_shape_operands(self):
        """Static tpop valid_shape smaller than physical shape should emit explicit operands."""
        span = ir.Span.unknown()
        tile_view = ir.TileView(valid_shape=[ir.ConstInt(0, pl.INDEX, span), ir.ConstInt(0, pl.INDEX, span)])
        # MemRef-less: a tpop result owns no general-pool buffer (InitMemRef leaves
        # it MemRef-less); codegen must emit no alloc_tile from the op name alone.
        tile_type = ir.TileType([16, 64], pl.FP32, None, tile_view, ir.MemorySpace.Vec)
        recv_tile = ir.Var("recv_tile", tile_type, span)
        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, tile_type, span)
        body = ir.SeqStmts([ir.AssignStmt(recv_tile, tpop_call, span)], span)
        func = ir.Function("empty_tpop", [], [], body, span, ir.FunctionType.AIV)

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "empty_tpop_program", span))
        tpop_line = next(line.strip() for line in mlir_code.splitlines() if "pto.tpop_from_aic" in line)

        assert "pto.tpop_from_aic(%c0_index, %c0_index) {split = 0}" in tpop_line
        assert "v_row=?" in tpop_line
        assert "v_col=?" in tpop_line

    def test_tpop_dynamic_valid_shape_rejects_bool_operand(self):
        """Dynamic tpop valid_shape operands must be integer or index typed."""
        span = ir.Span.unknown()
        valid_row = ir.Var("valid_row", ir.ScalarType(pl.BOOL), span)
        tile_view = ir.TileView(valid_shape=[valid_row, ir.ConstInt(64, pl.INT64, span)])
        # MemRef-less: a tpop result owns no general-pool buffer (InitMemRef leaves
        # it MemRef-less); codegen must emit no alloc_tile from the op name alone.
        tile_type = ir.TileType([16, 64], pl.FP32, None, tile_view, ir.MemorySpace.Vec)
        recv_tile = ir.Var("recv_tile", tile_type, span)
        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, tile_type, span)
        body = ir.SeqStmts([ir.AssignStmt(recv_tile, tpop_call, span)], span)
        func = ir.Function(
            "dynamic_tpop_bool_row",
            [(valid_row, ir.ParamDirection.In)],
            [],
            body,
            span,
            ir.FunctionType.AIV,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        with pytest.raises(
            ValueError, match="tpop valid_shape operand must be integer or index type, got i1"
        ):
            codegen.PTOCodegen().generate(ir.Program([func], "dynamic_tpop_bool_row_program", span))

    @pytest.mark.parametrize(
        ("tpush_op_name", "func_type", "memory_space"),
        [
            ("tile.tpush_to_aiv", ir.FunctionType.AIC, ir.MemorySpace.Acc),
            ("tile.tpush_to_aic", ir.FunctionType.AIV, ir.MemorySpace.Vec),
        ],
    )
    def test_tpush_uses_validshape_aliased_tile(self, tpush_op_name, func_type, memory_space):
        """set_validshape + cross-core tpush should push the in-place validShape tile handle."""
        span = ir.Span.unknown()

        src = ir.Var("src", ir.TensorType([32, 32], pl.FP32), span)
        valid_row = ir.Var("valid_row", ir.ScalarType(pl.INDEX), span)
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)

        zero = ir.ConstInt(0, pl.INDEX, span)
        shape_32 = ir.ConstInt(32, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([shape_32, shape_32], span)

        load_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 32 * 32 * 4, 0)
        load_view = ir.TileView(
            valid_shape=[shape_32, shape_32],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        load_type = ir.TileType([32, 32], pl.FP32, load_memref, load_view, memory_space)
        src_tile = ir.Var("src_tile", load_type, span)

        narrowed_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 32 * 32 * 4, 0)
        narrowed_view = ir.TileView(
            valid_shape=[valid_row, valid_col],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        narrowed_type = ir.TileType([32, 32], pl.FP32, narrowed_memref, narrowed_view, memory_space)
        narrowed_tile = ir.Var("narrowed_tile", narrowed_type, span)

        load_call = ir.Call(
            ir.Op("tile.load"),
            [src, offsets, shapes, shapes],
            {"target_memory": memory_space},
            load_type,
            span,
        )
        set_validshape_call = ir.Call(
            ir.Op("tile.set_validshape"),
            [src_tile, valid_row, valid_col],
            {},
            narrowed_type,
            span,
        )
        tpush_call = ir.Call(ir.Op(tpush_op_name), [narrowed_tile], {"split": 0}, ir.UnknownType(), span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(src_tile, load_call, span),
                ir.AssignStmt(narrowed_tile, set_validshape_call, span),
                ir.EvalStmt(tpush_call, span),
            ],
            span,
        )
        func = ir.Function(
            "narrow_then_push",
            [
                (src, ir.ParamDirection.In),
                (valid_row, ir.ParamDirection.In),
                (valid_col, ir.ParamDirection.In),
            ],
            [],
            body,
            span,
            func_type,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "narrow_then_push_program", span))

        narrowed_alloc_lines = [
            line.strip()
            for line in mlir_code.splitlines()
            if "%narrowed_tile" in line and "pto.alloc_tile" in line
        ]
        set_validshape_line = next(
            line.strip() for line in mlir_code.splitlines() if "pto.set_validshape" in line
        )
        pto_tpush_op_name = tpush_op_name.replace("tile.", "pto.")
        tpush_line = next(line.strip() for line in mlir_code.splitlines() if pto_tpush_op_name in line)

        assert not narrowed_alloc_lines, (
            "Expected set_validshape result to alias the payload tile without a second alloc_tile, got:\n"
            + "\n".join(narrowed_alloc_lines)
        )
        assert "%src_tile" in set_validshape_line, (
            f"Expected set_validshape to target source payload tile SSA, got:\n{set_validshape_line}"
        )
        assert "v_row=?" in set_validshape_line and "v_col=?" in set_validshape_line, (
            f"Expected dynamic tile type on set_validshape, got:\n{set_validshape_line}"
        )
        assert "%src_tile" in tpush_line, (
            f"Expected tpush to push the source tile after in-place set_validshape, got:\n{tpush_line}"
        )
        assert "%narrowed_tile" not in tpush_line, (
            f"Expected tpush not to use a payload-less set_validshape alias alloc, got:\n{tpush_line}"
        )
        assert "v_row=?" in tpush_line and "v_col=?" in tpush_line, (
            f"Expected tpush to follow main's always-dynamic alloc_tile type annotation, got:\n{tpush_line}"
        )

    def test_no_split_acc_to_vec_uses_full_box_for_tpush_and_tpop(self):
        """A partial Acc payload crosses the no-split C2V FIFO as one full physical box."""
        span = ir.Span.unknown()
        zero = ir.ConstInt(0, pl.INDEX, span)
        rows = ir.ConstInt(16, pl.INDEX, span)
        cols = ir.ConstInt(32, pl.INDEX, span)
        valid_cols = ir.ConstInt(24, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shape = ir.MakeTuple([rows, cols], span)
        valid_shape = ir.MakeTuple([rows, valid_cols], span)

        src = ir.Var("src", ir.TensorType([16, 32], pl.FP32), span)
        acc_memref = ir.MemRef(ir.MemorySpace.Acc, ir.ConstInt(0, pl.INT64, span), 16 * 32 * 4, 0)
        acc_type = ir.TileType(
            [rows, cols],
            pl.FP32,
            acc_memref,
            ir.TileView(valid_shape=[rows, valid_cols]),
            ir.MemorySpace.Acc,
        )
        acc_tile = ir.Var("acc_tile", acc_type, span)
        load_call = ir.Call(
            ir.Op("tile.load"),
            [src, offsets, shape, valid_shape],
            {"target_memory": ir.MemorySpace.Acc},
            acc_type,
            span,
        )
        push_call = ir.Call(
            ir.Op("tile.tpush_to_aiv"),
            [acc_tile],
            {"split": 0},
            ir.UnknownType(),
            span,
        )
        producer = ir.Function(
            "partial_acc_producer",
            [(src, ir.ParamDirection.In)],
            [],
            ir.SeqStmts(
                [ir.AssignStmt(acc_tile, load_call, span), ir.EvalStmt(push_call, span)],
                span,
            ),
            span,
            ir.FunctionType.AIC,
        )

        vec_type = ir.TileType(
            [rows, cols],
            pl.FP32,
            None,
            ir.TileView(valid_shape=[rows, valid_cols]),
            ir.MemorySpace.Vec,
        )
        popped = ir.Var("popped", vec_type, span)
        pop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, vec_type, span)
        consumer = ir.Function(
            "partial_vec_consumer",
            [],
            [],
            ir.SeqStmts([ir.AssignStmt(popped, pop_call, span)], span),
            span,
            ir.FunctionType.AIV,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        mlir_code = codegen.PTOCodegen().generate(
            ir.Program([producer, consumer], "partial_acc_to_vec_program", span)
        )
        producer_code = _extract_func_section(mlir_code, "partial_acc_producer")
        consumer_code = _extract_func_section(mlir_code, "partial_vec_consumer")

        producer_lines = [line.strip() for line in producer_code.splitlines()]
        push_index = next(i for i, line in enumerate(producer_lines) if "pto.tpush_to_aiv" in line)
        producer_validshape = [
            (i, line) for i, line in enumerate(producer_lines) if "pto.set_validshape" in line
        ]
        assert len(producer_validshape) == 2
        assert producer_validshape[0][0] < push_index < producer_validshape[1][0]
        assert ", %c16_index, %c32_index :" in producer_validshape[0][1]
        assert ", %c16_index, %c24_index :" in producer_validshape[1][1]

        consumer_lines = [line.strip() for line in consumer_code.splitlines()]
        pop_index = next(i for i, line in enumerate(consumer_lines) if "pto.tpop_from_aic" in line)
        reshape_index = next(i for i, line in enumerate(consumer_lines) if "pto.treshape" in line)
        assert "pto.tpop_from_aic(%c16_index, %c32_index) {split = 0}" in consumer_lines[pop_index]
        transport = re.search(r"^(%\w+) = pto\.tpop_from_aic", consumer_lines[pop_index])
        assert transport is not None
        assert pop_index < reshape_index
        assert f"pto.treshape {transport.group(1)}" in consumer_lines[reshape_index]
        assert "rows=16, cols=32, v_row=16, v_col=24" in consumer_lines[reshape_index]
        assert "pto.set_validshape" not in consumer_code, (
            "a frontend tpop result is not a locally bound PTOAS tile; logical shape "
            f"restoration must use a metadata-only treshape:\n{consumer_code}"
        )

    def _row_narrowed_acc_push_program(
        self,
        valid_rows,
        valid_cols,
        compact,
        split=0,
        rows=64,
        backend_type=BackendType.Ascend910,
        *,
        dtype=pl.INT32,
        lane_stride=None,
    ):
        """One AIC function that pushes a row-narrowed Acc tile into the C2V FIFO.

        ``valid_rows=None`` makes the row extent a runtime ``INDEX`` parameter.
        """
        span = ir.Span.unknown()
        zero = ir.ConstInt(0, pl.INDEX, span)
        box_rows = rows
        rows = ir.ConstInt(box_rows, pl.INDEX, span)
        cols = ir.ConstInt(128, pl.INDEX, span)
        runtime_rows = ir.Var("valid_row", ir.ScalarType(pl.INDEX), span) if valid_rows is None else None
        v_rows = runtime_rows if runtime_rows is not None else ir.ConstInt(valid_rows, pl.INDEX, span)
        v_cols = ir.ConstInt(valid_cols, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shape = ir.MakeTuple([rows, cols], span)
        valid_shape = ir.MakeTuple([v_rows, v_cols], span)

        src = ir.Var("src", ir.TensorType([box_rows, 128], dtype), span)
        acc_memref = ir.MemRef(ir.MemorySpace.Acc, ir.ConstInt(0, pl.INT64, span), box_rows * 128 * 4, 0)
        acc_type = ir.TileType(
            [rows, cols],
            dtype,
            acc_memref,
            ir.TileView(valid_shape=[v_rows, v_cols], compact=compact),
            ir.MemorySpace.Acc,
        )
        acc_tile = ir.Var("acc_tile", acc_type, span)
        load_call = ir.Call(
            ir.Op("tile.load"),
            [src, offsets, shape, valid_shape],
            {"target_memory": ir.MemorySpace.Acc},
            acc_type,
            span,
        )
        kwargs = {"split": split}
        if lane_stride is not None:
            kwargs["lane_stride"] = lane_stride
        push_call = ir.Call(ir.Op("tile.tpush_to_aiv"), [acc_tile], kwargs, ir.UnknownType(), span)
        producer = ir.Function(
            "narrow_acc_producer",
            [(src, ir.ParamDirection.In)]
            + ([(runtime_rows, ir.ParamDirection.In)] if runtime_rows is not None else []),
            [],
            ir.SeqStmts([ir.AssignStmt(acc_tile, load_call, span), ir.EvalStmt(push_call, span)], span),
            span,
            ir.FunctionType.AIC,
        )
        backend.reset_for_testing()
        backend.set_backend_type(backend_type)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([producer], "narrow_acc_push_program", span))
        return _extract_func_section(mlir_code, "narrow_acc_producer")

    def test_no_split_acc_to_vec_transport_keeps_the_producer_row_extent(self):
        """A row-narrowed compact Acc must reach TPUSH at its own row extent.

        ``mad`` lays the product out in L0C with an N-fractal stride taken from
        the L0A operand's *valid* rows (pto-isa ``TMatmul.hpp``:
        ``uint16_t m = aMatrix.GetValidRow()``), and ``TStoreAccNz2nd`` -- which
        is what TPUSH runs -- recomputes that stride as
        ``ceil(validRow/16)*16`` for a compact tile. Widening validRow to the
        physical box before the push therefore makes the fix-pipe walk L0C at a
        pitch ``mad`` never wrote at: with a 64-row box valid to 16 the push
        picks up N-fractal 4j for every fractal j (issue #2510). Columns are a
        different story and stay widened -- see the sibling test.
        """
        producer_code = self._row_narrowed_acc_push_program(16, 128, ir.CompactMode.normal)
        assert "pto.set_validshape" not in producer_code, (
            "a full-width row-narrowed Acc needs no transport widening at all; widening the rows "
            f"desynchronizes TPUSH from the pitch mad wrote at:\n{producer_code}"
        )

    def test_no_split_acc_to_vec_transport_widens_columns_only(self):
        """Columns still widen to the box; rows still do not.

        The FIFO slot's row pitch is the producer's physical column count, so a
        partially written column range leaves stale bytes *inside* the valid
        rows. That is why the column half of the transport normalization stays.
        """
        producer_code = self._row_narrowed_acc_push_program(16, 96, ir.CompactMode.normal)
        producer_lines = [line.strip() for line in producer_code.splitlines()]
        push_index = next(i for i, line in enumerate(producer_lines) if "pto.tpush_to_aiv" in line)
        validshape = [(i, line) for i, line in enumerate(producer_lines) if "pto.set_validshape" in line]
        assert len(validshape) == 2, producer_code
        assert validshape[0][0] < push_index < validshape[1][0]
        assert ", %c16_index, %c128_index :" in validshape[0][1], (
            f"transport must keep the producer's 16 valid rows and widen only the columns:\n{producer_code}"
        )
        assert ", %c16_index, %c96_index :" in validshape[1][1]

    def test_split_acc_to_vec_transport_keeps_the_producer_row_extent(self):
        """A compact narrowed Acc crosses a *split* boundary at its own row extent.

        The no-split rule holds at a split too: the push reads L0C at the pitch
        ``ceil(validRow/16)*16``, which matches what ``mad`` wrote only while
        validRow is the matmul's own. No lane needs a row past it -- a lane's band
        is ``clamp(V - lane * S, 0, S)`` rows from ``lane * S`` -- so widening the
        rows to the box only skews the valid ones (on device: 1792 of 2048 wrong
        for a 64-row box valid to 16 across ``pl.split(UP_DOWN)``).
        """
        for split in (1, 2):
            producer_code = self._row_narrowed_acc_push_program(16, 128, ir.CompactMode.normal, split=split)
            assert "pto.tpush_to_aiv" in producer_code
            assert "pto.set_validshape" not in producer_code, (
                f"a full-width compact Acc needs no transport widening at split={split}; widening its "
                f"rows desynchronizes TPUSH from the pitch mad wrote at:\n{producer_code}"
            )

    def test_split_acc_to_vec_transport_widens_columns_only(self):
        """At a split, columns still widen to the box while a compact Acc's rows do not."""
        producer_code = self._row_narrowed_acc_push_program(40, 96, ir.CompactMode.normal, split=1)
        producer_lines = [line.strip() for line in producer_code.splitlines()]
        push_index = next(i for i, line in enumerate(producer_lines) if "pto.tpush_to_aiv" in line)
        validshape = [(i, line) for i, line in enumerate(producer_lines) if "pto.set_validshape" in line]
        assert len(validshape) == 2, producer_code
        assert validshape[0][0] < push_index < validshape[1][0]
        assert ", %c40_index, %c128_index :" in validshape[0][1], (
            f"transport must keep the producer's 40 valid rows and widen only the columns:\n{producer_code}"
        )
        assert ", %c40_index, %c96_index :" in validshape[1][1]

    def test_split_acc_to_vec_non_compact_transport_keeps_the_full_box(self):
        """A non-compact Acc reads L0C at ``Rows`` whatever its validRow, so its box transport stays."""
        producer_code = self._row_narrowed_acc_push_program(16, 128, ir.CompactMode.null, split=1)
        validshape = [line.strip() for line in producer_code.splitlines() if "pto.set_validshape" in line]
        assert len(validshape) == 2, producer_code
        assert ", %c64_index, %c128_index :" in validshape[0], producer_code

    def test_split_acc_to_vec_transport_keeps_a_runtime_row_extent(self):
        """A runtime row extent crosses at its own value too; only the lanes' pops defer it."""
        producer_code = self._row_narrowed_acc_push_program(None, 128, ir.CompactMode.normal, split=1)
        assert "pto.tpush_to_aiv" in producer_code
        assert "pto.set_validshape" not in producer_code, (
            f"a runtime-extent compact Acc must not be widened to the box before TPUSH:\n{producer_code}"
        )

    def test_hardware_split_acc_to_vec_rejects_a_row_narrowed_compact_accumulator(self):
        """Where the push splits its rows in hardware, the box and the compact pitch conflict.

        Ascend950 hands each vector lane half of the TRANSPORTED rows (a dual-mode
        TMOV), so a split push has to carry the box to put lane 1 at the box half,
        and that re-derives the compact tile's L0C pitch from the box. The shape
        is refused there rather than lowered into skewed data.
        """
        with pytest.raises(
            ValueError, match="split push hands each vector lane half of the transported rows"
        ):
            self._row_narrowed_acc_push_program(
                16, 128, ir.CompactMode.normal, split=1, backend_type=BackendType.Ascend950
            )

    @pytest.mark.parametrize("valid_rows", [96, None])
    def test_hardware_split_at_valid_rows_preserves_accumulator_pitch(self, valid_rows):
        code = self._row_narrowed_acc_push_program(
            valid_rows,
            128,
            ir.CompactMode.normal,
            split=1,
            rows=192,
            backend_type=BackendType.Ascend950,
            dtype=pl.FP32,
        )
        assert "pto.tpush_to_aiv" in code
        assert "pto.set_validshape" not in code

    @pytest.mark.parametrize("rows, cols", [(17, 128), (16, 64)])
    def test_hardware_split_at_valid_rows_rejects_unsupported_extents(self, rows, cols):
        with pytest.raises(
            ValueError, match="split push hands each vector lane half of the transported rows"
        ):
            self._row_narrowed_acc_push_program(
                rows,
                cols,
                ir.CompactMode.normal,
                split=1,
                backend_type=BackendType.Ascend950,
                dtype=pl.FP32,
            )

    def test_hardware_split_preserves_an_explicit_box_partition(self):
        code = self._row_narrowed_acc_push_program(
            8,
            128,
            ir.CompactMode.normal,
            split=1,
            rows=16,
            backend_type=BackendType.Ascend950,
            dtype=pl.FP32,
            lane_stride=8,
        )
        assert "pto.set_validshape" in code

    def test_hardware_split_keeps_the_box_for_runtime_rows_in_one_fractal(self):
        code = self._row_narrowed_acc_push_program(
            None,
            128,
            ir.CompactMode.normal,
            split=1,
            rows=16,
            backend_type=BackendType.Ascend950,
            dtype=pl.FP32,
        )
        assert "pto.set_validshape" in code

    def test_hardware_split_acc_to_vec_non_compact_transport_keeps_the_full_box(self):
        """A non-compact Acc reads at ``Rows`` either way, so the hardware split keeps the box."""
        producer_code = self._row_narrowed_acc_push_program(
            16, 128, ir.CompactMode.null, split=1, backend_type=BackendType.Ascend950
        )
        validshape = [line.strip() for line in producer_code.splitlines() if "pto.set_validshape" in line]
        assert len(validshape) == 2, producer_code
        assert ", %c64_index, %c128_index :" in validshape[0], producer_code

    def test_hardware_split_acc_to_vec_allows_a_single_fractal_block_accumulator(self):
        """``ceil(validRow/16)*16 == Rows`` makes the box pitch the compact one, so it crosses."""
        producer_code = self._row_narrowed_acc_push_program(
            8, 128, ir.CompactMode.normal, split=1, rows=16, backend_type=BackendType.Ascend950
        )
        assert "pto.tpush_to_aiv" in producer_code

    def test_split_acc_to_vec_allows_a_single_fractal_block_accumulator(self):
        """``ceil(validRow/16)*16 == Rows`` leaves writer and reader agreeing anyway.

        The refusal is gated on the pitches actually differing, so the shapes that
        cross a split boundary correctly today — a `[16, N]` accumulator, whatever
        its valid rows — keep doing so.
        """
        producer_code = self._row_narrowed_acc_push_program(8, 128, ir.CompactMode.normal, split=1, rows=16)
        assert "pto.tpush_to_aiv" in producer_code

    @staticmethod
    def _static_acc_to_vec_pop_code(
        shape, valid_shape, *, split=1, lane_stride=None, backend_type=BackendType.Ascend910
    ):
        """Generate a static C2V consumer without running the split-lowering passes."""
        span = ir.Span.unknown()
        tile_type = ir.TileType(
            [ir.ConstInt(dim, pl.INDEX, span) for dim in shape],
            pl.FP32,
            None,
            ir.TileView(valid_shape=[ir.ConstInt(dim, pl.INDEX, span) for dim in valid_shape]),
            ir.MemorySpace.Vec,
        )
        kwargs = {"split": split}
        if lane_stride is not None:
            kwargs["lane_stride"] = lane_stride
        popped = ir.Var("popped", tile_type, span)
        pop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], kwargs, tile_type, span)
        consumer = ir.Function(
            "split_vec_consumer",
            [],
            [],
            ir.SeqStmts([ir.AssignStmt(popped, pop_call, span)], span),
            span,
            ir.FunctionType.AIV,
        )
        backend.reset_for_testing()
        backend.set_backend_type(backend_type)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([consumer], "split_acc_to_vec_program", span))
        return _extract_func_section(mlir_code, "split_vec_consumer")

    def test_split_acc_to_vec_pop_uses_full_box_for_tpop(self):
        """A split C2V pop transports the full per-lane box, not the logical valid box.

        The pinned pto-isa derives its GM slot rows/cols from GetValidRow and
        GetValidCol. Its GM row stride is validCol for UP_DOWN and twice
        validCol for LEFT_RIGHT. Narrowing validCol therefore changes the
        producer's physical row pitch and silently reads the wrong data.
        TPOP must preserve that pitch before treshape restores logical validity.
        This is the consumer-side counterpart of the producer-side widening in
        ``test_split_tpush_uses_full_box_transport_dims``.
        """
        # Per-lane consumer half of an UP_DOWN split: [8, 32] box, logical [8, 24].
        consumer_code = self._static_acc_to_vec_pop_code([8, 32], [8, 24])
        consumer_lines = [line.strip() for line in consumer_code.splitlines()]

        pop_index = next(i for i, line in enumerate(consumer_lines) if "pto.tpop_from_aic" in line)
        assert any("pto.treshape" in line for line in consumer_lines), (
            "a full-box split pop must expose its logical valid shape through pto.treshape:\n" + consumer_code
        )
        reshape_index = next(i for i, line in enumerate(consumer_lines) if "pto.treshape" in line)
        assert "pto.tpop_from_aic(%c8_index, %c32_index) {split = 1}" in consumer_lines[pop_index], (
            "a split C2V pop must transport the full per-lane box; a narrowed valid_col "
            f"mis-strides the GM slot read:\n{consumer_code}"
        )
        transport = re.search(r"^(%\w+) = pto\.tpop_from_aic", consumer_lines[pop_index])
        assert transport is not None
        assert pop_index < reshape_index
        assert f"pto.treshape {transport.group(1)}" in consumer_lines[reshape_index]
        assert "rows=8, cols=32, v_row=8, v_col=24" in consumer_lines[reshape_index], (
            f"the logical per-lane valid shape must be restored by treshape:\n{consumer_code}"
        )

    @pytest.mark.parametrize(
        ("split", "shape", "valid_shape", "stride_kind"),
        [
            (1, [32, 128], [20, 128], "default"),
            (1, [32, 128], [20, 128], "capacity"),
            (1, [32, 128], [20, 128], "logical"),
            (1, [32, 128], [20, 96], "default"),
            (1, [32, 128], [20, 96], "capacity"),
            (1, [32, 128], [20, 96], "logical"),
            (2, [32, 64], [32, 20], "default"),
            (2, [32, 64], [32, 20], "capacity"),
        ],
    )
    def test_split_acc_to_vec_pop_preserves_partition_stride(self, split, shape, valid_shape, stride_kind):
        """Narrow valid extents do not change the default physical partition stride.

        A balanced logical partition may instead carry an explicit smaller stride.
        Widening that split-axis extent to capacity would move lane 1's FIFO read.
        """
        axis = split - 1
        stride = {"default": None, "capacity": shape[axis], "logical": valid_shape[axis]}[stride_kind]
        consumer_code = self._static_acc_to_vec_pop_code(shape, valid_shape, split=split, lane_stride=stride)
        pop_line = next(line.strip() for line in consumer_code.splitlines() if "pto.tpop_from_aic" in line)
        transport_shape = list(shape)
        if stride_kind == "logical":
            transport_shape[axis] = stride
        rows, cols = transport_shape
        assert f"pto.tpop_from_aic(%c{rows}_index, %c{cols}_index) {{split = {split}}}" in pop_line, (
            f"the split-axis transport extent must preserve the partition stride:\n{consumer_code}"
        )
        reshape_lines = [line for line in consumer_code.splitlines() if "pto.treshape" in line]
        assert len(reshape_lines) == 1, consumer_code
        transport = re.search(r"^(%\w+) = pto\.tpop_from_aic", pop_line)
        assert transport is not None
        assert f"pto.treshape {transport.group(1)}" in reshape_lines[0], consumer_code
        assert (
            f"rows={shape[0]}, cols={shape[1]}, v_row={valid_shape[0]}, v_col={valid_shape[1]}"
            in reshape_lines[0]
        ), consumer_code

    def test_left_right_acc_to_vec_pop_rejects_stride_smaller_than_physical_columns(self):
        """LEFT_RIGHT cannot change its lane offset without also changing the GM row pitch."""
        with pytest.raises(ValueError, match="LEFT_RIGHT.*lane_stride.*physical"):
            self._static_acc_to_vec_pop_code([32, 64], [32, 20], split=2, lane_stride=20)

    def test_hardware_split_acc_to_vec_pop_rejects_stride_smaller_than_physical_rows(self):
        """A hardware split already places each lane at its physical row band."""
        with pytest.raises(ValueError, match="lane_stride.*physical.*hardware"):
            self._static_acc_to_vec_pop_code(
                [32, 128], [20, 128], lane_stride=20, backend_type=BackendType.Ascend950
            )

    @pytest.mark.parametrize(
        (
            "tpush_op_name",
            "func_type",
            "memory_space",
            "split",
        ),
        [
            ("tile.tpush_to_aiv", ir.FunctionType.AIC, ir.MemorySpace.Acc, 1),
            ("tile.tpush_to_aic", ir.FunctionType.AIV, ir.MemorySpace.Vec, 2),
        ],
    )
    def test_split_tpush_uses_full_box_transport_dims(self, tpush_op_name, func_type, memory_space, split):
        """Split tpush transports the full producer box on BOTH axes.

        Narrowing the split axis to the user's logical valid_shape on the
        producer side leaves consumer subblocks reading uninitialised slot
        memory — especially on LEFT_RIGHT splits, where the partial
        producer write fails to populate the slot column range subblock 0
        reads. The transport set_validshape must therefore reset both axes
        to the box shape; consumer-side localisation through
        LocalizeValidDimForSplit clamps the logical valid_shape back to
        the truthful per-subblock extent. See
        src/backend/common/pto_ops_common.cpp::EmitSplitTpushTransportValidShape.
        """
        transport_valid_shape = ", %c16_index, %c16_index :"
        span = ir.Span.unknown()

        src = ir.Var("src", ir.TensorType([16, 16], pl.FP32), span)
        valid_row = ir.Var("valid_row", ir.ScalarType(pl.INDEX), span)
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)

        zero = ir.ConstInt(0, pl.INDEX, span)
        shape_16 = ir.ConstInt(16, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([shape_16, shape_16], span)

        src_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 16 * 16 * 4, 0)
        src_view = ir.TileView(
            valid_shape=[shape_16, shape_16],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        src_type = ir.TileType([16, 16], pl.FP32, src_memref, src_view, memory_space)
        src_tile = ir.Var("src_tile", src_type, span)

        narrowed_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 16 * 16 * 4, 0)
        narrowed_view = ir.TileView(
            valid_shape=[valid_row, valid_col],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        narrowed_type = ir.TileType([16, 16], pl.FP32, narrowed_memref, narrowed_view, memory_space)
        narrowed_tile = ir.Var("narrowed_tile", narrowed_type, span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(
                    src_tile,
                    ir.Call(
                        ir.Op("tile.load"),
                        [src, offsets, shapes, shapes],
                        {"target_memory": memory_space},
                        src_type,
                        span,
                    ),
                    span,
                ),
                ir.AssignStmt(
                    narrowed_tile,
                    ir.Call(
                        ir.Op("tile.set_validshape"),
                        [src_tile, valid_row, valid_col],
                        {},
                        narrowed_type,
                        span,
                    ),
                    span,
                ),
                ir.EvalStmt(
                    ir.Call(
                        ir.Op(tpush_op_name),
                        [narrowed_tile],
                        {"split": split},
                        ir.UnknownType(),
                        span,
                    ),
                    span,
                ),
            ],
            span,
        )
        func = ir.Function(
            "split_narrow_then_push",
            [
                (src, ir.ParamDirection.In),
                (valid_row, ir.ParamDirection.In),
                (valid_col, ir.ParamDirection.In),
            ],
            [],
            body,
            span,
            func_type,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "split_narrow_then_push_program", span))

        set_validshape_lines = [
            line.strip() for line in mlir_code.splitlines() if "pto.set_validshape" in line
        ]
        tpush_line = next(
            line.strip() for line in mlir_code.splitlines() if tpush_op_name.replace("tile.", "pto.") in line
        )

        assert len(set_validshape_lines) == 3, (
            "Expected logical set_validshape, transport normalization, and logical restore, got:\n"
            + "\n".join(set_validshape_lines)
        )
        assert ", %arg1, %arg2 :" in set_validshape_lines[0], (
            f"Expected the initial logical validShape update, got:\n{set_validshape_lines[0]}"
        )
        assert transport_valid_shape in set_validshape_lines[1], (
            "Expected split tpush to normalize the non-split transport dimension, got:\n"
            f"{set_validshape_lines[1]}"
        )
        assert ", %arg1, %arg2 :" in set_validshape_lines[2], (
            f"Expected the logical validShape restore after tpush, got:\n{set_validshape_lines[2]}"
        )
        assert "%src_tile" in tpush_line and "%narrowed_tile" not in tpush_line, (
            f"Expected tpush to use the aliased source tile, got:\n{tpush_line}"
        )

    def test_no_split_dual_aiv_tpush_widens_cols_preserves_rows(self):
        """No-split dual-AIV tpush widens COLUMNS to the box but PRESERVES rows.

        On 910B the no-split dual-AIV dispatch runs a mixed root's producer on
        two AIV subblocks that share one FIFO slot while the single cube
        consumer pops the full slot. A producer that narrowed its valid_shape
        (e.g. set_validshape on a partial attention block) would otherwise leave
        the slot columns >= valid_col stale and feed garbage into the consumer
        matmul. So for split==0 functions carrying ``dual_aiv_dispatch`` the
        transport set_validshape widens the COLUMN axis to the box while
        PRESERVING the producer's row valid_shape -- so subblock 0's real push
        carries the full column box and subblock 1's valid_shape=(0, 0) replay
        stays a true 0-row no-op instead of racing garbage rows into subblock
        0's slot. Genuine split==1/2 paths widen both axes; plain split==0
        without ``dual_aiv_dispatch`` emits no transport at all. See
        src/backend/common/pto_ops_common.cpp::EmitSplitTpushTransportValidShape.
        """
        span = ir.Span.unknown()
        memory_space = ir.MemorySpace.Vec

        src = ir.Var("src", ir.TensorType([16, 16], pl.FP32), span)
        valid_row = ir.Var("valid_row", ir.ScalarType(pl.INDEX), span)
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)

        zero = ir.ConstInt(0, pl.INDEX, span)
        shape_16 = ir.ConstInt(16, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([shape_16, shape_16], span)

        src_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 16 * 16 * 4, 0)
        src_view = ir.TileView(
            valid_shape=[shape_16, shape_16],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        src_type = ir.TileType([16, 16], pl.FP32, src_memref, src_view, memory_space)
        src_tile = ir.Var("src_tile", src_type, span)

        narrowed_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 16 * 16 * 4, 0)
        narrowed_view = ir.TileView(
            valid_shape=[valid_row, valid_col],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        narrowed_type = ir.TileType([16, 16], pl.FP32, narrowed_memref, narrowed_view, memory_space)
        narrowed_tile = ir.Var("narrowed_tile", narrowed_type, span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(
                    src_tile,
                    ir.Call(
                        ir.Op("tile.load"),
                        [src, offsets, shapes, shapes],
                        {"target_memory": memory_space},
                        src_type,
                        span,
                    ),
                    span,
                ),
                ir.AssignStmt(
                    narrowed_tile,
                    ir.Call(
                        ir.Op("tile.set_validshape"),
                        [src_tile, valid_row, valid_col],
                        {},
                        narrowed_type,
                        span,
                    ),
                    span,
                ),
                ir.EvalStmt(
                    ir.Call(
                        ir.Op("tile.tpush_to_aic"),
                        [narrowed_tile],
                        {"split": 0},
                        ir.UnknownType(),
                        span,
                    ),
                    span,
                ),
            ],
            span,
        )
        func = ir.Function(
            "dual_aiv_narrow_then_push",
            [
                (src, ir.ParamDirection.In),
                (valid_row, ir.ParamDirection.In),
                (valid_col, ir.ParamDirection.In),
            ],
            [],
            body,
            span,
            ir.FunctionType.AIV,
            attrs={"dual_aiv_dispatch": True},
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        mlir_code = codegen.PTOCodegen().generate(
            ir.Program([func], "dual_aiv_narrow_then_push_program", span)
        )

        set_validshape_lines = [
            line.strip() for line in mlir_code.splitlines() if "pto.set_validshape" in line
        ]
        tpush_line = next(line.strip() for line in mlir_code.splitlines() if "pto.tpush_to_aic" in line)

        assert len(set_validshape_lines) == 3, (
            "Expected logical set_validshape, transport normalization, and logical restore, got:\n"
            + "\n".join(set_validshape_lines)
        )
        assert ", %arg1, %arg2 :" in set_validshape_lines[0], (
            f"Expected the initial logical validShape update, got:\n{set_validshape_lines[0]}"
        )
        # Column widened to the box (%c16_index); ROW preserved at the
        # producer's valid_row (%arg1) so the subblock-1 replay stays a 0-row
        # no-op rather than racing garbage rows into subblock 0's slot.
        assert ", %arg1, %c16_index :" in set_validshape_lines[1], (
            "Expected no-split dual-AIV transport to widen cols but preserve rows, got:\n"
            f"{set_validshape_lines[1]}"
        )
        assert ", %arg1, %arg2 :" in set_validshape_lines[2], (
            f"Expected the logical validShape restore after tpush, got:\n{set_validshape_lines[2]}"
        )
        assert "%src_tile" in tpush_line and "%narrowed_tile" not in tpush_line, (
            f"Expected tpush to use the aliased source tile, got:\n{tpush_line}"
        )

    def test_no_split_dual_aiv_zero_row_replay_emits_no_transport(self):
        """The subblock-1 replay (statically 0 rows) emits NO transport.

        BuildNoSplitLane1ReplayStmts zeroes the replay tile's valid_shape to
        (0, 0). A col-widening transport for a 0-row push moves no data yet
        (on 910B) perturbs the shared-slot dual-AIV merge -- emitting one
        regressed the cross_core_v2c_nosplit golden. So when the producer's
        transport rows are statically 0, EmitSplitTpushTransportValidShape
        skips the transport entirely; only the real, non-zero-row subblock-0
        push gets it. See
        src/backend/common/pto_ops_common.cpp::EmitSplitTpushTransportValidShape.
        """
        span = ir.Span.unknown()
        memory_space = ir.MemorySpace.Vec
        zero = ir.ConstInt(0, pl.INDEX, span)
        shape_16 = ir.ConstInt(16, pl.INDEX, span)

        src = ir.Var("src", ir.TensorType([16, 16], pl.FP32), span)
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([shape_16, shape_16], span)

        src_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 16 * 16 * 4, 0)
        src_view = ir.TileView(
            valid_shape=[shape_16, shape_16],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        src_type = ir.TileType([16, 16], pl.FP32, src_memref, src_view, memory_space)
        src_tile = ir.Var("src_tile", src_type, span)

        # Replay tile: statically 0 rows (as BuildNoSplitLane1ReplayStmts zeroes it).
        zeroed_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 16 * 16 * 4, 0)
        zeroed_view = ir.TileView(
            valid_shape=[zero, valid_col],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        zeroed_type = ir.TileType([16, 16], pl.FP32, zeroed_memref, zeroed_view, memory_space)
        zeroed_tile = ir.Var("zeroed_tile", zeroed_type, span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(
                    src_tile,
                    ir.Call(
                        ir.Op("tile.load"),
                        [src, offsets, shapes, shapes],
                        {"target_memory": memory_space},
                        src_type,
                        span,
                    ),
                    span,
                ),
                ir.AssignStmt(
                    zeroed_tile,
                    ir.Call(
                        ir.Op("tile.set_validshape"),
                        [src_tile, zero, valid_col],
                        {},
                        zeroed_type,
                        span,
                    ),
                    span,
                ),
                ir.EvalStmt(
                    ir.Call(
                        ir.Op("tile.tpush_to_aic"),
                        [zeroed_tile],
                        {"split": 0},
                        ir.UnknownType(),
                        span,
                    ),
                    span,
                ),
            ],
            span,
        )
        func = ir.Function(
            "dual_aiv_zero_row_replay",
            [(src, ir.ParamDirection.In), (valid_col, ir.ParamDirection.In)],
            [],
            body,
            span,
            ir.FunctionType.AIV,
            attrs={"dual_aiv_dispatch": True},
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        mlir_code = codegen.PTOCodegen().generate(
            ir.Program([func], "dual_aiv_zero_row_replay_program", span)
        )

        set_validshape_lines = [
            line.strip() for line in mlir_code.splitlines() if "pto.set_validshape" in line
        ]
        # No transport normalization / restore -- the 0-row replay push is
        # skipped, so the only set_validshape is the user's logical update and
        # none widen a column to the box constant (%c16_index).
        assert all("%c16_index" not in line for line in set_validshape_lines), (
            "Expected NO col-widening transport for the 0-row replay, got:\n"
            + "\n".join(set_validshape_lines)
        )
        assert len(set_validshape_lines) <= 1, (
            "Expected at most the logical set_validshape (no transport+restore pair), got:\n"
            + "\n".join(set_validshape_lines)
        )

    def test_tfree_stays_after_nested_control_flow_use(self):
        """Nested control-flow users of a tpop result must stay before tfree."""

        @pl.program
        class ConditionalTpopProgram:
            @pl.function(type=pl.FunctionType.AIC)
            def cube_consumer(
                self,
                flag: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
                pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf)

                received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=1)
                if flag == 0:
                    received_left = pl.move(received, target_memory=pl.MemorySpace.Left)
                    received_mat = pl.move(received_left, target_memory=pl.MemorySpace.Mat)
                    _: pl.Tensor[[16, 16], pl.FP16] = pl.store(received_mat, [0, 0], output)
                else:
                    received_right = pl.move(received, target_memory=pl.MemorySpace.Right)
                    received_mat_else = pl.move(received_right, target_memory=pl.MemorySpace.Mat)
                    _: pl.Tensor[[16, 16], pl.FP16] = pl.store(received_mat_else, [0, 0], output)

                pl.tfree_to_aiv(received)
                return output

        codes = self._compile_and_generate(ConditionalTpopProgram)
        aic_body = _extract_func_section(codes["cube_consumer"], "cube_consumer")

        assert aic_body.index("pto.tpop_from_aiv") < aic_body.index("scf.if"), (
            "The nested control-flow user should remain after the tpop"
        )
        assert aic_body.index("scf.if") < aic_body.index("pto.tfree_from_aiv"), (
            "tfree must remain after nested control-flow uses of the popped tile"
        )

    def test_tfree_rejects_mismatched_tpop_direction(self):
        """tfree must validate that its tile came from the matching tpop direction."""

        @pl.program
        class MismatchedTfreeProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def vector_consumer(self):
                pipe_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=pipe_buf)
                received: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
                pl.tfree_to_aiv(received)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "system.tfree_to_aiv requires its tile argument to come from tile.tpop_from_aiv, "
                "got tile.tpop_from_aic"
            ),
        ):
            self._compile_and_generate(MismatchedTfreeProgram)

    def test_tfree_rejects_mismatched_pipe_id(self):
        """Explicit tfree id must match the originating tpop pipe id."""

        @pl.program
        class MismatchedTfreeIdProgram:
            @pl.function(type=pl.FunctionType.AIC)
            def cube_consumer(self):
                pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
                pl.aic_initialize_pipe(
                    pl.const(0, pl.INT32),
                    pipe_buf,
                    dir_mask=2,
                    slot_size=512,
                    id=1,
                )
                received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(
                    split=0,
                    id=1,
                )
                pl.tfree_to_aiv(received, id=0)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "system.tfree_to_aiv pipe id 0 does not match originating tile.tpop_from_aiv pipe id 1"
            ),
        ):
            self._compile_and_generate(MismatchedTfreeIdProgram)

    def test_tpop_user_stays_after_if_defined_scalar_dependency(self):
        """A tpop user that depends on an if-defined scalar must not be hoisted before the if."""

        @pl.program
        class IfDefinedScalarProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def vector_consumer(
                self,
                flag: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pipe_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=pipe_buf)

                received: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
                if flag == 0:
                    scale = 1.0
                else:
                    scale = 2.0
                scaled = pl.tile.muls(received, scale)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(scaled, [0, 0], output)
                pl.tfree_to_aic(received)
                return updated

        codes = self._compile_and_generate(IfDefinedScalarProgram)
        aiv_body = _extract_func_section(codes["vector_consumer"], "vector_consumer")

        assert aiv_body.index("pto.tpop_from_aic") < aiv_body.index("scf.if"), (
            "The scalar-defining if should remain after the tpop"
        )
        assert aiv_body.index("scf.if") < aiv_body.index("pto.tmuls"), (
            "The tpop user must remain after the if-defined scalar dependency"
        )

    def test_bidirectional_vector(self):
        """Test Vector kernel keeps bidirectional communication on one default pipe."""
        codes = self._compile_and_generate(BidirectionalCrossCorProgram)
        vector_code = codes["vector_bidir"]

        assert vector_code, "Vector bidir MLIR should not be empty"
        # Buffer setup: C2V consumer reserves buffer, V2C producer imports peer buffer
        assert "pto.reserve_buffer" in vector_code, "Should reserve buffer for C2V"
        assert 'name = "c2v_slot_buffer"' in vector_code, "Should reference c2v_slot_buffer"
        assert "auto = false" in vector_code, "Should have auto = false for explicit base"
        assert "base = 8192" in vector_code, "Should have explicit base address (0x2000 = 8192)"
        assert "-> i32" in vector_code, "Buffer ops should return i32"
        assert "pto.import_reserved_buffer" in vector_code, "Should import peer buffer for V2C"
        assert "peer_func = @cube_bidir" in vector_code, "Should reference cube_bidir"
        # Bidirectional init
        assert "pto.aiv_initialize_pipe" in vector_code, "Should contain aiv_initialize_pipe"
        assert "dir_mask = 3" in vector_code, "Should have dir_mask = 3 (bidirectional)"
        assert "{id =" not in vector_code, "Legacy bidirectional default pipe should not emit explicit id"
        assert "c2v_consumer_buf = " in vector_code, "Should have c2v_consumer_buf as SSA reference"
        assert "v2c_consumer_buf = " in vector_code, "Should have v2c_consumer_buf as SSA reference"
        # V2C producer side: preprocess + push
        assert "pto.tadd" in vector_code, "Should do elementwise add (Vector op)"
        assert "pto.tpush_to_aic" in vector_code, "Should push to AIC"
        # C2V consumer side: receive matmul result + post-process
        assert "pto.tpop_from_aic" in vector_code, "Should pop from AIC"
        assert "pto.texp" in vector_code, "Should do exp post-processing (Vector op)"
        assert "pto.tfree_from_aic" in vector_code, "Should free C2V slot"

    def test_bidirectional_cube(self):
        """Test Cube kernel keeps bidirectional communication on one default pipe."""
        codes = self._compile_and_generate(BidirectionalCrossCorProgram)
        cube_code = codes["cube_bidir"]

        assert cube_code, "Cube bidir MLIR should not be empty"
        # Buffer setup: V2C consumer reserves buffer with explicit base, C2V producer imports peer buffer
        assert "pto.reserve_buffer" in cube_code, "Should reserve buffer for V2C"
        assert 'name = "v2c_slot_buffer"' in cube_code, "Should reference v2c_slot_buffer"
        assert "auto = false" in cube_code, "Should have auto = false for explicit base"
        assert "base = 4096" in cube_code, "Should have explicit base address (0x1000 = 4096)"
        assert "-> i32" in cube_code, "Buffer ops should return i32"
        assert "pto.import_reserved_buffer" in cube_code, "Should import peer buffer for C2V"
        assert "peer_func = @vector_bidir" in cube_code, "Should reference vector_bidir"
        # Bidirectional init with SSA consumer buffer references
        assert "pto.aic_initialize_pipe" in cube_code, "Should contain aic_initialize_pipe"
        assert "dir_mask = 3" in cube_code, "Should have dir_mask = 3 (bidirectional)"
        assert "{id =" not in cube_code, "Legacy bidirectional default pipe should not emit explicit id"
        assert "c2v_consumer_buf = " in cube_code, "Should have c2v_consumer_buf as SSA reference"
        assert "v2c_consumer_buf = " in cube_code, "Should have v2c_consumer_buf as SSA reference"
        # V2C consumer side: receive preprocessed data
        assert "pto.tpop_from_aiv" in cube_code, "Should pop from AIV"
        assert "pto.tfree_from_aiv" in cube_code, "Should free V2C slot"
        # C2V producer side: matmul + push back
        assert "pto.tpush_to_aiv" in cube_code, "Should push to AIV"
        assert "pto.tmatmul" in cube_code, "Should do matmul (Cube op)"

    def test_multiple_pipe_vector(self):
        """Test Vector kernel with two explicit V2C frontend pipes."""
        codes = self._compile_and_generate(MultiPipeSameDirectionCrossCoreProgram)
        vector_code = codes["vector_multi_pipe"]

        assert vector_code, "Vector multi-pipe MLIR should not be empty"
        # Buffer setup: V2C producer imports two peer buffers
        assert vector_code.count("pto.import_reserved_buffer") == 2
        assert 'name = "v2c_slot_buffer_0"' in vector_code
        assert 'name = "v2c_slot_buffer_1"' in vector_code
        assert "peer_func = @cube_multi_pipe" in vector_code, "Should reference cube_multi_pipe"
        # Multiple explicit V2C init ops
        assert "pto.aiv_initialize_pipe" in vector_code, "Should contain aiv_initialize_pipe"
        assert "dir_mask = 3" not in vector_code, "Explicit multi-pipe program should not combine dirs"
        assert "dir_mask = 1" not in vector_code
        assert "pto.aiv_initialize_pipe {id = 0, dir_mask = 2" in vector_code
        assert "pto.aiv_initialize_pipe {id = 1, dir_mask = 2" in vector_code
        assert "v2c_consumer_buf = " in vector_code, "Should have v2c_consumer_buf as SSA reference"
        # V2C producer side: preprocess + push
        assert "pto.tadd" in vector_code, "Should do elementwise add (Vector op)"
        assert "pto.tsub" in vector_code, "Should do elementwise sub (Vector op)"
        assert "pto.tpush_to_aic" in vector_code, "Should push to AIC"
        assert "pto.tpop_from_aic" not in vector_code
        assert "pto.tfree_from_aic" not in vector_code
        assert vector_code.count("{id = 0, split = 0}") == 1
        assert vector_code.count("{id = 1, split = 0}") == 1

    def test_multiple_pipe_cube(self):
        """Test Cube kernel with two explicit V2C frontend pipes."""
        codes = self._compile_and_generate(MultiPipeSameDirectionCrossCoreProgram)
        cube_code = codes["cube_multi_pipe"]

        assert cube_code, "Cube multi-pipe MLIR should not be empty"
        # Buffer setup: V2C consumer reserves two buffers with explicit bases
        assert cube_code.count("pto.reserve_buffer") == 2
        assert 'name = "v2c_slot_buffer_0"' in cube_code
        assert 'name = "v2c_slot_buffer_1"' in cube_code
        assert "auto = false" in cube_code, "Should have auto = false for explicit base"
        assert "base = 4096" in cube_code, "Should have explicit base address (0x1000 = 4096)"
        assert "base = 8192" in cube_code, "Should have explicit base address (0x2000 = 8192)"
        assert "-> i32" in cube_code, "Buffer ops should return i32"
        assert "pto.import_reserved_buffer" not in cube_code
        # Multiple explicit V2C init ops with SSA consumer buffer references
        assert "pto.aic_initialize_pipe" in cube_code, "Should contain aic_initialize_pipe"
        assert "dir_mask = 3" not in cube_code, "Explicit multi-pipe program should not combine dirs"
        assert "dir_mask = 1" not in cube_code
        assert "pto.aic_initialize_pipe {id = 0, dir_mask = 2" in cube_code
        assert "pto.aic_initialize_pipe {id = 1, dir_mask = 2" in cube_code
        assert "v2c_consumer_buf = " in cube_code, "Should have v2c_consumer_buf as SSA reference"
        # V2C consumer side: receive preprocessed data
        assert "pto.tpop_from_aiv" in cube_code, "Should pop from AIV"
        assert "pto.tfree_from_aiv" in cube_code, "Should free V2C slot"
        assert "pto.tmatmul" in cube_code, "Should do matmul (Cube op)"
        assert "pto.tpush_to_aiv" not in cube_code
        # tpop ids are inherited by tfree even when user code omits tfree id.
        assert cube_code.count("{id = 0, split = 0}") == 2
        assert cube_code.count("{id = 1, split = 0}") == 2

    def test_all_cross_core_pto_ops_covered(self):
        """Verify all 10 cross-core PTO operations are exercised across both test programs."""
        unidir_codes = self._compile_and_generate(CrossCoreTpushTpopProgram)
        bidir_codes = self._compile_and_generate(BidirectionalCrossCorProgram)
        all_code = "\n".join(unidir_codes.values()) + "\n" + "\n".join(bidir_codes.values())

        expected_ops = [
            "pto.tpush_to_aiv",
            "pto.tpush_to_aic",
            "pto.tpop_from_aic",
            "pto.tpop_from_aiv",
            "pto.tfree_from_aic",
            "pto.tfree_from_aiv",
            "pto.aic_initialize_pipe",
            "pto.aiv_initialize_pipe",
            "pto.reserve_buffer",
            "pto.import_reserved_buffer",
        ]
        for op in expected_ops:
            assert op in all_code, f"Expected PTO op '{op}' not found in generated MLIR"


class TestExpandMixedKernelCodegen:
    """Tests that PTO codegen works on AIC/AIV functions produced by expand_mixed_kernel."""

    @pytest.mark.parametrize("tile_polling", [False, True], ids=["scalar", "tile"])
    @pytest.mark.parametrize("keep_result", [False, True], ids=["unused-result", "used-result"])
    def test_while_condition_only_iter_arg_codegen(self, tmp_path, tile_polling, keep_result):
        """Mixed SPMD polling must retain condition-only state even with an unused result (#3009)."""

        @pl.jit
        def mixed_while_poll(
            a: pl.Tensor[[32, 16], pl.FP16],
            b: pl.Tensor[[16, 16], pl.FP16],
            scratch: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            flags: pl.InOut[pl.Tensor[[16], pl.INT32]],
            cube_out: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
            vector_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            TILE_POLLING: pl.constexpr,
            KEEP_RESULT: pl.constexpr,
        ):
            # Matmul makes this a mixed kernel; the nested AIV branches retain
            # the producer/consumer shape without requiring device execution.
            for worker in pl.spmd(2, sync_start=True, name_hint="mixed_while_poll"):
                ac = pl.load(a, [worker * 16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
                bc = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
                pl.store(pl.matmul(ac, bc, out_dtype=pl.FP32), [worker * 16, 0], cube_out)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    if lane == 0:
                        if worker == 0:
                            av = pl.load(a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
                            pl.store(pl.add(pl.cast(av, pl.FP32), 1.0), [0, 0], scratch)
                            pl.system.cacheinvalid()
                            pl.system.fence()
                            pl.write(flags, [0], pl.const(1, pl.INT32))
                        elif worker == 1:
                            if TILE_POLLING:
                                initial_tile = pl.load(flags, [0], [16], target_memory=pl.MemorySpace.Vec)
                                initial_ready = pl.tile.read(initial_tile, [0])
                            else:
                                initial_ready = pl.read(flags, [0])
                            for (ready,) in pl.while_(init_values=(initial_ready,)):
                                pl.cond(ready != 1)
                                pl.system.cacheinvalid()
                                if TILE_POLLING:
                                    flag_tile = pl.load(flags, [0], [16], target_memory=pl.MemorySpace.Vec)
                                    next_ready = pl.tile.read(flag_tile, [0])
                                else:
                                    next_ready = pl.read(flags, [0])
                                final_ready = pl.yield_(next_ready)
                            if KEEP_RESULT:
                                # This control used to hide the bug by making
                                # the loop result live independently of its condition.
                                pl.write(flags, [1], final_ready)
                            pl.system.cacheinvalid()
                            received = pl.load(scratch, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
                            pl.store(pl.mul(received, 2.0), [0, 0], vector_out)
            return cube_out, vector_out

        mixed_while_poll.compile(
            TILE_POLLING=tile_polling,
            KEEP_RESULT=keep_result,
            config=RunConfig(
                backend_type=BackendType.Ascend910,
                codegen_only=True,
                save_kernels=True,
                save_kernels_dir=str(tmp_path),
                dump_passes=True,
            ),
        )
        dumps = list((tmp_path / "passes_dump").glob("*after_ExpandMixedKernel.py"))
        assert dumps, "Expected an ExpandMixedKernel pass dump"
        expanded = "\n".join(path.read_text() for path in dumps)
        assert "__FREE_VAR" not in expanded, expanded
        assert "pl.while_(init_values=" in expanded, expanded

        pto_files = list(tmp_path.rglob("*.pto"))
        assert pto_files, "Expected generated PTO MLIR"
        pto = "\n".join(path.read_text() for path in pto_files)
        loop = re.search(
            r"scf\.while \([^\n]+\) : \(i32\) -> \(i32\) \{.*?scf\.condition\(.*?"
            r"\} do \{(?P<body>.*?)scf\.yield (?P<yielded>%\w+) : i32",
            pto,
            re.DOTALL,
        )
        assert loop is not None, pto
        # The backedge must yield the freshly read flag, so retaining a reload
        # while yielding stale state cannot satisfy this polling regression.
        body = loop.group("body")
        yielded = loop.group("yielded")
        reload_op = "pto.tgetval" if tile_polling else "pto.load"
        assert re.search(rf"^\s*{re.escape(yielded)} = {re.escape(reload_op)} ", body, re.MULTILINE), body
        if tile_polling:
            assert "pto.tload " in body, body

    @staticmethod
    def _expand_and_generate(program) -> dict[str, str]:
        """Apply full PTOAS passes with expand_mixed_kernel, then generate PTO MLIR per function.

        Uses all PTOAS strategy passes with expand_mixed_kernel inserted after
        ConvertTensorToTileOps (its intended pipeline position).

        Returns:
            dict mapping function name to generated MLIR code for InCore-variant functions.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend950)

        # Full PTOAS pipeline with expand_mixed_kernel at its intended position
        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.unroll_loops())
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())
        pipeline.add_pass(passes.outline_incore_scopes())
        pipeline.add_pass(passes.outline_cluster_scopes())
        pipeline.add_pass(passes.convert_tensor_to_tile_ops())
        pipeline.add_pass(passes.lower_composite_ops())
        pipeline.add_pass(passes.flatten_tile_nd_to_2d())
        pipeline.add_pass(passes.infer_tile_memory_space())
        pipeline.add_pass(passes.expand_mixed_kernel())
        pipeline.add_pass(passes.stamp_tfree_split())
        pipeline.add_pass(passes.init_mem_ref())
        pipeline.add_pass(passes.memory_reuse())
        pipeline.add_pass(passes.allocate_memory_addr())
        optimized = pipeline.run(program)

        result = {}
        codegen_instance = codegen.PTOCodegen()
        groups, ungrouped = _build_group_mapping(optimized)

        # Grouped: one module per group
        for group_name, members in groups.items():
            grouped_program = ir.Program(members, group_name, optimized.span)
            mlir_code = codegen_instance.generate(grouped_program)
            result[group_name] = mlir_code
            for func in members:
                result[func.name] = mlir_code

        # Ungrouped: one module per function
        for func in ungrouped:
            if not ir.is_incore_type(func.func_type):
                continue
            single = ir.Program([func], func.name, optimized.span)
            mlir_code = codegen_instance.generate(single)
            result[func.name] = mlir_code
        return result

    def test_tile_sub_is_vector_codegen(self):
        """tile.sub in mixed kernel should generate pto.tsub in AIV, pto.tmatmul in AIC."""

        @pl.program
        class MixedSubMatmul:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                x_sub: pl.Tile[[16, 128], pl.BF16] = pl.sub(x_tile, x_tile)
                x_sub_l1: pl.Tile[[16, 128], pl.BF16] = pl.move(x_sub, target_memory=pl.MemorySpace.Mat)
                s_sub_l0a: pl.Tile[[16, 128], pl.BF16] = pl.move(x_sub_l1, target_memory=pl.MemorySpace.Left)
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(s_sub_l0a, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        codes = self._expand_and_generate(MixedSubMatmul)

        # AIV function should contain pto.tsub (vector op)
        assert "main_incore_0_aiv" in codes, "AIV function should be generated"
        aiv_code = codes["main_incore_0_aiv"]
        assert "pto.tsub" in aiv_code, "AIV should contain pto.tsub for tile.sub"
        assert "pto.import_reserved_buffer" in aiv_code, "AIV should import the peer V2C buffer"
        assert "pto.aiv_initialize_pipe" in aiv_code, "AIV should initialize cross-core pipe"

        # AIC function should contain pto.tmatmul (cube op)
        assert "main_incore_0_aic" in codes, "AIC function should be generated"
        aic_code = codes["main_incore_0_aic"]
        aic_body = _extract_func_section(aic_code, "main_incore_0_aic")
        assert "pto.tmatmul" in aic_code, "AIC should contain pto.tmatmul for tile.matmul"
        assert "pto.reserve_buffer" in aic_code, "AIC should reserve the V2C consumer buffer"
        assert "auto = false" in aic_code, "Auto reserve_buffer should be resolved before PTO emission"
        assert "base = 0" in aic_code, "The first auto reserve_buffer should start from base 0"
        assert "pto.aic_initialize_pipe" in aic_code, "AIC should initialize cross-core pipe"
        assert aic_body.index("pto.aic_initialize_pipe") < aic_body.index("pto.tpop_from_aiv"), (
            "AIC initialize_pipe should be emitted before the first V2C tpop"
        )
        assert "pto.tfree_from_aiv" in aic_body, "AIC consumer should free the V2C slot"
        assert aic_body.index("pto.tpop_from_aiv") < aic_body.index("pto.tmov"), (
            "AIC should move the popped tile before freeing it"
        )
        assert aic_body.index("pto.tmov") < aic_body.index("pto.tfree_from_aiv"), (
            "AIC should free the popped tile after its direct use"
        )

        # tile.sub should NOT be in AIC
        assert "pto.tsub" not in aic_body, "AIC should not contain pto.tsub"

    def test_acc_slice_before_c2v_boundary_codegen(self):
        """Cube-side tile.slice feeding C2V boundary must survive mixed-kernel expansion."""

        @pl.program
        class SliceBeforeC2V:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                x_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                acc: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
                row: pl.Tile[[1, 128], pl.FP32] = pl.slice(acc, [1, 128], [0, 0])
                row_vec: pl.Tile[[1, 128], pl.FP32] = pl.move(
                    row,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[1, 128], pl.FP32] = pl.store(row_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                out_0: pl.Tensor[[1, 128], pl.FP32] = pl.create_tensor([1, 128], dtype=pl.FP32)
                z: pl.Tensor[[1, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        codes = self._expand_and_generate(SliceBeforeC2V)

        assert "main_incore_0_aic" in codes, "AIC function should be generated"
        aic_body = _extract_func_section(codes["main_incore_0_aic"], "main_incore_0_aic")
        assert "pto.subview" in aic_body, "AIC should keep the tile.slice producer before C2V push"
        assert "pto.tpush_to_aiv" in aic_body, "AIC should push the sliced row to AIV"
        assert aic_body.index("pto.subview") < aic_body.index("pto.tpush_to_aiv"), (
            "AIC should subview the row tile before pushing it across cores"
        )

        assert "main_incore_0_aiv" in codes, "AIV function should be generated"
        aiv_body = _extract_func_section(codes["main_incore_0_aiv"], "main_incore_0_aiv")
        assert "pto.tpop_from_aic" in aiv_body, "AIV should pop the sliced row from AIC"
        assert "pto.tstore" in aiv_body, "AIV should store the popped row to the output tensor"

    def test_partial_mx_scale_v2c_materializes_before_widening(self):
        """Partial MX slice carriers must reach PTO codegen without set_validshape on a view."""

        @pl.program
        class PartialMxScaleV2C:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
            ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
                quant, scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), group_axis=1)
                partial_scale = pl.slice(scale, [16, 2], [0, 0], valid_shape=[16, 1])
                _scale_mat = pl.move(
                    partial_scale,
                    target_memory=pl.Mem.Mat,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.row_major,
                )
                return pl.store(quant, [0, 0], out)

        codes = self._expand_and_generate(PartialMxScaleV2C)
        aiv_body = _extract_func_section(codes["main_incore_0_aiv"], "main_incore_0_aiv")
        subview_pos = aiv_body.index("pto.subview")
        materialize_pos = aiv_body.index("pto.tmov", subview_pos)
        widen_pos = aiv_body.index("pto.set_validshape")
        push_pos = aiv_body.index("pto.tpush_to_aic")
        assert subview_pos < materialize_pos < widen_pos < push_pos, aiv_body

    def test_bidirectional_mixed_kernel_keeps_combined_pipe(self):
        """Automatic bidirectional mixed kernels keep the legacy combined pipe."""

        @pl.program
        class BidirectionalMixed:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_sum = pl.add(x_tile, x_tile)
                x_sum_mat = pl.move(
                    x_sum,
                    target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                x_left = pl.move(x_sum_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0 = pl.store(z_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                out_0: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
                z: pl.Tensor[[16, 64], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        codes = self._expand_and_generate(BidirectionalMixed)
        aic_body = _extract_func_section(codes["main_incore_0_aic"], "main_incore_0_aic")
        aiv_body = _extract_func_section(codes["main_incore_0_aiv"], "main_incore_0_aiv")

        # Automatic setup always pins the ring depth explicitly (default 2), so PTOAS cannot
        # fall back to its own dir_mask-derived depth and overrun the reserved buffer.
        assert "pto.aic_initialize_pipe {dir_mask = 3, slot_size = 4096, slot_num = 2}" in aic_body
        assert "pto.aiv_initialize_pipe {dir_mask = 3, slot_size = 4096, slot_num = 2}" in aiv_body
        assert "{id =" not in aic_body
        assert "{id =" not in aiv_body
        assert "pto.tpush_to_aiv" in aic_body and "{split = 0}" in aic_body
        assert "pto.tpop_from_aiv {split = 0}" in aic_body
        assert "pto.tfree_from_aiv {split = 0}" in aic_body
        assert "pto.tpush_to_aic" in aiv_body and "{split = 0}" in aiv_body
        assert "pto.tpop_from_aic {split = 0}" in aiv_body
        assert "pto.tfree_from_aic {split = 0}" in aiv_body

    def test_multiple_pipe_ids_offset_gm_buffers_on_a2a3(self):
        """Explicit pipe ids share one GM workspace but point at disjoint regions."""

        @pl.program
        class MultiplePipeSetup:
            @pl.function(type=pl.FunctionType.AIC)
            def cube_kernel(self):
                c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_kernel")
                v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=32768, base=0x1000)
                pl.aic_initialize_pipe(c2v_peer, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, id=0)
                pl.aic_initialize_pipe(pl.const(0, pl.INT32), v2c_buf, dir_mask=2, slot_size=4096, id=1)

            @pl.function(type=pl.FunctionType.AIV)
            def vector_kernel(self):
                c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=65536, base=0x2000)
                v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_kernel")
                pl.aiv_initialize_pipe(c2v_buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, id=0)
                pl.aiv_initialize_pipe(pl.const(0, pl.INT32), v2c_peer, dir_mask=2, slot_size=4096, id=1)

            @pl.function(type=pl.FunctionType.Group)
            def group_func(self):
                self.cube_kernel()
                self.vector_kernel()

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self):
                self.group_func()

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        try:
            transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(
                MultiplePipeSetup
            )
            codegen_instance = codegen.PTOCodegen()
            groups, _ = _build_group_mapping(transformed)
            group_funcs = next(iter(groups.values()))
            mlir_code = codegen_instance.generate(
                ir.Program(group_funcs, "multiple_pipe_a2a3", transformed.span)
            )
        finally:
            backend.reset_for_testing()

        aic_body = _extract_func_section(mlir_code, "cube_kernel")
        aiv_body = _extract_func_section(mlir_code, "vector_kernel")

        for body in (aic_body, aiv_body):
            assert "pto.aic_initialize_pipe {id = 0, dir_mask = 1" in body or (
                "pto.aiv_initialize_pipe {id = 0, dir_mask = 1" in body
            )
            assert "id = 1, dir_mask = 2" in body
            assert re.search(r"pto.addptr %arg\d+", body)
            assert "arith.constant 16384 : index" in body
            assert re.search(r"gm_slot_buffer = %arg\d+ : !pto.ptr<f32>", body)
            assert body.count("gm_slot_buffer = %") == 2
            assert "dir_mask = 3" not in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
