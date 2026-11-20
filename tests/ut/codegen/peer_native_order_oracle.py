# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent instruction-order replay for straight-line native receive fixtures."""

import re
from dataclasses import dataclass
from graphlib import TopologicalSorter

_FLAG = re.compile(r"(set|wait)_flag\(PIPE_(\w+), PIPE_(\w+), EVENT_ID(\d+)\);")
_BARRIER = re.compile(r"pipe_barrier\(PIPE_(\w+)\);")
_POP = re.compile(
    r"TPOP<TPipe<\d+, Direction::\w+, (\d+), (\d+), (\d+), true>, "
    r"Tile<TileType::(Mat|Vec), (\w+), (\d+), (\d+), [^>]+>, "
    r"TileSplitAxis::TILE_NO_SPLIT>\((v\d+), (v\d+)\);"
)
_READ = re.compile(r"(TMOV|TADD|TEXP|TMULS)\((v\d+), ([^;]+)\);")
_STORE = re.compile(r"TSTORE\(v\d+, (v\d+)\);")
_CONSTANT = re.compile(r"const int(?:32|64)_t (v\d+) = (\d+);")
_ADDRESS = re.compile(r"uint64_t (v\d+) = \(uint64_t\) (v\d+);")
_TILE = re.compile(r"Tile<TileType::Vec, (\w+), (\d+), (\d+), [^>]+> (v\d+)(?: = .+)?;")
_ASSIGN = re.compile(r"TASSIGN\((v\d+), (v\d+)\);")
_PIPE = re.compile(r"auto (v\d+) = TPipe<[^>]+>\(v\d+, (v\d+), v\d+\);")
_VALID = re.compile(r"(v\d+)\.SetValidShape\((v\d+), (v\d+)\);")
_BYTES = {"half": 2, "bfloat16_t": 2, "float": 4}


@dataclass(frozen=True)
class Payload:
    name: str
    issue: int
    completion: int
    reads: tuple[tuple[int, int], ...]
    writes: tuple[tuple[int, int], ...]


def _overlaps(left, right):
    return any(a < d and c < b for a, b in left for c, d in right)


def _conflict(before, after):
    return _overlaps(before.writes, after.reads + after.writes) or _overlaps(before.reads, after.writes)


class NativeReceiveReplay:
    """The fixtures use one local ring and a single executed, nonempty lane."""

    def __init__(self, body, *, whole_vec_body=False):
        self.observed = {}
        self.required = {}
        self.last_issue = {}
        self.last_payload = {}
        self.commands = {}
        self.gates = {}
        self.all_gate = None
        self.live = {}
        self.last_consume = {}
        self.reused = []
        self.payloads = []
        self.pops = []
        self.readers = []
        self.captures = {}
        self.geometry = None
        self.whole_vec_body = whole_vec_body
        self.constants = {}
        self.tiles = {}
        self.windows = {}
        self.reserves = {}
        self.ordinary = 0
        for raw in body.splitlines():
            self.parse(raw.strip())
        assert not self.live, f"unconsumed flags: {self.live}"

    def node(self, *native, local=()):
        index = len(self.observed)
        self.required[index] = {n for n in native if n is not None}
        self.observed[index] = self.required[index] | {n for n in local if n is not None}
        return index

    def issue(self, pipe):
        issue = self.node(self.last_issue.get(pipe), local=(self.gates.get(pipe), self.all_gate))
        self.last_issue[pipe] = issue
        return issue

    def payload(self, name, pipe, *, reads=(), writes=()):
        issue = self.issue(pipe)
        completion = self.node(issue, self.last_payload.get(pipe))
        self.last_payload[pipe] = completion
        payload = Payload(name, issue, completion, reads, writes)
        for previous in self.payloads:
            if _conflict(previous, payload):
                self.required[issue].add(previous.completion)
        self.payloads.append(payload)
        return payload

    def flag(self, action, source, destination, event_id):
        key = source, destination, event_id
        if action == "set":
            assert key not in self.live, "event publication overwrites an unconsumed key"
            issue = self.issue(source)
            publish = self.node(local=(issue, self.last_payload.get(source), self.commands.get(source)))
            self.live[key] = publish
            self.commands[source] = publish
            if key in self.last_consume:
                self.reused.append((self.last_consume[key], publish))
        else:
            assert key in self.live, "wait has no matching publication"
            issue = self.issue(destination)
            wait = self.node(local=(issue, self.live.pop(key)))
            self.gates[destination] = wait
            self.commands[destination] = wait
            self.last_consume[key] = wait

    def barrier(self, pipe):
        if pipe == "ALL":
            self.all_gate = self.node(
                local=(*self.last_issue.values(), *self.last_payload.values(), *self.commands.values())
            )
        else:
            issue = self.issue(pipe)
            barrier = self.node(local=(issue, self.last_payload.get(pipe), self.commands.get(pipe)))
            self.gates[pipe] = barrier
            self.commands[pipe] = barrier

    def parse(self, line):
        if self.whole_vec_body and self.parse_placement(line):
            return
        if flag := _FLAG.fullmatch(line):
            self.flag(*flag.groups())
        elif barrier := _BARRIER.fullmatch(line):
            self.barrier(barrier[1])
        elif pop := _POP.fullmatch(line):
            stride, gm_depth, local_depth = map(int, pop.groups()[:3])
            space, dtype, rows, cols, handle, tile = pop.groups()[3:]
            geometry = stride, gm_depth, local_depth, space, handle
            assert self.geometry is None or self.geometry == geometry
            self.geometry = geometry
            ordinal = len(self.pops)
            size = int(rows) * int(cols) * _BYTES[dtype]
            assert 0 < size <= stride and 0 < local_depth <= gm_depth
            reserve = self.reserves[handle] if self.whole_vec_body else 0
            begin = reserve + ordinal % local_depth * stride
            window = (begin, begin + size)
            self.captures[tile] = (ordinal, window)
            self.windows[tile] = window
            self.pops.append(self.payload(f"pop_{ordinal}", "MTE2", writes=(window,)))
            self.readers.append([])
        elif line.startswith("TPOP<"):
            raise AssertionError(f"unrecognized native pop: {line}")
        elif read := _READ.fullmatch(line):
            macro, dst, inputs = read.groups()
            operands = set(re.findall(r"\bv\d+\b", inputs))
            captures = {self.captures[v] for v in operands if v in self.captures}
            writes = (self.windows[dst],) if self.whole_vec_body else ()
            pipe = "MTE1" if macro == "TMOV" else "V"
            if captures:
                ordinals, windows = zip(*sorted(captures), strict=True)
                if self.whole_vec_body:
                    windows = tuple(self.windows[v] for v in sorted(operands))
                label = "read_" + "_".join(f"{i}_{len(self.readers[i])}" for i in ordinals)
                payload = self.payload(label, pipe, reads=windows, writes=writes)
                for ordinal in ordinals:
                    self.readers[ordinal].append(payload)
            elif self.whole_vec_body:
                windows = tuple(self.windows[v] for v in sorted(operands))
                self.payload(f"ordinary_{self.ordinary}", pipe, reads=windows, writes=writes)
                self.ordinary += 1
        elif self.whole_vec_body and (store := _STORE.fullmatch(line)):
            # The only GM payload is this output write. It cannot conflict
            # with any local Vec interval or the separately allocated FIFO.
            self.payload("store", "MTE3", reads=(self.windows[store[1]],))
        elif self.whole_vec_body and re.match(r"T[A-Z]+[<(]", line) and not line.startswith("TFREE<"):
            raise AssertionError(f"unrecognized native payload: {line}")

    def parse_placement(self, line):
        if constant := _CONSTANT.fullmatch(line):
            self.constants[constant[1]] = int(constant[2])
        elif address := _ADDRESS.fullmatch(line):
            self.constants[address[1]] = self.constants[address[2]]
        elif tile := _TILE.fullmatch(line):
            dtype, rows, cols, var = tile.groups()
            self.tiles[var] = (int(rows), int(cols), _BYTES[dtype])
        elif binding := _ASSIGN.fullmatch(line):
            rows, cols, element = self.tiles[binding[1]]
            begin = self.constants[binding[2]]
            self.windows[binding[1]] = (begin, begin + rows * cols * element)
        elif pipe := _PIPE.fullmatch(line):
            self.reserves[pipe[1]] = self.constants[pipe[2]]
        elif valid := _VALID.fullmatch(line):
            assert self.tiles[valid[1]][:2] == tuple(self.constants[v] for v in valid.groups()[1:])
        else:
            return False
        return True

    @staticmethod
    def closure(graph):
        masks = {}
        for node in TopologicalSorter(graph).static_order():
            mask = 1 << node
            for dependency in graph[node]:
                mask |= masks[dependency]
            masks[node] = mask
        return masks

    def compare(self):
        required, observed = map(self.closure, (self.required, self.observed))
        actual = {
            (before.name, after.name)
            for before in self.payloads
            for after in self.payloads
            if observed[after.issue] & (1 << before.completion)
        }
        needed = {
            (before.name, after.name)
            for before in self.payloads
            for after in self.payloads
            if required[after.issue] & (1 << before.completion)
        }
        unconsumed = [(a, b) for a, b in self.reused if not observed[b] & (1 << a)]
        return needed - actual, actual - needed, unconsumed
