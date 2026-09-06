"""Host-only conformance and critical-path diagnostics for exported PTOAS DAGs.

These utilities explain a supplied graph; they neither repair the compiler
graph silently nor claim that a diagnostic graph predicts device latency.
"""

import hashlib
import heapq
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def orchestration_tensor_dependencies(text: str, kernel_names: Mapping[int, str]) -> dict[str, Any]:
    """Extract source-level task tensor dependencies, not runtime completion time.

    This deliberately does not collapse a mixed group to one child or turn a
    source submit site inside a loop into one executed task. Exact variable
    identity is used; alias analysis and explicit fence expansion remain open.
    """
    mixed = {
        name: [int(v) for v in re.findall(r"\d+", values)]
        for name, values in re.findall(r"MixedKernels\s+(\w+)\s*=\s*\{([^}]+)\}", text)
    }
    pattern = r"rt_submit_(?:aiv_task|aic_task|task)\(\s*(\w+)\s*,\s*(\w+)\s*\)"
    tasks = []
    for kernel, args in re.findall(pattern, text):
        ids = [int(kernel)] if kernel.isdigit() else mixed.get(kernel, [])
        if not ids or any(k not in kernel_names for k in ids):
            raise ValueError("orchestration submit has an unbound kernel")
        accesses = re.findall(rf"\b{re.escape(args)}\.add_(input|output|inout)\(\s*(\w+)\s*\)", text)
        expected = len(re.findall(rf"\b{re.escape(args)}\.add_(?:input|output|inout)\(", text))
        if len(accesses) != expected:
            raise ValueError("orchestration tensor argument is not a direct bound variable")
        tasks.append(
            dict(
                site=len(tasks),
                args=args,
                kernels=[kernel_names[k] for k in ids],
                reads=sorted({a for mode, a in accesses if mode in {"input", "inout"}}),
                writes=sorted({a for mode, a in accesses if mode in {"output", "inout"}}),
            )
        )
    if not tasks or len(re.findall(pattern, text)) != len(
        re.findall(r"\brt_submit_(?:aiv_task|aic_task|task)\(", text)
    ):
        raise ValueError("orchestration submit inventory incomplete")
    edges = []
    for i, first in enumerate(tasks):
        for second in tasks[i + 1 :]:
            for kind, left, right in (
                ("raw", "writes", "reads"),
                ("war", "reads", "writes"),
                ("waw", "writes", "writes"),
            ):
                for tensor in sorted(set(first[left]) & set(second[right])):
                    edges.append(dict(source=first["site"], target=second["site"], kind=kind, tensor=tensor))
    return dict(
        tasks=tasks,
        dependencies=edges,
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        explicit_dummy_sites=len(re.findall(r"\brt_submit_dummy_task\(", text)),
        invocation_model_complete=False,
        omissions=[
            "dynamic submit multiplicity",
            "tensor aliases",
            "explicit fence expansion",
            "runtime core/lane assignment and contention",
        ],
    )


def frontend_c2v_dependencies(
    pto_text: str,
    producer: str,
    consumer: str,
    schedules: Mapping[str, Mapping[str, Any]],
    graphs: Mapping[str, str],
) -> dict[str, Any]:
    """Recover a named single-producer C2V channel, without inventing lane timing.

    The peer reserve/import is the channel identity, not equality of addresses.
    Only a single static push is supported. Pop/free sites retain their branch
    and loop contexts; callers must expand runtime lanes before composing time.
    """
    starts = list(re.finditer(r"\bfunc\.func\s+@([\w.$]+)\s*\(", pto_text))
    functions = {
        m[1]: pto_text[m.start() : starts[i + 1].start() if i + 1 < len(starts) else len(pto_text)]
        for i, m in enumerate(starts)
    }
    if producer not in functions or consumer not in functions:
        raise ValueError("C2V function missing")
    first, second = functions[producer], functions[consumer]
    imports = re.findall(
        r'pto\.import_reserved_buffer\s*\{name\s*=\s*"([^"]+)",\s*peer_func\s*=\s*@([\w.$]+)', first
    )
    reserves = re.findall(r'pto\.reserve_buffer\s*\{name\s*=\s*"([^"]+)"', second)
    channels = [name for name, peer in imports if peer == consumer and name in reserves]
    if len(channels) != 1:
        raise ValueError("C2V requires one unambiguous named peer channel")
    configs = []
    for body, side in ((first, "aic"), (second, "aiv")):
        binding = re.search(
            rf'(%[\w.$]+)\s*=\s*pto\.(?:import_reserved_buffer|reserve_buffer)\s*\{{name\s*=\s*"{re.escape(channels[0])}"',
            body,
        )
        consumed = re.findall(r"c2v_consumer_buf\s*=\s*(%[\w.$]+)", body)
        if not binding or consumed != [binding[1]]:
            raise ValueError("C2V named peer storage is not bound to pipe initialization")
        init = re.findall(rf"pto\.{side}_initialize_pipe\s*\{{([^}}]+)\}}", body)
        if len(init) != 1:
            raise ValueError("C2V requires exactly one pipe initialization per function")
        config = {k: int(v) for k, v in re.findall(r"(dir_mask|slot_size|slot_num)\s*=\s*(\d+)", init[0])}
        if set(config) != {"dir_mask", "slot_size", "slot_num"} or config["dir_mask"] != 1:
            raise ValueError("C2V requires a complete unidirectional channel configuration")
        configs.append(config)
    if configs[0] != configs[1] or min(configs[0].values()) <= 0:
        raise ValueError("C2V peer slot configurations disagree")

    def sites(function, operation):
        nodes, _ = parse_weighted_graph(graphs[function])
        by_access = {n["access"]: i for i, n in nodes.items()}
        if len(by_access) != len(nodes):
            raise ValueError("C2V graph has ambiguous access identities")
        schedule = {
            n.get("operation", {}).get("pypto_access_order"): n
            for n in schedules[function]["nodes"]
            if n.get("kind") == "operation"
        }
        result = []
        for line in functions[function].splitlines():
            if not re.search(rf"\bpto\.{operation}\b", line):
                continue
            access = re.search(r"pypto\.access\.(\d+)", line)
            split = re.search(r"split\s*=\s*(\d+)", line)
            if not access or not split or int(split[1]) != 0:
                raise ValueError("C2V site lacks provenance or uses an unsupported split")
            order = int(access[1])
            if order not in by_access or order not in schedule:
                raise ValueError("C2V transfer site is missing from the schedule graph")
            node = schedule[order]
            result.append(
                dict(
                    function=function,
                    access_order=order,
                    node=by_access[order],
                    branch_stack=node.get("branch_stack", []),
                    loop_stack=node.get("loop_stack", []),
                )
            )
        return result

    pushes = sites(producer, "tpush_to_aiv")
    pops = sites(consumer, "tpop_from_aic")
    frees = sites(consumer, "tfree_from_aic")
    if len(pushes) != 1 or pushes[0]["loop_stack"] or pushes[0]["branch_stack"] or not pops:
        raise ValueError("C2V requires one unconditional non-loop push and at least one pop")
    if len(pops) != len(frees):
        raise ValueError("C2V pop/free inventory differs")
    edges = []
    for pop, free in zip(pops, frees, strict=True):
        if (pop["branch_stack"], pop["loop_stack"]) != (free["branch_stack"], free["loop_stack"]):
            raise ValueError("C2V pop/free scope mismatch")
        if pop["access_order"] >= free["access_order"]:
            raise ValueError("C2V release precedes pop")
        edges.extend(
            (
                dict(source=pushes[0], target=pop, kind="mixed-c2v-availability"),
                dict(source=pop, target=free, kind="mixed-slot-release"),
            )
        )
    return dict(
        schema_version=1,
        channel=channels[0],
        **configs[0],
        edges=edges,
        pto_sha256=hashlib.sha256(pto_text.encode()).hexdigest(),
        graph_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in graphs.items()},
        invocation_model_complete=False,
        requires_runtime_lane_expansion=True,
        latency_cycles=None,
    )


def parse_weighted_graph(text: str) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Read one fully weighted native text graph, including recurrence records."""
    headers = re.findall(
        r"(?m)^KernelScheduleGraph @(\S+) nodes=(\d+) dag_edges=(\d+) dependencies=(\d+)", text
    )
    if len(headers) != 1:
        raise ValueError("expected exactly one native graph header")
    nodes = {}
    edges = []
    for line in text.splitlines():
        if match := re.match(r"\s+node\[(\d+)\] (.*)", line):
            node_id = int(match[1])
            fields = dict(re.findall(r"(\w+)=([^ ]+)", match[2]))
            if node_id in nodes or fields.get("duration_resolved") != "true":
                raise ValueError(f"duplicate or unresolved node {node_id}")
            nodes[node_id] = {
                **fields,
                "cycles": int(fields["duration_cycles"]),
                "access": int(fields["pypto_access_order"]),
                "loop_depth": int(fields["loop_depth"]),
            }
        elif match := re.match(
            r"\s+edge (\d+) -> (\d+) kind=(\S+) distance=(\d+) latency_cycles=(\d+)", line
        ):
            edges.append(
                dict(
                    source=int(match[1]),
                    target=int(match[2]),
                    kind=match[3],
                    distance=int(match[4]),
                    latency=int(match[5]),
                    recurrence_loop_depth=(
                        int(depth[1]) if (depth := re.search(r"loop-depth=(\d+)", line)) else 0
                    ),
                )
            )
    if set(nodes) != set(range(int(headers[0][1]))) or len(edges) != int(headers[0][3]):
        raise ValueError("graph node/dependency counts do not match its header")
    if any(edge["source"] not in nodes or edge["target"] not in nodes for edge in edges):
        raise ValueError("graph edge names an absent node")
    return nodes, edges


def dag_witness(nodes: Mapping[int, Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the acyclic LP, one witness, and per-node finish/suffix times.

    Positive-distance edges are explicitly excluded, not folded into the DAG.
    Duplicate endpoints use their maximum delay, matching native scoring.
    """
    pairs: dict[tuple[int, int], int] = {}
    for edge in edges:
        if edge.get("distance", 0):
            continue
        pair = (int(edge["source"]), int(edge["target"]))
        if any(node not in nodes for node in pair):
            raise ValueError(f"edge {pair} names an absent node")
        delay = int(edge.get("latency", 0))
        if delay < 0:
            raise ValueError(f"edge {pair} has a negative delay")
        pairs[pair] = max(delay, pairs.get(pair, 0))
    children: dict[int, list[tuple[int, int]]] = defaultdict(list)
    parents: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (source, target), delay in sorted(pairs.items()):
        children[source].append((target, delay))
        parents[target].append((source, delay))
    pending = {node: len(parents[node]) for node in nodes}
    ready = sorted(node for node in nodes if not pending[node])
    order = []
    finish, predecessor = {}, {}
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        best, parent = max(((finish[p] + delay, p) for p, delay in parents[node]), default=(0, None))
        cycles = int(nodes[node]["cycles"])
        if cycles < 0:
            raise ValueError(f"node {node} has a negative duration")
        finish[node], predecessor[node] = best + cycles, parent
        for child, _ in children[node]:
            pending[child] -= 1
            if not pending[child]:
                heapq.heappush(ready, child)
    if len(order) != len(nodes):
        raise ValueError("distance-zero graph contains a cycle")
    suffix = {}
    for node in reversed(order):
        suffix[node] = int(nodes[node]["cycles"]) + max(
            (delay + suffix[child] for child, delay in children[node]), default=0
        )
    latency, tail = max(((value, node) for node, value in finish.items()), default=(0, None))
    path = []
    while tail is not None:
        path.append(tail)
        tail = predecessor[tail]
    return dict(longest_path_cycles=latency, critical_path=path[::-1], finish=finish, suffix=suffix)


def single_writer_raw_dependencies(
    nodes: Mapping[int, Mapping[str, Any]],
    catalog: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recover required RAW edges from explicit allocation identity, not addresses.

    Only complete, single-writer allocation phases with a known covering write
    and dominating structural scope qualify. Multi-writer/ambiguous cases are
    not inferred. This is a conformance witness, not a complete SSA exporter.
    """
    by_access = {node["access"]: node_id for node_id, node in nodes.items()}
    if len(by_access) != len(nodes):
        raise ValueError("allocation audit requires unique operation access orders")
    scopes = {
        node.get("operation", {}).get("pypto_access_order"): node
        for node in schedule.get("nodes", [])
        if node.get("kind") == "operation"
    }
    result = []
    for allocation in catalog:
        if not allocation.get("complete"):
            continue
        accesses = allocation["accesses"]
        writes = [a for a in accesses if a["mode"] == "write"]
        if len(writes) != 1:
            continue
        writer = writes[0]
        for reader in accesses:
            if reader["mode"] != "read" or reader["order"] <= writer["order"]:
                continue
            if any(a["order"] not in by_access or not a.get("range_known") for a in (writer, reader)):
                continue
            if writer.get("pool") != reader.get("pool"):
                continue
            if (
                writer["offset"] > reader["offset"]
                or writer["offset"] + writer["size"] < reader["offset"] + reader["size"]
            ):
                continue
            source, target = by_access[writer["order"]], by_access[reader["order"]]
            before, after = scopes.get(writer["order"]), scopes.get(reader["order"])
            if before is None or after is None:
                continue
            if any(
                after.get(key, [])[: len(before.get(key, []))] != before.get(key, [])
                for key in ("loop_stack", "branch_stack")
            ):
                continue
            result.append(
                dict(
                    source=source,
                    target=target,
                    kind="allocation-raw",
                    distance=0,
                    latency=0,
                    allocation=allocation["buffer"],
                    source_access=writer["order"],
                    target_access=reader["order"],
                )
            )
    return result
