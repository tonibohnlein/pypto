# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate command-graph semantics of the strict emitted-C++ adapter."""

import pytest
from pypto import ir

from .sync_cpp_order import command_graph, cpp_trace
from .sync_order_metrics import compare_order
from .test_finite_sync_analysis import reference


def calls():
    result = []
    for op, pipe, reads, writes in [
        ("buffer.load", ir.PipeType.MTE2, [], [0]),
        ("buffer.load", ir.PipeType.MTE2, [], [1]),
        ("buffer.matmul", ir.PipeType.M, [0], []),
    ]:
        call = ir.Call(ir.get_op(op), [], ir.TupleType([]), ir.Span.unknown())
        result.append((call, {}, pipe, reads, writes))
    return result


def test_late_publication_adds_order_without_changing_command_count():
    payloads = calls()
    early = [
        ["payload", "TLOAD"],
        ["set_flag", "PIPE_MTE2", "PIPE_M", "0"],
        ["payload", "TLOAD"],
        ["wait_flag", "PIPE_MTE2", "PIPE_M", "0"],
        ["payload", "TMATMUL"],
    ]
    late = [early[0], early[2], early[1], early[3], early[4]]
    a, _ = command_graph(payloads, early, range(6))
    b, _ = command_graph(payloads, late, range(6))
    required = reference([(p, True, r, w) for _, _, p, r, w in payloads])[2]
    result = compare_order([c[2] for c in payloads], required, b, a)
    assert result["relation"] == "after-dominates"
    assert result["removed"] == [(1, 2)]
    assert result["before"]["excess_completion_start"] == 1
    assert result["after"]["excess_completion_start"] == 0
    tail, facts = command_graph(payloads, early + [["return_barrier", "PIPE_ALL"]], range(6))
    assert tail == a  # Return completion is not an extra internal payload pair.
    assert facts["return_barriers"] == 1


def test_all_barrier_does_not_consume_flag_and_unmatched_wait_refuses():
    with pytest.raises(AssertionError, match="outstanding"):
        command_graph([], [["set_flag", "PIPE_MTE2", "PIPE_M", "0"], ["barrier", "PIPE_ALL"]], range(6))
    with pytest.raises(AssertionError, match="without publication"):
        command_graph([], [["wait_flag", "PIPE_MTE2", "PIPE_M", "0"]], range(6))


def test_primed_protocol_requires_causal_consumption_before_republication():
    events = [["set_flag", "PIPE_MTE2", "PIPE_M", "0"], ["wait_flag", "PIPE_MTE2", "PIPE_M", "0"]]
    command_graph([], events, range(6))
    with pytest.raises(AssertionError, match="consumption"):
        command_graph([], events + events, range(6))
    # Explicit reverse acknowledgment proves the rearm; the checker adds no
    # reuse edge of its own and must distinguish these two programs.
    ack = [["set_flag", "PIPE_M", "PIPE_MTE2", "0"], ["wait_flag", "PIPE_M", "PIPE_MTE2", "0"]]
    command_graph([], events + ack + events, range(6))


def test_cpp_trace_preserves_parameter_promotions_and_range(tmp_path):
    source = tmp_path / "input.cpp"
    source.write_text(
        "AICORE void test(int32_t v0) {\n"
        "uint32_t v1 = 1;\n"
        "if (v0 < v1) {\n"
        "set_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);\n"
        "}\n}\n"
    )
    # int32(-1) promotes to uint32 here; replacing it with int64 would instead
    # take the branch. The host adapter must preserve the original signature.
    assert cpp_trace(source, tmp_path / "trace", [-1]) == []
    with pytest.raises(AssertionError, match="ABI type"):
        cpp_trace(source, tmp_path / "out_of_range", [1 << 31])
    source.write_text(source.read_text().replace("int32_t v0", "uint32_t v0"))
    with pytest.raises(AssertionError, match="ABI type"):
        cpp_trace(source, tmp_path / "unsigned", [-1])
