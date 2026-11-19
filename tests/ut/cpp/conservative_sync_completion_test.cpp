/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "src/ir/transforms/utils/conservative_sync_completion.h"

#include <cassert>
#include <vector>

namespace {
using pypto::ir::PipeType;
using pypto::ir::sync::CompletionFacts;
using pypto::ir::sync::CompletionSummary;
using pypto::ir::sync::CompletionSyncEdge;
using pypto::ir::sync::CompletionTransferResult;
using pypto::ir::sync::CompletionVertex;
using pypto::ir::sync::PipeToMask;
using pypto::ir::sync::SolveCompletionTransfer;

void CheckLinearTransitiveRelayAndWrite() {
  std::vector<CompletionVertex> vertices(3);
  vertices[0].write_pipes = PipeToMask(PipeType::MTE2);
  vertices[0].post_syncs = {{PipeType::MTE2, PipeType::S, /*bidirectional=*/false}};
  vertices[1].pre_syncs = {{PipeType::S, PipeType::V, /*bidirectional=*/false}};
  vertices[2].pre_syncs = {{PipeType::V, PipeType::MTE3, /*bidirectional=*/true}};

  const CompletionTransferResult res = SolveCompletionTransfer(vertices, {{0, 1}, {1, 2}}, 0, 2);
  assert(res.work.sccs == 3);
  assert(res.work.non_trivial_sccs == 0);
  assert(!res.summary.has_bypass_path);
  assert(res.summary.IsRelayed(PipeType::MTE2, PipeType::MTE3));
  assert(res.summary.IsRelayed(PipeType::S, PipeType::MTE3));
  // Bidirectional edge at vertex 2 relays MTE3 -> V, not MTE3 -> MTE2.
  assert(res.summary.IsRelayed(PipeType::MTE3, PipeType::V));
  assert(!res.summary.IsRelayed(PipeType::MTE3, PipeType::MTE2));
  assert(res.summary.IsPostWriteSafe(PipeType::MTE2, PipeType::MTE3));
  assert(!res.summary.IsPostWriteSafe(PipeType::MTE2, PipeType::MTE1));
}

void CheckConditionalBranchMeet() {
  // Branch 1: write(MTE2) + sync(MTE2 -> V)
  // Branch 2: empty bypass (no write, no sync)
  {
    std::vector<CompletionVertex> vertices(4);
    vertices[1].write_pipes = PipeToMask(PipeType::MTE2);
    vertices[1].post_syncs = {{PipeType::MTE2, PipeType::V, false}};
    const auto res = SolveCompletionTransfer(vertices, {{0, 1}, {0, 2}, {1, 3}, {2, 3}}, 0, 3);
    assert(res.summary.has_bypass_path);
    assert(!res.summary.IsRelayed(PipeType::MTE2, PipeType::V));
    // Every path that wrote MTE2 synced MTE2 -> V.
    assert((res.summary.active & PipeToMask(PipeType::MTE2)) != 0);
    assert(res.summary.IsPostWriteSafe(PipeType::MTE2, PipeType::V));
  }

  // Branch 1: write(MTE2) + sync(MTE2 -> V)
  // Branch 2: write(MTE2) with NO sync
  {
    std::vector<CompletionVertex> vertices(4);
    vertices[1].write_pipes = PipeToMask(PipeType::MTE2);
    vertices[1].post_syncs = {{PipeType::MTE2, PipeType::V, false}};
    vertices[2].write_pipes = PipeToMask(PipeType::MTE2);
    const auto res = SolveCompletionTransfer(vertices, {{0, 1}, {0, 2}, {1, 3}, {2, 3}}, 0, 3);
    assert(!res.summary.IsPostWriteSafe(PipeType::MTE2, PipeType::V));
  }
}

void CheckLoopSccPostWriteAndKillOrdering() {
  // Optional loop: hdr(0) -> body(1: write MTE2 -> sync MTE2->V) -> hdr(0), hdr(0) -> exit(2).
  {
    std::vector<CompletionVertex> vertices(3);
    vertices[1].write_pipes = PipeToMask(PipeType::MTE2);
    vertices[1].post_syncs = {{PipeType::MTE2, PipeType::V, true}};
    const auto res = SolveCompletionTransfer(vertices, {{0, 1}, {1, 0}, {0, 2}}, 0, 2);
    assert(res.work.non_trivial_sccs == 1);
    assert(res.summary.has_bypass_path);
    // 0-trip bypass prevents guaranteed inbound relay MTE2 -> V across the loop.
    assert(!res.summary.IsRelayed(PipeType::MTE2, PipeType::V));
    // Yet any iteration that executes completes MTE2 -> V before exiting.
    assert(res.summary.IsPostWriteSafe(PipeType::MTE2, PipeType::V));
  }

  // Buggy loop: sync(MTE2 -> V) BEFORE write(MTE2). Last write is not completed!
  {
    std::vector<CompletionVertex> vertices(3);
    vertices[1].pre_syncs = {{PipeType::MTE2, PipeType::V, true}};
    vertices[1].write_pipes = PipeToMask(PipeType::MTE2);
    const auto res = SolveCompletionTransfer(vertices, {{0, 1}, {1, 0}, {0, 2}}, 0, 2);
    assert(res.work.non_trivial_sccs == 1);
    assert(!res.summary.IsPostWriteSafe(PipeType::MTE2, PipeType::V));
  }

  // Do-while (>= 1 trip) loop: sync(MTE2 -> S) -> sync(S -> V) in body, backedge to body, exit.
  {
    std::vector<CompletionVertex> vertices(3);
    vertices[1].pre_syncs = {{PipeType::MTE2, PipeType::S, false}};
    vertices[1].post_syncs = {{PipeType::S, PipeType::V, false}};
    const auto res = SolveCompletionTransfer(vertices, {{0, 1}, {1, 1}, {1, 2}}, 0, 2);
    assert(res.work.non_trivial_sccs == 1);
    assert(!res.summary.has_bypass_path);
    assert(res.summary.IsRelayed(PipeType::MTE2, PipeType::V));
  }
}

void CheckNestedSummaryComposition() {
  // Child region: writes V, syncs (S -> V -> MTE3), no bypass.
  std::vector<CompletionVertex> child_v(2);
  child_v[0].pre_syncs = {{PipeType::S, PipeType::V, false}};
  child_v[0].write_pipes = PipeToMask(PipeType::V);
  child_v[1].pre_syncs = {{PipeType::V, PipeType::MTE3, false}};
  const CompletionSummary child_summary = SolveCompletionTransfer(child_v, {{0, 1}}, 0, 1).summary;
  assert(child_summary.IsRelayed(PipeType::S, PipeType::MTE3));
  assert(child_summary.IsPostWriteSafe(PipeType::V, PipeType::MTE3));

  // Parent: write(MTE2), sync(MTE2 -> S), child_summary, check MTE2->MTE3 and V->MTE3.
  std::vector<CompletionVertex> parent_v(3);
  parent_v[0].write_pipes = PipeToMask(PipeType::MTE2);
  parent_v[0].post_syncs = {{PipeType::MTE2, PipeType::S, false}};
  parent_v[1].nested_summary = child_summary;
  const CompletionSummary parent_summary = SolveCompletionTransfer(parent_v, {{0, 1}, {1, 2}}, 0, 2).summary;
  assert(parent_summary.IsRelayed(PipeType::MTE2, PipeType::MTE3));
  assert(parent_summary.IsPostWriteSafe(PipeType::MTE2, PipeType::MTE3));
  assert(parent_summary.IsPostWriteSafe(PipeType::V, PipeType::MTE3));
}

}  // namespace

int main() {
  CheckLinearTransitiveRelayAndWrite();
  CheckConditionalBranchMeet();
  CheckLoopSccPostWriteAndKillOrdering();
  CheckNestedSummaryComposition();
  return 0;
}
