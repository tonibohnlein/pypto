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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_PLACEMENT_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_PLACEMENT_H_

#include <functional>
#include <map>
#include <set>
#include <utility>

#include "pypto/core/logging.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"

namespace pypto::ir::sync {
using SyncSiteKey = std::pair<size_t, size_t>;

// Both logical tokens and physical handoffs use the same original payload cuts.
// The admitted initialization branch is one always-executed site: wrap the
// entire branch, never publish from only one arm. Zero-trip bodies are untouched.
class SyncPlacement : public IRMutator {
 public:
  explicit SyncPlacement(const CompactSyncInput& input) : input_(input) {
    CHECK(input.owner) << "Synchronization realization requires the original occurrence owner";
    if (input.region_tree) {
      std::function<void(const SyncRegion&)> collect = [&](const SyncRegion& region) {
        if (region.repetition && !region.repetition->count) {
          absent_loops_.insert(region.repetition->loop.get());
          return;
        }
        for (const auto& child : region.children) collect(child);
      };
      collect(*input.region_tree);
    }
    for (size_t r = 0; r < input.segments.size(); ++r) {
      const auto& segment = input.segments[r];
      if (segment.original_loop) loops_.emplace(segment.original_loop.get(), r);
      for (const auto& frame : segment.nested_frames) loops_.emplace(frame.loop.get(), r);
      for (size_t s = 0; s < segment.sites.size(); ++s) {
        const auto& site = segment.sites[s];
        CHECK(anchors_.emplace(site.original.get(), SyncSiteKey{r, s}).second)
            << "Shared payload call needs a distinct occurrence anchor";
        if (site.continuation)
          CHECK(anchors_.emplace(site.continuation.get(), SyncSiteKey{r, s}).second)
              << "Shared conditional payload needs a distinct occurrence anchor";
      }
    }
  }

 protected:
  const CompactSyncInput& input_;
  std::map<const ForStmt*, size_t> loops_;
  const SyncSite& Payload(SyncSiteKey site) const { return input_.segments[site.first].sites[site.second]; }
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    return RewriteSyncSequence(op, [this](const StmtPtr& child) { return VisitStmt(child); });
  }
  virtual StmtPtr Wrap(const StmtPtr& payload, SyncSiteKey site) = 0;
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    auto it = anchors_.find(call.get());
    INTERNAL_CHECK(it != anchors_.end()) << "Payload was not retained by the importer";
    return Wrap(op, it->second);
  }
  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    StmtPtr first = op->then_body_;
    if (auto seq = As<SeqStmts>(first)) {
      INTERNAL_CHECK(seq->stmts_.size() == 1) << "Changed admitted initialization branch";
      first = seq->stmts_.front();
    }
    auto eval = As<EvalStmt>(first);
    INTERNAL_CHECK(eval) << "Missing initialization payload";
    auto call = As<Call>(eval->expr_);
    auto it = anchors_.find(call.get());
    INTERNAL_CHECK(it != anchors_.end() && Payload(it->second).continuation) << "Unqualified original branch";
    return Wrap(op, it->second);
  }
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (absent_loops_.count(op.get())) return op;
    if (!input_.segments[loops_.at(op.get())].count) return op;
    return IRMutator::VisitStmt_(op);
  }

 private:
  std::map<const Call*, SyncSiteKey> anchors_;
  std::set<const ForStmt*> absent_loops_;
};
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_PLACEMENT_H_
