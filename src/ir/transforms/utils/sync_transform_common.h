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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_TRANSFORM_COMMON_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_TRANSFORM_COMMON_H_

#include <cstdint>
#include <optional>
#include <set>
#include <unordered_set>
#include <utility>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto::ir::sync {

inline int64_t PositiveMod64(int64_t a, int64_t m) {
  const int64_t r = a % m;
  return r < 0 ? r + m : r;
}

inline int64_t PositiveMod128(__int128 a, int64_t m) {
  const __int128 r = a % m;
  return static_cast<int64_t>(r < 0 ? r + m : r);
}

inline __int128 FloorDiv128(__int128 a, __int128 b) {
  __int128 q = a / b;
  const __int128 r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) --q;
  return q;
}

inline __int128 FloorMod128(__int128 a, __int128 b) {
  __int128 r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) r += b;
  return r;
}

inline bool IsPureBlockQueryCall(const CallPtr& call) {
  return call && call->args_.empty() &&
         (IsOp(call, "buffer.get_block_idx") || IsOp(call, "buffer.get_block_num") ||
          IsOp(call, "buffer.get_subblock_idx"));
}

inline bool IsSyncNonPayloadCall(const CallPtr& call) {
  return IsPureBlockQueryCall(call) || IsOp(call, "buffer.alloc") || IsOp(call, "buffer.alloc_multi") ||
         IsOp(call, "buffer.subview") || IsOp(call, "buffer.reshape") || IsOp(call, "buffer.slot") ||
         IsOp(call, "buffer.set_validshape");
}

inline std::optional<PipeType> TryGetSyncPipeKwarg(const CallPtr& call, const char* key) {
  if (!call) return std::nullopt;
  for (const auto& [name, value] : call->kwargs_) {
    if (name == key) return static_cast<PipeType>(pypto::AnyCast<int>(value, name));
  }
  return std::nullopt;
}

inline PipeType RequireSyncPipeKwarg(const CallPtr& call, const char* key,
                                     const char* error_msg = "Missing logical synchronization pipe") {
  const auto pipe = TryGetSyncPipeKwarg(call, key);
  CHECK(pipe.has_value()) << error_msg;
  return *pipe;
}

inline bool ContainsSyncTokenType(const TypePtr& type) {
  if (!type) return false;
  if (As<SyncTokenType>(type)) return true;
  if (const auto tup = As<TupleType>(type)) {
    for (const auto& elem : tup->types_) {
      if (ContainsSyncTokenType(elem)) return true;
    }
  }
  return false;
}

// Collect distinct cross-pipe `sync.wait` edges `(src_pipe, dst_pipe)` in `stmt`.
inline std::set<std::pair<PipeType, PipeType>> CollectSyncWaitDirections(const StmtPtr& stmt) {
  std::set<std::pair<PipeType, PipeType>> dirs;
  if (!stmt) return dirs;
  class Collector : public IRVisitor {
   public:
    explicit Collector(std::set<std::pair<PipeType, PipeType>>& out) : out_(out) {}
    void VisitExpr(const ExprPtr& expression) override {
      if (expression && seen_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
    }

   protected:
    void VisitExpr_(const CallPtr& call) override {
      if (IsOp(call, "sync.wait") && !call->args_.empty()) {
        if (const auto tok = As<SyncTokenType>(call->args_[0]->GetType())) {
          const PipeType dst = RequireSyncPipeKwarg(call, "dst_pipe");
          if (tok->pipe_ != dst) out_.emplace(tok->pipe_, dst);
        }
      }
      IRVisitor::VisitExpr_(call);
    }

   private:
    std::set<std::pair<PipeType, PipeType>>& out_;
    std::unordered_set<const Expr*> seen_;
  } collector(dirs);
  collector.VisitStmt(stmt);
  return dirs;
}

template <typename SiteMap>
inline CallPtr FindSiteCallInStmt(const StmtPtr& node, const SiteMap& sites) {
  if (!node) return nullptr;
  if (const auto sequence = As<SeqStmts>(node)) {
    for (const auto& child : sequence->stmts_) {
      if (auto found = FindSiteCallInStmt(child, sites)) return found;
    }
    return nullptr;
  }
  if (const auto eval = As<EvalStmt>(node)) {
    const auto candidate = As<Call>(eval->expr_);
    if (candidate && sites.count(candidate.get())) return candidate;
  }
  return nullptr;
}

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_TRANSFORM_COMMON_H_
