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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_STATEMENT_REWRITE_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_STATEMENT_REWRITE_H_
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "pypto/ir/stmt.h"
namespace pypto::ir::sync {
// Gather original nested sequences once. Re-flattening at each ancestor would
// repeatedly copy preserved statements when token leaves disappear.
inline StmtPtr RewriteSyncSequence(const SeqStmtsPtr& sequence,
                                   const std::function<StmtPtr(const StmtPtr&)>& rewrite) {
  std::vector<StmtPtr> output;
  std::function<void(const StmtPtr&)> append = [&](const StmtPtr& statement) {
    if (auto nested = As<SeqStmts>(statement)) {
      for (const auto& child : nested->stmts_) append(child);
      return;
    }
    auto result = rewrite(statement);
    if (auto replacement = As<SeqStmts>(result))
      output.insert(output.end(), replacement->stmts_.begin(), replacement->stmts_.end());
    else
      output.push_back(std::move(result));
  };
  for (const auto& child : sequence->stmts_) append(child);
  return std::make_shared<SeqStmts>(output, sequence->span_);
}
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_STATEMENT_REWRITE_H_
