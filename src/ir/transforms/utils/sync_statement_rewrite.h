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
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
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
// Rewrite only statement occurrences in an already qualified synchronization
// owner. The generic mutator's typed hooks dispatch expressions directly and
// therefore bypass a VisitExpr override. Own every supported container here;
// preserve guards, metadata, loop binders and signatures by identity. Unknown
// statement kinds are preserved, not recursively interpreted or admitted.
class SyncStatementMutator : public IRMutator {
 public:
  StmtPtr VisitStmt(const StmtPtr& statement) override {
    if (!statement) {
      return statement;
    }
    switch (statement->GetKind()) {
      case ObjectKind::SeqStmts:
      case ObjectKind::IfStmt:
      case ObjectKind::ForStmt:
      case ObjectKind::WhileStmt:
      case ObjectKind::AssignStmt:
      case ObjectKind::EvalStmt:
      case ObjectKind::ReturnStmt:
      case ObjectKind::YieldStmt:
      case ObjectKind::InCoreScopeStmt:
      case ObjectKind::ClusterScopeStmt:
      case ObjectKind::GraphScopeStmt:
      case ObjectKind::RegScopeStmt:
      case ObjectKind::HierarchyScopeStmt:
      case ObjectKind::SpmdScopeStmt:
      case ObjectKind::SplitAivScopeStmt:
      case ObjectKind::RuntimeScopeStmt:
      case ObjectKind::CommDomainScopeStmt:
      case ObjectKind::SimtScopeStmt:
        // Preserve virtual typed hooks in derived command/return rewriters.
        return IRMutator::VisitStmt(statement);
      default:
        return statement;
    }
  }
  ExprPtr VisitExpr(const ExprPtr& expression) override { return expression; }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& sequence) override {
    return RewriteSyncSequence(sequence, [this](const StmtPtr& child) { return VisitStmt(child); });
  }
  StmtPtr VisitStmt_(const IfStmtPtr& branch) override {
    const auto then_body = VisitStmt(branch->then_body_);
    auto else_body = branch->else_body_;
    if (else_body) {
      else_body = VisitStmt(*else_body);
    }
    if (then_body == branch->then_body_ && else_body == branch->else_body_) {
      return branch;
    }
    auto result = MutableCopy(branch);
    result->then_body_ = then_body;
    result->else_body_ = std::move(else_body);
    return result;
  }
  StmtPtr VisitStmt_(const AssignStmtPtr& statement) override { return statement; }
  StmtPtr VisitStmt_(const EvalStmtPtr& statement) override { return statement; }
  StmtPtr VisitStmt_(const ReturnStmtPtr& statement) override { return statement; }
  StmtPtr VisitStmt_(const YieldStmtPtr& statement) override { return statement; }
  StmtPtr VisitStmt_(const ForStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const WhileStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const InCoreScopeStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const ClusterScopeStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const GraphScopeStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const RegScopeStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const HierarchyScopeStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const SpmdScopeStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const SplitAivScopeStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const CommDomainScopeStmtPtr& statement) override { return RewriteBody(statement); }
  StmtPtr VisitStmt_(const SimtScopeStmtPtr& statement) override { return RewriteBody(statement); }

 private:
  template <typename Statement>
  StmtPtr RewriteBody(const std::shared_ptr<const Statement>& original) {
    const auto body = VisitStmt(original->body_);
    if (body == original->body_) {
      return original;
    }
    // Copy the concrete subtype, including scope-specific evidence and attrs.
    auto result = MutableCopy(original);
    result->body_ = body;
    return result;
  }
};
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_STATEMENT_REWRITE_H_
