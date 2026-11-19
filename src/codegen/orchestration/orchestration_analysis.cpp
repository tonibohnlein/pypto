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

#include "pypto/codegen/orchestration/orchestration_analysis.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

std::string GetSSABaseName(const std::string& name) { return auto_name::GetCompatibleBaseName(name); }

// IsBuiltinOp now lives in ir::op_predicates and is thin-forwarded from the
// header; codegen call sites still read unqualified via that forward.

bool IsTensorOp(const std::string& op_name) { return op_name.find("tensor.") == 0; }

bool IsArrayOp(const std::string& op_name) { return op_name.find("array.") == 0; }

std::string FormatConstIntValue(const ConstIntPtr& c, const std::string& cpp_type) {
  int64_t v = c->value_;
  if (cpp_type != "int64_t") {
    return "static_cast<" + cpp_type + ">(" + std::to_string(v) + ")";
  }
  return std::to_string(v);
}

std::string FormatConstFloatValue(const ConstFloatPtr& c, const std::string& cpp_type) {
  double v = c->value_;
  if (cpp_type == "float") {
    return std::to_string(static_cast<float>(v));
  }
  return std::to_string(v);  // double
}

// ---------------------------------------------------------------------------
// OrchestrationInfoCollector
// ---------------------------------------------------------------------------

void OrchestrationInfoCollector::VisitStmt_(const AssignStmtPtr& assign) {
  // The tuple key is registered against the *binding Var* (stable identity),
  // never the call pointer — a SubmitToCallView is transient, so keying on it
  // would desynchronise this analysis from the codegen lookup (which now also
  // keys on the binding Var).
  if (auto call = AsCallOrSubmitView(assign->value_)) {
    if (!IsBuiltinOp(call->op_->name_) && !IsOp(call, "tensor.create")) {
      if (As<TupleType>(call->GetType())) {
        std::string unique_key = "_tc_" + std::to_string(tuple_call_counter_++);
        tuple_var_to_key[assign->var_.get()] = unique_key;
      }
    }
  } else if (As<MakeTuple>(assign->value_)) {
    std::string unique_key = "_tc_" + std::to_string(tuple_call_counter_++);
    tuple_var_to_key[assign->var_.get()] = unique_key;
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_ref = AsVarLike(tuple_get->tuple_)) {
      auto it = tuple_var_to_key.find(tuple_ref.get());
      if (it != tuple_var_to_key.end()) {
        call_tuple_elements[it->second].push_back({tuple_get->index_, assign->var_.get()});
      }
    }
  }
  IRVisitor::VisitStmt_(assign);
}

// ---------------------------------------------------------------------------
// VarLineageCollector
// ---------------------------------------------------------------------------

VarLineageCollector::VarLineageCollector(ProgramPtr program) : program_(std::move(program)) {}

void VarLineageCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    var_to_param[param.get()] = param.get();
  }
}

void VarLineageCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = for_stmt->iter_args_[i];
    const Var* param = ResolveExpr(iter_arg->initValue_);
    if (param) {
      var_to_param[iter_arg.get()] = param;
      // Only propagate buffer lineage to the return_var for Tensor-type carries.
      // Scalar carries (e.g. a loop counter ``idx = batch_base + inner``) are
      // value-typed: the body may overwrite them with a freshly computed value
      // that has no relationship to the init param.  Propagating param lineage
      // to a Scalar return_var would make the return->param map bind a Scalar
      // return element to a param index, causing EmitTensorAlias to emit
      // ``const Tensor&`` for an int64_t variable (issue #1580).
      if (i < for_stmt->return_vars_.size() && AsTensorTypeLike(for_stmt->return_vars_[i]->GetType())) {
        var_to_param[for_stmt->return_vars_[i].get()] = param;
      }
    }
  }
  IRVisitor::VisitStmt_(for_stmt);
}

void VarLineageCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = while_stmt->iter_args_[i];
    const Var* param = ResolveExpr(iter_arg->initValue_);
    if (param) {
      var_to_param[iter_arg.get()] = param;
      // Same guard as ForStmt: only propagate to Tensor return_vars.
      if (i < while_stmt->return_vars_.size() && AsTensorTypeLike(while_stmt->return_vars_[i]->GetType())) {
        var_to_param[while_stmt->return_vars_[i].get()] = param;
      }
    }
  }
  IRVisitor::VisitStmt_(while_stmt);
}

void VarLineageCollector::VisitStmt_(const AssignStmtPtr& assign) {
  if (auto src_var = AsVarLike(assign->value_)) {
    const Var* param = ResolveVar(src_var.get());
    if (param) {
      var_to_param[assign->var_.get()] = param;
    }
  } else if (auto call = AsCallOrSubmitView(assign->value_)) {
    // Propagate lineage through function calls: the result inherits lineage
    // from the Out/InOut argument. This covers sequential SPMD submissions
    // like: out = self.kernel(a, b, out) where `out` is the output buffer.
    //
    // Group/Spmd wrappers carry their effective directions in the signature —
    // DeriveCallDirections materialized them — so a plain field read covers
    // every callee kind.
    if (!IsBuiltinOp(call->op_->name_)) {
      auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
      if (callee) {
        const auto& effective_dirs = callee->param_directions_;
        // Prefer tracing through the Out/InOut arg the callee actually
        // returns (multi-Out kernels would otherwise be mis-traced to the
        // first Out, leaking scratch-buffer lineage onto the result Var).
        std::optional<size_t> returned_idx = ir::return_lineage::ExplicitReturnedParamIndex(callee);
        for (size_t i = 0; i < effective_dirs.size() && i < call->args_.size(); ++i) {
          if (effective_dirs[i] != ParamDirection::Out && effective_dirs[i] != ParamDirection::InOut) {
            continue;
          }
          if (returned_idx.has_value() && i != *returned_idx) {
            continue;
          }
          if (auto arg_var = AsVarLike(call->args_[i])) {
            const Var* param = ResolveVar(arg_var.get());
            if (param) {
              var_to_param[assign->var_.get()] = param;
              break;
            }
          }
        }
      }
    }
  }
  IRVisitor::VisitStmt_(assign);
}

const Var* VarLineageCollector::ResolveVar(const Var* var) const {
  auto it = var_to_param.find(var);
  return it != var_to_param.end() ? it->second : nullptr;
}

const Var* VarLineageCollector::ResolveExpr(const ExprPtr& expr) const {
  if (auto var = AsVarLike(expr)) {
    return ResolveVar(var.get());
  }
  return nullptr;
}

}  // namespace codegen
}  // namespace pypto
