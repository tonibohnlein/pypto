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

#include <memory>
#include <utility>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::pass {
namespace {
bool IsOwnedLeaf(const FunctionPtr& function) {
  return (function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV) &&
         !function->requires_runtime_binding_ && !function->HasAttr(kAttrExternalSource) &&
         !function->HasAttr(kAttrBuiltinTemplateDir);
}
}  // namespace

Pass InsertSync() {
  // Invocation evidence belongs to this program, never to a process-global
  // sidecar. Standalone failure leaves the immutable input program untouched.
  return CreateProgramPass(
      [](const ProgramPtr& program) {
        auto* context = PassContext::Current();
        CHECK(context && context->GetAutoSyncDisjointGM())
            << "InsertSync requires the explicit disjoint GM argument contract";
        const auto target = context->GetBackendHandler()->GetPtoTargetArch();
        const auto evidence = sync::CollectSyncInvocationEvidence(program);
        auto functions = program->functions_;
        for (auto& [global, function] : functions) {
          CheckLogicalSync(function);
          CHECK(!sync::HasLogicalSync(function)) << "InsertSync refuses preexisting logical synchronization";
          if (!IsOwnedLeaf(function)) continue;
          CHECK(function->ir_stage_ == FunctionIRStage::Buffer &&
                context->GetMemoryPlanner() != MemoryPlanner::PtoAS)
              << "InsertSync requires physically placed Buffer IR";
          auto bounds = evidence.parameters.find(global->name_);
          const auto parameters =
              bounds == evidence.parameters.end() ? sync::SyncParameterBounds{} : bounds->second;
          auto input = sync::ImportPlacedSync(function, true, target, parameters);
          auto analysis = sync::AnalyzeCompactSync(std::move(input));
          function = sync::RealizeCompactSync(analysis).function;
        }
        return std::make_shared<Program>(functions, program->name_, program->span_);
      },
      "InsertSync", PassProperties{.produced = {IRProperty::LogicalSyncValid}});
}

Pass OptimizeSync() {
  return CreateProgramPass(
      [](const ProgramPtr& program) {
        for (const auto& [global, function] : program->functions_) CheckLogicalSync(function);
        return program;
      },
      "OptimizeSync", PassProperties{.produced = {IRProperty::LogicalSyncValid}});
}

Pass LowerSync() {
  return CreateFunctionPass(
      [](const FunctionPtr& function) {
        CheckLogicalSync(function);
        if (!IsOwnedLeaf(function)) return function;
        auto* context = PassContext::Current();
        CHECK(context) << "LowerSync requires a target PassContext";
        auto result =
            sync::LowerLogicalSync(function, context->GetBackendHandler()->GetPtoTargetArch(), std::nullopt);
        // Codegen may use the placed-access qualification only after all actual
        // token uses have been lowered and the downstream manual marker exists.
        if (result.set_sites && result.wait_sites) return sync::SyncOutcome(result.function, true, "");
        return result.function;
      },
      "LowerSync", PassProperties{.produced = {IRProperty::LogicalSyncValid}});
}
}  // namespace pypto::ir::pass
