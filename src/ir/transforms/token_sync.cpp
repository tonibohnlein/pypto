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
#include <optional>
#include <string>
#include <utility>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/scoped_logical_sync.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::pass {
namespace {
bool IsOwnedLeaf(const FunctionPtr& function) {
  return (function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV) &&
         !function->requires_runtime_binding_ && !function->HasAttr(kAttrExternalSource) &&
         !function->HasAttr(kAttrBuiltinTemplateDir);
}
FunctionPtr InsertLogicalSync(const FunctionPtr& function, const std::string& target,
                              const sync::SyncParameterBounds& parameters,
                              std::optional<std::pair<int64_t, int64_t>> blocks) {
  if (blocks) {
    const auto invocation = sync::AnalyzeSyncInvocation(function, true, target, parameters, *blocks);
    return invocation.cases.size() == 1
               ? sync::RealizeCompactSync(*invocation.cases.front().mathematics).function
               : sync::RealizeLogicalInvocation(invocation).function;
  }
  auto input = sync::ImportPlacedSync(function, true, target, parameters, blocks);
  return sync::RealizeCompactSync(sync::AnalyzeCompactSync(std::move(input))).function;
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
          auto block = evidence.block_indices.find(global->name_);
          const auto blocks =
              block == evidence.block_indices.end() ? std::nullopt : std::make_optional(block->second);
          function = InsertLogicalSync(function, target, parameters, blocks);
          sync::CheckGeneratedLogicalSync(function);
        }
        return std::make_shared<Program>(functions, program->name_, program->span_);
      },
      "InsertSync", PassProperties{.produced = {IRProperty::LogicalSyncValid}});
}

Pass OptimizeSync() {
  return CreateProgramPass(
      [](const ProgramPtr& program) {
        auto functions = program->functions_;
        for (auto& [global, function] : functions) function = sync::OptimizeLogicalSync(function);
        return std::make_shared<Program>(functions, program->name_, program->span_);
      },
      "OptimizeSync", PassProperties{.produced = {IRProperty::LogicalSyncValid}});
}

Pass LowerSync() {
  return CreateProgramPass(
      [](const ProgramPtr& program) {
        // Re-establish caller facts from this token program. Neither insertion's
        // old analysis nor a process-global sidecar authorizes physical lowering.
        for (const auto& [global, function] : program->functions_) CheckLogicalSync(function);
        const auto evidence = sync::CollectSyncInvocationEvidence(program);
        auto* context = PassContext::Current();
        CHECK(context) << "LowerSync requires a target PassContext";
        auto functions = program->functions_;
        for (auto& [global, function] : functions) {
          if (!IsOwnedLeaf(function)) continue;
          auto bound = evidence.parameters.find(global->name_);
          const auto parameters =
              bound == evidence.parameters.end() ? sync::SyncParameterBounds{} : bound->second;
          auto block = evidence.block_indices.find(global->name_);
          const auto blocks =
              block == evidence.block_indices.end() ? std::nullopt : std::make_optional(block->second);
          auto result = sync::LowerLogicalSync(function, context->GetBackendHandler()->GetPtoTargetArch(),
                                               std::nullopt, parameters, blocks);
          // Mark only a complete physical plan. Functions without commands
          // continue to receive downstream PTOAS synchronization as before.
          function = result.set_sites && result.wait_sites ? sync::SyncOutcome(result.function, true, "")
                                                           : result.function;
        }
        return std::make_shared<Program>(functions, program->name_, program->span_);
      },
      "LowerSync", PassProperties{.produced = {IRProperty::LogicalSyncValid}});
}

Pass RealizeSync() {
  auto transform = [](const FunctionPtr& function, const sync::SyncParameterBounds& bounds,
                      const std::optional<std::pair<int64_t, int64_t>>& block_bounds) -> FunctionPtr {
    if (!IsInCoreType(function->func_type_)) return function;
    // These functions do not enter PTO code generation; their external owner
    // remains responsible for synchronization. Do not claim PTOAS delegation.
    if (function->requires_runtime_binding_ || function->HasAttr(kAttrExternalSource) ||
        function->HasAttr(kAttrBuiltinTemplateDir))
      return function;
    auto* ctx = PassContext::Current();
    CHECK(ctx && ctx->GetEnablePyptoAutoSync()) << "RealizeSync requires enable_pypto_autosync=True";
    if (function->func_type_ != FunctionType::AIC && function->func_type_ != FunctionType::AIV)
      return sync::SyncOutcome(function, false, "PyPTO synchronization supports only AIC/AIV leaves");
    if (function->ir_stage_ != FunctionIRStage::Buffer || ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS)
      return sync::SyncOutcome(function, false, "PyPTO synchronization requires physically placed Buffer IR");
    if (!ctx->GetAutoSyncDisjointGM())
      return sync::SyncOutcome(function, false, "Disjoint GM argument contract was not supplied");
    auto target = ctx->GetBackendHandler()->GetPtoTargetArch();
    if (target != "a2a3")
      return sync::SyncOutcome(function, false,
                               "PyPTO physical synchronization supports only manual a2a3 macros");
    // Only publish after complete analysis AND realization. A qualification or
    // resource refusal returns the original body, never partially emitted IR.
    // Internal errors are deliberately not caught. The strict retained-result
    // APIs still throw on these user-facing input/proof refusals.
    try {
      // The same logical producer and actual-token consumer as the public
      // stages. Insertion preserves callers, payloads and scalar coordinates,
      // so this transaction may reuse its own freshly collected evidence.
      const auto logical = InsertLogicalSync(function, target, bounds, block_bounds);
      sync::CheckGeneratedLogicalSync(logical);
      const auto optimized = sync::OptimizeLogicalSync(logical);
      sync::CheckGeneratedLogicalSync(optimized);
      const auto physical = sync::LowerLogicalSync(optimized, target, std::nullopt, bounds, block_bounds);
      // PTOAS recognizes literal flags, not empty plans or barriers, as manual
      // ownership. Do not fabricate a dummy flag or mix partial ownership.
      if (!physical.set_sites || !physical.wait_sites)
        return sync::SyncOutcome(function, false,
                                 "Candidate has no SET/WAIT marker; delegate the original kernel");
      return sync::SyncOutcome(physical.function, true, "");
    } catch (const pypto::ValueError& error) {
      return sync::SyncOutcome(function, false, error.what());
    }
  };
  // Range evidence belongs to this closed program's call sites, not mutable
  // function metadata. Do not infer bounds when the standalone API has none.
  return CreateProgramPass(
      [transform](const ProgramPtr& program) {
        // Never delegate unresolved logical IR or reprocess an owned physical
        // plan, even when an earlier eligibility test would skip that function.
        for (const auto& [global, function] : program->functions_) {
          CHECK(!sync::HasLogicalSync(function))
              << "RealizeSync refuses preexisting logical synchronization; use LowerSync";
          CHECK(!function->GetAttr<bool>(kAttrPyptoAutosync))
              << "RealizeSync cannot reprocess an owned physical plan";
        }
        auto evidence = sync::CollectSyncInvocationEvidence(program);
        auto functions = program->functions_;
        for (auto& [global, function] : functions) {
          auto block = evidence.block_indices.find(global->name_);
          function = transform(
              function, evidence.parameters[global->name_],
              block == evidence.block_indices.end() ? std::nullopt : std::make_optional(block->second));
        }
        return std::make_shared<Program>(functions, program->name_, program->span_);
      },
      "RealizeSync", PassProperties{});
}
}  // namespace pypto::ir::pass
