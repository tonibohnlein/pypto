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

#ifndef PYTHON_BINDINGS_MODULES_TESTING_SYNC_SUPPORT_H_
#define PYTHON_BINDINGS_MODULES_TESTING_SYNC_SUPPORT_H_

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "src/ir/transforms/utils/physical_sync_schedule.h"

namespace pypto::ir::sync {
std::optional<int64_t> EvaluateSyncCoordinate(const SyncCoordinateProgram& program, size_t root,
                                              const std::map<const ForStmt*, int64_t>& original_values,
                                              const std::map<size_t, int64_t>& parameters,
                                              std::optional<int64_t> block);
}  // namespace pypto::ir::sync

// Testing-owned adapters. Retained mathematics first produces actual tokens,
// then uses production LowerSync. Synthetic pure command forests exercise the
// same scheduler/emitter; they supply no order or resource evidence.
// None is a production route or an independent correctness oracle.
namespace pypto::python::sync_testing {
using ir::CallPtr;
using ir::ExprPtr;
using ir::FunctionPtr;
using ir::Span;
using ir::StmtPtr;
using ir::sync::CompactSyncResult;
using ir::sync::PhysicalSyncRealization;
using ir::sync::SyncInvocationAnalysis;
using ir::sync::SyncSchedulePolicy;
namespace physical = ir::sync::physical;
struct DemandInvocation {
  FunctionPtr function;
  std::vector<PhysicalSyncRealization> cases;
};
PhysicalSyncRealization RealizeDemandPlan(const CompactSyncResult&, const std::string&,
                                          const std::optional<std::vector<int>>&,
                                          const SyncSchedulePolicy& = {});
DemandInvocation RealizeDemandInvocation(const SyncInvocationAnalysis&, const std::string&,
                                         const std::optional<std::vector<int>>&,
                                         const SyncSchedulePolicy& = {});
physical::ControlPtr ImportCommands(const StmtPtr&, PhysicalSyncRealization&);
StmtPtr MaterializeValues(const FunctionPtr&, const StmtPtr&);
std::vector<StmtPtr> ShareGuards(const std::vector<StmtPtr>&, PhysicalSyncRealization&);
struct CommandCase {
  std::pair<int64_t, int64_t> blocks;
  StmtPtr body;
};
std::vector<StmtPtr> EmitCases(const ExprPtr&, const physical::Intervals&, const std::vector<CommandCase>&,
                               PhysicalSyncRealization&, const Span&);
std::pair<std::vector<physical::Intervals>, PhysicalSyncRealization> InspectGroups(
    const FunctionPtr&, const std::vector<CommandCase>&);
PhysicalSyncRealization EmitBoundaries(const FunctionPtr&,
                                       const std::vector<std::tuple<CallPtr, StmtPtr, StmtPtr>>&, bool,
                                       bool = false);
std::optional<ir::sync::PointBoundary> BoundaryAt(const ir::sync::CompactSyncView& view, size_t storage,
                                                  int64_t row, int64_t column,
                                                  const std::map<size_t, int64_t>& parameters,
                                                  std::optional<int64_t> block_index = std::nullopt);
std::optional<ir::sync::PointBoundary> BoundarySlot(const ir::sync::CompactSyncView& view, size_t storage,
                                                    int64_t slot, int64_t row, int64_t column,
                                                    const std::map<size_t, int64_t>& parameters = {},
                                                    std::optional<int64_t> block_index = std::nullopt);
std::vector<ir::sync::Crossing> ExpandRetainedDemands(const ir::sync::CompactSyncView& view,
                                                      size_t max_payloads);
std::shared_ptr<const CompactSyncResult> SelectInvocationCase(const SyncInvocationAnalysis& analysis,
                                                              int64_t block_index,
                                                              const std::map<size_t, int64_t>& parameters);
}  // namespace pypto::python::sync_testing
#endif  // PYTHON_BINDINGS_MODULES_TESTING_SYNC_SUPPORT_H_
