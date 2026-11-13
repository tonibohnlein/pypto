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

#ifndef PYPTO_BINDINGS_MODULES_TESTING_SYNC_SUPPORT_H_
#define PYPTO_BINDINGS_MODULES_TESTING_SYNC_SUPPORT_H_

#include "src/ir/transforms/utils/physical_sync_schedule.h"

// Testing-owned adapters. They supply mathematical callbacks or synthetic pure
// command forests to the SAME allocator/scheduler/emitter used by LowerSync.
// None is a production route or an independent correctness oracle.
namespace pypto::python::sync_testing {
using namespace ir;
using namespace ir::sync;
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
}  // namespace pypto::python::sync_testing
#endif  // PYPTO_BINDINGS_MODULES_TESTING_SYNC_SUPPORT_H_
