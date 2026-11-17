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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_SYNC_CONTRACT_H_
#define PYPTO_BACKEND_COMMON_BUFFER_SYNC_CONTRACT_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/sync/isa_macro_resources.h"

namespace pypto::backend {
// Additional accesses of the selected native implementation, not operands or
// allocator reservations. Physical byte coordinates share the ordinary storage
// domain. A conservative enclosure must not become an exact footprint/selector.
// Factory-owned BufferSyncImplementation publishes this inventory immutably.
struct BufferSyncImplicitEffect {
  ir::MemorySpace space;
  int64_t begin, end;
  ir::BufferAccess data;
  const char* provenance;
  bool exact_footprint;
};
// Operation identity shared with Buffer emission. This does not qualify effects,
// target execution, aliases, readiness or synchronization ownership.
const char* BufferNativeOperation(const ir::CallPtr& call);

// Factory-owned variant evidence. Resource lookup cannot rebind a selection to
// another core/target or fabricate an audited signal-free macro.
class BufferSyncImplementation {
 public:
  ir::PipeType Pipe() const { return pipe_; }
  bool AccumulatorSizeEligible() const { return accumulator_size_eligible_; }
  // Only a selected blocking scalar instruction may publish its SSA result
  // after the protected payload cut. NoAccess or Value alone proves readiness.
  bool ProducesReadyScalar() const { return produces_ready_scalar_; }
  std::vector<sync::IsaResourceClaim> Claims() const;
  const std::vector<BufferSyncImplicitEffect>& ImplicitEffects() const { return implicit_effects_; }

 private:
  friend BufferSyncImplementation SelectBufferSyncImplementation(const ir::CallPtr&, ir::FunctionType,
                                                                 const std::string&);
  BufferSyncImplementation(ir::PipeType pipe, std::string macro, ir::FunctionType core, std::string target)
      : pipe_(pipe), macro_(std::move(macro)), core_(core), target_(std::move(target)) {}
  ir::PipeType pipe_;
  std::string macro_;
  ir::FunctionType core_;
  std::string target_;
  bool accumulator_size_eligible_ = false;
  bool produces_ready_scalar_ = false;
  bool mat_to_operand_extract_ = false;
  bool acc_to_mat_insert_ = false;
  bool audited_signal_free_variant_ = false;
  std::vector<BufferSyncImplicitEffect> implicit_effects_;
};
// Input is a registry-validated Buffer call. Select only existing single-pipe
// variants; geometry/readiness/modes remain the importer's obligation. Unknown
// targets retain conservative mathematics without accumulator protection only
// when their effects are target-independent. Target-specific implicit effects
// require a selected contract; Claims() always refuses unknown physical targets.
BufferSyncImplementation SelectBufferSyncImplementation(const ir::CallPtr& call, ir::FunctionType core,
                                                        const std::string& target);

// Static payload qualification shared by exact and conservative consumers.
// The original immutable call owns every operand, including Tensor GM operands,
// descriptor metadata and scalar coordinate expressions. Effects come from the
// registry after call validation; none are inferred from an operand's type or
// from a missing effect declaration. Native implicit effects are separately
// retained by the selected implementation. This establishes neither operand
// footprint coverage nor readiness, alias separation, loop membership or reuse.
// Those capabilities remain separate; in particular there is no trip count or
// normalized occurrence frame in this contract.
enum class BufferSyncOperationKind { Payload, Descriptor, ReadyScalar };

class BufferSyncOperation {
 public:
  const ir::CallPtr& Operation() const { return operation_; }
  BufferSyncOperationKind Kind() const { return kind_; }
  // Qualified metadata/SPMD recipes issue no asynchronous payload or flags.
  bool SignalFree() const { return kind_ != BufferSyncOperationKind::Payload; }
  const BufferSyncImplementation& Implementation() const {
    CHECK(implementation_) << "Descriptor/scalar contracts have no payload implementation";
    return *implementation_;
  }
  const std::vector<ir::BufferArgEffect>& ArgumentEffects() const { return argument_effects_; }
  const ir::BufferResultSpec& Result() const { return result_; }

 private:
  friend BufferSyncOperation QualifyBufferSyncOperation(const ir::CallPtr&, ir::FunctionType,
                                                        const std::string&);
  friend std::optional<BufferSyncOperation> QualifyBufferSyncNonPayload(const ir::CallPtr&,
                                                                        const std::string&);
  BufferSyncOperation(ir::CallPtr operation, std::optional<BufferSyncImplementation> implementation,
                      std::vector<ir::BufferArgEffect> effects, ir::BufferResultSpec result,
                      BufferSyncOperationKind kind)
      : operation_(std::move(operation)),
        implementation_(std::move(implementation)),
        argument_effects_(std::move(effects)),
        result_(result),
        kind_(kind) {}
  ir::CallPtr operation_;
  std::optional<BufferSyncImplementation> implementation_;
  std::vector<ir::BufferArgEffect> argument_effects_;
  ir::BufferResultSpec result_;
  BufferSyncOperationKind kind_;
};
// Qualifies the existing single-pipe payload subset. Unsupported implementations
// refuse; malformed IR/registration errors propagate. Allocations and views are
// not payloads and retain their existing result/alias/metadata interpretation.
BufferSyncOperation QualifyBufferSyncOperation(const ir::CallPtr& call, ir::FunctionType core,
                                               const std::string& target);

// Placement evidence is separate from selected instruction semantics. A supplied
// half-open hull must enclose every admitted address of this operand; absence
// means unavailable, never empty. Only matching local spaces require a hull.
struct BufferSyncByteWindow {
  int64_t begin, end;
};
void RequireBufferSyncWorkspaceSeparation(
    const BufferSyncOperation& operation,
    const std::function<std::optional<BufferSyncByteWindow>(const ir::ExprPtr&)>& window);

// Recognize the existing emitted alloc/view/valid-metadata/SPMD recipes. Registry
// effects/result behavior remain authoritative. The selected native recipe
// additionally establishes synchronous descriptor evaluation or an ABI-ready
// scalar result, and no signal ownership. This never infers safety from NoAccess
// alone. Runtime descriptor bounds remain execution preconditions. Unknown forms
// return no contract; malformed known forms propagate validation errors.
std::optional<BufferSyncOperation> QualifyBufferSyncNonPayload(const ir::CallPtr& call,
                                                               const std::string& target);

struct LocalSyncExecutionContract {
  bool scalar_prefix_serialized;
  const char* terminal_completion_op;
  bool set_completes_source_prefix;
  bool wait_consumes_before_advance;
};
// A missing target contract never certifies completion or physical ownership.
const LocalSyncExecutionContract& RequireLocalSyncExecutionContract(const std::string& target);
}  // namespace pypto::backend
#endif  // PYPTO_BACKEND_COMMON_BUFFER_SYNC_CONTRACT_H_
