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

#ifndef SRC_IR_TRANSFORMS_UTILS_SCOPED_TOKEN_ORDER_H_
#define SRC_IR_TRANSFORMS_UTILS_SCOPED_TOKEN_ORDER_H_

#include <memory>
#include <vector>

#include "src/ir/transforms/utils/physical_sync_allocation.h"

namespace pypto::ir::sync {
// A sufficient event-order subgraph of actual scoped token uses. No access
// bridges or demand reduction are performed. Clipped/nonuniform edges remain
// handoff obligations. They can prove individual queries, or uniform reuse when
// an affine witness covers an entire carry class; one sampled query is not a
// uniform certificate.
struct ScopedTokenWork;
// A universally quantified resource question, NOT an edge in the token graph.
// Axes advance the two original ordinals over the same named common frames;
// fixed inner suffixes remain in the bases. Every tuple is queried.
// Proves C(consumer payload) -> I(producer payload), a sufficient WAIT -> SET
// bridge at the established direct cuts, not generic command-cut reachability.
// False means unproved, including unsupported same-payload/intra-cut relations.
struct ScopedEventTransition {
  Occurrence consumer, producer;
  std::vector<ScopedDemandAxis> axes;
};
class ScopedTokenOrder {
 public:
  explicit ScopedTokenOrder(std::shared_ptr<const ScopedSyncProtocol> protocol);
  ~ScopedTokenOrder();
  const CompactSyncInput& Input() const;
  const std::shared_ptr<const ScopedSyncProtocol>& Protocol() const;
  const std::vector<PhysicalSyncFamily>& Families() const;
  // Callbacks borrow this immutable owner and are used synchronously.
  DirectSyncOrder View() const;
  // False means unproved in the selected subgraph, not absent in actual IR.
  bool Precedes(const Occurrence& source, bool completion_source, const Occurrence& target,
                bool completion_target) const;
  bool Uniform(size_t family) const;
  // Prove reuse for a proposed physical family against retained actual tokens.
  // The caller separately proves that the candidate preserves token uses.
  bool CanReuseFamily(const PhysicalSyncFamily& family) const;
  bool ProveTransition(const ScopedEventTransition& transition) const;
  size_t SummaryNodes() const;
  size_t InterfaceEntries() const;
  size_t WitnessAttempts() const;
  size_t CandidateVisits() const;
  size_t BinaryProbes() const;

 private:
  ScopedTokenOrder(std::shared_ptr<const ScopedSyncProtocol> protocol,
                   std::optional<std::vector<PhysicalSyncFamily>> prepared);
  friend std::unique_ptr<ScopedTokenOrder> PrepareScopedTokenOrder(std::shared_ptr<const ScopedSyncProtocol>,
                                                                   ScopedTokenWork*);
  class State;
  std::unique_ptr<State> state_;
};
// Optional exact family normalization for physical preparation. Parsing remains
// observable separately; failed reuse qualification retains the original owner.
std::unique_ptr<ScopedTokenOrder> PrepareScopedTokenOrder(std::shared_ptr<const ScopedSyncProtocol> protocol,
                                                          ScopedTokenWork* work = nullptr);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SCOPED_TOKEN_ORDER_H_
