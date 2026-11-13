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
// handoff obligations but cannot silently become uniform reuse evidence.
class ScopedTokenOrder {
 public:
  ScopedTokenOrder(CompactSyncInput input, std::vector<PhysicalSyncFamily> families);
  ~ScopedTokenOrder();
  const CompactSyncInput& Input() const;
  const std::vector<PhysicalSyncFamily>& Families() const;
  // Callbacks borrow this immutable owner and are used synchronously.
  DirectSyncOrder View() const;
  // False means unproved in the selected subgraph, not absent in actual IR.
  bool Precedes(const Occurrence& source, bool completion_source, const Occurrence& target,
                bool completion_target) const;
  bool Uniform(size_t family) const;
  size_t SummaryNodes() const;
  size_t InterfaceEntries() const;

 private:
  class State;
  std::unique_ptr<State> state_;
};
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SCOPED_TOKEN_ORDER_H_
