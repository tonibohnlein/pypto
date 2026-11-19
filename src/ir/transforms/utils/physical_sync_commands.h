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

#ifndef SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_COMMANDS_H_
#define SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_COMMANDS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
class SyncTokenNames;
}  // namespace pypto::ir::sync

namespace pypto::ir::sync::physical {
// Internal command descriptors retain original coordinates and assigned IDs.
// All implementation visitors and sharing tables remain in the command translation unit.
using Intervals = std::vector<std::pair<int64_t, int64_t>>;

using AxisId = uint32_t;

enum class AxisKind : uint8_t {
  LaunchBlock,  // Dominating block coordinate over [blocks.first, blocks.second]
  LoopFrame,    // Normalized loop digit d in [0, count) for iv = start_expr_or_lit + d * step
  LoopFirst,    // Digit 1 on the first visit, 0 on all later visits
  LoopLast,     // Digit 1 on the last visit, 0 on all earlier visits
};

struct AxisInfo {
  AxisId id = 0;
  AxisKind kind = AxisKind::LoopFrame;
  VarPtr var;                // loop->loop_var_ or dominating block Var
  ForStmtPtr loop;           // Original loop, if this is a loop axis
  int64_t count = 1;         // >= 1; normalized digit domain is [0, count)
  int64_t start = 0;         // Literal frame/block start, or iv threshold for LoopLast
  int64_t step = 1;          // >= 1
  ExprPtr start_expression;  // Original symbolic start for LoopFrame or LoopFirst
};

// Transaction-owned deterministic registry of finite coordinate axes.
class GuardDomainArena {
 public:
  AxisId RegisterLaunchBlock(const VarPtr& block_var, std::pair<int64_t, int64_t> bounds);
  AxisId RegisterLoopFrame(const SyncLoopFrame& frame);
  AxisId RegisterLoopBoundary(const ForStmtPtr& loop, AxisKind kind, int64_t last_threshold);
  const AxisInfo& Axis(AxisId id) const { return axes_.at(id); }
  size_t NumAxes() const { return axes_.size(); }

 private:
  std::vector<AxisInfo> axes_;
  std::map<std::tuple<const Var*, uint8_t, int64_t, int64_t, int64_t, const Expr*>, AxisId> index_;
};

// Canonical non-empty arithmetic progression {first + k * stride < end} subseteq [0, axis_count).
// Canonical Invariants:
//   (C1) 0 <= first < end <= axis_count, stride >= 1, (end - 1 - first) % stride == 0.
//   (C2) Tight upper bound: end == Last() + 1 == first + (Count() - 1) * stride + 1.
//   (C3) Singleton normalization: if Count() == 1 (i.e. end - first == 1), then stride == 1.
struct AxisAP {
  int64_t first = 0;
  int64_t end = 1;
  int64_t stride = 1;

  int64_t Count() const { return (end - 1 - first) / stride + 1; }
  int64_t Last() const { return end - 1; }
  bool IsFull(int64_t axis_count) const { return first == 0 && end == axis_count && stride == 1; }
  bool IsSingleton() const { return end - first == 1; }
  bool operator==(const AxisAP& o) const { return first == o.first && end == o.end && stride == o.stride; }
  bool operator!=(const AxisAP& o) const { return !(*this == o); }
  bool operator<(const AxisAP& o) const {
    return std::tie(first, end, stride) < std::tie(o.first, o.end, o.stride);
  }
  static std::optional<AxisAP> MakeCanonical(int64_t raw_first, int64_t raw_end, int64_t raw_stride,
                                             int64_t axis_count);
};

struct AxisConstraint {
  AxisId axis = 0;
  AxisAP ap;  // Invariant (I1): strictly proper (!ap.IsFull(arena.Axis(axis).count))
  bool operator==(const AxisConstraint& o) const { return axis == o.axis && ap == o.ap; }
  bool operator!=(const AxisConstraint& o) const { return !(*this == o); }
  bool operator<(const AxisConstraint& o) const { return std::tie(axis, ap) < std::tie(o.axis, o.ap); }
};

// Compact bitmask over {0 .. modulus-1}. Inline for modulus <= 64.
class ResidueMask {
 public:
  explicit ResidueMask(int64_t modulus = 1, uint64_t bits = 0);
  bool Empty() const;
  bool Full() const;
  size_t PopCount() const;
  bool Test(int64_t r) const;
  void Set(int64_t r);
  ResidueMask operator&(const ResidueMask& o) const;
  ResidueMask operator|(const ResidueMask& o) const;
  ResidueMask Complement() const;
  ResidueMask Shift(int64_t delta) const;  // (r + delta) mod M
  std::optional<int64_t> SmallestPeriod() const;
  bool operator==(const ResidueMask& o) const;
  bool operator!=(const ResidueMask& o) const { return !(*this == o); }
  bool operator<(const ResidueMask& o) const;
  size_t Hash() const;

 private:
  int64_t modulus_ = 1;
  uint64_t inline_bits_ = 0;
  std::vector<uint64_t> ext_bits_;
  void MaskUnusedBits();
};

struct RankTerm {
  AxisId axis = 0;
  int64_t first = 0;
  int64_t step = 1;
  int64_t coeff = 1;
  bool operator==(const RankTerm& o) const {
    return std::tie(axis, first, step, coeff) == std::tie(o.axis, o.first, o.step, o.coeff);
  }
  bool operator<(const RankTerm& o) const {
    return std::tie(axis, first, step, coeff) < std::tie(o.axis, o.first, o.step, o.coeff);
  }
};

// Coupled multi-axis modular rank constraint:
//   (base + sum_t coeff_t * ((d_{axis_t} - first_t) / step_t)) mod modulus in residues
struct RankCongruence {
  int64_t modulus = 2;
  int64_t base = 0;
  std::vector<RankTerm> terms;
  ResidueMask residues;
  bool operator==(const RankCongruence& o) const {
    return modulus == o.modulus && base == o.base && terms == o.terms && residues == o.residues;
  }
  bool operator!=(const RankCongruence& o) const { return !(*this == o); }
  bool operator<(const RankCongruence& o) const {
    return std::tie(modulus, base, terms, residues) < std::tie(o.modulus, o.base, o.terms, o.residues);
  }
};

// Sparse canonical Cartesian progression box (+ coupled multi-axis rank congruences).
struct RadixBox {
  std::vector<AxisConstraint> constraints;  // Sorted by AxisId, all proper subsets
  std::vector<RankCongruence> ranks;        // Sorted; supports intersection closure
  bool IsUnconstrained() const { return constraints.empty() && ranks.empty(); }
  AxisAP AxisRange(AxisId axis, const GuardDomainArena& arena) const;
  bool operator==(const RadixBox& o) const { return constraints == o.constraints && ranks == o.ranks; }
  bool operator!=(const RadixBox& o) const { return !(*this == o); }
  bool operator<(const RadixBox& o) const {
    return std::tie(constraints, ranks) < std::tie(o.constraints, o.ranks);
  }
};

// Canonical domain-aware guard carried on immutable CommandControl.
class RadixBoxGuard {
 public:
  static constexpr size_t kMaxGuardBoxes = 16;

  RadixBoxGuard() = default;
  static RadixBoxGuard False() { return {}; }
  static RadixBoxGuard True();
  static RadixBoxGuard FromBox(std::shared_ptr<const GuardDomainArena> arena, std::optional<RadixBox> box);
  static RadixBoxGuard FromBoxes(std::shared_ptr<const GuardDomainArena> arena, std::vector<RadixBox> boxes);
  static RadixBoxGuard OpaqueDiagnostic(ExprPtr expr, std::vector<const Var*> inputs, bool qualified);

  bool IsEmpty() const { return kind_ == Kind::kFalse; }
  bool IsTautology() const {
    return kind_ == Kind::kBoxes && boxes_.size() == 1 && boxes_[0].IsUnconstrained();
  }
  bool IsOpaque() const { return kind_ == Kind::kOpaqueExpr; }
  bool IsDisjoint() const { return disjoint_; }
  const std::shared_ptr<const GuardDomainArena>& Arena() const { return arena_; }
  const std::vector<RadixBox>& Boxes() const { return boxes_; }
  const ExprPtr& OpaqueExpr() const { return opaque_; }
  const std::vector<const Var*>& OpaqueInputs() const { return opaque_inputs_; }
  bool PredicateQualified() const { return kind_ != Kind::kOpaqueExpr || opaque_qualified_; }
  bool IsFirstIterationOf(const VarPtr& loop_var, int64_t lr_start, int64_t lr_step) const;

 private:
  enum class Kind : uint8_t { kFalse, kBoxes, kOpaqueExpr };
  Kind kind_ = Kind::kFalse;
  std::shared_ptr<const GuardDomainArena> arena_;
  std::vector<RadixBox> boxes_;
  ExprPtr opaque_;
  std::vector<const Var*> opaque_inputs_;
  bool opaque_qualified_ = true;
  bool disjoint_ = true;
};

// Core RadixBoxGuard algebra & lowering primitives (exposed for scheduling & unit testing).
std::optional<AxisAP> CRTIntersect(const AxisAP& a, const AxisAP& b, int64_t axis_count);
std::optional<RadixBox> IntersectBox(const RadixBox& a, const RadixBox& b, const GuardDomainArena& arena);
std::optional<unsigned __int128> BoxCardinality(const RadixBox& box, const std::vector<AxisId>& universe,
                                                const GuardDomainArena& arena);
bool CoalesceBoxes(std::vector<RadixBox>& boxes, const GuardDomainArena& arena);
std::optional<std::vector<RadixBox>> SubtractBox(const RadixBox& s, const RadixBox& g,
                                                 const GuardDomainArena& arena);

RadixBoxGuard IntersectGuards(const RadixBoxGuard& a, const RadixBoxGuard& b);
std::optional<RadixBoxGuard> UnionGuards(const RadixBoxGuard& a, const RadixBoxGuard& b);
std::optional<RadixBoxGuard> SubtractGuards(const RadixBoxGuard& s, const RadixBoxGuard& g);
std::optional<unsigned __int128> GuardCardinality(const RadixBoxGuard& g,
                                                  const std::vector<AxisId>& universe);
std::vector<AxisId> ReferencedAxes(const RadixBoxGuard& a, const RadixBoxGuard& b = {},
                                   const RadixBoxGuard& c = {});
bool AreDisjointGuardsInScope(const RadixBoxGuard& a, const RadixBoxGuard& b, const RadixBoxGuard& scope);
bool AreComplementaryGuardsInScope(const RadixBoxGuard& a, const RadixBoxGuard& b,
                                   const RadixBoxGuard& scope);
bool EquivalentGuardsInScope(const RadixBoxGuard& a, const RadixBoxGuard& b, const RadixBoxGuard& scope);
bool SameGuard(const RadixBoxGuard& a, const RadixBoxGuard& b);
RadixBoxGuard SimplifyGuardGiven(const RadixBoxGuard& g, const RadixBoxGuard& scope);
RadixBoxGuard RestrictGuardAxis(const RadixBoxGuard& g, AxisId axis, const AxisAP& range,
                                const std::shared_ptr<const GuardDomainArena>& arena);
ExprPtr LowerRadixGuardToExpr(const RadixBoxGuard& g, const RadixBoxGuard& scope, const Span& span);
std::vector<ExprPtr> LowerBoxToSplitPredicates(const RadixBox& box, const RadixBoxGuard& scope,
                                               const GuardDomainArena& arena, const Span& span);

struct CommandControl;
using ControlPtr = std::shared_ptr<const CommandControl>;
struct FrozenCommandForest;
struct ControlForest : std::vector<ControlPtr> {
  using std::vector<ControlPtr>::vector;
  mutable std::shared_ptr<const FrozenCommandForest> frozen;
};
struct CommandControl {
  RadixBoxGuard guard = RadixBoxGuard::True();
  ExprPtr command;
  ControlForest body, alternative;
  Span span;
  // Cached requirements of this node's validated predicate, not readiness
  // evidence. Production must bind them to an original-cut scope below.
  std::vector<const Var*> predicate_inputs;
  // False only for diagnostic pure syntax whose arithmetic premise is absent.
  // Such a predicate stays opaque; production scope qualification rejects it.
  bool predicate_qualified = true;
};
// No generated SSA values are allocated while planning. A slot belongs only to
// one effective sequence scope; definitions appear at their first evaluation.
struct ScalarRecipe {
  ExprPtr original;
  std::shared_ptr<const ScalarRecipe> left, right;
  std::optional<size_t> slot;
};
using ScalarRecipePtr = std::shared_ptr<const ScalarRecipe>;
struct ScalarBinding {
  size_t slot;
  ScalarRecipePtr value;
};
struct FrozenCommand {
  ControlPtr original;
  ExprPtr condition_expr;
  ScalarRecipePtr condition;
  std::vector<ScalarBinding> definitions;
  std::array<size_t, 6> scalar_cost{};
  std::shared_ptr<const FrozenCommandForest> body, alternative;
};
struct FrozenCommandForest {
  std::vector<ControlPtr> roots;
  RadixBoxGuard scope = RadixBoxGuard::True();
  std::vector<FrozenCommand> commands;
};
std::shared_ptr<const FrozenCommandForest> FreezeCommandForest(
    const ControlForest& forest, PhysicalSyncRealization& work,
    const RadixBoxGuard& scope = RadixBoxGuard::True());
std::vector<StmtPtr> MaterializeFrozenCommands(const ControlForest& forest, SyncTokenNames& names,
                                               PhysicalSyncRealization& work,
                                               const RadixBoxGuard& scope = RadixBoxGuard::True());
struct CommandPredicateScope {
  FunctionPtr owner;
  // Share entry facts once per owner, rather than copying every formal at
  // every cut. Local facts contain only active original loop IVs.
  std::shared_ptr<const std::set<const Var*>> entry_ready;
  std::set<const Var*> ready;
  // Arbitrary SSA definitions and branch-private values are not admitted.
};
void RequireCommandScope(const ControlForest& forest, const CommandPredicateScope& scope,
                         PhysicalSyncRealization& work);
struct CaseCommands {
  std::pair<int64_t, int64_t> blocks;
  ControlPtr body;
};

struct LoopBoundaryDomain {
  struct Case {
    std::pair<int64_t, int64_t> blocks;
    int64_t count;
  };
  ForStmtPtr loop;
  // Positive trip counts proved for the original invocation cases. The cut
  // execution domain already excludes any absent loop.
  std::vector<Case> cases;
};

struct PhysicalCuts {
  ControlForest before, after;
  std::vector<CaseCommands> before_cases, after_cases;
  // Upper bound on the invocation coordinates reaching this ORIGINAL cut.
  // Only a retained zero-visit proof removes coordinates. Missing evidence
  // leaves the full domain; a present cut with no commands is never absence.
  std::vector<std::pair<int64_t, int64_t>> domain;
  std::set<const Var*> uniform_variables;
  bool uniform_checked = false;
  const ForStmt* enclosing_loop = nullptr;
  // Present for production inventories. Diagnostic-only imports explicitly
  // supply readiness externally and do not manufacture production evidence.
  std::optional<CommandPredicateScope> predicate_scope;
};
using CutInventory = std::map<const Call*, std::shared_ptr<PhysicalCuts>>;

bool SameControl(const IRNodePtr& left, const IRNodePtr& right);
bool SameControl(const ControlPtr& left, const ControlPtr& right);
std::optional<int64_t> OriginalLoopBoundary(const ForStmtPtr& loop, int64_t start, int64_t step,
                                            int64_t count, const ExprPtr& start_expr, int64_t digit);
ControlPtr ControlSequence(ControlForest body, const Span& span, PhysicalSyncRealization& work);
ControlPtr ControlBranch(RadixBoxGuard guard, ControlPtr body, std::optional<ControlPtr> alternative,
                         const Span& span, PhysicalSyncRealization& work);
ControlPtr ControlBranch(ExprPtr condition, ControlPtr body, std::optional<ControlPtr> alternative,
                         const Span& span, PhysicalSyncRealization& work);
ControlPtr ControlCommand(ExprPtr command, const Span& span, PhysicalSyncRealization& work);
ControlPtr RebuildControl(const ControlPtr& original, RadixBoxGuard guard, ExprPtr command,
                          ControlForest body, ControlForest alternative, PhysicalSyncRealization& work);
void FlattenControlSequence(const ControlPtr& node, ControlForest& commands, PhysicalSyncRealization& work);
// Endpoint accounting applies only before invocation-case composition.
ControlForest ShareCommandGuards(const ControlForest& commands, PhysicalSyncRealization& work,
                                 bool endpoint = true, bool pair_else_branches = true,
                                 const RadixBoxGuard& scope = RadixBoxGuard::True());
void AppendCase(std::vector<CaseCommands>& cases, std::pair<int64_t, int64_t> blocks,
                const ControlForest& commands, const Span& span, PhysicalSyncRealization& work);
Intervals SubtractIntervals(const Intervals& domain, const Intervals& absent);
ControlForest ScheduleCommandCases(const ExprPtr& block, const Intervals& possible,
                                   const std::vector<CaseCommands>& source, PhysicalSyncRealization& work,
                                   const Span& span, const std::set<const Var*>& uniform_variables,
                                   const std::shared_ptr<GuardDomainArena>& arena = nullptr,
                                   const LoopBoundaryDomain* boundary_domain = nullptr);
struct CaseCodeSize {
  size_t commands = 0, guards = 0, expressions = 0, statements = 0;
  explicit CaseCodeSize(const ControlForest& body, const RadixBoxGuard& scope = RadixBoxGuard::True());
  bool Improves(const CaseCodeSize& old) const;

 private:
  void Visit(const ControlPtr& node, const RadixBoxGuard& scope);
};

// Shared guard hashing helper for scheduling.
size_t HashRadixBoxGuard(const RadixBoxGuard& guard);

}  // namespace pypto::ir::sync::physical

#endif  // SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_COMMANDS_H_
