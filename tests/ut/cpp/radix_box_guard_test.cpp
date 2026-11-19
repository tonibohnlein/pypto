/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "src/ir/transforms/utils/physical_sync_commands.h"

namespace {
using pypto::DataType;
using pypto::ir::Add;
using pypto::ir::And;
using pypto::ir::As;
using pypto::ir::CallPtr;
using pypto::ir::ConstBool;
using pypto::ir::ConstInt;
using pypto::ir::Eq;
using pypto::ir::EvalStmt;
using pypto::ir::ExprPtr;
using pypto::ir::FloorDiv;
using pypto::ir::FloorMod;
using pypto::ir::ForKind;
using pypto::ir::ForStmt;
using pypto::ir::ForStmtPtr;
using pypto::ir::Ge;
using pypto::ir::Gt;
using pypto::ir::Le;
using pypto::ir::Lt;
using pypto::ir::Mul;
using pypto::ir::Ne;
using pypto::ir::OpRegistry;
using pypto::ir::Or;
using pypto::ir::PipeType;
using pypto::ir::ScalarType;
using pypto::ir::SeqStmts;
using pypto::ir::Span;
using pypto::ir::Sub;
using pypto::ir::Var;
using pypto::ir::VarPtr;
using pypto::ir::sync::PhysicalSyncRealization;
using pypto::ir::sync::SyncLoopFrame;
using pypto::ir::sync::physical::AreComplementaryGuardsInScope;
using pypto::ir::sync::physical::AreDisjointGuardsInScope;
using pypto::ir::sync::physical::AxisAP;
using pypto::ir::sync::physical::AxisConstraint;
using pypto::ir::sync::physical::AxisId;
using pypto::ir::sync::physical::BoxCardinality;
using pypto::ir::sync::physical::CaseCodeSize;
using pypto::ir::sync::physical::CoalesceBoxes;
using pypto::ir::sync::physical::ControlBranch;
using pypto::ir::sync::physical::ControlCommand;
using pypto::ir::sync::physical::ControlForest;
using pypto::ir::sync::physical::ControlPtr;
using pypto::ir::sync::physical::CRTIntersect;
using pypto::ir::sync::physical::EquivalentGuardsInScope;
using pypto::ir::sync::physical::FreezeCommandForest;
using pypto::ir::sync::physical::GuardCardinality;
using pypto::ir::sync::physical::GuardDomainArena;
using pypto::ir::sync::physical::IntersectBox;
using pypto::ir::sync::physical::IntersectGuards;
using pypto::ir::sync::physical::LowerBoxToSplitPredicates;
using pypto::ir::sync::physical::LowerRadixGuardToExpr;
using pypto::ir::sync::physical::RadixBox;
using pypto::ir::sync::physical::RadixBoxGuard;
using pypto::ir::sync::physical::RankCongruence;
using pypto::ir::sync::physical::RankTerm;
using pypto::ir::sync::physical::ResidueMask;
using pypto::ir::sync::physical::SameGuard;
using pypto::ir::sync::physical::ShareCommandGuards;
using pypto::ir::sync::physical::SimplifyGuardGiven;
using pypto::ir::sync::physical::SubtractBox;
using pypto::ir::sync::physical::SubtractGuards;
using pypto::ir::sync::physical::UnionGuards;

int64_t FloorDivI(int64_t a, int64_t b) {
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) --q;
  return q;
}

int64_t FloorModI(int64_t a, int64_t b) {
  int64_t r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) r += b;
  return r;
}

int64_t EvalIntExpr(const ExprPtr& expr, const std::map<const Var*, int64_t>& env) {
  if (auto ci = As<ConstInt>(expr)) return ci->value_;
  if (auto v = As<Var>(expr)) return env.at(v.get());
  if (auto a = As<Add>(expr)) return EvalIntExpr(a->left_, env) + EvalIntExpr(a->right_, env);
  if (auto s = As<Sub>(expr)) return EvalIntExpr(s->left_, env) - EvalIntExpr(s->right_, env);
  if (auto m = As<Mul>(expr)) return EvalIntExpr(m->left_, env) * EvalIntExpr(m->right_, env);
  if (auto fd = As<FloorDiv>(expr)) {
    return FloorDivI(EvalIntExpr(fd->left_, env), EvalIntExpr(fd->right_, env));
  }
  if (auto fm = As<FloorMod>(expr)) {
    return FloorModI(EvalIntExpr(fm->left_, env), EvalIntExpr(fm->right_, env));
  }
  assert(false && "Unsupported integer expr in test evaluator");
  return 0;
}

bool EvalBoolExpr(const ExprPtr& expr, const std::map<const Var*, int64_t>& env) {
  if (auto cb = As<ConstBool>(expr)) return cb->value_;
  if (auto eq = As<Eq>(expr)) return EvalIntExpr(eq->left_, env) == EvalIntExpr(eq->right_, env);
  if (auto ne = As<Ne>(expr)) return EvalIntExpr(ne->left_, env) != EvalIntExpr(ne->right_, env);
  if (auto lt = As<Lt>(expr)) return EvalIntExpr(lt->left_, env) < EvalIntExpr(lt->right_, env);
  if (auto le = As<Le>(expr)) return EvalIntExpr(le->left_, env) <= EvalIntExpr(le->right_, env);
  if (auto gt = As<Gt>(expr)) return EvalIntExpr(gt->left_, env) > EvalIntExpr(gt->right_, env);
  if (auto ge = As<Ge>(expr)) return EvalIntExpr(ge->left_, env) >= EvalIntExpr(ge->right_, env);
  if (auto a = As<And>(expr)) return EvalBoolExpr(a->left_, env) && EvalBoolExpr(a->right_, env);
  if (auto o = As<Or>(expr)) return EvalBoolExpr(o->left_, env) || EvalBoolExpr(o->right_, env);
  assert(false && "Unsupported bool expr in test evaluator");
  return false;
}

ForStmtPtr MakeTestLoop(const std::string& name, int64_t start, int64_t count, int64_t step) {
  auto idx_type = std::make_shared<ScalarType>(DataType::INDEX);
  auto var = std::make_shared<Var>(name, idx_type, Span::unknown());
  auto s_expr = std::make_shared<ConstInt>(start, DataType::INDEX, Span::unknown());
  auto stop_expr = std::make_shared<ConstInt>(start + count * step, DataType::INDEX, Span::unknown());
  auto step_expr = std::make_shared<ConstInt>(step, DataType::INDEX, Span::unknown());
  auto body = std::make_shared<SeqStmts>(std::vector<pypto::ir::StmtPtr>{}, Span::unknown());
  return std::make_shared<ForStmt>(var, s_expr, stop_expr, step_expr, std::vector<pypto::ir::IterArgPtr>{},
                                   body, std::vector<VarPtr>{}, Span::unknown(), ForKind::Sequential);
}

bool BoxContains(const RadixBox& box, AxisId a0, int64_t d0, AxisId a1, int64_t d1,
                 const GuardDomainArena& arena) {
  auto ap0 = box.AxisRange(a0, arena);
  if (d0 < ap0.first || d0 >= ap0.end || (d0 - ap0.first) % ap0.stride != 0) return false;
  auto ap1 = box.AxisRange(a1, arena);
  if (d1 < ap1.first || d1 >= ap1.end || (d1 - ap1.first) % ap1.stride != 0) return false;
  for (const auto& rc : box.ranks) {
    int64_t val = rc.base;
    for (const auto& t : rc.terms) {
      int64_t d = (t.axis == a0) ? d0 : d1;
      int64_t p = (d - t.first) / t.step;
      val += t.coeff * p;
    }
    int64_t r = ((val % rc.modulus) + rc.modulus) % rc.modulus;
    if (!rc.residues.Test(r)) return false;
  }
  return true;
}

bool GuardContains(const RadixBoxGuard& g, AxisId a0, int64_t d0, AxisId a1, int64_t d1,
                   const GuardDomainArena& arena) {
  if (g.IsEmpty()) return false;
  if (g.IsTautology()) return true;
  for (const auto& box : g.Boxes()) {
    if (BoxContains(box, a0, d0, a1, d1, arena)) return true;
  }
  return false;
}

ExprPtr MakeFlag(const char* op_name, PipeType src, PipeType dst, int id) {
  return OpRegistry::GetInstance().Create(op_name, {},
                                          {{"src_pipe", static_cast<int>(src)},
                                           {"dst_pipe", static_cast<int>(dst)},
                                           {"event_id", id},
                                           {"core_type", std::string("aic")}},
                                          Span::unknown());
}

void TestCRTIntersectExhaustive() {
  for (int64_t N : {1, 2, 3, 5, 8, 12}) {
    for (int64_t f1 = 0; f1 < N; ++f1) {
      for (int64_t s1 = 1; s1 <= 6; ++s1) {
        for (int64_t e1 = f1 + 1; e1 <= N; ++e1) {
          auto ap1 = AxisAP::MakeCanonical(f1, e1, s1, N);
          if (!ap1) continue;
          for (int64_t f2 = 0; f2 < N; ++f2) {
            for (int64_t s2 = 1; s2 <= 6; ++s2) {
              for (int64_t e2 = f2 + 1; e2 <= N; ++e2) {
                auto ap2 = AxisAP::MakeCanonical(f2, e2, s2, N);
                if (!ap2) continue;
                std::vector<int64_t> expected;
                for (int64_t d = 0; d < N; ++d) {
                  bool in1 = d >= ap1->first && d < ap1->end && (d - ap1->first) % ap1->stride == 0;
                  bool in2 = d >= ap2->first && d < ap2->end && (d - ap2->first) % ap2->stride == 0;
                  if (in1 && in2) expected.push_back(d);
                }
                auto got = CRTIntersect(*ap1, *ap2, N);
                if (expected.empty()) {
                  assert(!got.has_value());
                } else {
                  assert(got.has_value());
                  assert(static_cast<size_t>(got->Count()) == expected.size());
                  std::vector<int64_t> actual;
                  for (int64_t d = got->first; d < got->end; d += got->stride) actual.push_back(d);
                  assert(actual == expected);
                  if (got->IsSingleton()) assert(got->stride == 1);
                }
              }
            }
          }
        }
      }
    }
  }
}

void TestBoxAlgebraAndTruthTables() {
  for (int64_t N0 : {3, 6, 10}) {
    for (int64_t N1 : {3, 7, 11}) {
      auto arena = std::make_shared<GuardDomainArena>();
      auto l0 = MakeTestLoop("i", 5, N0, 2);
      auto l1 = MakeTestLoop("j", 1, N1, 3);
      AxisId a0 = arena->RegisterLoopFrame(SyncLoopFrame{l0, N0, 5, 2, nullptr});
      AxisId a1 = arena->RegisterLoopFrame(SyncLoopFrame{l1, N1, 1, 3, nullptr});
      std::vector<AxisId> universe{a0, a1};

      // RankCongruence 1-rank & 2-rank DP vs brute force over moduli 2..6.
      for (int64_t M : {2, 3, 4, 5, 6}) {
        ResidueMask m0(M);
        m0.Set(0);
        if (M >= 3) m0.Set(M - 1);
        RankCongruence rc0{M, 1, {RankTerm{a0, 0, 1, 1}, RankTerm{a1, 0, 1, 2 % M}}, m0};

        int64_t M2 = (M <= 4) ? 2 : 3;
        ResidueMask m1(M2);
        m1.Set(0);
        RankCongruence rc1{M2, 0, {RankTerm{a0, 0, 1, 1}, RankTerm{a1, 0, 1, 1}}, m1};

        RadixBox b1;
        b1.constraints.push_back(AxisConstraint{a0, *AxisAP::MakeCanonical(1, N0, 2, N0)});
        b1.ranks.push_back(rc0);

        RadixBox b2;
        b2.constraints.push_back(AxisConstraint{a1, *AxisAP::MakeCanonical(0, N1 - 1, 1, N1)});
        b2.ranks.push_back(rc1);

        // BoxCardinality 1-rank
        unsigned __int128 bf1 = 0;
        for (int64_t d0 = 0; d0 < N0; ++d0)
          for (int64_t d1 = 0; d1 < N1; ++d1)
            if (BoxContains(b1, a0, d0, a1, d1, *arena)) ++bf1;
        auto card1 = BoxCardinality(b1, universe, *arena);
        assert(card1.has_value() && *card1 == bf1);

        // IntersectBox producing 2 ranks (when M * M2 <= 64)
        auto inter = IntersectBox(b1, b2, *arena);
        unsigned __int128 bf_inter = 0;
        for (int64_t d0 = 0; d0 < N0; ++d0)
          for (int64_t d1 = 0; d1 < N1; ++d1)
            if (BoxContains(b1, a0, d0, a1, d1, *arena) && BoxContains(b2, a0, d0, a1, d1, *arena))
              ++bf_inter;
        if (bf_inter == 0) {
          assert(!inter.has_value());
        } else {
          assert(inter.has_value());
          if (M * M2 <= 64) {
            auto ci = BoxCardinality(*inter, universe, *arena);
            assert(ci.has_value() && *ci == bf_inter);
          }
          // Lowering & truth-table verification across original loop vars.
          auto g_inter = RadixBoxGuard::FromBox(arena, *inter);
          ExprPtr lowered = LowerRadixGuardToExpr(g_inter, RadixBoxGuard::True(), Span::unknown());
          for (int64_t d0 = 0; d0 < N0; ++d0) {
            for (int64_t d1 = 0; d1 < N1; ++d1) {
              std::map<const Var*, int64_t> env{{l0->loop_var_.get(), 5 + d0 * 2},
                                                {l1->loop_var_.get(), 1 + d1 * 3}};
              bool expected = BoxContains(*inter, a0, d0, a1, d1, *arena);
              assert(EvalBoolExpr(lowered, env) == expected);
            }
          }
        }

        // SubtractBox verification on 1-rank boxes
        auto g1 = RadixBoxGuard::FromBox(arena, b1);
        auto g2 = RadixBoxGuard::FromBox(arena, b2);
        auto diff = SubtractGuards(g1, g2);
        if (diff.has_value()) {
          for (int64_t d0 = 0; d0 < N0; ++d0) {
            for (int64_t d1 = 0; d1 < N1; ++d1) {
              bool expected =
                  BoxContains(b1, a0, d0, a1, d1, *arena) && !BoxContains(b2, a0, d0, a1, d1, *arena);
              assert(GuardContains(*diff, a0, d0, a1, d1, *arena) == expected);
            }
          }
        }
      }
    }
  }
}

void TestSimplifyGivenAndLoweringShortcuts() {
  // #85 §1 multi-box S_else trace: [0, 4) x [0, 4), parent = {i == 0 && j == 0}.
  auto arena = std::make_shared<GuardDomainArena>();
  auto li = MakeTestLoop("i", 0, 4, 1);
  auto lj = MakeTestLoop("j", 0, 4, 1);
  AxisId ai = arena->RegisterLoopFrame(SyncLoopFrame{li, 4, 0, 1, nullptr});
  AxisId aj = arena->RegisterLoopFrame(SyncLoopFrame{lj, 4, 0, 1, nullptr});

  RadixBox parent_box;
  parent_box.constraints.push_back(AxisConstraint{ai, AxisAP{0, 1, 1}});
  parent_box.constraints.push_back(AxisConstraint{aj, AxisAP{0, 1, 1}});
  auto g_parent = RadixBoxGuard::FromBox(arena, parent_box);
  auto s_else = SubtractGuards(RadixBoxGuard::True(), g_parent);
  assert(s_else.has_value() && s_else->Boxes().size() == 2);

  // G_child = {i == 0, j >= 1} -> Phase 1 drops j >= 1 leaving {i == 0}!
  RadixBox child1;
  child1.constraints.push_back(AxisConstraint{ai, AxisAP{0, 1, 1}});
  child1.constraints.push_back(AxisConstraint{aj, AxisAP{1, 4, 1}});
  auto simp1 = SimplifyGuardGiven(RadixBoxGuard::FromBox(arena, child1), *s_else);
  assert(simp1.Boxes().size() == 1 && simp1.Boxes()[0].constraints.size() == 1);
  assert(simp1.Boxes()[0].constraints[0].axis == ai &&
         simp1.Boxes()[0].constraints[0].ap == (AxisAP{0, 1, 1}));

  // G_child = {i == 0, j in [1, 3)} -> Phase 2 omits redundant j >= 1, emits i == 0 && j < 3!
  RadixBox child2;
  child2.constraints.push_back(AxisConstraint{ai, AxisAP{0, 1, 1}});
  child2.constraints.push_back(AxisConstraint{aj, AxisAP{1, 3, 1}});
  auto preds2 = LowerBoxToSplitPredicates(child2, *s_else, *arena, Span::unknown());
  assert(preds2.size() == 2);
  assert(As<Eq>(preds2[0]) != nullptr);
  assert(As<Lt>(preds2[1]) != nullptr);

  // Overlapping prime-implicant relaxation regression (startup-2-2-1):
  // [0, 2) x [0, 2), G = {[i == 1]} U {[i == 0, j == 1]} (size 3 != 4).
  auto arena2 = std::make_shared<GuardDomainArena>();
  auto lk = MakeTestLoop("k_pair", 3, 2, 2);
  auto lk0 = MakeTestLoop("k0", 0, 2, 1);
  AxisId ak = arena2->RegisterLoopFrame(SyncLoopFrame{lk, 2, 3, 2, nullptr});
  AxisId ak0 = arena2->RegisterLoopFrame(SyncLoopFrame{lk0, 2, 0, 1, nullptr});
  RadixBox b_k1;
  b_k1.constraints.push_back(AxisConstraint{ak, AxisAP{1, 2, 1}});
  RadixBox b_k0_1;
  b_k0_1.constraints.push_back(AxisConstraint{ak, AxisAP{0, 1, 1}});
  b_k0_1.constraints.push_back(AxisConstraint{ak0, AxisAP{1, 2, 1}});
  auto g3 = RadixBoxGuard::FromBoxes(arena2, {b_k1, b_k0_1});
  assert(!g3.IsTautology());
  auto card3 = GuardCardinality(g3, {ak, ak0});
  assert(card3.has_value() && *card3 == 3);
  auto g3_simp = SimplifyGuardGiven(g3, RadixBoxGuard::True());
  assert(!g3_simp.IsTautology());
  auto card3s = GuardCardinality(g3_simp, {ak, ak0});
  assert(card3s.has_value() && *card3s == 3);
  // Relaxed ExprPtr drops k_pair == 0 inside the Or: (k_pair == 5 || k0 == 1).
  ExprPtr expr3 = LowerRadixGuardToExpr(g3_simp, RadixBoxGuard::True(), Span::unknown());
  assert(As<Or>(expr3) != nullptr);
  for (int64_t d0 = 0; d0 < 2; ++d0) {
    for (int64_t d1 = 0; d1 < 2; ++d1) {
      std::map<const Var*, int64_t> env{{lk->loop_var_.get(), 3 + d0 * 2}, {lk0->loop_var_.get(), d1}};
      assert(EvalBoolExpr(expr3, env) == !(d0 == 0 && d1 == 0));
    }
  }

  // Ne complement shortcut & duplicate-box deduplication + multi-way periodic collapse.
  auto arena3 = std::make_shared<GuardDomainArena>();
  auto lx = MakeTestLoop("x", 0, 6, 1);
  AxisId ax = arena3->RegisterLoopFrame(SyncLoopFrame{lx, 6, 0, 1, nullptr});
  RadixBox r0, r1, r2;
  r0.constraints.push_back(AxisConstraint{ax, *AxisAP::MakeCanonical(0, 6, 3, 6)});
  r1.constraints.push_back(AxisConstraint{ax, *AxisAP::MakeCanonical(1, 6, 3, 6)});
  r2.constraints.push_back(AxisConstraint{ax, *AxisAP::MakeCanonical(2, 6, 3, 6)});
  auto g01 = UnionGuards(RadixBoxGuard::FromBox(arena3, r0), RadixBoxGuard::FromBox(arena3, r1));
  assert(g01.has_value());
  ExprPtr ne_expr = LowerRadixGuardToExpr(*g01, RadixBoxGuard::True(), Span::unknown());
  assert(As<Ne>(ne_expr) != nullptr);
  for (int64_t d = 0; d < 6; ++d) {
    std::map<const Var*, int64_t> env{{lx->loop_var_.get(), d}};
    assert(EvalBoolExpr(ne_expr, env) == (d % 3 != 2));
  }
  // Multi-way periodic hull collapse of all 3 residue classes -> True!
  auto g_all = RadixBoxGuard::FromBoxes(arena3, {r0, r1, r2, r0});
  assert(g_all.IsTautology());
  // Duplicate boxes alone deduplicate to 1 box.
  auto g_dup = RadixBoxGuard::FromBoxes(arena3, {r0, r0});
  assert(g_dup.Boxes().size() == 1);

  // Redundant Ge/Lt omission on periodic AxisAP (start=5, step=2, count=4 -> d in {1, 3} emits only
  // FloorMod).
  auto arena4 = std::make_shared<GuardDomainArena>();
  auto lp = MakeTestLoop("p", 5, 4, 2);
  AxisId ap_ax = arena4->RegisterLoopFrame(SyncLoopFrame{lp, 4, 5, 2, nullptr});
  RadixBox odd_b;
  odd_b.constraints.push_back(AxisConstraint{ap_ax, *AxisAP::MakeCanonical(1, 4, 2, 4)});
  auto odd_preds = LowerBoxToSplitPredicates(odd_b, RadixBoxGuard::True(), *arena4, Span::unknown());
  assert(odd_preds.size() == 1);
  assert(As<Eq>(odd_preds[0]) != nullptr && As<FloorMod>(As<Eq>(odd_preds[0])->left_) != nullptr);
}

void TestLargeTripScalingAndIsFirstIteration() {
  auto arena = std::make_shared<GuardDomainArena>();
  constexpr int64_t kLarge = 4000000;
  auto blk_var =
      std::make_shared<Var>("block", std::make_shared<ScalarType>(DataType::INDEX), Span::unknown());
  auto l0 = MakeTestLoop("outer", 4, kLarge, 3);
  auto l1 = MakeTestLoop("inner", 0, kLarge, 1);
  AxisId ab = arena->RegisterLaunchBlock(blk_var, {0, kLarge - 1});
  AxisId a0 = arena->RegisterLoopFrame(SyncLoopFrame{l0, kLarge, 4, 3, nullptr});
  AxisId a1 = arena->RegisterLoopFrame(SyncLoopFrame{l1, kLarge, 0, 1, nullptr});

  RadixBox b;
  b.constraints.push_back(AxisConstraint{ab, *AxisAP::MakeCanonical(100, kLarge - 100, 1, kLarge)});
  b.constraints.push_back(AxisConstraint{a0, *AxisAP::MakeCanonical(0, kLarge, 2, kLarge)});
  b.constraints.push_back(AxisConstraint{a1, *AxisAP::MakeCanonical(1, kLarge, 3, kLarge)});
  ResidueMask mask(6);
  mask.Set(0);
  mask.Set(2);
  b.ranks.push_back(RankCongruence{6, 0, {RankTerm{a0, 0, 1, 1}, RankTerm{a1, 0, 1, 1}}, mask});
  auto g = RadixBoxGuard::FromBox(arena, b);
  auto card = GuardCardinality(g, {ab, a0, a1});
  assert(card.has_value() && *card > 0);
  auto simp = SimplifyGuardGiven(g, RadixBoxGuard::True());
  assert(!simp.IsEmpty() && !simp.IsTautology());

  // Phase-relative IsFirstIterationOf (#86 Finding A5).
  RadixBox first_phase;
  first_phase.constraints.push_back(AxisConstraint{a0, AxisAP{2, 3, 1}});
  auto g_first = RadixBoxGuard::FromBox(arena, first_phase);
  assert(g_first.IsFirstIterationOf(l0->loop_var_, 4 + 2 * 3, 3));
  assert(!g_first.IsFirstIterationOf(l0->loop_var_, 4, 3));
}

void TestNormalizeNodesLatentBugAndNonAdjacent() {
  auto arena = std::make_shared<GuardDomainArena>();
  auto li = MakeTestLoop("i", 0, 4, 1);
  auto lj = MakeTestLoop("j", 0, 4, 1);
  AxisId ai = arena->RegisterLoopFrame(SyncLoopFrame{li, 4, 0, 1, nullptr});
  AxisId aj = arena->RegisterLoopFrame(SyncLoopFrame{lj, 4, 0, 1, nullptr});
  PhysicalSyncRealization work;

  auto g_i0 = [&]() {
    RadixBox b;
    b.constraints.push_back(AxisConstraint{ai, AxisAP{0, 1, 1}});
    return RadixBoxGuard::FromBox(arena, b);
  }();
  auto g_ige1 = [&]() {
    RadixBox b;
    b.constraints.push_back(AxisConstraint{ai, AxisAP{1, 4, 1}});
    return RadixBoxGuard::FromBox(arena, b);
  }();
  auto g_i2 = [&]() {
    RadixBox b;
    b.constraints.push_back(AxisConstraint{ai, AxisAP{2, 3, 1}});
    return RadixBoxGuard::FromBox(arena, b);
  }();

  // 1. Latent Bug L1 (#85 §2.1): [if (i==0) set(0), set(1), if (i==0) wait(0)] MUST NOT
  // hoist wait(0) backward before set(1); instead, only set(0) may slide forward past set(1),
  // and [if (i==0) wait(2), set(1), if (i==0) wait(0)] must stay unmerged (size 3).
  {
    auto n0 = ControlBranch(
        g_i0,
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 0), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto n1 =
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 1), Span::unknown(), work);
    auto n2 = ControlBranch(
        g_i0,
        ControlCommand(MakeFlag("system.wait_flag", PipeType::MTE2, PipeType::M, 0), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto out = ShareCommandGuards({n0, n1, n2}, work, false, true);
    assert(out.size() == 2);
    assert(out[0]->command != nullptr);
    assert(out[1]->guard.Boxes().size() == 1 && out[1]->body.size() == 1);
    assert(out[1]->body[0]->body.size() == 2);
    assert(CaseCodeSize(out).commands == 3);
  }
  {
    auto n0 = ControlBranch(
        g_i0,
        ControlCommand(MakeFlag("system.wait_flag", PipeType::MTE2, PipeType::M, 2), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto n1 =
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 1), Span::unknown(), work);
    auto n2 = ControlBranch(
        g_i0,
        ControlCommand(MakeFlag("system.wait_flag", PipeType::MTE2, PipeType::M, 0), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto out = ShareCommandGuards({n0, n1, n2}, work, false, true);
    assert(out.size() == 3);
  }

  // 2. NA-1: [if (i==0) set(0), set(1), if (i==0) set(2)] merges cleanly across set(1).
  {
    auto n0 = ControlBranch(
        g_i0,
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 0), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto n1 =
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 1), Span::unknown(), work);
    auto n2 = ControlBranch(
        g_i0,
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 2), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto out = ShareCommandGuards({n0, n1, n2}, work, false, true);
    assert(out.size() == 2);
  }

  // 3. NA-2: [if (i==0) set(0), set(1), if (i==2) set(0)] merges to single-box {0, 2}.
  // On count=4, last+stride == 4 >= count so only i % 2 == 0 is needed (1 guard).
  {
    auto n0 = ControlBranch(
        g_i0,
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 0), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto n1 =
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 1), Span::unknown(), work);
    auto n2 = ControlBranch(
        g_i2,
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 0), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto out = ShareCommandGuards({n0, n1, n2}, work, false, true);
    assert(out.size() == 2);
    assert(CaseCodeSize(out).guards == 1);  // i % 2 == 0 over [0, 4)
  }

  // 4. Two-stage pair_else_branches: false preserves separate arms; true pairs into if/else.
  {
    auto n0 = ControlBranch(
        g_i0,
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 0), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto n1 = ControlBranch(
        g_ige1,
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 2), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto pre = ShareCommandGuards({n0, n1}, work, false, false);
    assert(pre.size() == 2);
    auto post = ShareCommandGuards(pre, work, false, true);
    assert(post.size() == 1 && !post[0]->alternative.empty());
  }

  // 5. NA-4: outer-axis factoring across gap + non-null Span preservation.
  {
    const Span custom_span("test.py", 42, 1);
    RadixBox b_00, b_0g;
    b_00.constraints.push_back(AxisConstraint{ai, AxisAP{0, 1, 1}});
    b_00.constraints.push_back(AxisConstraint{aj, AxisAP{0, 1, 1}});
    b_0g.constraints.push_back(AxisConstraint{ai, AxisAP{0, 1, 1}});
    b_0g.constraints.push_back(AxisConstraint{aj, AxisAP{1, 4, 1}});
    auto n0 = ControlBranch(
        RadixBoxGuard::FromBox(arena, b_00),
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 0), custom_span, work),
        std::nullopt, custom_span, work);
    auto mid = ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 2), custom_span, work);
    auto n2 = ControlBranch(
        RadixBoxGuard::FromBox(arena, b_0g),
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 1), custom_span, work),
        std::nullopt, custom_span, work);
    auto out = ShareCommandGuards({n0, mid, n2}, work, false, true);
    assert(out.size() == 2);
    assert(CaseCodeSize(out).guards == 2);  // if (i == 0) { if (j == 0) set(0) else set(1) }
    assert(out[1]->span.begin_line_ == 42);
  }

  // 6. !IsDisjoint() when >16 overlapping boxes + NA-4 refusal preserves commands without move-corruption.
  {
    auto arena_nd = std::make_shared<GuardDomainArena>();
    auto blk_v = std::make_shared<Var>("blk", std::make_shared<ScalarType>(DataType::INDEX), Span::unknown());
    auto la0 = MakeTestLoop("a0", 0, 40, 1);
    auto la1 = MakeTestLoop("a1", 0, 40, 1);
    auto la2 = MakeTestLoop("a2", 0, 40, 1);
    AxisId ab = arena_nd->RegisterLaunchBlock(blk_v, {0, 39});
    AxisId ax0 = arena_nd->RegisterLoopFrame(SyncLoopFrame{la0, 40, 0, 1, nullptr});
    AxisId ax1 = arena_nd->RegisterLoopFrame(SyncLoopFrame{la1, 40, 0, 1, nullptr});
    AxisId ax2 = arena_nd->RegisterLoopFrame(SyncLoopFrame{la2, 40, 0, 1, nullptr});
    std::vector<RadixBox> overlap_boxes;
    for (int64_t k = 0; k < 17; ++k) {
      RadixBox bx;
      bx.constraints.push_back(AxisConstraint{ax0, AxisAP{k, k + 10, 1}});
      bx.constraints.push_back(AxisConstraint{ax1, AxisAP{k, k + 10, 1}});
      overlap_boxes.push_back(std::move(bx));
    }
    auto g_nd = RadixBoxGuard::FromBoxes(arena_nd, std::move(overlap_boxes));
    assert(!g_nd.IsDisjoint() && !g_nd.IsTautology() && !g_nd.IsEmpty());
    assert(!GuardCardinality(g_nd, {ax0, ax1}).has_value());
    auto g_nd_simp = SimplifyGuardGiven(g_nd, RadixBoxGuard::True());
    assert(!g_nd_simp.IsTautology() && !g_nd_simp.IsEmpty());

    // Build NA-4 sibling nodes sharing outer_c = (ab == 0) under !IsDisjoint() scope g_nd.
    // Both nodes hit only box k=0 (a0==0, a1==0), so each stays 1-box; inner_scope = g_nd & (ab==0)
    // stays !IsDisjoint(), so SimplifyGuardGiven refuses to drop (ab==0) and NA-4 refuses (after >= before).
    RadixBox b_n0, b_n1;
    b_n0.constraints.push_back(AxisConstraint{ab, AxisAP{0, 1, 1}});
    b_n0.constraints.push_back(AxisConstraint{ax0, AxisAP{0, 1, 1}});
    b_n0.constraints.push_back(AxisConstraint{ax1, AxisAP{0, 1, 1}});
    b_n1.constraints.push_back(AxisConstraint{ab, AxisAP{0, 1, 1}});
    b_n1.constraints.push_back(AxisConstraint{ax0, AxisAP{0, 1, 1}});
    b_n1.constraints.push_back(AxisConstraint{ax1, AxisAP{0, 1, 1}});
    b_n1.constraints.push_back(AxisConstraint{ax2, AxisAP{0, 1, 1}});
    auto nn0 = ControlBranch(
        RadixBoxGuard::FromBox(arena_nd, b_n0),
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 0), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto nn1 = ControlBranch(
        RadixBoxGuard::FromBox(arena_nd, b_n1),
        ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 1), Span::unknown(), work),
        std::nullopt, Span::unknown(), work);
    auto out_refused = ShareCommandGuards({nn0, nn1}, work, false, true, g_nd);
    assert(out_refused.size() == 2);
    assert(CaseCodeSize(out_refused, g_nd).commands == 2);
    assert(!out_refused[0]->body.empty() && !out_refused[1]->body.empty());
  }
}

void TestScheduleCasesAndHashing() {
  using pypto::ir::sync::physical::CaseCommands;
  using pypto::ir::sync::physical::ControlSequence;
  using pypto::ir::sync::physical::HashRadixBoxGuard;
  using pypto::ir::sync::physical::Intervals;
  using pypto::ir::sync::physical::ScheduleCommandCases;

  PhysicalSyncRealization work;
  auto blk = std::make_shared<Var>("block", std::make_shared<ScalarType>(DataType::INDEX), Span::unknown());
  auto cmdA = ControlSequence(
      {ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 0), Span::unknown(), work)},
      Span::unknown(), work);
  const Intervals possible{{0, 1}, {5, 7}};

  // 1. possible=[(0,1),(5,7)], cases=[(0,1,cmdA),(5,6,cmdA)] -> block < 7 (1 comparison under True).
  {
    std::vector<CaseCommands> cases{{{0, 1}, cmdA}, {{5, 6}, cmdA}};
    auto forest = ScheduleCommandCases(blk, possible, cases, work, Span::unknown(), {});
    assert(forest.size() == 1);
    auto frozen = FreezeCommandForest(forest, work);
    assert(frozen->commands.size() == 1);
    auto lt = As<Lt>(frozen->commands[0].condition_expr);
    assert(lt != nullptr && lt->left_ == blk);
    assert(As<ConstInt>(lt->right_) && As<ConstInt>(lt->right_)->value_ == 7);
  }

  // 2. possible=[(0,1),(5,7)], cases=[(0,0,cmdA),(5,7,cmdA)] -> block != 1 (1 comparison under True).
  {
    std::vector<CaseCommands> cases{{{0, 0}, cmdA}, {{5, 7}, cmdA}};
    auto forest = ScheduleCommandCases(blk, possible, cases, work, Span::unknown(), {});
    assert(forest.size() == 1);
    auto frozen = FreezeCommandForest(forest, work);
    assert(frozen->commands.size() == 1);
    auto ne = As<Ne>(frozen->commands[0].condition_expr);
    assert(ne != nullptr && ne->left_ == blk);
    assert(As<ConstInt>(ne->right_) && As<ConstInt>(ne->right_)->value_ == 1);
  }

  // 3. HashRadixBoxGuard distinguishes single-residue masks r==0 vs r==1.
  {
    auto arena = std::make_shared<GuardDomainArena>();
    auto li = MakeTestLoop("i", 0, 6, 1);
    auto lj = MakeTestLoop("j", 0, 6, 1);
    AxisId ai = arena->RegisterLoopFrame(SyncLoopFrame{li, 6, 0, 1, nullptr});
    AxisId aj = arena->RegisterLoopFrame(SyncLoopFrame{lj, 6, 0, 1, nullptr});
    ResidueMask m0(3), m1(3);
    m0.Set(0);
    m1.Set(1);
    RadixBox b0, b1;
    b0.ranks.push_back(RankCongruence{3, 0, {RankTerm{ai, 0, 1, 1}, RankTerm{aj, 0, 1, 1}}, m0});
    b1.ranks.push_back(RankCongruence{3, 0, {RankTerm{ai, 0, 1, 1}, RankTerm{aj, 0, 1, 1}}, m1});
    auto g0 = RadixBoxGuard::FromBox(arena, b0);
    auto g1 = RadixBoxGuard::FromBox(arena, b1);
    assert(HashRadixBoxGuard(g0) != HashRadixBoxGuard(g1));
  }
}

void TestLaunchBlockCommutationFence() {
  using pypto::ir::sync::physical::ControlSequence;
  PhysicalSyncRealization work;
  auto blk = std::make_shared<Var>("block", std::make_shared<ScalarType>(DataType::INDEX), Span::unknown());
  auto arena = std::make_shared<GuardDomainArena>();
  const AxisId block_axis = arena->RegisterLaunchBlock(blk, {0, 1});
  RadixBox b0, b1;
  b0.constraints.push_back(AxisConstraint{block_axis, *AxisAP::MakeCanonical(0, 1, 1, 2)});
  b1.constraints.push_back(AxisConstraint{block_axis, *AxisAP::MakeCanonical(1, 2, 1, 2)});
  const auto g_b0 = RadixBoxGuard::FromBox(arena, b0);
  const auto g_b1 = RadixBoxGuard::FromBox(arena, b1);

  auto set0 =
      ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::V, 0), Span::unknown(), work);
  auto set1 =
      ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::V, 1), Span::unknown(), work);
  auto wait0 =
      ControlCommand(MakeFlag("system.wait_flag", PipeType::MTE2, PipeType::V, 0), Span::unknown(), work);
  auto wait1 =
      ControlCommand(MakeFlag("system.wait_flag", PipeType::MTE2, PipeType::V, 1), Span::unknown(), work);
  auto wait2 =
      ControlCommand(MakeFlag("system.wait_flag", PipeType::MTE2, PipeType::V, 2), Span::unknown(), work);

  // 1. case_only_fixture seam: [set(0), if (block == 0) set(1), wait(0), if (block == 0) wait(1)]
  // wait(1) must NOT slide backward across non-disjoint wait(0) to merge with set(1).
  {
    auto br_set1 = ControlBranch(g_b0, ControlSequence(ControlForest{set1}, Span::unknown(), work),
                                 std::nullopt, Span::unknown(), work);
    auto br_wait1 = ControlBranch(g_b0, ControlSequence(ControlForest{wait1}, Span::unknown(), work),
                                  std::nullopt, Span::unknown(), work);
    ControlForest out = ShareCommandGuards(ControlForest{set0, br_set1, wait0, br_wait1}, work, false, true);
    assert(out.size() == 4);
    assert(out[0]->command != nullptr && out[1]->command == nullptr && out[1]->body.size() == 1);
    assert(out[2]->command != nullptr && out[3]->command == nullptr && out[3]->body.size() == 1);
  }

  // 2. Same-kind LaunchBlock branches separated by unguarded wait(0): order on block==0 is preserved.
  {
    auto br_w1 = ControlBranch(g_b0, ControlSequence(ControlForest{wait1}, Span::unknown(), work),
                               std::nullopt, Span::unknown(), work);
    auto br_w2 = ControlBranch(g_b0, ControlSequence(ControlForest{wait2}, Span::unknown(), work),
                               std::nullopt, Span::unknown(), work);
    ControlForest out = ShareCommandGuards(ControlForest{br_w1, wait0, br_w2}, work, false, true);
    assert(out.size() == 3);
  }

  // 3. Disjoint LaunchBlock intervening branch allows NA-1 coalescing without altering any case trace.
  {
    auto br_w1 = ControlBranch(g_b0, ControlSequence(ControlForest{wait1}, Span::unknown(), work),
                               std::nullopt, Span::unknown(), work);
    auto br_w0_other = ControlBranch(g_b1, ControlSequence(ControlForest{wait0}, Span::unknown(), work),
                                     std::nullopt, Span::unknown(), work);
    auto br_w2 = ControlBranch(g_b0, ControlSequence(ControlForest{wait2}, Span::unknown(), work),
                               std::nullopt, Span::unknown(), work);
    ControlForest out_no_else =
        ShareCommandGuards(ControlForest{br_w1, br_w0_other, br_w2}, work, false, false);
    assert(out_no_else.size() == 2);
    ControlForest out = ShareCommandGuards(ControlForest{br_w1, br_w0_other, br_w2}, work, false, true);
    assert(out.size() == 1 && !out[0]->alternative.empty());
  }

  // 4. Exhaustive LaunchBlock branches [if (block==0) set(1), wait(0), if (block==1) set(1)]
  // must NOT lift set(1) past wait(0) (non-disjoint).
  {
    auto br_s1_b0 = ControlBranch(g_b0, ControlSequence(ControlForest{set1}, Span::unknown(), work),
                                  std::nullopt, Span::unknown(), work);
    auto br_s1_b1 = ControlBranch(g_b1, ControlSequence(ControlForest{set1}, Span::unknown(), work),
                                  std::nullopt, Span::unknown(), work);
    ControlForest out = ShareCommandGuards(ControlForest{br_s1_b0, wait0, br_s1_b1}, work, false, true);
    assert(out.size() == 3);
  }
}

void TestRefactorAAuditFixes() {
  PhysicalSyncRealization work;
  auto set0 =
      ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::V, 0), Span::unknown(), work);
  auto set1 =
      ControlCommand(MakeFlag("system.set_flag", PipeType::MTE2, PipeType::M, 1), Span::unknown(), work);
  auto wait0 =
      ControlCommand(MakeFlag("system.wait_flag", PipeType::MTE2, PipeType::V, 0), Span::unknown(), work);

  // [A-P1.1] NormalizeNodes Step 7A: 3-way partition where prefix stripping empties 1 arm
  // must NOT turn the remaining 2nd non-empty arm into an unconditional else.
  {
    auto arena = std::make_shared<GuardDomainArena>();
    auto li = MakeTestLoop("i", 0, 3, 1);
    AxisId ai = arena->RegisterLoopFrame(SyncLoopFrame{li, 3, 0, 1, nullptr});
    auto mk_arm = [&](int64_t val, std::vector<ControlPtr> cmds) {
      RadixBox bx;
      bx.constraints.push_back(AxisConstraint{ai, AxisAP{val, val + 1, 1}});
      ControlForest cf(cmds.begin(), cmds.end());
      return ControlBranch(RadixBoxGuard::FromBox(arena, bx),
                           ControlSequence(std::move(cf), Span::unknown(), work), std::nullopt,
                           Span::unknown(), work);
    };
    ControlForest out = ShareCommandGuards(
        ControlForest{mk_arm(0, {set0, set1}), mk_arm(1, {set0}), mk_arm(2, {set0, wait0})}, work, false,
        true);
    auto frozen = FreezeCommandForest(out, work);
    assert(frozen != nullptr);
    size_t branch_count = 0;
    for (const auto& cmd : frozen->commands) {
      if (cmd.body && !cmd.body->commands.empty()) {
        ++branch_count;
        assert(cmd.condition_expr != nullptr);
        assert(!cmd.alternative || cmd.alternative->commands.empty());
      }
    }
    assert(branch_count == 2);
  }

  // [A-P1.2] CoalesceBoxes with ranks preserves exact union truth table.
  {
    auto arena = std::make_shared<GuardDomainArena>();
    auto li = MakeTestLoop("i", 0, 8, 1);
    AxisId ai = arena->RegisterLoopFrame(SyncLoopFrame{li, 8, 0, 1, nullptr});
    ResidueMask mask(2);
    mask.Set(0);
    RankCongruence rc{2, 0, {RankTerm{ai, 0, 1, 1}}, mask};
    RadixBox b_even;
    b_even.constraints.push_back(AxisConstraint{ai, *AxisAP::MakeCanonical(0, 5, 2, 8)});
    b_even.ranks.push_back(rc);
    RadixBox b_odd;
    b_odd.constraints.push_back(AxisConstraint{ai, *AxisAP::MakeCanonical(1, 6, 2, 8)});
    b_odd.ranks.push_back(rc);
    std::vector<RadixBox> boxes{b_even, b_odd};
    CoalesceBoxes(boxes, *arena);
    auto g = RadixBoxGuard::FromBoxes(arena, std::move(boxes));
    for (int64_t d = 0; d < 8; ++d) {
      bool expected = (d <= 5) && ((d % 2) == 0);
      assert(GuardContains(g, ai, d, ai, d, *arena) == expected);
    }
  }

  // [A-P1.3] SubtractBox uses eff_ap = CRTIntersect(s_ap, g_ap) and avoids int64 overflow at INT64_MAX.
  {
    auto arena = std::make_shared<GuardDomainArena>();
    auto li = MakeTestLoop("i", 0, 8, 1);
    AxisId ai = arena->RegisterLoopFrame(SyncLoopFrame{li, 8, 0, 1, nullptr});
    RadixBox s_box, g_box;
    s_box.constraints.push_back(AxisConstraint{ai, *AxisAP::MakeCanonical(0, 5, 2, 8)});  // {0, 2, 4}
    g_box.constraints.push_back(AxisConstraint{ai, *AxisAP::MakeCanonical(2, 6, 3, 8)});  // {2, 5} -> eff={2}
    auto diff = SubtractBox(s_box, g_box, *arena);
    assert(diff.has_value());
    for (int64_t d = 0; d < 8; ++d) {
      bool in_diff = false;
      for (const auto& bx : *diff) {
        if (BoxContains(bx, ai, d, ai, d, *arena)) in_diff = true;
      }
      assert(in_diff == (d == 0 || d == 4));
    }

    auto arena_max = std::make_shared<GuardDomainArena>();
    auto l_max = MakeTestLoop("imax", 0, INT64_MAX, 1);
    AxisId am = arena_max->RegisterLoopFrame(SyncLoopFrame{l_max, INT64_MAX, 0, 1, nullptr});
    RadixBox s_max, g_max;
    s_max.constraints.push_back(
        AxisConstraint{am, *AxisAP::MakeCanonical(INT64_MAX - 5, INT64_MAX, 2, INT64_MAX)});
    g_max.constraints.push_back(
        AxisConstraint{am, *AxisAP::MakeCanonical(INT64_MAX - 3, INT64_MAX, 2, INT64_MAX)});
    auto diff_max = SubtractBox(s_max, g_max, *arena_max);
    assert(diff_max.has_value() && diff_max->size() == 1);
    assert(BoxContains((*diff_max)[0], am, INT64_MAX - 5, am, INT64_MAX - 5, *arena_max));
    assert(!BoxContains((*diff_max)[0], am, INT64_MAX - 3, am, INT64_MAX - 3, *arena_max));
  }

  // [A-P2.5] RadixBoxGuard::FromBox canonicalizes dead rank arms to False(), tightens 1-term rank hull,
  // and BoxCardinality normalizes negative rc.base.
  {
    auto arena = std::make_shared<GuardDomainArena>();
    auto li = MakeTestLoop("i", 0, 8, 1);
    AxisId ai = arena->RegisterLoopFrame(SyncLoopFrame{li, 8, 0, 1, nullptr});

    ResidueMask m23(4);
    m23.Set(2);
    m23.Set(3);
    RadixBox dead_box;
    dead_box.constraints.push_back(AxisConstraint{ai, *AxisAP::MakeCanonical(0, 2, 1, 8)});  // d in {0, 1}
    dead_box.ranks.push_back(RankCongruence{4, 0, {RankTerm{ai, 0, 1, 1}}, m23});
    assert(RadixBoxGuard::FromBox(arena, dead_box).IsEmpty());

    ResidueMask m01(4);
    m01.Set(0);
    m01.Set(1);
    RadixBox hull_box;
    hull_box.ranks.push_back(RankCongruence{4, 0, {RankTerm{ai, 0, 1, 1}}, m01});  // active: {0, 1, 4, 5}
    auto g_hull = RadixBoxGuard::FromBox(arena, hull_box);
    assert(g_hull.Boxes().size() == 1);
    assert(g_hull.Boxes()[0].AxisRange(ai, *arena).Last() == 5);

    auto lj = MakeTestLoop("j", 0, 4, 1);
    AxisId aj = arena->RegisterLoopFrame(SyncLoopFrame{lj, 4, 0, 1, nullptr});
    ResidueMask m_zero(3);
    m_zero.Set(0);
    RadixBox neg_base_box;
    neg_base_box.ranks.push_back(
        RankCongruence{3, -1, {RankTerm{ai, 0, 1, 1}, RankTerm{aj, 0, 1, 1}}, m_zero});
    auto card = BoxCardinality(neg_base_box, {ai, aj}, *arena);
    unsigned __int128 bf = 0;
    for (int64_t d0 = 0; d0 < 8; ++d0) {
      for (int64_t d1 = 0; d1 < 4; ++d1) {
        if (FloorModI(-1 + d0 + d1, 3) == 0) ++bf;
      }
    }
    assert(card.has_value() && *card == bf);
  }

  // [A-P2.6] LowerBoxToSplitPredicates returns {} on singleton axis matching outer scope.
  {
    auto arena = std::make_shared<GuardDomainArena>();
    auto li = MakeTestLoop("i", 0, 6, 1);
    AxisId ai = arena->RegisterLoopFrame(SyncLoopFrame{li, 6, 0, 1, nullptr});
    RadixBox b_single;
    b_single.constraints.push_back(AxisConstraint{ai, AxisAP{2, 3, 1}});
    auto g_single = RadixBoxGuard::FromBox(arena, b_single);
    assert(LowerBoxToSplitPredicates(b_single, g_single, *arena, Span::unknown()).empty());
  }

  // [A-P2.8] OriginalBoundary handles negative-start loop directly without subtracting negative start.
  {
    auto arena = std::make_shared<GuardDomainArena>();
    auto lneg = MakeTestLoop("neg", -4, 4, 2);  // values: -4, -2, 0, 2 (digits 0..3)
    AxisId an = arena->RegisterLoopFrame(SyncLoopFrame{lneg, 4, -4, 2, nullptr});
    RadixBox bx;
    bx.constraints.push_back(AxisConstraint{an, *AxisAP::MakeCanonical(1, 4, 1, 4)});  // d >= 1 <=> neg >= -2
    auto g = RadixBoxGuard::FromBox(arena, bx);
    ExprPtr lowered = LowerRadixGuardToExpr(g, RadixBoxGuard::True(), Span::unknown());
    assert(lowered != nullptr);
    for (int64_t d = 0; d < 4; ++d) {
      std::map<const Var*, int64_t> env{{lneg->loop_var_.get(), -4 + d * 2}};
      assert(EvalBoolExpr(lowered, env) == (d >= 1));
    }
  }
}

}  // namespace

int main() {
  TestCRTIntersectExhaustive();
  TestBoxAlgebraAndTruthTables();
  TestSimplifyGivenAndLoweringShortcuts();
  TestLargeTripScalingAndIsFirstIteration();
  TestNormalizeNodesLatentBugAndNonAdjacent();
  TestScheduleCasesAndHashing();
  TestLaunchBlockCommutationFence();
  TestRefactorAAuditFixes();
  return 0;
}
