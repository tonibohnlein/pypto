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

#include "src/ir/transforms/utils/peer_fifo_pair.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/codegen/gm_pipe_layout.h"
#include "pypto/codegen/gm_pipe_layout_evidence.h"
#include "pypto/core/error.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "src/ir/transforms/utils/peer_gm_invocation.h"
#include "src/ir/transforms/utils/peer_local_effects.h"
#include "src/ir/transforms/utils/peer_local_plan.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"

namespace pypto::ir::sync {
namespace {

using cross_core_pipe::AnalyzePeerBorrowUses;
using cross_core_pipe::AnalyzePeerTransportEffects;
using cross_core_pipe::CollectCrossCorePipeMetadata;
using cross_core_pipe::CrossCorePipeMetadata;
using cross_core_pipe::PeerPipeBinding;
using cross_core_pipe::PeerTransportEffects;
using cross_core_pipe::PeerTransportKind;
using cross_core_pipe::PeerTransportSite;
using transform_utils::FlattenToStmts;
using transform_utils::GetCallFromStmt;

struct Endpoint {
  WrapperCallInfo call;
  PeerTransportEffects effects;
  std::unordered_map<const Var*, CallPtr> declarations;
  std::set<std::string> reserve_names;

  explicit Endpoint(WrapperCallInfo c)
      : call(std::move(c)), effects(AnalyzePeerTransportEffects(call.inner_callee)) {}

  bool CollectDeclarations() {
    for (const auto& stmt : FlattenToStmts(call.inner_callee->body_)) {
      const auto assign = As<AssignStmt>(stmt);
      const auto decl = GetCallFromStmt(stmt);
      if (!assign || !decl) continue;
      if (IsOp(decl, "system.reserve_buffer") || IsOp(decl, "buffer.peer_reserve")) {
        if (!reserve_names.insert(decl->GetKwarg<std::string>("name", "")).second) return false;
      } else if (IsOp(decl, "buffer.peer_reserve_addr") && decl->args_.size() == 1) {
        const auto reserve = Declaration(decl->args_[0]);
        if (!IsOp(reserve, "buffer.peer_reserve")) {
          return false;
        }
        declarations.emplace(assign->var_.get(), reserve);
        continue;
      } else if (!IsOp(decl, "system.import_peer_buffer")) {
        continue;
      }
      declarations.emplace(assign->var_.get(), decl);
    }
    return true;
  }

  CallPtr Declaration(const ExprPtr& arg) const {
    auto found = declarations.find(AsVarLike(arg).get());
    return found == declarations.end() ? nullptr : found->second;
  }
};

bool Linked(const Endpoint& writer, const PeerPipeBinding& outgoing, const Endpoint& reader,
            const PeerPipeBinding& incoming, size_t direction) {
  const auto import = writer.Declaration(outgoing.initializer->args_[direction]);
  const auto reserve = reader.Declaration(incoming.initializer->args_[direction]);
  return import && reserve && IsOp(import, "system.import_peer_buffer") &&
         (IsOp(reserve, "system.reserve_buffer") || IsOp(reserve, "buffer.peer_reserve")) &&
         !reserve->GetKwarg<std::string>("name", "").empty() &&
         import->GetKwarg<std::string>("name", "") == reserve->GetKwarg<std::string>("name", "") &&
         import->GetKwarg<std::string>("peer_func", "") == reader.call.inner_callee->name_;
}

// A stream word is an ordered list of transfer types, bracketed by loops
// whose bounds are the same values in the enclosing Group. Keep the two
// directions independent: a producer and consumer can interleave their
// streams differently. Balance by itself does not establish credit progress.
using Stream = std::pair<int, int>;  // logical pipe ID and direction (C2V = 1)
enum class StepKind { Message, Begin, End, Rendezvous };
struct Step {
  StepKind kind;
  Stream wire;
  std::string signature;
  bool operator==(const Step& other) const {
    return std::tie(kind, wire, signature) == std::tie(other.kind, other.wire, other.signature);
  }
};
using Word = std::vector<Step>;
using Words = std::map<Stream, Word>;
// Native pipe IDs are nonnegative. This entry retains issue order across
// wires and MIX barrier cuts for the bounded-credit check.
constexpr Stream kIssueOrder{-1, 0};
constexpr size_t kMaxStreamWord = 4096;

class ProjectStreams {
 public:
  ProjectStreams(const Endpoint& endpoint, const FunctionPtr& group, std::optional<int> subblock = {})
      : subblock_(subblock), cube_(endpoint.call.inner_callee->func_type_ == FunctionType::AIC) {
    const auto& params = endpoint.call.inner_callee->params_;
    const auto& args = endpoint.call.inner_call->args_;
    for (size_t i = 0; i < std::min(params.size(), args.size()); ++i) {
      if (auto constant = transform_utils::EvalConstInt(args[i])) {
        bounds_[params[i].get()] = std::to_string(*constant);
      } else {
        const auto actual = AsVarLike(args[i]);
        const auto it = std::find(group->params_.begin(), group->params_.end(), actual);
        if (it != group->params_.end())
          bounds_[params[i].get()] = "arg" + std::to_string(it - group->params_.begin());
      }
    }
    for (const auto& site : endpoint.effects.sites) sites_[site.statement.get()] = &site;
  }

  std::string error;
  Words Run(const StmtPtr& stmt, bool terminal = true) {
    if (!stmt || !error.empty()) return {};
    if (const auto it = sites_.find(stmt.get()); it != sites_.end()) {
      const auto& site = *it->second;
      if (site.kind == PeerTransportKind::Free) return {};
      const int dir = (cube_ == (site.kind == PeerTransportKind::Push)) ? 1 : 2;
      std::string shape = site.descriptor->dtype_.ToString();
      for (const auto extent : site.descriptor->shape_) shape += "x" + std::to_string(extent);
      // Both AIV lanes send the group signal for a no-split push. Only a
      // statically zero row proves that this lane will leave the shared GM
      // slot untouched: native emission can widen a narrow column extent.
      if (subblock_ && !cube_ && site.kind == PeerTransportKind::Push) {
        const auto rows = site.descriptor->valid_shape_.front();
        const auto call = GetCallFromStmt(site.statement);
        if (!cross_core_pipe::IsBufferPeerTransport(call) && rows < 0) {
          return Fail("AIV push has a dynamic valid row extent");
        }
        shape = site.empty_payload ? "empty" : "writes";
      }
      const Stream wire{site.pipe_id, dir};
      Step message{StepKind::Message, wire, shape};
      return {{wire, {message}}, {kIssueOrder, {message}}};
    }
    if (const auto seq = As<SeqStmts>(stmt)) {
      Words result;
      for (size_t i = 0; i < seq->stmts_.size(); ++i) {
        const auto part = Run(seq->stmts_[i], terminal && i + 1 == seq->stmts_.size());
        for (const auto& [stream, word] : part) {
          auto& output = result[stream];
          if (word.size() > kMaxStreamWord - output.size())
            return Fail("stream word exceeds analysis budget");
          output.insert(output.end(), word.begin(), word.end());
        }
      }
      return result;
    }
    if (const auto branch = As<IfStmt>(stmt)) {
      if (subblock_) {
        if (const auto eq = As<Eq>(branch->condition_)) {
          const auto left = Bound(eq->left_), right = Bound(eq->right_);
          if (!left.empty() && !right.empty() && (left == "subblock" || right == "subblock")) {
            const auto& other = left == "subblock" ? right : left;
            if (other == "0" || other == "1") {
              return Run(other == std::to_string(*subblock_) ? branch->then_body_
                                                             : branch->else_body_.value_or(nullptr),
                         terminal);
            }
          }
        }
      }
      const auto first = Run(branch->then_body_, terminal);
      const auto second = branch->else_body_ ? Run(*branch->else_body_, terminal) : Words{};
      if (first != second) return Fail("conditional arms have different peer streams");
      return first;
    }
    if (const auto loop = As<ForStmt>(stmt)) {
      auto body = Run(loop->body_, false);
      if (body.empty()) return {};
      if (loop->kind_ != ForKind::Sequential) return Fail("peer loop is not sequential");
      if (auto trips = transform_utils::EvalConstTripCount(loop)) {
        if (*trips == 0) return {};
        for (auto& [stream, word] : body) {
          if (static_cast<uint64_t>(*trips) > kMaxStreamWord / word.size())
            return Fail("stream word exceeds analysis budget");
          const auto iteration = word;
          for (int64_t i = 1; i < *trips; ++i) word.insert(word.end(), iteration.begin(), iteration.end());
        }
      } else {
        auto start = Bound(loop->start_), stop = Bound(loop->stop_), step = Bound(loop->step_);
        if (start.empty() || stop.empty() || step.empty())
          return Fail("peer loop bounds lack a common Group value");
        for (auto& [stream, word] : body) {
          if (word.size() > kMaxStreamWord - 2) return Fail("stream word exceeds analysis budget");
          word.insert(word.begin(), {StepKind::Begin, {}, start + "," + stop + "," + step});
          word.push_back({StepKind::End, {}, {}});
        }
      }
      return body;
    }
    const auto call = GetCallFromStmt(stmt);
    if (call && (IsOp(call, "system.set_flag") || IsOp(call, "system.wait_flag")))
      return Fail("authored event constraints require joint planning");
    if (call && IsOp(call, "system.syncall")) {
      if (!cross_core_pipe::IsHardMixedSyncAll(call)) return Fail("unmodeled peer rendezvous");
      return {{kIssueOrder, {{StepKind::Rendezvous, {}, {}}}}};
    }
    if (auto assignment = As<AssignStmt>(stmt)) {
      auto value = Bound(assignment->value_);
      if (!value.empty()) bounds_[assignment->var_.get()] = std::move(value);
      return {};
    }
    if (As<EvalStmt>(stmt) || As<YieldStmt>(stmt) || (terminal && As<ReturnStmt>(stmt))) return {};
    return Fail("unmodeled control or early exit in peer schedule");
  }

 private:
  std::optional<int> subblock_;
  bool cube_;
  std::unordered_map<const Stmt*, const PeerTransportSite*> sites_;
  std::unordered_map<const Var*, std::string> bounds_;

  Words Fail(const std::string& message) {
    if (error.empty()) error = message;
    return {};
  }
  std::string Bound(const ExprPtr& value) const {
    if (auto integer = transform_utils::EvalConstInt(value)) return std::to_string(*integer);
    if (const auto call = As<Call>(value);
        IsOp(call, "tile.get_block_idx") || IsOp(call, "buffer.get_block_idx")) {
      return "block";
    }
    if (subblock_) {
      if (const auto call = As<Call>(value);
          IsOp(call, "tile.get_subblock_idx") || IsOp(call, "buffer.get_subblock_idx")) {
        return "subblock";
      }
    }
    const auto it = bounds_.find(AsVarLike(value).get());
    return it == bounds_.end() ? "" : it->second;
  }
};

// Both no-split lanes address the same GM slot. Retain lane identity while
// projecting each outbound ordinal, including complementary branch arms.
// Nominal shape and equal notification counts alone allow two active writers.
bool HasSingleVectorWriter(const Endpoint& vec, const FunctionPtr& group) {
  ProjectStreams lane0(vec, group, 0), lane1(vec, group, 1);
  const auto first = lane0.Run(vec.call.inner_callee->body_);
  const auto second = lane1.Run(vec.call.inner_callee->body_);
  if (!lane0.error.empty() || !lane1.error.empty()) return false;
  for (const auto& [wire, a] : first) {
    if (wire.second != core_affinity::kDirMaskV2C) continue;
    const auto it = second.find(wire);
    if (it == second.end() || a.size() != it->second.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      const auto& b = it->second[i];
      if (a[i].kind == StepKind::Message && b.kind == StepKind::Message) {
        if ((a[i].signature == "writes") == (b.signature == "writes")) return false;
      } else if (!(a[i] == b)) {
        return false;
      }
    }
  }
  return true;
}

// A2/A3 pushes wait for a credit at batch boundaries n >= depth. By then
// the peer must have popped n - depth + period tiles to have posted enough
// credits. Pops wait for the corresponding ready signal. Each predicate is
// monotone in the other endpoint's ordinal, so issuing either enabled step
// cannot disable its peer. Equal AIV branch words let both lanes advance
// together, including a zero-valid lane which still sends native signals.
std::string CheckFifoProgress(const Word& cube, const Word& vec,
                              const std::map<int, PeerPipeBinding>& pipes) {
  const std::array<const Word*, 2> words{&cube, &vec};
  std::array<size_t, 2> pc{};
  struct Counts {
    int64_t sent = 0;
    int64_t received = 0;
  };
  std::map<Stream, Counts> counts;
  bool in_loop = false;
  for (;;) {
    std::array<const Step*, 2> next{};
    for (size_t side = 0; side < 2; ++side) {
      if (pc[side] < words[side]->size()) next[side] = &(*words[side])[pc[side]];
    }
    if (!next[0] && !next[1]) return "";
    if (next[0] && next[1] && next[0]->kind != StepKind::Message && *next[0] == *next[1]) {
      if (next[0]->kind == StepKind::Rendezvous && in_loop)
        return "MIX rendezvous in a symbolic loop needs launch-wide trip agreement";
      if (next[0]->kind != StepKind::Rendezvous) {
        const bool entering = next[0]->kind == StepKind::Begin;
        if (entering == in_loop) return "nested symbolic FIFO loop lacks a progress summary";
        // At both cuts each stream has drained and starts at batch residue
        // zero. Its next credit test depends only on new sent-received, so
        // the same proof applies for zero, one or any number of iterations.
        for (const auto& [wire, state] : counts) {
          const auto& pipe = pipes.at(wire.first);
          if (state.sent != state.received || state.sent % pipe.CreditPeriod() != 0 ||
              pipe.transport_slots % pipe.CreditPeriod() != 0)
            return "symbolic FIFO loop crosses an unclosed credit batch";
        }
        in_loop = entering;
      }
      ++pc[0];
      ++pc[1];
      continue;
    }
    bool advanced = false;
    for (size_t side = 0; side < 2; ++side) {
      const auto* step = next[side];
      if (!step || step->kind != StepKind::Message) continue;
      auto& state = counts[step->wire];
      const auto& pipe = pipes.at(step->wire.first);
      const bool send = (step->wire.second == 1) == (side == 0);
      if (send) {
        const int period = pipe.CreditPeriod();
        // Odd, non-divisible ring depths have a distinct startup phase.
        if (pipe.transport_slots % period != 0)
          return "FIFO progress requires a whole number of credit batches per ring";
        if (state.sent % period == 0 && state.sent - state.received > pipe.transport_slots - period) continue;
        ++state.sent;
      } else {
        if (state.received == state.sent) continue;
        ++state.received;
      }
      advanced = true;
      ++pc[side];
    }
    if (!advanced) return "FIFO ready/credit or rendezvous cycle blocks peer progress";
  }
}

// Both endpoints have already matched each logical pipe and slot geometry.
std::string ValidateWorkspace(codegen::gm_pipe::LayoutEvidence& evidence, const FunctionPtr& group,
                              const std::map<int, PeerPipeBinding>& pipes) {
  try {
    // The injected tensor's placeholder shape is unrelated to its physical
    // size. Use the same forwarding and placement evidence as the emitter.
    const auto context = evidence.BindRoot(group);
    const auto& layout = context->Layout();
    for (const auto& [id, pipe] : pipes) {
      const auto base = layout.offsets.find({id, pipe.direction_mask});
      if (base == layout.offsets.end()) return "peer pipe absent from launch workspace";
      const auto rings = codegen::gm_pipe::RingCountForDirMask(pipe.direction_mask);
      const auto ring_bytes = pipe.transport_slots * pipe.slot_bytes;
      if (!rings || ring_bytes > (layout.bytes - base->second) / rings)
        return "peer ring exceeds the managed launch workspace";
      // A2/A3 forms the bidirectional TPipe's second base in signed int.
      if (rings == 2 && ring_bytes > std::numeric_limits<int32_t>::max())
        return "second directional ring offset exceeds native TPipe entry field";
    }
    return {};
  } catch (const ValueError& error) {
    return error.what();
  }
}

std::string CheckPair(codegen::gm_pipe::LayoutEvidence& workspace, const FunctionPtr& group, Endpoint& cube,
                      Endpoint& vec) {
  for (auto* side : {&cube, &vec}) {
    if (!side->effects.unavailable.empty()) return side->effects.unavailable;
    if (!side->CollectDeclarations()) return "duplicate local FIFO reserve name";
  }
  if (cube.effects.pipes.size() != vec.effects.pipes.size() || cube.effects.pipes.empty())
    return "endpoints initialize different FIFO IDs";
  for (const auto& [id, c] : cube.effects.pipes) {
    const auto other = vec.effects.pipes.find(id);
    if (other == vec.effects.pipes.end()) return "endpoints initialize different FIFO IDs";
    const auto& v = other->second;
    if (c.direction_mask != v.direction_mask || c.transport_slots != v.transport_slots ||
        c.slot_bytes != v.slot_bytes)
      return "endpoint direction, transport depth or slot stride differs";
    if (((c.direction_mask & 1) && !Linked(cube, c, vec, v, 0)) ||
        ((c.direction_mask & 2) && !Linked(vec, v, cube, c, 1)))
      return "FIFO import does not reference the receiving endpoint's reserve";
  }
  const auto storage = ValidateWorkspace(workspace, group, cube.effects.pipes);
  std::string report = storage.empty() ? "physical FIFO workspace matched; " : storage + "; ";
  ProjectStreams producer(cube, group), consumer(vec, group);
  auto first = producer.Run(cube.call.inner_callee->body_);
  auto second = consumer.Run(vec.call.inner_callee->body_);
  if (!producer.error.empty()) return producer.error;
  if (!consumer.error.empty()) return consumer.error;
  auto cube_order = first.extract(kIssueOrder);
  auto vec_order = second.extract(kIssueOrder);
  if (first.empty() || first != second)
    return "endpoint send/receive stream counts, loop domains or message types differ";
  for (const auto& [wire, word] : first) {
    if (wire.second == core_affinity::kDirMaskV2C) {
      report += HasSingleVectorWriter(vec, group) ? "V2C epochs have a single nonempty sender; "
                                                  : "V2C single-writer evidence unavailable; ";
      break;
    }
  }
  const auto stalled = CheckFifoProgress(cube_order.mapped(), vec_order.mapped(), cube.effects.pipes);
  return report +
         (stalled.empty() ? "FIFO ready/credit progress proved for the paired issue streams" : stalled);
}

struct PeerDispatch {
  FunctionPtr group;
  std::vector<FunctionPtr> functions;
  std::vector<WrapperCallInfo> endpoints;  // An emission unit with exactly one AIC and one AIV.
};

std::vector<PeerDispatch> CollectPeerDispatches(const ProgramPtr& program) {
  std::vector<PeerDispatch> dispatches;
  for (const auto& [global, group] : program->functions_) {
    if (group->func_type_ != FunctionType::Group) continue;
    const auto calls = CollectInnerCalls(group, program);
    const auto direct = FlattenToStmts(group->body_);
    const auto plain = [](const ExprPtr& value) {
      return AsVarLike(value) || As<ConstInt>(value) || As<ConstFloat>(value) || As<ConstBool>(value);
    };
    std::set<std::string> straight;
    bool regular = calls.size() == 2;
    for (const auto& stmt : direct) {
      ExprPtr value;
      if (auto assign = As<AssignStmt>(stmt)) {
        value = assign->value_;
        if (plain(value)) continue;
      } else if (auto eval = As<EvalStmt>(stmt)) {
        value = eval->expr_;
      } else if (auto ret = As<ReturnStmt>(stmt); ret && stmt == direct.back()) {
        if (std::all_of(ret->value_.begin(), ret->value_.end(), plain)) continue;
        if (ret->value_.size() == 1) value = ret->value_.front();
      }
      const auto call = transform_utils::AsCallOrSubmitView(value);
      // CollectInnerCalls omits opaque callees. Check every actual Group
      // statement and operand before granting ownership of the emitted unit.
      if (!call || !As<GlobalVar>(call->op_) || call->HasAttr(kAttrManualDepEdges) ||
          call->HasAttr(kAttrCoreNum) || call->HasAttr(kAttrSyncStart) || call->HasAttr(kAttrPredicate) ||
          call->HasAttr("allow_early_resolve") ||
          !std::all_of(call->args_.begin(), call->args_.end(), plain) ||
          !straight.insert(call->op_->name_).second) {
        regular = false;
      }
    }
    regular &= straight.size() == 2;
    std::optional<WrapperCallInfo> cube, vec;
    PeerDispatch dispatch{group, {}, {}};
    bool has_fifo = false;
    for (const auto& call : calls) {
      const auto type = call.inner_callee->func_type_;
      if (type != FunctionType::AIC && type != FunctionType::AIV) continue;
      auto& endpoint = type == FunctionType::AIC ? cube : vec;
      regular &= !endpoint && straight.count(call.inner_callee->name_) != 0 &&
                 call.inner_call->args_.size() == call.inner_callee->params_.size();
      endpoint = call;
      dispatch.functions.push_back(call.inner_callee);
      CrossCorePipeMetadata metadata;
      CollectCrossCorePipeMetadata(FlattenToStmts(call.inner_callee->body_), metadata);
      has_fifo |= metadata.HasAnySetup() || metadata.HasCrossCoreOps();
    }
    if (!has_fifo) continue;
    if (regular && cube && vec) dispatch.endpoints = {*cube, *vec};
    dispatches.push_back(std::move(dispatch));
  }
  return dispatches;
}

// A shared expression used by two issue statements denotes two instructions.
// The physical scheduler indexes commands by the top-level call. Give each
// repeated issue its own anchor while preserving operands, binders and spans.
class DistinctIssueCuts : public SyncStatementMutator {
 public:
  FunctionPtr Run(const FunctionPtr& fn) {
    const auto body = VisitStmt(fn->body_);
    if (!changed_) return fn;
    auto result = MutableCopy(fn);
    result->body_ = body;
    return result;
  }

 private:
  std::set<const Call*> seen_;
  bool changed_ = false;
  CallPtr Distinct(const ExprPtr& expression) {
    const auto call = As<Call>(expression);
    if (!call || seen_.insert(call.get()).second) return nullptr;
    changed_ = true;
    return MutableCopy(call);
  }
  StmtPtr VisitStmt_(const AssignStmtPtr& stmt) override {
    const auto copy = Distinct(stmt->value_);
    return copy ? std::make_shared<AssignStmt>(stmt->var_, copy, stmt->span_) : stmt;
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& stmt) override {
    const auto copy = Distinct(stmt->expr_);
    return copy ? std::make_shared<EvalStmt>(copy, stmt->span_) : stmt;
  }
};

FunctionPtr PlanLocalEndpoint(const WrapperCallInfo& call, const FunctionPtr& group,
                              const SyncInvocationEvidence& invocation, PeerInvocationGmEvidence& gm,
                              codegen::gm_pipe::LayoutEvidence& layout, PeerGmEffects& group_effects) {
  const auto fn = DistinctIssueCuts().Run(call.inner_callee);
  CHECK(!fn->requires_runtime_binding_ && !fn->HasAttr(kAttrExternalSource) &&
        !fn->HasAttr(kAttrBuiltinTemplateDir))
      << "external endpoint has its own synchronization owner";
  const auto& owners = gm.CallersOf(fn->name_);
  CHECK(owners.size() == 1 && owners.count(group->name_))
      << "peer leaf is reachable outside this mixed dispatch";
  const auto nested = invocation.callees.find(fn->name_);
  CHECK(nested == invocation.callees.end() || nested->second.empty()) << "unqualified peer helper";
  const auto transport = AnalyzePeerTransportEffects(fn);
  CHECK(transport.unavailable.empty()) << transport.unavailable;
  // The placed workspace and its separation from ordinary GM are required to
  // describe this endpoint's hidden FIFO accesses, independently of its peer.
  CHECK(transport.pipes.size() == 1) << "local FIFO workspace requires one qualified native pipe";
  const auto storage = ValidateWorkspace(layout, group, transport.pipes);
  CHECK(storage.empty()) << storage;
  if (fn->ir_stage_ == FunctionIRStage::Functional) {
    const auto borrow = AnalyzePeerBorrowUses(fn, transport);
    CHECK(borrow.unavailable.empty()) << borrow.unavailable;
  }

  const auto workspace = std::find_if(group->params_.begin(), group->params_.end(), [](const VarPtr& formal) {
    return formal->name_hint_ == "__gm_pipe_buffer";
  });
  CHECK(workspace != group->params_.end()) << "Group has no managed GM workspace";
  const auto workspace_formal = static_cast<size_t>(workspace - group->params_.begin());
  std::optional<SyncDispatchCoordinates> coordinates;
  if (fn->ir_stage_ == FunctionIRStage::Buffer) {
    coordinates = gm.TryLaunchCoordinates(group, workspace_formal);
  }
  std::unordered_map<const Var*, size_t> positions;
  for (size_t i = 0; i < group->params_.size(); ++i) positions.emplace(group->params_[i].get(), i);
  if (coordinates) {
    std::set<size_t> endpoint_widths;
    for (size_t i = 0; i < fn->params_.size(); ++i) {
      const auto input = AsVarLike(call.inner_call->args_[i]);
      const auto formal = positions.find(input.get());
      if (formal != positions.end() && coordinates->width_parameters.count(formal->second) &&
          As<ScalarType>(fn->params_[i]->GetType()) &&
          GetScalarDtype(fn->params_[i]) == GetScalarDtype(input)) {
        endpoint_widths.insert(i);
      }
    }
    coordinates->width_parameters = std::move(endpoint_widths);
  }
  const auto parameters = invocation.parameters.find(fn->name_);
  const auto local = ImportPeerLocalEffects(
      fn, transport, "a2a3",
      parameters == invocation.parameters.end() ? SyncParameterBounds{} : parameters->second, coordinates);
  std::set<size_t> touched;
  for (const auto index : local.gm.formals) {
    const auto root = AsVarLike(call.inner_call->args_.at(index));
    const auto pos = positions.find(root.get());
    CHECK(pos != positions.end() && root != *workspace)
        << "peer GM access lacks an original Group tensor separate from the workspace";
    touched.insert(pos->second);
    if (local.gm.scalar_reads.count(index)) {
      group_effects.scalar_reads.insert(pos->second);
    }
    if (local.gm.writes.count(index)) {
      group_effects.writes.insert(pos->second);
    }
  }
  gm.Check(group, touched, workspace_formal);
  for (const auto index : local.gm.host_extent_requirements) {
    const auto argument = AsVarLike(call.inner_call->args_[index]);
    const auto from = As<TensorType>(argument->GetType());
    const auto to = As<TensorType>(fn->params_[index]->GetType());
    CHECK(from && to && from->dtype_ == to->dtype_ &&
          tile_view_semantics::ShapeExprListsEquivalent(from->shape_, to->shape_))
        << "Peer scalar GM address requires the original Group tensor shape";
    gm.CheckHostRankOneExtent(group, positions.at(argument.get()));
  }
  return RealizePeerLocalPlan(local);
}

}  // namespace

std::vector<PeerLocalDecision> PlanPeerLocalUnits(const ProgramPtr& program,
                                                  const SyncInvocationEvidence& invocation) {
  const auto dispatches = CollectPeerDispatches(program);
  std::vector<PeerLocalDecision> decisions;
  if (dispatches.empty()) return decisions;
  codegen::gm_pipe::LayoutEvidence layout(program);
  PeerInvocationGmEvidence gm(program, invocation);
  const auto* context = PassContext::Current();
  const bool disjoint_gm = context && context->GetAutoSyncDisjointGM();
  for (const auto& dispatch : dispatches) {
    if (dispatch.endpoints.empty()) {
      for (const auto& fn : dispatch.functions)
        decisions.push_back({fn, "mixed emission unit requires one unconditional call to each peer"});
      continue;
    }

    std::vector<PeerLocalDecision> local;
    std::string failures;
    PeerGmEffects effects;
    for (const auto& endpoint : dispatch.endpoints) {
      try {
        local.push_back({PlanLocalEndpoint(endpoint, dispatch.group, invocation, gm, layout, effects), {}});
      } catch (const ValueError& error) {
        auto detail = "local planning for " + endpoint.inner_callee->name_ + ": " + error.what();
        if (!failures.empty()) failures += "; ";
        failures += detail;
        local.push_back({endpoint.inner_callee, std::move(detail)});
      }
    }
    if (failures.empty()) {
      try {
        gm.CheckScalarReads(dispatch.group, effects, disjoint_gm);
      } catch (const ValueError& error) {
        failures = error.what();
      }
    }
    for (size_t i = 0; i < local.size(); ++i) {
      if (!failures.empty() && local[i].fallback_reason.empty()) {
        local[i] = {dispatch.endpoints[i].inner_callee,
                    "local plan verified; shared PTOAS invocation requires fallback: " + failures};
      }
      decisions.push_back(std::move(local[i]));
    }
  }
  return decisions;
}

std::map<std::string, std::string> InspectPeerFifoProtocols(const ProgramPtr& program) {
  std::map<std::string, std::string> reports;
  codegen::gm_pipe::LayoutEvidence layout(program);
  for (const auto& dispatch : CollectPeerDispatches(program)) {
    if (dispatch.endpoints.empty()) {
      reports[dispatch.group->name_] = "diagnostic requires one unconditional call to each peer";
      continue;
    }
    Endpoint cube(dispatch.endpoints[0]), vec(dispatch.endpoints[1]);
    reports[dispatch.group->name_] = CheckPair(layout, dispatch.group, cube, vec);
  }
  return reports;
}
}  // namespace pypto::ir::sync
