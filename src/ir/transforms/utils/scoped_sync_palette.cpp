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

#include "src/ir/transforms/utils/scoped_sync_palette.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/scoped_token_families.h"

namespace pypto::ir::sync {
const char* ScopedPaletteStatusName(ScopedPaletteStatus status) {
  switch (status) {
    case ScopedPaletteStatus::Certified:
      return "certified";
    case ScopedPaletteStatus::UnsupportedGrammar:
      return "unsupported-grammar";
    case ScopedPaletteStatus::ArithmeticOverflow:
      return "arithmetic-overflow";
    case ScopedPaletteStatus::UnprovedReuse:
      return "unproved-reuse";
    case ScopedPaletteStatus::Constructed:
      return "constructed";
  }
  INTERNAL_CHECK(false) << "Invalid palette status";
  return "invalid";
}

namespace {
using Digit = ScopedPublicationDigit;
struct Refusal {
  ScopedPaletteStatus status;
  std::string reason;
};
int64_t Checked(__int128 value) {
  if (value < 0 || value > std::numeric_limits<int64_t>::max())
    throw Refusal{ScopedPaletteStatus::ArithmeticOverflow, "Publication rank arithmetic exceeds int64"};
  return static_cast<int64_t>(value);
}
void Require(bool condition, const char* reason) {
  if (!condition) throw Refusal{ScopedPaletteStatus::UnsupportedGrammar, reason};
}
struct Node {
  enum Kind { Leaf, Sequence, Repeat } kind = Sequence;
  int64_t count = 0, trips = 1;
  const ForStmt* frame = nullptr;
  size_t use = 0;
  std::vector<Digit> digits;
  std::vector<std::shared_ptr<Node>> children;
  std::vector<int64_t> ends;
};
using NodePtr = std::shared_ptr<Node>;
struct Publication {
  size_t use;
  std::vector<Digit> digits;
};
int64_t Last(const Digit& d) { return Checked(d.first + static_cast<__int128>(d.step) * (d.count - 1)); }
}  // namespace
struct ScopedPublicationSchedule::State {
  std::shared_ptr<const ScopedSyncProtocol> protocol;
  PipeType source, target;
  NodePtr root;
  std::vector<ScopedPublicationRankPiece> pieces;
};
ScopedPublicationSchedule::ScopedPublicationSchedule(std::shared_ptr<const State> state)
    : state_(std::move(state)) {}
const std::shared_ptr<const ScopedSyncProtocol>& ScopedPublicationSchedule::Protocol() const {
  return state_->protocol;
}
PipeType ScopedPublicationSchedule::SourcePipe() const { return state_->source; }
PipeType ScopedPublicationSchedule::TargetPipe() const { return state_->target; }
int64_t ScopedPublicationSchedule::Publications() const { return state_->root->count; }
const std::vector<ScopedPublicationRankPiece>& ScopedPublicationSchedule::Pieces() const {
  return state_->pieces;
}

struct ScopedPaletteBuilder {
  const ScopedTokenOrder& order;
  const CompactSyncInput& input;
  const std::vector<PhysicalSyncFamily>& uses;
  ScopedPaletteResult result;
  std::shared_ptr<ScopedPublicationSchedule::State> state;
  int64_t budget;
  std::vector<size_t> selected;
  std::vector<std::vector<size_t>> cuts;
  std::map<const SyncRegion*, std::map<size_t, size_t>> routes;
  std::vector<size_t> Index(const SyncRegion& region) {
    if (region.leaf) return {*region.leaf};
    std::vector<size_t> descendants;
    for (size_t i = 0; i < region.children.size(); ++i)
      for (auto leaf : Index(region.children[i])) {
        routes[&region].emplace(leaf, i);
        ++result.work.routing_entries;
        descendants.push_back(leaf);
      }
    return descendants;
  }
  ScopedPaletteBuilder(const ScopedTokenOrder& o, PipeType source, PipeType target, int64_t b)
      : order(o),
        input(o.Input()),
        uses(o.Protocol()->Families()),
        state(std::make_shared<ScopedPublicationSchedule::State>()),
        budget(b) {
    state->protocol = o.Protocol();
    state->source = source;
    state->target = target;
  }
  NodePtr Sequence(std::vector<NodePtr> children) {
    auto node = std::make_shared<Node>();
    for (auto& child : children)
      if (child->count) {
        node->count = Checked(static_cast<__int128>(node->count) + child->count);
        node->children.push_back(std::move(child));
        node->ends.push_back(node->count);
      }
    ++result.work.grammar_nodes;
    return node;
  }
  NodePtr Repeat(NodePtr child, const ForStmt* frame, int64_t count) {
    auto node = std::make_shared<Node>();
    node->kind = Node::Repeat;
    node->frame = frame;
    node->trips = count;
    node->count = Checked(static_cast<__int128>(child->count) * count);
    node->children.push_back(std::move(child));
    ++result.work.grammar_nodes;
    return node;
  }
  Digit Domain(size_t id, size_t axis) {
    ++result.work.domain_visits;
    const auto& domain = uses[id].source_domain;
    Require(domain && axis < domain->ranges.size(), "Missing exact source progression");
    const auto [first, end] = domain->ranges[axis];
    const auto step = domain->steps.empty() ? 1 : domain->steps.at(axis);
    Require(first >= 0 && end > first && step > 0, "Malformed publication progression");
    return {first, step, 1 + (end - first - 1) / step};
  }
  bool Active(size_t id, const std::vector<Digit>& context) {
    for (size_t i = 0; i < context.size(); ++i) {
      const auto d = Domain(id, i);
      const auto& c = context[i];
      if (c.first < d.first || c.first > Last(d) || (c.first - d.first) % d.step) return false;
      Require(Last(c) <= Last(d) && (c.count == 1 || c.step % d.step == 0),
              "Phase partition did not preserve complete activity");
    }
    return true;
  }
  std::vector<size_t> Filter(const std::vector<size_t>& ids, const std::vector<Digit>& context) {
    std::vector<size_t> out;
    for (auto id : ids)
      if (Active(id, context)) out.push_back(id);
    return out;
  }
  NodePtr Loop(const SyncLoopFrame& frame, const std::vector<size_t>& ids, std::vector<Digit> context,
               const std::function<NodePtr(const std::vector<size_t>&, const std::vector<Digit>&)>& body) {
    std::vector<NodePtr> parts;
    if (frame.count == 0 || ids.empty()) return Sequence({});
    std::set<int64_t> cuts{0, frame.count};
    int64_t period = 1;
    const auto axis = context.size();
    for (auto id : ids) {
      const auto& frames = input.segments[uses[id].source_region].nested_frames;
      Require(axis < frames.size() && frames[axis].loop.get() == frame.loop.get(),
              "Publication frame differs from original schedule");
      auto d = Domain(id, axis);
      cuts.insert(d.first);
      cuts.insert(Checked(static_cast<__int128>(Last(d)) + 1));
      if (d.count > 1) period = Checked(static_cast<__int128>(period / std::gcd(period, d.step)) * d.step);
    }
    std::vector<int64_t> boundaries(cuts.begin(), cuts.end());
    for (size_t i = 1; i < boundaries.size(); ++i) {
      auto first = boundaries[i - 1], length = boundaries[i] - first;
      Require(first >= 0 && boundaries[i] <= frame.count, "Publication domain escapes original loop");
      const auto cycles = length / period, tail = length % period;
      auto phases = [&](int64_t begin, int64_t phases_count, int64_t trips) {
        std::vector<NodePtr> children;
        for (int64_t phase = 0; phase < phases_count; ++phase) {
          ++result.work.phase_visits;
          auto next = context;
          next.push_back({begin + phase, period, trips});
          auto active = Filter(ids, next);
          if (!active.empty()) children.push_back(body(active, next));
        }
        return Sequence(std::move(children));
      };
      if (cycles) parts.push_back(Repeat(phases(first, period, cycles), frame.loop.get(), cycles));
      if (tail) parts.push_back(phases(Checked(first + static_cast<__int128>(cycles) * period), tail, 1));
    }
    return Sequence(std::move(parts));
  }
  NodePtr Region(const SyncRegion& region, const std::vector<size_t>& ids, const std::vector<Digit>& context,
                 bool body = false) {
    if (region.repetition && !body)
      return Loop(*region.repetition, ids, context,
                  [&](const auto& subset, const auto& next) { return Region(region, subset, next, true); });
    if (region.leaf) {
      const auto index = *region.leaf;
      const auto& segment = input.segments[index];
      if (context.size() < segment.nested_frames.size())
        return Loop(segment.nested_frames[context.size()], ids, context,
                    [&](const auto& subset, const auto& next) { return Region(region, subset, next, true); });
      std::vector<NodePtr> leaves;
      const std::set<size_t> active(ids.begin(), ids.end());
      for (auto id : cuts[index])
        if (active.count(id)) {
          auto leaf = std::make_shared<Node>();
          leaf->kind = Node::Leaf;
          leaf->count = 1;
          leaf->use = id;
          leaf->digits = context;
          leaves.push_back(std::move(leaf));
          ++result.work.grammar_nodes;
        }
      return Sequence(std::move(leaves));
    }
    std::vector<NodePtr> children;
    std::map<size_t, std::vector<size_t>> subsets;
    for (auto id : ids) subsets[routes.at(&region).at(uses[id].source_region)].push_back(id);
    for (const auto& [child, subset] : subsets)
      children.push_back(Region(region.children[child], subset, context));
    return Sequence(std::move(children));
  }
  size_t FrameIndex(const Publication& p, const ForStmt* frame) const {
    const auto& frames = input.segments[uses[p.use].source_region].nested_frames;
    for (size_t i = 0; i < frames.size(); ++i)
      if (frames[i].loop.get() == frame) return i;
    throw Refusal{ScopedPaletteStatus::UnsupportedGrammar, "Publication lost enclosing repetition frame"};
  }
  Publication At(const NodePtr& node, int64_t index) {
    ++result.work.boundary_node_visits;
    Require(index >= 0 && index < node->count, "Publication boundary rank out of range");
    if (node->kind == Node::Leaf) return {node->use, node->digits};
    if (node->kind == Node::Sequence) {
      const auto child = static_cast<size_t>(std::upper_bound(node->ends.begin(), node->ends.end(), index) -
                                             node->ends.begin());
      return At(node->children[child], index - (child ? node->ends[child - 1] : 0));
    } else {
      const auto& child = node->children.front();
      auto p = At(child, index % child->count);
      auto& d = p.digits[FrameIndex(p, node->frame)];
      d.first = Checked(d.first + static_cast<__int128>(index / child->count) * d.step);
      d.count = 1;
      return p;
    }
  }
  std::vector<Digit> Target(const Publication& p) const {
    const auto& f = uses[p.use];
    const auto& domain = *f.target_domain;
    std::vector<Digit> out;
    for (size_t i = 0; i < domain.ranges.size(); ++i) {
      const auto [first, end] = domain.ranges[i];
      if (i < f.axes.size()) {
        Require(f.axes[i].source_step == f.axes[i].target_step,
                "Matching coordinates require equal normalized progression steps");
        const auto source_first = f.source_domain->ranges[i].first;
        Require(i < p.digits.size(), "Matching map lost source axis");
        auto d = p.digits[i];
        d.first = Checked(static_cast<__int128>(d.first) - source_first + first);
        const auto step = domain.steps.empty() ? 1 : domain.steps[i];
        Require(d.first >= first && Last(d) < end && (d.first - first) % step == 0 &&
                    (d.count == 1 || d.step % step == 0),
                "Matching rank map escapes WAIT activity");
        out.push_back(d);
      } else {
        const auto step = domain.steps.empty() ? 1 : domain.steps[i];
        Require(end - first <= step, "Non-prefix varying matching coordinate");
        out.push_back({first, 1, 1});
      }
    }
    return out;
  }
  int64_t Ordinal(size_t region, const std::vector<Digit>& digits) const {
    const auto& frames = input.segments[region].nested_frames;
    Require(frames.size() == digits.size(), "Publication coordinate arity mismatch");
    int64_t value = 0;
    for (size_t i = 0; i < frames.size(); ++i)
      value = Checked(static_cast<__int128>(value) * frames[i].count + digits[i].first);
    return value;
  }
  bool Transition(const Publication& before, const Publication& after) {
    const auto& a = uses[before.use];
    const auto& b = uses[after.use];
    auto old = Target(before);
    const auto& next = after.digits;
    const auto& af = input.segments[a.target_region].nested_frames;
    const auto& bf = input.segments[b.source_region].nested_frames;
    ScopedEventTransition query{{a.target_region, Ordinal(a.target_region, old), a.target_site},
                                {b.source_region, Ordinal(b.source_region, next), b.source_site},
                                {}};
    size_t last = 0;
    for (size_t i = 0; i < old.size(); ++i)
      if (old[i].count > 1) last = std::max(last, i + 1);
    for (size_t i = 0; i < next.size(); ++i)
      if (next[i].count > 1) last = std::max(last, i + 1);
    for (size_t i = 0; i < last; ++i) {
      Require(i < old.size() && i < next.size() && af[i].loop == bf[i].loop &&
                  old[i].count == next[i].count && (old[i].count == 1 || old[i].step == next[i].step),
              "Reuse transition needs unsupported independent occurrence frames");
      int64_t as = 1, bs = 1;
      for (size_t j = i + 1; j < af.size(); ++j) as = Checked(static_cast<__int128>(as) * af[j].count);
      for (size_t j = i + 1; j < bf.size(); ++j) bs = Checked(static_cast<__int128>(bs) * bf[j].count);
      auto step = old[i].count == 1 ? 1 : old[i].step;
      query.axes.push_back({af[i].loop, old[i].count, Checked(static_cast<__int128>(as) * step),
                            Checked(static_cast<__int128>(bs) * step), step, step});
    }
    result.work.proof_axis_entries += query.axes.size();
    ++result.work.transition_queries;
    return order.ProveTransition(query);
  }
  bool Verify(const NodePtr& node) {
    for (const auto& child : node->children)
      if (!Verify(child)) return false;
    if (node->kind == Node::Sequence) {
      int64_t prefix = 0;
      for (const auto& child : node->children) {
        const auto end = Checked(static_cast<__int128>(prefix) + child->count);
        for (auto i = std::max<int64_t>(0, prefix - budget); i < prefix; ++i) {
          const auto target = Checked(static_cast<__int128>(i) + budget);
          if (target >= prefix && target < end)
            if (!Transition(At(node, i), At(node, target))) return false;
        }
        prefix = end;
      }
    } else if (node->kind == Node::Repeat && node->children.front()->count) {
      const auto& body = node->children.front();
      const auto p = body->count;
      for (auto r = std::max<int64_t>(0, p - budget); r < p; ++r) {
        const auto shifted = Checked(static_cast<__int128>(r) + budget);
        const auto d = shifted / p;
        if (d == 0 || d >= node->trips) continue;
        auto before = At(body, r), after = At(body, shifted % p);
        auto& a = before.digits[FrameIndex(before, node->frame)];
        auto& b = after.digits[FrameIndex(after, node->frame)];
        a.count = b.count = node->trips - d;
        b.first = Checked(b.first + static_cast<__int128>(d) * b.step);
        if (!Transition(before, after)) return false;
      }
    }
    return true;
  }
  void Pieces(const NodePtr& node, int64_t base, std::map<const ForStmt*, int64_t> coefficients) {
    if (node->kind == Node::Leaf) {
      ScopedPublicationRankPiece piece;
      piece.raw_use = node->use;
      piece.rank_base = base;
      piece.source = node->digits;
      piece.target = Target({node->use, node->digits});
      for (const auto& frame : input.segments[uses[node->use].source_region].nested_frames)
        piece.rank_coefficients.push_back(coefficients[frame.loop.get()]);
      state->pieces.push_back(std::move(piece));
      ++result.work.rank_pieces;
    } else if (node->kind == Node::Repeat) {
      coefficients[node->frame] = node->children.front()->count;
      Pieces(node->children.front(), base, std::move(coefficients));
    } else {
      for (const auto& child : node->children) {
        Pieces(child, base, coefficients);
        base = Checked(static_cast<__int128>(base) + child->count);
      }
    }
  }
  ScopedPaletteResult Build() {
    try {
      cuts.resize(input.segments.size());
      for (const auto& cut : order.Protocol()->Cuts())
        cuts[cut.region].insert(cuts[cut.region].end(), cut.after.begin(), cut.after.end());

      if (input.region_tree) Index(*input.region_tree);
      Require(state->source != state->target, "Palette requires a directional cross-pipe protocol");
      Require(input.region_tree.has_value(), "Publication ranking requires original structured schedule");
      for (size_t i = 0; i < uses.size(); ++i) {
        const auto& f = uses[i];
        if (input.segments[f.source_region].sites[f.source_site].pipe == state->source &&
            input.segments[f.target_region].sites[f.target_site].pipe == state->target) {
          Require(f.source_domain && f.target_domain, "Raw token use lacks exact endpoint domains");
          selected.push_back(i);
        }
      }
      state->root = Region(*input.region_tree, selected, {});
      Pieces(state->root, 0, {});
      std::vector<int64_t> volumes(uses.size());
      for (const auto& piece : state->pieces) {
        int64_t volume = 1;
        for (const auto& d : piece.source) volume = Checked(static_cast<__int128>(volume) * d.count);
        volumes[piece.raw_use] = Checked(static_cast<__int128>(volumes[piece.raw_use]) + volume);
      }
      for (auto id : selected)
        Require(volumes[id] == uses[id].count, "Publication partition lost raw use multiplicity");
      result.schedule =
          std::shared_ptr<const ScopedPublicationSchedule>(new ScopedPublicationSchedule(state));
      result.status = ScopedPaletteStatus::Constructed;
    } catch (const Refusal& refusal) {
      result.status = refusal.status;
      result.reason = refusal.reason;
    }
    return result;
  }
  ScopedPaletteResult Certify(std::shared_ptr<const ScopedPublicationSchedule> schedule) {
    try {
      Require(budget > 0, "Palette budget must be positive");
      Require(schedule && schedule->Protocol().get() == order.Protocol().get(),
              "Publication schedule belongs to a different immutable token protocol");
      result.schedule = std::move(schedule);
      if (!Verify(result.schedule->state_->root)) {
        result.status = ScopedPaletteStatus::UnprovedReuse;
        result.reason = "Actual-token order does not prove a complete same-lane transition";
      } else {
        result.status = ScopedPaletteStatus::Certified;
        auto certificate =
            std::shared_ptr<ScopedPaletteCertificate>(new ScopedPaletteCertificate(result.schedule, budget));
        result.certificate = std::move(certificate);
      }
    } catch (const Refusal& refusal) {
      result.status = refusal.status;
      result.reason = refusal.reason;
    }
    return result;
  }
};
ScopedPaletteResult BuildScopedPublicationSchedule(const ScopedTokenOrder& order, PipeType source,
                                                   PipeType target) {
  return ScopedPaletteBuilder(order, source, target, 0).Build();
}
ScopedPaletteResult CertifyScopedPalette(const ScopedTokenOrder& order,
                                         std::shared_ptr<const ScopedPublicationSchedule> schedule,
                                         int64_t budget) {
  const auto source = schedule ? schedule->SourcePipe() : static_cast<PipeType>(0);
  const auto target = schedule ? schedule->TargetPipe() : static_cast<PipeType>(0);
  return ScopedPaletteBuilder(order, source, target, budget).Certify(std::move(schedule));
}
}  // namespace pypto::ir::sync
