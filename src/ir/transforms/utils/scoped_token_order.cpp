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

// Order from supplied token uses, independently of storage-demand construction.
// Every selected edge is uniform at its named owner; exceptional uses remain
// emission obligations but are absent from this sufficient proof subgraph.
// Memoization belongs to each immutable body/seam, keyed by repetition count
// and batched relative ports. B concrete external endpoints propagate along
// O(B log K) binary paths. Interface construction and row work are separate;
// this is not a linear-in-source compiler bound. Port-set copying/comparison,
// per-leaf ancestor paths and interval indexing are charged separately. Summary
// counters count distinct nonempty nodes/rows, excluding repeated cache aliases.
// Exceptional token relations additionally supply one-edge native/token/native
// witnesses. Point requests cost O(F_pair log(records) * frame_depth); a uniform
// reuse certificate adds one such request per mixed-radix carry class and exact
// affine range checks. F_pair counts only families in the requested direction.
// No request enumerates dynamic trips. These witnesses are sufficient, not a
// complete periodic command-graph closure.
#include "src/ir/transforms/utils/scoped_token_order.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/numerical_sync_merge.h"
#include "src/ir/transforms/utils/rotating_sync_analysis.h"
#include "src/ir/transforms/utils/scoped_sync_palette.h"
#include "src/ir/transforms/utils/scoped_token_families.h"

namespace pypto::ir::sync {
namespace {
using Ordinal = NumericalOrdinal;
using Event = NumericalEvent;
using Ports = std::set<Event>;
using Node = std::shared_ptr<const NumericalNode>;
using Edge = std::pair<Ordinal, Ordinal>;  // C(source) -> I(target), local body ordinals.
Ordinal Product(Ordinal value, int64_t count) {
  Ordinal output;
  CHECK(value >= 0 && count >= 0 && !__builtin_mul_overflow(value, Ordinal(count), &output))
      << "Scoped token reference length overflows";
  return output;
}
struct Spec {
  enum Kind { Leaf, Join, Repeat } kind;
  size_t leaf = 0;
  ForStmtPtr frame;
  int64_t count = 1;
  Spec* left = nullptr;
  Spec* right = nullptr;
  Ordinal length = 0;
  std::array<std::optional<Ordinal>, kPipeCount> first{}, last{};
  std::array<Ordinal, kPipeCount> native_counts{};
  std::set<Edge> edges;
  std::array<std::vector<size_t>, kPipeCount> native_sites;
  // The owner itself is part of the cache identity: equal counts/ports in
  // distinct reset bodies do not share order or seam evidence.
  std::map<std::pair<int64_t, Ports>, Node> cache;
};
struct Path {
  Ordinal first = 0;
  std::vector<Spec*> ancestors;
  std::vector<std::pair<Spec*, Ordinal>> frames;
};
void NativePorts(Ports& ports, const std::array<std::optional<Ordinal>, kPipeCount>& first,
                 const std::array<std::optional<Ordinal>, kPipeCount>& last) {
  for (size_t p = 0; p < kPipeCount; ++p) {
    if (first[p]) {
      for (bool completion : {false, true}) {
        ports.emplace(*first[p], completion);
        ports.emplace(*last[p], completion);
      }
    }
  }
}
}  // namespace
class ScopedTokenOrder::State {
 public:
  std::shared_ptr<const ScopedSyncProtocol> protocol;
  const CompactSyncInput& input;
  std::optional<std::vector<PhysicalSyncFamily>> prepared;
  const std::vector<PhysicalSyncFamily>& families;
  std::vector<bool> uniform;
  size_t nodes = 0, entries = 0;
  mutable size_t query_steps = 0;
  mutable size_t witness_attempts = 0, candidate_visits = 0, binary_probes = 0;

  State(std::shared_ptr<const ScopedSyncProtocol> source,
        std::optional<std::vector<PhysicalSyncFamily>> supplied)
      : protocol(std::move(source)),
        input(protocol->Input()),
        prepared(std::move(supplied)),
        families(prepared ? *prepared : protocol->Families()),
        paths_(input.segments.size()),
        cores_(input.segments.size()) {
    protocol->ValidateFor(input);
    CHECK(input.region_tree) << "Scoped token order requires original source regions";
    std::vector<const ForStmt*> frames;
    root_ = Structure(*input.region_tree, frames);
    Path path;
    Index(root_, path);
    for (const auto& family : families) {
      CheckFamily(family);
      uniform.push_back(Install(family));
      const auto source = input.segments[family.source_region].sites[family.source_site].pipe;
      const auto target = input.segments[family.target_region].sites[family.target_site].pipe;
      directional_families_[source][target].push_back(uniform.size() - 1);
    }
    for (size_t r = 0; r < cores_.size(); ++r) {
      auto& core = cores_[r];
      core.count = 1;
      core.core_frontiers = true;
      for (const auto& site : input.segments[r].sites) core.pipes.push_back(site.pipe);
      const auto* leaf = paths_[r].ancestors.back();
      for (const auto& [a, b] : leaf->edges)
        core.generators.push_back({static_cast<size_t>(a), static_cast<size_t>(b), 0});
      BuildPeriodicCoreFrontiers(core);
    }
    order_ = Build(root_, root_->kind == Spec::Repeat ? root_->count : 1, {});
  }
  Ordinal Encode(const Occurrence& occurrence) const {
    const auto [r, visit, site] = occurrence;
    CHECK(r < input.segments.size() && visit >= 0 && visit < input.segments[r].count &&
          site < input.segments[r].sites.size())
        << "Invalid scoped token occurrence";
    const auto& path = paths_[r];
    auto output = NumericalAdd(path.first, site);
    auto value = visit;
    for (auto i = path.frames.rbegin(); i != path.frames.rend(); ++i) {
      CHECK(i->first->count > 0) << "Scoped occurrence belongs to an empty frame";
      output = NumericalAdd(output, Product(i->second, value % i->first->count));
      value /= i->first->count;
    }
    CHECK(!value) << "Scoped occurrence has an incompatible original frame";
    return output;
  }
  bool UniformQuery(const Occurrence& a, bool ac, const Occurrence& b, bool bc) const {
    NumericalOrderQuery query{
        [this](size_t r, Ordinal x, bool xc, Ordinal y, bool yc) { return LeafQuery(r, x, xc, y, yc); },
        query_steps};
    return QueryNumericalOrder(query, order_, Encode(a), ac, Encode(b), bc);
  }
  bool Query(const Occurrence& a, bool ac, const Occurrence& b, bool bc) const {
    if (UniformQuery(a, ac, b, bc)) return true;
    const auto [ar, av, as] = a;
    const auto [br, bv, bs] = b;
    const auto ap = input.segments[ar].sites[as].pipe, bp = input.segments[br].sites[bs].pipe;
    const auto first = Encode(a), last = Encode(b);
    ++witness_attempts;
    for (const auto index : directional_families_[ap][bp]) {
      ++candidate_visits;
      const auto& edge = families[index];
      const auto record = FirstSourceAtOrAfter(edge, first);
      if (record < edge.count && Encode(FamilyOccurrence(edge, record, true)) <= last) return true;
    }
    return false;
  }
  bool Reuse(size_t index) const { return ReuseFamily(families.at(index)); }
  bool ReuseFamily(const PhysicalSyncFamily& f) const {
    // Candidate membership/equivalence is a separate preparation obligation.
    // Validate its original-coordinate schema, but never add candidate edges
    // to the retained token graph to establish its own resource certificate.
    CheckFamily(f);
    // All proof edges are translation-invariant at their own named repeat.
    // Reference-forward paths between successive uses cannot leave their
    // contiguous repeat and return. Each mixed-radix carry class therefore has
    // one representative: lower digits last->first, carrying digit 0->1.
    // The obligation itself may have a clipped domain and be absent from the
    // proof graph. Its reuse witness still uses only uniformly installed edges;
    // CheckFamily fixes the same original common-prefix translation coordinates.
    int64_t suffix_target = 0;
    for (size_t i = f.axes.size(); i-- > 0;) {
      const auto& axis = f.axes[i];
      if (axis.count > 1) {
        const Occurrence previous{f.target_region, f.target_first + suffix_target, f.target_site};
        const Occurrence next{f.source_region, f.source_first + axis.source_stride, f.source_site};
        // The old representative proof only applies to unit-step translations.
        // Periodic domains instead need a witness for the entire carry class.
        if (!(UnitAxes(f) && UniformQuery(previous, true, next, false)) &&
            !TokenReuseWitness(f, i, previous, next))
          return false;
      }
      const __int128 total = __int128(suffix_target) + __int128(axis.count - 1) * axis.target_stride;
      CHECK(total <= std::numeric_limits<int64_t>::max()) << "Scoped reuse coordinate overflows";
      suffix_target = static_cast<int64_t>(total);
    }
    return f.count == 1 || !f.axes.empty();
  }
  bool Barrier(size_t index) const {
    const auto& f = families.at(index);
    const auto source = Encode({f.source_region, f.source_first, f.source_site});
    const auto target = Encode({f.target_region, f.target_first, f.target_site});
    const auto pipe = input.segments[f.target_region].sites[f.target_site].pipe;
    if (input.segments[f.source_region].sites[f.source_site].pipe != pipe ||
        NativeRank(root_, target, pipe) - NativeRank(root_, source, pipe) != 1)
      return false;
    // CheckFamily proves each axis stays inside its original digit without
    // carrying. Native rank is affine on that exact coordinate rectangle.
    // Equal coefficients preserve adjacency over every record, including phase
    // subsets absent from the uniform token proof graph.
    for (size_t i = 0; i < f.axes.size(); ++i) {
      if (f.axes[i].count == 1) continue;
      const auto* a = paths_[f.source_region].frames[i].first;
      const auto* b = paths_[f.target_region].frames[i].first;
      if (Product(a->left->native_counts[pipe], AxisStep(f, i, false)) !=
          Product(b->left->native_counts[pipe], AxisStep(f, i, true)))
        return false;
    }
    return true;
  }

 private:
  std::vector<std::unique_ptr<Spec>> specs_;
  std::vector<Path> paths_;
  std::vector<CompactSegmentResult> cores_;
  Spec* root_ = nullptr;
  Node order_;
  std::set<const NumericalNode*> counted_;
  std::array<std::array<std::vector<size_t>, kPipeCount>, kPipeCount> directional_families_;

 public:
  void ValidateFamilies(const std::vector<PhysicalSyncFamily>& supplied) const {
    CHECK(supplied.size() == families.size()) << "Protocol family inventory changed";
    for (size_t i = 0; i < families.size(); ++i) {
      const auto& a = families[i];
      const auto& b = supplied[i];
      CHECK(std::tie(a.source_region, a.source_site, a.target_region, a.target_site, a.source_first,
                     a.target_first, a.count) == std::tie(b.source_region, b.source_site, b.target_region,
                                                          b.target_site, b.source_first, b.target_first,
                                                          b.count) &&
            a.source_domain == b.source_domain && a.target_domain == b.target_domain &&
            a.protocol_uses == b.protocol_uses && b.ids.empty() && a.axes.size() == b.axes.size())
          << "Protocol handoff, domain or identity changed";
      for (size_t j = 0; j < a.axes.size(); ++j) {
        const auto& x = a.axes[j];
        const auto& y = b.axes[j];
        CHECK(x.frame == y.frame && x.count == y.count && x.source_stride == y.source_stride &&
              x.target_stride == y.target_stride && x.source_step == y.source_step &&
              x.target_step == y.target_step)
            << "Protocol matching coordinates changed";
      }
    }
  }

 private:
  int64_t AxisStep(const PhysicalSyncFamily& f, size_t i, bool target) const {
    const auto r = target ? f.target_region : f.source_region;
    auto stride = input.segments[r].count;
    for (size_t j = 0; j <= i; ++j) stride /= paths_[r].frames[j].first->count;
    const auto supplied = target ? f.axes[i].target_stride : f.axes[i].source_stride;
    CHECK(stride > 0 && supplied > 0 && supplied % stride == 0)
        << "Scoped token axis is not an original-coordinate progression";
    return supplied / stride;
  }
  bool UnitAxes(const PhysicalSyncFamily& f) const {
    for (size_t i = 0; i < f.axes.size(); ++i)
      if (AxisStep(f, i, false) != 1 || AxisStep(f, i, true) != 1) return false;
    return true;
  }
  Occurrence FamilyOccurrence(const PhysicalSyncFamily& f, int64_t record, bool target) const {
    const auto occurrence = DecodeSyncFamilyEndpoint(
        f, record, target ? SyncEndpointKind::Target : SyncEndpointKind::Source, SyncEmptyAxes::Singleton);
    CHECK(occurrence) << "Scoped token family record has incompatible coordinates";
    return *occurrence;
  }
  int64_t FirstSourceAtOrAfter(const PhysicalSyncFamily& edge, Ordinal position) const {
    int64_t lo = 0, hi = edge.count;
    while (lo < hi) {
      ++binary_probes;
      const auto mid = lo + (hi - lo) / 2;
      if (Encode(FamilyOccurrence(edge, mid, false)) < position)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }
  // Translate the historical same-family carry question into the same query
  // schema used by palette successors across different actual token records.
  bool TokenReuseWitness(const PhysicalSyncFamily& use, size_t carry, const Occurrence& previous,
                         const Occurrence& next) const {
    ScopedEventTransition transition{previous, next, {}};
    for (size_t j = 0; j <= carry; ++j) {
      const auto& axis = use.axes[j];
      transition.axes.push_back({axis.frame, axis.count - (j == carry), axis.target_stride,
                                 axis.source_stride, axis.target_step, axis.source_step});
    }
    return ProveTransition(transition);
  }

 public:
  bool ProveTransition(const ScopedEventTransition& transition) const {
    const auto [pr, pv, ps] = transition.consumer;
    const auto [nr, nv, ns] = transition.producer;
    PhysicalSyncFamily query{pr, ps, nr, ns, pv, nv, 1, {}};
    query.axes = transition.axes;
    __int128 count = 1;
    for (const auto& axis : query.axes) {
      CHECK(axis.count > 0 && axis.source_step > 0 && axis.target_step > 0)
          << "Invalid resource transition domain";
      count *= axis.count;
      CHECK(count <= std::numeric_limits<int64_t>::max()) << "Resource transition volume overflows";
    }
    query.count = static_cast<int64_t>(count);
    const bool forward = CheckFamily(query, false);
    for (size_t i = 0; i < query.axes.size(); ++i)
      CHECK(AxisStep(query, i, false) == query.axes[i].source_step &&
            AxisStep(query, i, true) == query.axes[i].target_step)
          << "Resource transition step differs from its original-coordinate map";
    if (!forward) return false;
    if (query.count == 1) return Query(transition.consumer, true, transition.producer, false);
    return NativeTokenWitness(query);
  }

 private:
  // Prove the complete query rectangle with a single actual token edge between
  // two native chains. Queries never become evidence in the retained graph.
  bool NativeTokenWitness(const PhysicalSyncFamily& query, size_t split_budget = 2) const {
    if (query.count == 1) {
      return Query({query.source_region, query.source_first, query.source_site}, true,
                   {query.target_region, query.target_first, query.target_site}, false);
    }
    const auto pr = query.source_region, ps = query.source_site;
    const auto nr = query.target_region, ns = query.target_site;
    const auto previous_pipe = input.segments[pr].sites[ps].pipe;
    const auto next_pipe = input.segments[nr].sites[ns].pipe;
    const auto previous_position = Encode({pr, query.source_first, ps});
    const auto next_position = Encode({nr, query.target_first, ns});
    ++witness_attempts;
    std::vector<int64_t> counts;
    std::vector<Ordinal> before_steps, after_steps;
    for (size_t j = 0; j < query.axes.size(); ++j) {
      counts.push_back(query.axes[j].count);
      before_steps.push_back(Product(paths_[pr].frames[j].second, AxisStep(query, j, false)));
      after_steps.push_back(Product(paths_[nr].frames[j].second, AxisStep(query, j, true)));
    }
    std::optional<std::pair<size_t, int64_t>> split_candidate;
    for (const auto index : directional_families_[previous_pipe][next_pipe]) {
      ++candidate_visits;
      const auto& edge = families[index];
      // Source occurrences are strictly increasing in mixed-radix record order.
      // Find the earliest edge that can follow this carry class's first consumer.
      const auto lo = FirstSourceAtOrAfter(edge, previous_position);
      if (lo == edge.count) continue;
      const auto source = Encode(FamilyOccurrence(edge, lo, false));
      const auto target = Encode(FamilyOccurrence(edge, lo, true));
      if (target > next_position) continue;
      std::vector<int64_t> digits(edge.axes.size());
      auto record = lo;
      for (size_t j = edge.axes.size(); j-- > 0;) {
        digits[j] = record % edge.axes[j].count;
        record /= edge.axes[j].count;
      }
      Ordinal before = source - previous_position, after = next_position - target;
      bool compatible = true;
      for (size_t j = 0; j < counts.size() && compatible; ++j) {
        if (counts[j] == 1) continue;
        size_t e = 0;
        while (e < edge.axes.size() && edge.axes[e].frame != query.axes[j].frame) ++e;
        if (e == edge.axes.size()) {
          compatible = false;
          break;
        }
        const auto step = AxisStep(query, j, false), edge_step = AxisStep(edge, e, false);
        if (step % edge_step) {
          if (split_budget > 0 && edge_step % step == 0 && edge_step / step <= 4 && counts[j] > 1) {
            const auto ratio = edge_step / step;
            if (!split_candidate || ratio < split_candidate->second) split_candidate = {j, ratio};
          }
          compatible = false;
          break;
        }
        const auto ratio = step / edge_step;
        if (Ordinal(digits[e]) + Product(ratio, counts[j] - 1) >= edge.axes[e].count) {
          compatible = false;
          break;
        }
        const auto source_step =
            Product(paths_[edge.source_region].frames[e].second, AxisStep(edge, e, false));
        const auto target_step =
            Product(paths_[edge.target_region].frames[e].second, AxisStep(edge, e, true));
        const auto before_delta = Product(source_step, ratio) - before_steps[j];
        const auto after_delta = after_steps[j] - Product(target_step, ratio);
        if (before_delta < 0) before -= Product(-before_delta, counts[j] - 1);
        if (after_delta < 0) after -= Product(-after_delta, counts[j] - 1);
        if (before < 0 || after < 0) compatible = false;
      }
      if (compatible) return true;
    }
    if (split_candidate && split_budget > 0) {
      const auto [j, k] = *split_candidate;
      for (int64_t r = 0; r < k && r < counts[j]; ++r) {
        PhysicalSyncFamily sub = query;
        const int64_t sub_count = (counts[j] - r + k - 1) / k;
        sub.source_first += r * query.axes[j].source_stride;
        sub.target_first += r * query.axes[j].target_stride;
        sub.axes[j].count = sub_count;
        sub.axes[j].source_stride *= k;
        sub.axes[j].target_stride *= k;
        sub.axes[j].source_step *= k;
        sub.axes[j].target_step *= k;
        sub.count = (query.count / counts[j]) * sub_count;
        if (!NativeTokenWitness(sub, split_budget - 1)) return false;
      }
      return true;
    }
    return false;
  }

  Spec* New(Spec::Kind kind) {
    auto item = std::make_unique<Spec>();
    item->kind = kind;
    auto* pointer = item.get();
    specs_.push_back(std::move(item));
    return pointer;
  }
  Spec* Join(Spec* left, Spec* right) {
    auto* out = New(Spec::Join);
    out->left = left;
    out->right = right;
    out->length = NumericalAdd(left->length, right->length);
    for (size_t p = 0; p < kPipeCount; ++p) {
      out->native_counts[p] = NumericalAdd(left->native_counts[p], right->native_counts[p]);
      out->first[p] = left->first[p];
      if (!out->first[p] && right->first[p]) out->first[p] = NumericalAdd(left->length, *right->first[p]);
      out->last[p] = right->last[p] ? std::optional<Ordinal>(NumericalAdd(left->length, *right->last[p]))
                                    : left->last[p];
    }
    return out;
  }
  Spec* Repeat(Spec* body, const SyncLoopFrame& frame) {
    auto* out = New(Spec::Repeat);
    out->left = body;
    out->frame = frame.loop;
    out->count = frame.count;
    out->length = Product(body->length, frame.count);
    if (frame.count) {
      for (size_t p = 0; p < kPipeCount; ++p) {
        out->native_counts[p] = Product(body->native_counts[p], frame.count);
        out->first[p] = body->first[p];
        if (body->last[p])
          out->last[p] = NumericalAdd(Product(body->length, frame.count - 1), *body->last[p]);
      }
    }
    return out;
  }
  Spec* Sequence(const std::vector<Spec*>& children, size_t begin, size_t end) {
    if (begin == end) return New(Spec::Join);  // Empty source interval.
    if (end == begin + 1) return children[begin];
    const auto middle = begin + (end - begin) / 2;
    return Join(Sequence(children, begin, middle), Sequence(children, middle, end));
  }
  Spec* Structure(const SyncRegion& region, std::vector<const ForStmt*>& frames) {
    if (region.leaf) {
      const auto r = *region.leaf;
      CHECK(r < input.segments.size() && region.children.empty() && !region.repetition)
          << "Invalid scoped token source leaf";
      const auto& segment = input.segments[r];
      CHECK(segment.nested_frames.size() >= frames.size()) << "Scoped token leaf lost original frames";
      for (size_t i = 0; i < frames.size(); ++i)
        CHECK(segment.nested_frames[i].loop.get() == frames[i]) << "Scoped token source frame mismatch";
      auto* out = New(Spec::Leaf);
      out->leaf = r;
      out->length = segment.sites.size();
      for (size_t s = 0; s < segment.sites.size(); ++s) {
        const auto pipe = segment.sites[s].pipe;
        if (!out->first[pipe]) out->first[pipe] = s;
        out->last[pipe] = s;
        out->native_sites[pipe].push_back(s);
        ++out->native_counts[pipe];
      }
      for (size_t i = segment.nested_frames.size(); i-- > frames.size();)
        out = Repeat(out, segment.nested_frames[i]);
      return out;
    }
    if (region.repetition) frames.push_back(region.repetition->loop.get());
    std::vector<Spec*> children;
    for (const auto& child : region.children) children.push_back(Structure(child, frames));
    auto* out = Sequence(children, 0, children.size());
    if (region.repetition) {
      frames.pop_back();
      out = Repeat(out, *region.repetition);
    }
    return out;
  }
  void Index(Spec* spec, Path& path) {
    path.ancestors.push_back(spec);
    if (spec->kind == Spec::Leaf) {
      CHECK(paths_[spec->leaf].ancestors.empty()) << "Duplicate scoped token source leaf";
      const auto& original = input.segments[spec->leaf].nested_frames;
      CHECK(original.size() == path.frames.size()) << "Scoped token coordinate frame arity mismatch";
      for (size_t i = 0; i < original.size(); ++i)
        CHECK(original[i].loop == path.frames[i].first->frame &&
              original[i].count == path.frames[i].first->count)
            << "Scoped token coordinate frame identity mismatch";
      paths_[spec->leaf] = path;
    } else if (spec->kind == Spec::Repeat) {
      path.frames.emplace_back(spec, spec->left->length);
      Index(spec->left, path);
      path.frames.pop_back();
    } else if (spec->left) {
      Index(spec->left, path);
      const auto first = path.first;
      path.first = NumericalAdd(first, spec->left->length);
      Index(spec->right, path);
      path.first = first;
    }
    path.ancestors.pop_back();
  }
  std::vector<int64_t> Digits(size_t r, int64_t visit) const {
    std::vector<int64_t> out(paths_[r].frames.size());
    for (size_t i = out.size(); i-- > 0;) {
      const auto count = paths_[r].frames[i].first->count;
      CHECK(count > 0) << "Scoped token family has an empty source frame";
      out[i] = visit % count;
      visit /= count;
    }
    CHECK(!visit) << "Scoped token visit exceeds original frame inventory";
    return out;
  }
  bool CheckFamily(const PhysicalSyncFamily& f, bool require_forward = true) const {
    const auto a = Encode({f.source_region, f.source_first, f.source_site});
    const auto b = Encode({f.target_region, f.target_first, f.target_site});
    Ordinal minimum = b - a;
    __int128 volume = 1;
    const auto& sp = paths_[f.source_region];
    const auto& tp = paths_[f.target_region];
    const auto sd = Digits(f.source_region, f.source_first), td = Digits(f.target_region, f.target_first);
    CHECK(f.axes.size() <= std::min(sp.frames.size(), tp.frames.size())) << "Scoped token axes exceed frames";
    int64_t ss = input.segments[f.source_region].count, ts = input.segments[f.target_region].count;
    for (size_t i = 0; i < f.axes.size(); ++i) {
      const auto& axis = f.axes[i];
      ss /= sp.frames[i].first->count;
      ts /= tp.frames[i].first->count;
      CHECK(axis.frame == sp.frames[i].first->frame && axis.frame == tp.frames[i].first->frame &&
            axis.count > 0 && ss > 0 && ts > 0 && axis.source_stride > 0 && axis.target_stride > 0 &&
            axis.source_stride % ss == 0 && axis.target_stride % ts == 0 &&
            axis.source_stride / ss == axis.target_stride / ts &&
            Ordinal(sd[i]) + Product(axis.source_stride / ss, axis.count - 1) < sp.frames[i].first->count &&
            Ordinal(td[i]) + Product(axis.target_stride / ts, axis.count - 1) < tp.frames[i].first->count)
          << "Scoped token family has incompatible axes";
      volume *= axis.count;
      CHECK(volume <= std::numeric_limits<int64_t>::max()) << "Scoped token family volume overflows";
      const auto delta = Product(tp.frames[i].second, axis.target_stride / ts) -
                         Product(sp.frames[i].second, axis.source_stride / ss);
      if (delta < 0) minimum += Product(-delta, axis.count - 1) * -1;
    }
    CHECK(volume == f.count) << "Scoped token family volume differs from its coordinate map";
    CHECK(!require_forward || minimum > 0) << "Scoped token family is not uniformly reference-forward";
    return minimum > 0;
  }
  // The source tree's repeat ancestors are the exact translation group at
  // the edge's owner. Full ancestor domains are required, except at one
  // adjacent-repeat seam. Fixed inner suffixes remain fixed body endpoints.
  bool Install(const PhysicalSyncFamily& f) {
    if (!UnitAxes(f)) return false;
    const auto& sp = paths_[f.source_region];
    const auto& tp = paths_[f.target_region];
    const auto sd = Digits(f.source_region, f.source_first), td = Digits(f.target_region, f.target_first);
    size_t common = 0;
    while (common < std::min(sp.frames.size(), tp.frames.size()) &&
           sp.frames[common].first == tp.frames[common].first)
      ++common;
    Spec* seam = nullptr;
    for (size_t i = 0; i < common; ++i) {
      const auto count = sp.frames[i].first->count;
      const auto axis_count = i < f.axes.size() ? f.axes[i].count : 1;
      if (seam) {
        if (axis_count != 1) return false;
      } else if (sd[i] == 0 && td[i] == 0 && axis_count == count) {
        continue;
      } else if (sd[i] == 0 && td[i] == 1 && axis_count == count - 1) {
        seam = sp.frames[i].first;
      } else {
        return false;
      }
    }
    size_t depth = 0;
    while (depth < std::min(sp.ancestors.size(), tp.ancestors.size()) &&
           sp.ancestors[depth] == tp.ancestors[depth])
      ++depth;
    CHECK(depth) << "Scoped token endpoints have no common original region";
    auto* owner = seam ? seam : sp.ancestors[depth - 1];
    auto relative = [&](const Path& path, const std::vector<int64_t>& digits, size_t site) {
      Ordinal value = site;
      bool below = false;
      size_t frame = 0;
      for (size_t i = 0; i < path.ancestors.size(); ++i) {
        const auto* current = path.ancestors[i];
        if (current == owner) below = true;
        if (current->kind == Spec::Repeat) {
          if (below && current != seam)
            value = NumericalAdd(value, Product(current->left->length, digits[frame]));
          ++frame;
        } else if (below && current->kind == Spec::Join && i + 1 < path.ancestors.size() &&
                   current->right == path.ancestors[i + 1]) {
          value = NumericalAdd(value, current->left->length);
        }
      }
      return value;
    };
    const auto source = relative(sp, sd, f.source_site), target = relative(tp, td, f.target_site);
    CHECK(seam || source < target) << "Scoped token body edge is not reference-forward";
    owner->edges.emplace(source, target);
    return true;
  }
  bool LeafQuery(size_t r, Ordinal a, bool ac, Ordinal b, bool bc) const {
    if (a == b && ac == bc) return true;
    return cores_[r].Precedes(0, static_cast<size_t>(a), ac, 0, static_cast<size_t>(b), bc);
  }
  Node Leaf(Spec* spec) {
    auto node = std::make_shared<NumericalNode>();
    node->length = spec->length;
    node->leaf = spec->leaf;
    node->first = spec->first;
    node->last = spec->last;
    const auto& sites = input.segments[spec->leaf].sites;
    for (size_t s = 0; s < sites.size(); ++s) {
      node->native_sites[sites[s].pipe].push_back(s);
      for (size_t c = 0; c < 2; ++c) node->chains[2 * sites[s].pipe + c].push_back(s);
    }
    IndexNumericalPositions(*node);
    for (size_t c = 0; c < kNumericalChains; ++c) {
      node->forward[c].resize(node->chains[c].size());
      for (size_t d = 0; d < kNumericalChains; ++d) {
        size_t target = 0;
        for (size_t source = 0; source < node->chains[c].size(); ++source) {
          while (target < node->chains[d].size() &&
                 !LeafQuery(spec->leaf, node->chains[c][source], c % 2, node->chains[d][target], d % 2))
            ++target;
          node->forward[c][source][d] = target;
        }
      }
    }
    InvertNumericalRows(*node);
    return node;
  }
  Node Merge(Node left, Node right, const Ports& ports, const std::vector<Edge>& edges) {
    auto node = std::make_shared<NumericalNode>();
    node->left = left;
    node->right = right;
    node->length = NumericalAdd(left->length, right->length);
    std::vector<NumericalCrossing> crossings;
    for (size_t p = 0; p < kPipeCount; ++p) {
      node->first[p] = left->first[p];
      if (!node->first[p] && right->first[p]) node->first[p] = NumericalAdd(left->length, *right->first[p]);
      node->last[p] = right->last[p] ? std::optional<Ordinal>(NumericalAdd(left->length, *right->last[p]))
                                     : left->last[p];
      if (left->last[p] && right->first[p])
        for (bool completion : {false, true})
          crossings.push_back({left->positions.at({*left->last[p], completion}),
                               right->positions.at({*right->first[p], completion}), true});
    }
    for (const auto& [a, b] : edges)
      crossings.push_back(
          {left->positions.at({a, true}), right->positions.at({b - left->length, false}), false});
    BuildNumericalMergeOrder(node, ports, std::move(crossings));
    return node;
  }
  Node Build(Spec* spec, int64_t count, Ports ports) {
    const auto length = spec->kind == Spec::Repeat ? Product(spec->left->length, count) : spec->length;
    if (!length) return std::make_shared<NumericalNode>();
    auto first = spec->first, last = spec->last;
    if (spec->kind == Spec::Repeat) {
      for (size_t p = 0; p < kPipeCount; ++p)
        if (last[p]) last[p] = NumericalAdd(Product(spec->left->length, count - 1), *spec->left->last[p]);
    }
    NativePorts(ports, first, last);
    // Static leaf interfaces already contain every site; do not retain several
    // identical tables merely because callers requested different subsets.
    if (spec->kind == Spec::Leaf) {
      for (size_t s = 0; s < input.segments[spec->leaf].sites.size(); ++s) {
        ports.emplace(s, false);
        ports.emplace(s, true);
      }
    }
    const auto key = std::make_pair(count, ports);
    const auto cached = spec->cache.find(key);
    if (cached != spec->cache.end()) return cached->second;
    Node result;
    if (spec->kind == Spec::Leaf) {
      result = Leaf(spec);
    } else if (spec->kind == Spec::Repeat && count == 1) {
      result = Build(spec->left, spec->left->kind == Spec::Repeat ? spec->left->count : 1, ports);
    } else {
      const auto left_count = spec->kind == Spec::Repeat ? count / 2 : 1;
      const auto split =
          spec->kind == Spec::Repeat ? Product(spec->left->length, left_count) : spec->left->length;
      std::vector<Edge> edges;
      for (const auto& [a, b] : spec->edges)
        edges.emplace_back(spec->kind == Spec::Repeat ? NumericalAdd(split - spec->left->length, a) : a,
                           spec->kind == Spec::Repeat ? NumericalAdd(split, b) : b);
      Ports lp, rp;
      auto place = [&](Event event) {
        CHECK(event.first >= 0 && event.first < length) << "Scoped token order port exceeds its owner";
        if (event.first < split)
          lp.insert(event);
        else
          rp.emplace(event.first - split, event.second);
      };
      for (auto port : ports) place(port);
      for (const auto& [a, b] : edges) {
        CHECK(a < split && b >= split) << "Scoped token edge crosses the wrong original boundary";
        place({a, true});
        place({b, false});
      }
      auto left =
          Build(spec->kind == Spec::Repeat ? spec : spec->left,
                spec->kind == Spec::Repeat ? left_count
                                           : (spec->left->kind == Spec::Repeat ? spec->left->count : 1),
                lp);
      auto right =
          Build(spec->kind == Spec::Repeat ? spec : spec->right,
                spec->kind == Spec::Repeat ? count - left_count
                                           : (spec->right->kind == Spec::Repeat ? spec->right->count : 1),
                rp);
      result = Merge(left, right, ports, edges);
    }
    if (counted_.insert(result.get()).second) {
      ++nodes;
      for (const auto& chain : result->chains) entries += chain.size() * kNumericalChains * 2;
    }
    spec->cache.emplace(key, result);
    return result;
  }
  Ordinal NativeRank(const Spec* spec, Ordinal target, PipeType pipe) const {
    if (spec->kind == Spec::Leaf) {
      const auto& sites = spec->native_sites[pipe];
      const auto at = std::lower_bound(sites.begin(), sites.end(), target);
      return static_cast<Ordinal>(at - sites.begin());
    }
    if (spec->kind == Spec::Repeat) {
      const auto length = spec->left->length;
      const auto visit = target / length;
      CHECK(visit <= std::numeric_limits<int64_t>::max()) << "Scoped native rank visit overflows";
      return NumericalAdd(Product(spec->left->native_counts[pipe], static_cast<int64_t>(visit)),
                          NativeRank(spec->left, target % length, pipe));
    }
    if (target < spec->left->length) return NativeRank(spec->left, target, pipe);
    return NumericalAdd(spec->left->native_counts[pipe],
                        NativeRank(spec->right, target - spec->left->length, pipe));
  }
};
ScopedTokenOrder::ScopedTokenOrder(std::shared_ptr<const ScopedSyncProtocol> protocol)
    : ScopedTokenOrder(std::move(protocol), std::nullopt) {}
ScopedTokenOrder::ScopedTokenOrder(std::shared_ptr<const ScopedSyncProtocol> protocol,
                                   std::optional<std::vector<PhysicalSyncFamily>> prepared)
    : state_(std::make_unique<State>(std::move(protocol), std::move(prepared))) {}
ScopedTokenOrder::~ScopedTokenOrder() = default;
const CompactSyncInput& ScopedTokenOrder::Input() const { return state_->input; }
const std::shared_ptr<const ScopedSyncProtocol>& ScopedTokenOrder::Protocol() const {
  return state_->protocol;
}
const std::vector<PhysicalSyncFamily>& ScopedTokenOrder::Families() const { return state_->families; }
bool ScopedTokenOrder::Precedes(const Occurrence& a, bool ac, const Occurrence& b, bool bc) const {
  return state_->Query(a, ac, b, bc);
}
bool ScopedTokenOrder::Uniform(size_t i) const { return state_->uniform.at(i); }
bool ScopedTokenOrder::ProveTransition(const ScopedEventTransition& transition) const {
  return state_->ProveTransition(transition);
}
bool ScopedTokenOrder::CanReuseFamily(const PhysicalSyncFamily& family) const {
  return state_->ReuseFamily(family);
}
size_t ScopedTokenOrder::SummaryNodes() const { return state_->nodes; }
size_t ScopedTokenOrder::InterfaceEntries() const { return state_->entries; }
size_t ScopedTokenOrder::WitnessAttempts() const { return state_->witness_attempts; }
size_t ScopedTokenOrder::CandidateVisits() const { return state_->candidate_visits; }
size_t ScopedTokenOrder::BinaryProbes() const { return state_->binary_probes; }
DirectSyncOrder ScopedTokenOrder::View() const {
  return {[this](const Occurrence& a, const Occurrence& b) { return state_->Query(a, true, b, false); },
          [this](const Occurrence& a, const Occurrence& b) { return state_->Encode(a) < state_->Encode(b); },
          [](size_t, size_t, size_t) -> std::optional<__int128> { return std::nullopt; },
          [this](size_t i) { return state_->Reuse(i); },
          [this](size_t i) { return state_->Barrier(i); },
          false,
          state_->protocol,
          [this](const std::vector<PhysicalSyncFamily>& families) { state_->ValidateFamilies(families); },
          [this](PipeType source, PipeType target, int64_t capacity) {
            auto result = BuildScopedPublicationSchedule(*this, source, target);
            if (!result.schedule) return result;
            const auto schedule = result.schedule;
            auto work = result.work;
            size_t attempts = 0;
            result.status = ScopedPaletteStatus::UnprovedReuse;
            result.reason = "No eligible directional IDs supplied";
            for (int64_t budget = 1; budget <= capacity; ++budget) {
              result = CertifyScopedPalette(*this, schedule, budget);
              ++attempts;
              work.transition_queries += result.work.transition_queries;
              work.boundary_node_visits += result.work.boundary_node_visits;
              work.proof_axis_entries += result.work.proof_axis_entries;
              if (result.certificate) break;
            }
            result.work = work;
            result.attempted_budgets = attempts;
            return result;
          }};
}
std::unique_ptr<ScopedTokenOrder> PrepareScopedTokenOrder(std::shared_ptr<const ScopedSyncProtocol> protocol,
                                                          ScopedTokenWork* work) {
  auto original = std::make_unique<ScopedTokenOrder>(std::move(protocol));
  std::unique_ptr<ScopedTokenOrder> accepted;
  const ScopedTokenOrder* active = original.get();
  for (;;) {
    auto normalized = NormalizeScopedTokenFamilies(
        active->Input(), active->Families(),
        [&](const PhysicalSyncFamily& candidate) { return active->CanReuseFamily(candidate); }, work);
    if (normalized.size() == active->Families().size()) break;
    auto next =
        std::unique_ptr<ScopedTokenOrder>(new ScopedTokenOrder(original->Protocol(), std::move(normalized)));
    // Normalization preserves actual uses, but this sufficient proof graph may
    // change. Never replace an available preparation with an unproved one.
    bool valid = true;
    for (const auto& family : next->Families()) {
      const auto direction = SyncFamilyPipes(next->Input(), family);
      if (direction.first != direction.second && family.count > 1 && !next->CanReuseFamily(family)) {
        valid = false;
        break;
      }
    }
    if (!valid) break;
    accepted = std::move(next);
    active = accepted.get();
  }
  return accepted ? std::move(accepted) : std::move(original);
}
}  // namespace pypto::ir::sync
