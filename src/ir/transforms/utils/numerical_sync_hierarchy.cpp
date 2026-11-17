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

#include "src/ir/transforms/utils/numerical_sync_hierarchy.h"

#include <algorithm>
#include <chrono>
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
#include "src/ir/transforms/utils/numerical_sync_merge.h"
#include "src/ir/transforms/utils/numerical_sync_phase.h"
#include "src/ir/transforms/utils/rotating_sync_analysis.h"
namespace pypto::ir::sync {
namespace {
using Ordinal = NumericalOrdinal;
using Node = NumericalNode;
using NodePtr = std::shared_ptr<const Node>;
using Row = NumericalRow;
Ordinal Length(int64_t count, size_t sites) { return Ordinal(count) * sites; }
int64_t Scope(const SyncSegment& segment) {
  if (!segment.original_loop) return 1;
  return segment.nested_frames.empty() ? segment.count : segment.nested_frames.back().count;
}
int64_t Repetitions(const SyncSegment& segment) {
  Ordinal repetitions = 1;
  for (size_t i = 0; i + 1 < segment.nested_frames.size(); ++i) {
    repetitions *= segment.nested_frames[i].count;
    CHECK(repetitions <= std::numeric_limits<int64_t>::max()) << "Numerical repetition count overflow";
  }
  return static_cast<int64_t>(repetitions);
}
int64_t Mod(Ordinal value, int64_t modulus) {
  const auto residue = value % modulus;
  return static_cast<int64_t>(residue < 0 ? residue + modulus : residue);
}
NumericalBoundary LeafBoundary(const CompactSyncView& result, size_t leaf, const NumericalCell& cell) {
  const auto selected = RotatingBoundary(result.State().input, result.State().segments[leaf], cell.storage,
                                         cell.slot, cell.row, cell.column, leaf);
  const size_t sites = result.State().input.segments[leaf].sites.size();
  auto ordinal = [&](const std::optional<Occurrence>& occurrence) -> std::optional<Ordinal> {
    if (!occurrence) return std::nullopt;
    return Length(std::get<1>(*occurrence), sites) + std::get<2>(*occurrence);
  };
  NumericalBoundary out;
  out.first_writer = ordinal(selected.first_writer);
  out.last_writer = ordinal(selected.last_writer);
  for (size_t p = 0; p < kPipeCount; ++p) {
    out.first_readers[p] = ordinal(selected.first_readers[p]);
    out.last_readers[p] = ordinal(selected.last_readers[p]);
  }
  return out;
}
// Check a complete domain schedule before replacing explicit cells by finite
// crossing witnesses. Every writing phase must fill [0, rows*row_bytes) once;
// intervening finite readers cover every row. Other domains are untouched.
std::optional<CollectiveSyncDomain> CollectiveDomain(const CompactSyncInput& input, size_t storage,
                                                     CompactSyncBuilder& work) {
  const auto& domain = input.storage[storage];
  if (!domain.byte_coordinates || domain.slot_count != 1) return std::nullopt;
  CollectiveSyncDomain proof;
  std::optional<PipeType> fill_pipe;
  std::vector<std::pair<int64_t, int64_t>> reads;
  std::set<int64_t> witnesses;
  std::set<size_t> starts, ends, touched;
  size_t last_reader = 0;
  auto covered = [&]() {
    if (reads.empty()) return false;
    std::sort(reads.begin(), reads.end());
    int64_t end = 0;
    for (const auto& interval : reads) {
      if (interval.first > end) return false;
      end = std::max(end, interval.second);
    }
    return end == domain.columns;
  };
  for (size_t leaf = 0; leaf < input.segments.size(); ++leaf) {
    const auto& segment = input.segments[leaf];
    const auto count = Scope(segment);
    std::vector<std::pair<const SyncSite*, const SyncAccess*>> effects;
    size_t writes = 0;
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses) {
        ++work.Diagnostics().collective_access_visits;
        if (access.storage == storage) {
          effects.emplace_back(&site, &access);
          writes += access.write;
        }
      }
    if (effects.empty()) continue;
    touched.insert(leaf);
    if (writes) {
      if (effects.size() != 1 || writes != 1 || segment.sites.size() != 1 || count <= 0) return std::nullopt;
      const auto& [site, pointer] = effects.front();
      const auto& access = *pointer;
      if (!access.exact_footprint || access.after_first || access.row || access.rows != 1 || access.column ||
          access.row_step || access.column_step != access.columns || access.columns <= 0 ||
          Ordinal(access.columns) * count != domain.columns ||
          access.coordinate_frame != segment.original_loop)
        return std::nullopt;
      if (fill_pipe && (*fill_pipe != site->pipe || proof.row_bytes != access.columns || !covered()))
        return std::nullopt;
      if (fill_pipe) ends.insert(last_reader);
      starts.insert(leaf);
      fill_pipe = site->pipe;
      proof.row_bytes = access.columns;
      proof.rows = count;
      proof.fill_leaves.push_back(leaf);
      reads.clear();
    } else {
      if (!fill_pipe || count != 1) return std::nullopt;
      for (const auto& [site, pointer] : effects) {
        const auto& access = *pointer;
        if (site->pipe == *fill_pipe || !access.exact_footprint || access.after_first || access.row ||
            access.rows != 1 || access.row_step || access.column_step || access.column < 0 ||
            access.columns <= 0 || access.column % proof.row_bytes || access.columns % proof.row_bytes ||
            Ordinal(access.column) + access.columns > domain.columns)
          return std::nullopt;
        const auto end = access.column + access.columns;
        reads.emplace_back(access.column, end);
        witnesses.insert(access.column);
        witnesses.insert(end - proof.row_bytes);
      }
      last_reader = leaf;
    }
  }
  if (!fill_pipe || !covered()) return std::nullopt;
  ends.insert(last_reader);
  if (input.region_tree) {
    // Every source repeat touching this domain must contain whole phases.
    // The original row-fill loop is already a leaf; no repeated ancestor is
    // exempt. Even a zero/one-count partial phase cannot borrow the proof.
    bool qualified = true;
    using Range = std::optional<std::pair<size_t, size_t>>;
    std::function<Range(const SyncRegion&)> check = [&](const SyncRegion& region) -> Range {
      if (region.leaf)
        return touched.count(*region.leaf) ? Range{{*region.leaf, *region.leaf}} : std::nullopt;
      Range range;
      for (const auto& child : region.children) {
        const auto part = check(child);
        if (!part) continue;
        if (!range) range = part;
        range->second = part->second;
      }
      // starts/ends delimit phases whose ENTIRE intervening reader coverage
      // was checked above. Source intervals are contiguous original regions.
      if (range && region.repetition && (!starts.count(range->first) || !ends.count(range->second)))
        qualified = false;
      return range;
    };
    check(*input.region_tree);
    if (!qualified) return std::nullopt;
  }
  witnesses.insert(0);
  witnesses.insert(domain.columns - proof.row_bytes);
  proof.witness_columns.assign(witnesses.begin(), witnesses.end());
  return proof;
}
std::vector<NumericalCell> InterfaceCells(CompactSyncBuilder& result, NumericalSyncHierarchy& hierarchy) {
  const auto& input = result.State().input;
  std::vector<bool> written(input.storage.size());
  using Effect = std::pair<const SyncSegment*, const SyncAccess*>;
  std::map<size_t, std::vector<Effect>> effects;
  for (const auto& segment : input.segments)
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses) {
        ++result.Diagnostics().interface_access_visits;
        if (access.write) written.at(access.storage) = true;
        effects[access.storage].emplace_back(&segment, &access);
      }
  for (const auto& [storage, domain_effects] : effects) {
    if (!written[storage]) continue;
    if (auto proof = CollectiveDomain(input, storage, result)) {
      ++result.Diagnostics().collective_domains;
      result.Diagnostics().collective_witnesses += proof->witness_columns.size();
      hierarchy.collective_domains.emplace(storage, std::move(*proof));
    }
  }
  std::map<size_t, size_t> private_owner;
  for (size_t region = 0; region < result.State().segments.size(); ++region)
    for (const auto& [storage, certificate] : result.State().segments[region].visit_owned) {
      if (hierarchy.collective_domains.count(storage)) continue;
      const auto [owner, inserted] = private_owner.emplace(storage, region);
      CHECK(inserted || owner->second == region)
          << "Visit-owned storage has multiple region owners; cross-reset selector export unavailable";
    }
  std::vector<NumericalCell> cells;
  for (const auto& [storage, domain_effects] : effects) {
    if (const auto proof = hierarchy.collective_domains.find(storage);
        proof != hierarchy.collective_domains.end()) {
      for (auto column : proof->second.witness_columns) cells.push_back({storage, 0, 0, column});
      continue;
    }
    if (const auto owner = private_owner.find(storage); owner != private_owner.end()) {
      const auto& region = input.segments[owner->second];
      const bool own_frame =
          region.nested_frames.size() == 1 && region.nested_frames.front().loop == region.original_loop &&
          region.nested_frames.front().count == region.count && region.reset_period == region.count;
      // A leaf's own ordinal frame is not an enclosing replay. No ancestor,
      // including a singleton ancestor, may borrow this visit-local proof.
      CHECK(region.nested_frames.empty() || own_frame)
          << "Visit-owned storage cannot be omitted across repeated or reset invocations";
      for (const auto& [segment, access] : domain_effects) {
        ++result.Diagnostics().interface_access_visits;
        CHECK(segment == &region)
            << "Visit-owned storage has an outside read or write; crossing selector export unavailable";
        CHECK((!access->row_step && !access->column_step) || access->coordinate_frame == region.original_loop)
            << "Visit-owned translation belongs to another occurrence frame";
      }
      ++result.Diagnostics().omitted_bridge_domains;
      continue;
    }
    // The shared alias-qualified physical inventory proves that no region
    // writes this domain. Its point selectors remain available on demand.
    if (!written[storage]) continue;
    std::set<int64_t> rows, columns, slots;
    for (const auto& [segment, pointer] : domain_effects) {
      ++result.Diagnostics().interface_access_visits;
      const auto& access = *pointer;
      if (!segment->count) continue;
      const auto count = Scope(*segment);
      CHECK(!access.row_step && !access.column_step)
          << "Moving geometry sharing a written regional domain lacks a complete selector partition";
      rows.insert(access.row);
      rows.insert(access.row + access.rows);
      columns.insert(access.column);
      columns.insert(access.column + access.columns);
      const int64_t banks = input.storage[storage].slot_count;
      const int64_t period = banks / std::gcd(access.slot_step, banks);
      // Explicit complete-interface output work. Enumerate visited physical
      // slots only; never the trip count beyond one orbit or a joint period.
      for (int64_t visit = 0; visit < std::min(count, period); ++visit) {
        ++result.Diagnostics().interface_slot_visits;
        slots.insert(Mod(Ordinal(access.slot_first) + Ordinal(visit) * access.slot_step, banks));
      }
    }
    std::vector<int64_t> r(rows.begin(), rows.end()), c(columns.begin(), columns.end());
    for (int64_t slot : slots)
      for (size_t i = 1; i < r.size(); ++i)
        for (size_t j = 1; j < c.size(); ++j) cells.push_back({storage, slot, r[i - 1], c[j - 1]});
  }
  result.Diagnostics().interface_cells = cells.size();
  return cells;
}
bool LeafQuery(const CompactSyncView& result, size_t leaf, Ordinal a, bool sc, Ordinal b, bool tc) {
  if (a == b && sc == tc) return true;
  if (result.State().hierarchy) {
    const auto provider = result.State().hierarchy->finite_providers.find(leaf);
    if (provider != result.State().hierarchy->finite_providers.end())
      return provider->second.order.Precedes(static_cast<size_t>(a), sc, static_cast<size_t>(b), tc);
  }
  const auto sites = result.State().input.segments[leaf].sites.size();
  INTERNAL_CHECK(sites) << "Nonempty numerical leaf has no sites";
  return result.State().segments[leaf].Precedes(
      static_cast<int64_t>(a / sites), static_cast<size_t>(a % sites), sc, static_cast<int64_t>(b / sites),
      static_cast<size_t>(b % sites), tc);
}
NodePtr BuildLeaf(CompactSyncBuilder& result, const std::vector<NumericalCell>& cells, size_t leaf) {
  auto node = std::make_shared<Node>();
  node->leaf = leaf;
  const auto& input = result.State().input.segments[leaf];
  const auto count = result.State().segments[leaf].internal_scope_count;
  node->length = Length(count, input.sites.size());
  for (size_t c = 0; c < cells.size(); ++c) {
    if (result.State().segments[leaf].rotating) {
      node->storage.push_back(LeafBoundary(result, leaf, cells[c]));
    } else {
      // Composition consumes exact conflict-class boundaries. A moving read
      // can intersect a whole-write class without touching its representative
      // point; point selectors below must NOT replace this class summary.
      const auto boundary = result.State().segments[leaf].body.Boundary(c);
      INTERNAL_CHECK(boundary) << "Invariant leaf omitted a supplied class";
      NumericalBoundary out;
      auto at = [&](const std::optional<size_t>& site, bool last) -> std::optional<Ordinal> {
        return site && count
                   ? std::optional<Ordinal>(Length(last ? count - 1 : 0, input.sites.size()) + *site)
                   : std::nullopt;
      };
      out.first_writer = at(boundary->first_writer, false);
      out.last_writer = at(boundary->last_writer, true);
      for (size_t p = 0; p < kPipeCount; ++p) {
        out.first_readers[p] = at(boundary->first_readers[p], false);
        out.last_readers[p] = at(boundary->last_readers[p], true);
      }
      node->storage.push_back(out);
    }
  }
  if (count) {
    for (size_t s = 0; s < input.sites.size(); ++s) {
      const auto p = input.sites[s].pipe;
      node->native_sites[p].push_back(s);
      if (!node->first[p]) node->first[p] = s;
      node->last[p] = Length(count - 1, input.sites.size()) + s;
    }
  }
  const auto matrix = static_cast<size_t>(PipeType::M);
  if (node->first[matrix]) {
    const auto first_site = static_cast<size_t>(*node->first[matrix] % input.sites.size());
    const auto last_site = static_cast<size_t>(*node->last[matrix] % input.sites.size());
    const auto& first = input.sites[first_site];
    if (!first.initializes_accumulator) node->first_m_continuation = first.protected_accumulator;
    node->last_m_qualified = input.sites[last_site].protected_accumulator;
  }
  for (const auto& event : NumericalPorts(*node, result.Diagnostics().port_normalization_visits)) {
    const auto site = static_cast<size_t>(event.first % input.sites.size());
    const size_t chain = 2 * static_cast<size_t>(input.sites[site].pipe) + event.second;
    node->chains[chain].push_back(event.first);
  }
  IndexNumericalPositions(*node);
  for (size_t c = 0; c < kNumericalChains; ++c) {
    node->forward[c].resize(node->chains[c].size());
    for (size_t d = 0; d < kNumericalChains; ++d) {
      size_t target = 0;
      for (size_t source = 0; source < node->chains[c].size(); ++source) {
        while (target < node->chains[d].size()) {
          ++result.Diagnostics().interface_queries;
          if (LeafQuery(result, leaf, node->chains[c][source], c % 2, node->chains[d][target], d % 2)) break;
          ++target;
        }
        node->forward[c][source][d] = target;
      }
    }
  }
  InvertNumericalRows(*node);
  CountNumericalInterface(*node, result);
  return node;
}
NodePtr Repeat(NodePtr body, int64_t count, CompactSyncBuilder& result, NodePtr* seam = nullptr) {
  NodePtr output = std::make_shared<Node>();
  if (!count || !body->length) return output;
  while (count) {
    if (count & 1) output = MergeNumericalNodes(output, body, result);
    count /= 2;
    if (count) {
      body = MergeNumericalNodes(body, body, result);
      if (seam && !*seam) *seam = body;
    }
  }
  return output;
}
NodePtr Compose(const std::vector<NodePtr>& nodes, size_t begin, size_t end, const std::string& order,
                CompactSyncBuilder& result) {
  if (begin == end) return std::make_shared<Node>();
  if (end - begin == 1) return nodes[begin];
  if (order == "left") {
    auto node = nodes[begin];
    for (size_t i = begin + 1; i < end; ++i) node = MergeNumericalNodes(node, nodes[i], result);
    return node;
  }
  if (order == "right") {
    auto node = nodes[end - 1];
    for (size_t i = end - 1; i-- > begin;) node = MergeNumericalNodes(nodes[i], node, result);
    return node;
  }
  const size_t middle = begin + (end - begin) / 2;
  return MergeNumericalNodes(Compose(nodes, begin, middle, order, result),
                             Compose(nodes, middle, end, order, result), result);
}
NumericalSourceLayout BuildSourceLayout(const SyncRegion& source, CompactSyncBuilder& result,
                                        std::set<size_t>& leaves) {
  ++result.Diagnostics().source_region_nodes;
  NumericalSourceLayout out;
  if (source.leaf) {
    CHECK(source.children.empty() && !source.repetition && *source.leaf < result.State().segments.size())
        << "Invalid source region leaf";
    CHECK(leaves.insert(*source.leaf).second) << "Source leaf has more than one original occurrence path";
    out.leaf = source.leaf;
    out.length = out.body_length = Length(result.State().segments[*source.leaf].internal_scope_count,
                                          result.State().input.segments[*source.leaf].sites.size());
  } else {
    if (source.repetition) {
      CHECK(source.repetition->count >= 0) << "Negative source repetition count";
      out.count = source.repetition->count;
      out.frame = source.repetition->loop;
    }
    out.offsets.push_back(0);
    for (const auto& child : source.children) {
      out.children.push_back(BuildSourceLayout(child, result, leaves));
      out.body_length = NumericalAdd(out.body_length, out.children.back().length);
      out.offsets.push_back(out.body_length);
    }
    CHECK(!__builtin_mul_overflow(out.body_length, Ordinal(out.count), &out.length))
        << "Source repetition length exceeds wide range";
  }
  return out;
}
void IndexSourcePaths(const NumericalSourceLayout& source, NumericalOccurrencePath& path,
                      NumericalSyncHierarchy& hierarchy, CompactSyncBuilder& result) {
  if (source.leaf) {
    const auto leaf = *source.leaf;
    path.frames.emplace_back(result.State().segments[leaf].internal_scope_count,
                             result.State().input.segments[leaf].sites.size());
    Ordinal visits = 1;
    for (const auto& [count, stride] : path.frames) {
      CHECK(!__builtin_mul_overflow(visits, Ordinal(count), &visits)) << "Source visit count overflow";
    }
    CHECK(visits == result.State().input.segments[leaf].count)
        << "Source path disagrees with leaf visit inventory";
    result.Diagnostics().source_coordinate_entries += path.frames.size();
    hierarchy.source_paths.at(leaf) = path;
    path.frames.pop_back();
    return;
  }
  const auto first = path.first;
  path.frames.emplace_back(source.count, source.body_length);
  for (size_t i = 0; i < source.children.size(); ++i) {
    path.first = NumericalAdd(first, source.offsets[i]);
    IndexSourcePaths(source.children[i], path, hierarchy, result);
  }
  path.frames.pop_back();
  path.first = first;
}
NodePtr BuildSourceNode(const SyncRegion& source, NumericalSourceLayout& layout,
                        const std::vector<NodePtr>& leaves, const std::string& order,
                        CompactSyncBuilder& result) {
  if (source.leaf) {
    layout.body = layout.summary = leaves.at(*source.leaf);
    return layout.summary;
  }
  std::vector<NodePtr> children;
  for (size_t i = 0; i < source.children.size(); ++i)
    children.push_back(BuildSourceNode(source.children[i], layout.children[i], leaves, order, result));
  auto body = Compose(children, 0, children.size(), order, result);
  layout.body = body;
  if (source.startup && source.repetition && source.repetition->count > 1 && body->length) {
    const auto [leaf, site] = *source.startup;
    const auto& initializer = result.State().input.segments[leaf].sites[site];
    // First-touch qualification proves that the first RMW read has no internal
    // predecessor, so the rows/covers are identical. Only incoming footprint
    // protection differs. Keep the startup node immutable, including when the
    // enclosing invocation repeats or a summary is shared by binary doubling.
    // An empty initializer leaf cannot establish a continuation entry.
    if (result.State().segments[leaf].internal_scope_count) {
      auto continuation = std::make_shared<Node>(*body);
      continuation->first_m_continuation = initializer.protected_accumulator;
      auto tail = Repeat(continuation, source.repetition->count - 1, result, &layout.seam);
      layout.summary = MergeNumericalNodes(body, tail, result);
      // With only two visits the final merge is the seam. Otherwise the
      // continuation double already supplies it: startup and continuation
      // have identical internal rows, exit evidence and storage boundaries.
      // Only their incoming first-M qualification differs, irrelevant on the
      // seam's left side. The right side is continuation in either case.
      if (!layout.seam) layout.seam = layout.summary;
      return layout.summary;
    }
  }
  layout.summary = source.repetition ? Repeat(body, source.repetition->count, result, &layout.seam) : body;
  return layout.summary;
}
Occurrence DecodeSource(const CompactSyncView& result, Ordinal ordinal) {
  const auto* node = &*result.State().hierarchy->source_layout;
  CHECK(ordinal >= 0 && ordinal < node->length) << "Invalid original source occurrence ordinal";
  Ordinal visit = 0;
  while (!node->leaf) {
    INTERNAL_CHECK(node->body_length) << "Nonempty source use has an empty body";
    visit = visit * node->count + ordinal / node->body_length;
    ordinal %= node->body_length;
    const auto upper = std::upper_bound(node->offsets.begin(), node->offsets.end(), ordinal);
    INTERNAL_CHECK(upper != node->offsets.begin() && upper != node->offsets.end())
        << "Invalid source child position";
    const auto index = static_cast<size_t>(upper - node->offsets.begin() - 1);
    ordinal -= node->offsets[index];
    node = &node->children[index];
  }
  const auto leaf = *node->leaf;
  const auto sites = result.State().input.segments[leaf].sites.size();
  visit = visit * result.State().segments[leaf].internal_scope_count + ordinal / sites;
  CHECK(visit < result.State().input.segments[leaf].count) << "Decoded source visit outside original frame";
  return {leaf, static_cast<int64_t>(visit), static_cast<size_t>(ordinal % sites)};
}
Occurrence Decode(const CompactSyncView& result, Ordinal ordinal) {
  if (result.State().hierarchy->source_layout) return DecodeSource(result, ordinal);
  const auto& offsets = result.State().hierarchy->segment_offsets;
  const auto upper = std::upper_bound(offsets.begin(), offsets.end(), ordinal);
  CHECK(upper != offsets.begin() && upper != offsets.end()) << "Invalid numerical occurrence ordinal";
  const size_t region = static_cast<size_t>(upper - offsets.begin() - 1);
  const auto& segment = result.State().input.segments[region];
  const auto relative = ordinal - offsets[region];
  INTERNAL_CHECK(!segment.sites.empty()) << "Nonempty numerical region has no sites";
  return {region, static_cast<int64_t>(relative / segment.sites.size()),
          static_cast<size_t>(relative % segment.sites.size())};
}
Ordinal Encode(const CompactSyncView& result, const Occurrence& occurrence) {
  const auto [region, visit, site] = occurrence;
  CHECK(region < result.State().input.segments.size()) << "Invalid numerical region";
  const auto& segment = result.State().input.segments[region];
  CHECK(visit >= 0 && visit < segment.count && site < segment.sites.size())
      << "Invalid numerical synchronization occurrence";
  if (result.State().hierarchy->source_layout) {
    const auto& path = result.State().hierarchy->source_paths[region];
    auto ordinal = NumericalAdd(path.first, site);
    auto remaining = visit;
    for (auto frame = path.frames.rbegin(); frame != path.frames.rend(); ++frame) {
      CHECK(frame->first > 0) << "Occurrence in an empty original frame";
      ordinal = NumericalAdd(ordinal, Ordinal(remaining % frame->first) * frame->second);
      remaining /= frame->first;
    }
    CHECK(!remaining) << "Occurrence exceeds original coordinate arity";
    return ordinal;
  }
  return NumericalAdd(result.State().hierarchy->segment_offsets[region],
                      Length(visit, segment.sites.size()) + site);
}

int64_t PhaseInteger(Ordinal value) {
  CHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max()) << "Phase coordinate overflow";
  return static_cast<int64_t>(value);
}
NodePtr BuildPhaseSource(const SyncRegion& source, NumericalSourceLayout& layout,
                         const std::map<size_t, NodePtr>& providers, const std::string& order,
                         CompactSyncBuilder& result, NumericalSyncHierarchy& hierarchy) {
  if (source.leaf) {
    layout.body = layout.summary = providers.at(hierarchy.finite_source_providers.at(*source.leaf));
    return layout.summary;
  }
  if (source.repetition) {
    auto found = hierarchy.phase_schedules.find(source.repetition->loop.get());
    if (found != hierarchy.phase_schedules.end()) {
      auto& phase = found->second;
      auto compose = [&](const std::vector<size_t>& ids) {
        std::vector<NodePtr> nodes;
        for (auto id : ids) nodes.push_back(providers.at(id));
        return Compose(nodes, 0, nodes.size(), order, result);
      };
      phase.prefix = compose(phase.prefix_providers);
      phase.cycle = compose(phase.cycle_providers);
      phase.tail = compose(phase.tail_providers);
      phase.repeated = Repeat(phase.cycle, phase.cycles, result, &phase.seam);
      layout.summary =
          MergeNumericalNodes(MergeNumericalNodes(phase.prefix, phase.repeated, result), phase.tail, result);
      INTERNAL_CHECK(layout.summary->length == layout.length) << "Phase summary lost original occurrences";
      return layout.summary;
    }
  }
  std::vector<NodePtr> children;
  for (size_t i = 0; i < source.children.size(); ++i)
    children.push_back(
        BuildPhaseSource(source.children[i], layout.children[i], providers, order, result, hierarchy));
  layout.body = Compose(children, 0, children.size(), order, result);
  layout.summary =
      source.repetition ? Repeat(layout.body, source.repetition->count, result, &layout.seam) : layout.body;
  return layout.summary;
}
CompactSyncBuilder AnalyzePhasedSource(CompactSyncInput input, const std::string& order) {
  const auto begin = std::chrono::steady_clock::now();
  CompactSyncBuilder result;
  result.State().input = std::move(input);
  IndexSyncStorageAccesses(result);
  auto hierarchy = std::make_shared<NumericalSyncHierarchy>();
  result.State().hierarchy = hierarchy;
  for (size_t storage = 0; storage < result.State().input.storage.size(); ++storage) {
    if (result.State().input.storage[storage].space != MemorySpace::DDR) continue;
    const auto& directory = result.State().storage_access_index[storage];
    if (!directory.written) {
      hierarchy->phase_readonly_domains.insert(storage);
      ++result.Diagnostics().omitted_bridge_domains;
    } else {
      bool external = false;
      for (const auto& [r, s, a] : directory.accesses)
        external |= static_cast<bool>(result.State().input.segments[r].sites[s].accesses[a].external);
      if (external) {
        CHECK(directory.accesses.size() == 1)
            << "External output ownership requires one complete write family and no other accesses";
        const auto [r, s, a] = directory.accesses.front();
        const auto& segment = result.State().input.segments[r];
        const auto& access = segment.sites[s].accesses[a];
        CHECK(access.write && access.exact_footprint && !access.after_first && access.external &&
              (access.external->row_major_tiles || access.external->swizzled_tiles || segment.count <= 1))
            << "External output has no exact injective tile ownership certificate";
        if (segment.count <= 1) {
          hierarchy->phase_owned_outputs.insert(storage);
          ++result.Diagnostics().omitted_bridge_domains;
          continue;
        }
        const auto owner_frame = access.external->row_major_tiles
                                     ? access.external->row_major_tiles->frame
                                     : access.external->swizzled_tiles->frame;
        const auto owner_row_scale = access.external->row_major_tiles
                                         ? access.external->row_major_tiles->row_scale
                                         : access.external->swizzled_tiles->row_scale;
        const auto owner_column_scale = access.external->row_major_tiles
                                            ? access.external->row_major_tiles->column_scale
                                            : access.external->swizzled_tiles->column_scale;
        bool found = false;
        for (const auto& frame : segment.nested_frames) {
          if (frame.loop == owner_frame) {
            CHECK(!found) << "Duplicate output tile coordinate frame";
            found = true;
          } else {
            CHECK(frame.count <= 1)
                << "External tile ownership does not cover repeated enclosing or inner visits";
          }
        }
        CHECK(found && access.rows == owner_row_scale && access.columns == owner_column_scale)
            << "External output ownership has incompatible frame or tile extents";
        hierarchy->phase_owned_outputs.insert(storage);
        ++result.Diagnostics().omitted_bridge_domains;
      } else {
        for (const auto& [r, s, a] : directory.accesses) {
          const auto& access = result.State().input.segments[r].sites[s].accesses[a];
          CHECK(!access.row_step && !access.column_step)
              << "Phased moving GM writes need a scoped external ownership certificate";
        }
      }
    }
  }
  hierarchy->segment_offsets.push_back(0);
  for (const auto& source : result.State().input.segments) {
    CompactSegmentResult descriptor;
    descriptor.count = source.count;
    descriptor.internal_scope_count = Scope(source);
    for (const auto& site : source.sites) descriptor.pipes.push_back(site.pipe);
    result.State().segments.push_back(std::move(descriptor));
    hierarchy->segment_offsets.push_back(
        NumericalAdd(hierarchy->segment_offsets.back(), Length(source.count, source.sites.size())));
  }
  std::set<size_t> leaves;
  hierarchy->source_layout = BuildSourceLayout(*result.State().input.region_tree, result, leaves);
  hierarchy->source_paths.resize(result.State().input.segments.size());
  NumericalOccurrencePath path;
  IndexSourcePaths(*hierarchy->source_layout, path, *hierarchy, result);
  PlanNumericalPhases(*result.State().input.region_tree, result, *hierarchy);
  BuildNumericalPhaseCells(result, *hierarchy);
  const auto partitioned = std::chrono::steady_clock::now();
  result.Diagnostics().partition_seconds = std::chrono::duration<double>(partitioned - begin).count();
  const auto providers = BuildNumericalPhaseProviders(result, *hierarchy);
  const auto reduced = std::chrono::steady_clock::now();
  result.Diagnostics().reduction_seconds = std::chrono::duration<double>(reduced - partitioned).count();
  hierarchy->root = BuildPhaseSource(*result.State().input.region_tree, *hierarchy->source_layout, providers,
                                     order, result, *hierarchy);
  result.Diagnostics().composition_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - reduced).count();
  return result;
}
// Export only retained merge records: leaf demand families remain compact.
void ExportNodeCrossings(const CompactSyncView& result, const NodePtr& node, Ordinal offset,
                         std::vector<Crossing>& out) {
  if (!node->length || !node->left) return;
  ExportNodeCrossings(result, node->left, offset, out);
  const auto right_offset = NumericalAdd(offset, node->left->length);
  ExportNodeCrossings(result, node->right, right_offset, out);
  for (const auto& crossing : node->crossings) {
    if (crossing.native) continue;
    const auto a = NumericalAdd(offset, node->left->chains[crossing.source.first][crossing.source.second]);
    const auto b =
        NumericalAdd(right_offset, node->right->chains[crossing.target.first][crossing.target.second]);
    out.emplace_back(Decode(result, a), Decode(result, b));
  }
}
// Export the unshared SOURCE tree once. A record's Cartesian axes name original
// loops; per-leaf strides encode the different numbers of inner visits on each
// side. Never walk every occurrence of a shared binary-repeat summary.
class ScopedExporter {
  struct Axis {
    ForStmtPtr frame;
    int64_t count;
    int64_t step = 1;
  };
  const CompactSyncView& result_;
  std::shared_ptr<ScopedSyncDemands> output_ = std::make_shared<ScopedSyncDemands>();
  std::vector<std::map<const ForStmt*, int64_t>> strides_;
  std::vector<Axis> axes_;

  void Record(Ordinal source, Ordinal target) {
    ScopedDemandFamily family{DecodeSource(result_, source), DecodeSource(result_, target), {}};
    const auto sr = std::get<0>(family.source), tr = std::get<0>(family.target);
    for (const auto& axis : axes_) {
      INTERNAL_CHECK(axis.count > 0) << "Internal error: empty scoped demand domain";
      family.axes.push_back(
          {axis.frame, axis.count, PhaseInteger(Ordinal(strides_.at(sr).at(axis.frame.get())) * axis.step),
           PhaseInteger(Ordinal(strides_.at(tr).at(axis.frame.get())) * axis.step), axis.step, axis.step});
      ++output_->coordinate_entries;
    }
    // Both maps must be injective. Otherwise publication multiplicity needs a
    // different matching contract, not a first-label or first-endpoint shortcut.
    Ordinal source_span = 0, target_span = 0;
    for (auto axis = family.axes.rbegin(); axis != family.axes.rend(); ++axis) {
      INTERNAL_CHECK(axis->source_stride > source_span && axis->target_stride > target_span)
          << "Internal error: noninjective scoped endpoint map";
      source_span += Ordinal(axis->count - 1) * axis->source_stride;
      target_span += Ordinal(axis->count - 1) * axis->target_stride;
    }
    INTERNAL_CHECK(
        Ordinal(std::get<1>(family.source)) + source_span < result_.State().input.segments[sr].count &&
        Ordinal(std::get<1>(family.target)) + target_span < result_.State().input.segments[tr].count)
        << "Internal error: scoped endpoint exceeds original invocation domain";
    output_->families.push_back(std::move(family));
  }
  void Crossings(const NodePtr& node, Ordinal offset) {
    if (!node->left) return;
    ++output_->visited_merge_nodes;
    const auto right = NumericalAdd(offset, node->left->length);
    for (const auto& crossing : node->crossings) {
      if (crossing.native) continue;
      Record(NumericalAdd(offset, node->left->chains[crossing.source.first][crossing.source.second]),
             NumericalAdd(right, node->right->chains[crossing.target.first][crossing.target.second]));
    }
  }
  void Sequence(const NodePtr& node, Ordinal offset, const std::set<const Node*>& children) {
    if (!node->length || children.count(node.get()) || !node->left) return;
    Sequence(node->left, offset, children);
    Sequence(node->right, NumericalAdd(offset, node->left->length), children);
    Crossings(node, offset);
  }
  void Finite(const NodePtr& node, Ordinal offset) {
    if (!node || !node->length) return;
    if (!node->left) {
      const auto provider = result_.State().hierarchy->finite_providers.find(node->leaf);
      INTERNAL_CHECK(provider != result_.State().hierarchy->finite_providers.end())
          << "Missing finite phase provider";
      for (const auto& [a, b] : provider->second.order.demands)
        Record(NumericalAdd(offset, a), NumericalAdd(offset, b));
      return;
    }
    Finite(node->left, offset);
    Finite(node->right, NumericalAdd(offset, node->left->length));
    Crossings(node, offset);
  }
  void Visit(const NumericalSourceLayout& source, Ordinal offset) {
    ++output_->visited_source_nodes;
    if (!source.length) return;
    if (source.frame) {
      const auto found = result_.State().hierarchy->phase_schedules.find(source.frame.get());
      if (found != result_.State().hierarchy->phase_schedules.end()) {
        const auto& phase = found->second;
        Finite(phase.prefix, offset);
        const auto cycles = NumericalAdd(offset, phase.prefix->length);
        if (phase.cycles) {
          axes_.push_back({source.frame, phase.cycles, phase.period});
          Finite(phase.cycle, cycles);
          axes_.pop_back();
        }
        if (phase.cycles > 1) {
          axes_.push_back({source.frame, phase.cycles - 1, phase.period});
          Crossings(phase.seam, cycles);
          axes_.pop_back();
        }
        Finite(phase.tail, NumericalAdd(cycles, phase.repeated->length));
        Sequence(source.summary, offset, {phase.prefix.get(), phase.repeated.get(), phase.tail.get()});
        return;
      }
    }
    if (source.leaf) {
      const auto leaf = *source.leaf;
      if (result_.State().hierarchy->finite_source_providers.count(leaf)) {
        Finite(source.summary, offset);
        return;
      }
      const auto& segment = result_.State().input.segments[leaf];
      const auto& analysis = result_.State().segments[leaf];
      for (const auto& demand : analysis.demands) {
        const auto count = analysis.internal_scope_count - demand.distance;
        if (count <= 0) continue;
        if (segment.original_loop) axes_.push_back({segment.original_loop, count});
        Record(NumericalAdd(offset, demand.source),
               NumericalAdd(offset, Length(demand.distance, segment.sites.size()) + demand.target));
        if (segment.original_loop) axes_.pop_back();
      }
      return;
    }
    if (source.frame) axes_.push_back({source.frame, source.count});
    std::set<const Node*> child_summaries;
    for (size_t i = 0; i < source.children.size(); ++i) {
      child_summaries.insert(source.children[i].summary.get());
      Visit(source.children[i], NumericalAdd(offset, source.offsets[i]));
    }
    // Use complete retained sequence merges. An untouched/empty sibling can
    // carry obligations from a more distant preceding child.
    Sequence(source.body, offset, child_summaries);
    if (source.frame) axes_.pop_back();
    if (source.count > 1) {
      INTERNAL_CHECK(source.frame && source.seam) << "Internal error: missing retained repeat seam";
      axes_.push_back({source.frame, source.count - 1});
      Crossings(source.seam, offset);
      axes_.pop_back();
    }
  }

 public:
  explicit ScopedExporter(const CompactSyncView& result) : result_(result) {
    for (const auto& segment : result.State().input.segments) {
      strides_.emplace_back();
      Ordinal stride = 1;
      for (auto frame = segment.nested_frames.rbegin(); frame != segment.nested_frames.rend(); ++frame) {
        INTERNAL_CHECK(stride <= std::numeric_limits<int64_t>::max())
            << "Internal error: source coordinate stride overflow";
        strides_.back().emplace(frame->loop.get(), static_cast<int64_t>(stride));
        stride *= frame->count;
      }
    }
  }
  std::shared_ptr<const ScopedSyncDemands> Run() {
    Visit(*result_.State().hierarchy->source_layout, 0);
    return output_;
  }
};
}  // namespace

CompactSyncBuilder AnalyzeNumericalSync(CompactSyncInput input, const std::string& order) {
  const bool structured = input.region_tree.has_value();
  if (structured) {
    if (input.segments.empty() &&
        std::any_of(input.storage.begin(), input.storage.end(),
                    [](const auto& s) { return s.slot_count > 1; }))
      return AnalyzePhasedSource(std::move(input), order);
    for (const auto& segment : input.segments)
      for (const auto& site : segment.sites)
        for (const auto& access : site.accesses)
          if (access.slot_coordinate_frame && access.slot_coordinate_frame != segment.original_loop)
            return AnalyzePhasedSource(std::move(input), order);
  }
  for (const auto& segment : input.segments)
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses)
        CHECK(!access.external) << "External coordinate geometry requires a qualified regional consumer";
  bool moving_write = false, slot_access = false;
  for (const auto& segment : input.segments)
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses) {
        slot_access |= input.storage[access.storage].slot_count > 1;
        moving_write |= (input.storage[access.storage].byte_coordinates ||
                         input.storage[access.storage].space == MemorySpace::DDR) &&
                        access.write && (access.row_step || access.column_step);
      }
  // Local rotating leaves retain their existing periodic analysis even when
  // wrapped in structured repetition. The rotating reducer still qualifies
  // every effect and refuses an unbound ancestor slot frame. Moving DDR domains
  // use its visit-owned proof; route selection supplies neither certificate.
  const bool invariant_leaves = structured && !moving_write && !slot_access;
  CompactSyncBuilder result;
  if (invariant_leaves) {
    // The initial structured adapter accepts invariant conflict classes only.
    // Slot geometry still needs its own scoped interface proof.
    for (const auto& storage : input.storage)
      CHECK(storage.slot_count == 1) << "Structured rotating source export is not yet qualified";
    result = AnalyzeInvariantSyncLeaves(std::move(input));
  } else {
    result.State().input = std::move(input);
    IndexSyncStorageAccesses(result);
    for (size_t r = 0; r < result.State().input.segments.size(); ++r) {
      const auto& original = result.State().input.segments[r];
      result.State().segments.push_back(
          AnalyzeRotatingSegment(result.State().input, r, Scope(original), result));
      // SourceLayout owns structured repetition; do not duplicate visits in
      // leaf mathematics or mistake a finite reader for its enclosing loop.
      result.State().segments.back().repetitions = structured ? 1 : Repetitions(original);
    }
  }
  auto hierarchy = std::make_shared<NumericalSyncHierarchy>();
  hierarchy->segment_offsets.push_back(0);
  for (const auto& original : result.State().input.segments)
    hierarchy->segment_offsets.push_back(
        NumericalAdd(hierarchy->segment_offsets.back(), Length(original.count, original.sites.size())));
  const auto start = std::chrono::steady_clock::now();
  if (invariant_leaves) {
    for (const auto& cell : result.State().storage_classes)
      hierarchy->cells.push_back({cell.storage, 0, cell.row, cell.column});
    result.Diagnostics().interface_cells = hierarchy->cells.size();
  } else {
    hierarchy->cells = InterfaceCells(result, *hierarchy);
  }
  std::vector<NodePtr> nodes;
  for (size_t r = 0; r < result.State().segments.size(); ++r) {
    auto leaf = BuildLeaf(result, hierarchy->cells, r);
    nodes.push_back(Repeat(leaf, result.State().segments[r].repetitions, result));
  }
  if (structured) {
    std::set<size_t> leaves;
    hierarchy->source_layout = BuildSourceLayout(*result.State().input.region_tree, result, leaves);
    CHECK(leaves.size() == result.State().segments.size()) << "Source tree omitted retained leaves";
    hierarchy->source_paths.resize(result.State().segments.size());
    NumericalOccurrencePath path;
    IndexSourcePaths(*hierarchy->source_layout, path, *hierarchy, result);
    hierarchy->root =
        BuildSourceNode(*result.State().input.region_tree, *hierarchy->source_layout, nodes, order, result);
  } else {
    hierarchy->root = Compose(nodes, 0, nodes.size(), order, result);
  }
  INTERNAL_CHECK(hierarchy->root->length == hierarchy->segment_offsets.back())
      << "Numerical repetition lost original occurrence coordinates";
  result.State().hierarchy = hierarchy;
  // Crossings remain owned by the hierarchy. A missing flat export is not an
  // empty answer; clients can audit bounded expansion without reanalysis.
  result.Diagnostics().composition_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}
bool NumericalReferenceBefore(const CompactSyncView& result, const Occurrence& a, const Occurrence& b) {
  return Encode(result, a) < Encode(result, b);
}
bool NumericalPrecedes(const CompactSyncView& result, const Occurrence& a, bool sc, const Occurrence& b,
                       bool tc) {
  const auto source = Encode(result, a), target = Encode(result, b);
  if (source > target || (source == target && sc == tc)) return false;
  const NumericalOrderQuery query{[&](size_t leaf, Ordinal a, bool ac, Ordinal b, bool bc) {
                                    return LeafQuery(result, leaf, a, ac, b, bc);
                                  },
                                  result.Diagnostics().hierarchy_query_steps};
  return QueryNumericalOrder(query, result.State().hierarchy->root, source, sc, target, tc);
}
Occurrence DecodeNumericalOccurrence(const CompactSyncView& result, __int128 ordinal) {
  return Decode(result, ordinal);
}
std::optional<std::vector<Crossing>> ExportNumericalCrossings(const CompactSyncView& result) {
  if (result.State().input.region_tree) return std::nullopt;
  // Check the complete scope before exposing any records. A repeated DAG use
  // cannot be represented by the ordinary single-interval family coordinates.
  for (size_t r = 0; r < result.State().segments.size(); ++r) {
    const auto& segment = result.State().segments[r];
    if (segment.repetitions != 1 || segment.count != result.State().input.segments[r].count)
      return std::nullopt;
  }
  if (!result.State().hierarchy) return result.State().crossing_demands;
  std::vector<Crossing> out;
  ExportNodeCrossings(result, result.State().hierarchy->root, 0, out);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}
std::shared_ptr<const ScopedSyncDemands> ExportNumericalScopedDemands(const CompactSyncView& result) {
  if (!result.State().hierarchy || !result.State().hierarchy->source_layout) return nullptr;
  return ScopedExporter(result).Run();
}
}  // namespace pypto::ir::sync
