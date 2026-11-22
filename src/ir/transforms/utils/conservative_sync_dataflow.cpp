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

#include "src/ir/transforms/utils/conservative_sync_dataflow.h"

#include <algorithm>
#include <array>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/buffer_runtime_valid_analysis.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/utils/sync_footprint_resolver.h"
#include "src/ir/transforms/utils/sync_loop_trip_proof.h"
#include "src/ir/transforms/utils/sync_scalar_readiness.h"
#include "src/ir/transforms/utils/sync_transform_common.h"

namespace pypto::ir::sync {
namespace {
void Union(ConservativeAccessSet& into, const ConservativeAccessSet& from) {
  into.insert(from.begin(), from.end());
}

MemorySpace Space(const ExprPtr& operand) {
  if (As<TensorType>(operand->GetType())) return MemorySpace::DDR;
  if (const auto buffer = As<BufferType>(operand->GetType())) return buffer->memory_space_;
  CHECK_SPAN(false, operand->span_)
      << "Conservative synchronization: data operand has no qualified storage space";
  return MemorySpace::DDR;
}

// Each catalog entry is a conservative connected component of overlapping
// enclosures / fine 2D or strided descriptors. Unknown effects are wildcards.
struct RawEffect {
  MemorySpace space;
  PipeType pipe;
  bool write;
  std::optional<backend::BufferSyncByteWindow> enclosure;
  ExprPtr original_root;
  bool candidate_full_overwrite = false;
  std::optional<StridedByteFootprint> vec_lattice;
  std::optional<MatrixBoxFootprint> matrix_box;
  IterationFiber fiber;
};

class StorageClasses {
 public:
  static constexpr size_t kMaxFineClusterDescriptors = 32;

  void Build(const std::vector<std::vector<RawEffect>>& sites, const std::vector<SyncCanonicalRoot>& roots) {
    effect_domains_.assign(sites.size(), {});
    for (size_t s = 0; s < sites.size(); ++s) {
      effect_domains_[s].assign(sites[s].size(), 0);
    }

    struct IndexedEffect {
      backend::BufferSyncByteWindow hull;
      size_t site_idx;
      size_t effect_idx;
    };
    std::map<MemorySpace, std::vector<IndexedEffect>> by_space;
    for (size_t s = 0; s < sites.size(); ++s) {
      for (size_t e = 0; e < sites[s].size(); ++e) {
        if (sites[s][e].enclosure) {
          by_space[sites[s][e].space].push_back({*sites[s][e].enclosure, s, e});
        }
      }
    }

    for (auto& [space, entries] : by_space) {
      // Stage 1: O(E log E) 1D byte-hull sweep.
      std::sort(entries.begin(), entries.end(), [](const IndexedEffect& a, const IndexedEffect& b) {
        return std::tie(a.hull.begin, a.hull.end, a.site_idx, a.effect_idx) <
               std::tie(b.hull.begin, b.hull.end, b.site_idx, b.effect_idx);
      });

      struct Cluster {
        backend::BufferSyncByteWindow hull;
        std::vector<std::pair<size_t, size_t>> members;
      };
      std::vector<Cluster> clusters;
      for (const auto& entry : entries) {
        if (!clusters.empty() && entry.hull.begin < clusters.back().hull.end) {
          clusters.back().hull.end = std::max(clusters.back().hull.end, entry.hull.end);
          clusters.back().members.emplace_back(entry.site_idx, entry.effect_idx);
        } else {
          clusters.push_back({entry.hull, {{entry.site_idx, entry.effect_idx}}});
        }
      }

      // Stage 2: Intra-cluster fine partitioning (2D MatrixBox & GCD StridedByteFootprint).
      for (const auto& cluster : clusters) {
        auto emit_single = [&](const backend::BufferSyncByteWindow& hull, bool uniform_geom,
                               const std::vector<std::pair<size_t, size_t>>& members) {
          const size_t id = domains.size() + 1;
          domains.push_back({id, space, hull, nullptr});
          domain_uniform_geom.push_back(uniform_geom);
          for (const auto& [si, ei] : members) effect_domains_[si][ei] = id;
        };

        if (cluster.members.size() == 1) {
          emit_single(cluster.hull, true, cluster.members);
          continue;
        }

        auto partition_and_emit = [&](size_t uniq_count, const std::vector<size_t>& member_gid,
                                      const auto& may_overlap) {
          std::vector<size_t> parent(uniq_count);
          std::iota(parent.begin(), parent.end(), 0);
          auto find_root = [&](size_t x) {
            while (parent[x] != x) {
              parent[x] = parent[parent[x]];
              x = parent[x];
            }
            return x;
          };
          for (size_t i = 0; i < uniq_count; ++i) {
            for (size_t j = i + 1; j < uniq_count; ++j) {
              const size_t ri = find_root(i), rj = find_root(j);
              if (ri != rj && may_overlap(i, j)) {
                parent[rj] = ri;
              }
            }
          }
          std::map<size_t, std::vector<std::pair<size_t, size_t>>> comp_members;
          std::map<size_t, size_t> comp_uniq_count;
          std::map<size_t, backend::BufferSyncByteWindow> comp_hull;
          for (size_t i = 0; i < uniq_count; ++i) {
            ++comp_uniq_count[find_root(i)];
          }
          for (size_t idx = 0; idx < cluster.members.size(); ++idx) {
            const size_t root = find_root(member_gid[idx]);
            const auto [si, ei] = cluster.members[idx];
            comp_members[root].push_back({si, ei});
            const auto& enc = *sites[si][ei].enclosure;
            auto it = comp_hull.find(root);
            if (it == comp_hull.end()) {
              comp_hull.emplace(root, enc);
            } else {
              it->second.begin = std::min(it->second.begin, enc.begin);
              it->second.end = std::max(it->second.end, enc.end);
            }
          }
          // Deterministic order by component hull (begin, end, root).
          std::vector<size_t> comp_order;
          comp_order.reserve(comp_members.size());
          for (const auto& [root, _] : comp_members) comp_order.push_back(root);
          std::sort(comp_order.begin(), comp_order.end(), [&](size_t a, size_t b) {
            const auto& ha = comp_hull.at(a);
            const auto& hb = comp_hull.at(b);
            return std::tie(ha.begin, ha.end, a) < std::tie(hb.begin, hb.end, b);
          });
          for (const size_t root : comp_order) {
            emit_single(comp_hull.at(root), comp_uniq_count.at(root) == 1, comp_members.at(root));
          }
        };

        if (space == MemorySpace::Vec) {
          bool all_have = true;
          std::vector<StridedByteFootprint> uniq;
          std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, size_t> geom_to_id;
          std::vector<size_t> member_gid;
          member_gid.reserve(cluster.members.size());
          for (const auto& [si, ei] : cluster.members) {
            const auto& lat = sites[si][ei].vec_lattice;
            if (!lat) {
              all_have = false;
              break;
            }
            const auto key = std::make_tuple(lat->base, lat->pitch, lat->rows, lat->width);
            auto [it, ins] = geom_to_id.emplace(key, uniq.size());
            if (ins) {
              if (uniq.size() >= kMaxFineClusterDescriptors) {
                all_have = false;
                break;
              }
              uniq.push_back(*lat);
            }
            member_gid.push_back(it->second);
          }
          if (!all_have || uniq.size() == 1) {
            emit_single(cluster.hull, all_have && uniq.size() == 1, cluster.members);
            continue;
          }
          partition_and_emit(uniq.size(), member_gid, [&](size_t i, size_t j) {
            return StridedByteFootprint::MayOverlap(uniq[i], uniq[j]);
          });
        } else {
          // Mat / Left / Right / Acc: 2D MatrixBoxFootprint partitioning.
          bool all_have = true;
          std::vector<MatrixBoxFootprint> uniq;
          std::map<std::tuple<size_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, size_t>
              box_to_id;
          std::vector<size_t> member_gid;
          member_gid.reserve(cluster.members.size());
          for (const auto& [si, ei] : cluster.members) {
            const auto& box = sites[si][ei].matrix_box;
            if (!box) {
              all_have = false;
              break;
            }
            const auto key = std::make_tuple(box->root_id, box->slot_lo, box->slot_hi, box->row_begin,
                                             box->row_end, box->col_begin, box->col_end);
            auto [it, ins] = box_to_id.emplace(key, uniq.size());
            if (ins) {
              if (uniq.size() >= kMaxFineClusterDescriptors) {
                all_have = false;
                break;
              }
              uniq.push_back(*box);
            }
            member_gid.push_back(it->second);
          }
          if (!all_have || uniq.size() == 1) {
            emit_single(cluster.hull, all_have && uniq.size() == 1, cluster.members);
            continue;
          }
          partition_and_emit(uniq.size(), member_gid, [&](size_t i, size_t j) {
            return MatrixBoxFootprint::MayOverlap(uniq[i], uniq[j], roots);
          });
        }
      }
    }

    // Original lexical effect order gives deterministic IDs, independent of
    // pointer ordering and hash iteration. Roots are retained by the catalog.
    for (size_t s = 0; s < sites.size(); ++s) {
      for (size_t e = 0; e < sites[s].size(); ++e) {
        const auto& effect = sites[s][e];
        if (effect.original_root) {
          auto it = roots_.find(effect.original_root.get());
          if (it == roots_.end()) {
            const auto id = domains.size() + 1;
            it = roots_.emplace(effect.original_root.get(), id).first;
            domains.push_back({id, effect.space, {}, effect.original_root});
            domain_uniform_geom.push_back(false);
          }
          effect_domains_[s][e] = it->second;
        }
      }
    }
  }

  size_t Domain(size_t site_idx, size_t effect_idx) const { return effect_domains_[site_idx][effect_idx]; }
  bool HasUniformGeometry(size_t domain_id) const {
    return domain_id > 0 && domain_id <= domain_uniform_geom.size() && domain_uniform_geom[domain_id - 1];
  }

  std::vector<ConservativeStorageDomain> domains;
  std::vector<bool> domain_uniform_geom;

 private:
  std::vector<std::vector<size_t>> effect_domains_;
  std::unordered_map<const Expr*, size_t> roots_;
};

constexpr size_t kAllPipeLaneCount = kPhysicalPipeCount + 1;  // MTE1..FIX + ALL.
constexpr int64_t kMaxAffineFiberSpan = 8;
constexpr int64_t kMaxModularFiberModulus = 16;

struct FiberKey {
  IterationFiber::Kind kind = IterationFiber::Kind::Wildcard;
  const ForStmt* frame = nullptr;
  int64_t modulus = 0;
  int64_t offset = 0;
  int64_t extent = 0;
  int64_t coeff = 0;
  uint8_t slice_axis = 0;
  IterationFiberBasis basis;

  static FiberKey Wildcard() { return {}; }
  static FiberKey From(const IterationFiber& f) {
    if (f.kind == IterationFiber::Kind::Wildcard || !f.frame) return {};
    return {f.kind, f.frame, f.modulus, f.offset, f.extent, f.coeff, f.slice_axis, f.basis};
  }

  bool operator==(const FiberKey& o) const {
    return kind == o.kind && frame == o.frame && modulus == o.modulus && offset == o.offset &&
           extent == o.extent && coeff == o.coeff && slice_axis == o.slice_axis && basis == o.basis;
  }
  bool operator<(const FiberKey& o) const {
    return std::tie(kind, frame, modulus, offset, extent, coeff, slice_axis, basis) <
           std::tie(o.kind, o.frame, o.modulus, o.offset, o.extent, o.coeff, o.slice_axis, o.basis);
  }
};

struct SiteEffectFiber {
  MemorySpace space = MemorySpace::DDR;
  PipeType pipe = PipeType::S;
  bool write = false;
  size_t domain = 0;
  FiberKey fiber;
};

using AffineDomainFrameKey = std::tuple<MemorySpace, size_t, const ForStmt*>;
using AffineDomainBoundsMap = std::map<AffineDomainFrameKey, std::pair<int64_t, int64_t>>;

bool FibersConflict(const FiberKey& a, const FiberKey& b) {
  // Wildcard conflicts with ALL fibers including CompletedPast(L).
  if (a.kind == IterationFiber::Kind::Wildcard || b.kind == IterationFiber::Kind::Wildcard) {
    return true;
  }
  if (a.frame != b.frame || !(a.basis == b.basis)) return true;
  if (a.kind == IterationFiber::Kind::CompletedPast || b.kind == IterationFiber::Kind::CompletedPast) {
    return false;
  }
  if (a.kind == IterationFiber::Kind::ModularSlot && b.kind == IterationFiber::Kind::ModularSlot) {
    return a.modulus != b.modulus || a.offset == b.offset;
  }
  if (a.kind == IterationFiber::Kind::AffineSlice && b.kind == IterationFiber::Kind::AffineSlice) {
    if (a.slice_axis != b.slice_axis || a.coeff != b.coeff) return true;
    return std::max<__int128>(a.offset, b.offset) <
           std::min<__int128>(__int128(a.offset) + a.extent, __int128(b.offset) + b.extent);
  }
  return true;
}

bool MustCoverFiber(const FiberKey& kill_fiber, const FiberKey& target_fiber) {
  if (kill_fiber.kind == IterationFiber::Kind::Wildcard) return true;
  if (target_fiber.kind == IterationFiber::Kind::Wildcard) return false;
  if (kill_fiber.frame != target_fiber.frame || !(kill_fiber.basis == target_fiber.basis)) return false;
  if (kill_fiber.kind == IterationFiber::Kind::ModularSlot &&
      target_fiber.kind == IterationFiber::Kind::ModularSlot) {
    return kill_fiber.modulus == target_fiber.modulus && kill_fiber.offset == target_fiber.offset;
  }
  if (kill_fiber.kind == IterationFiber::Kind::AffineSlice &&
      target_fiber.kind == IterationFiber::Kind::AffineSlice) {
    return kill_fiber.slice_axis == target_fiber.slice_axis && kill_fiber.coeff == target_fiber.coeff &&
           kill_fiber.offset <= target_fiber.offset &&
           __int128(target_fiber.offset) + target_fiber.extent <=
               __int128(kill_fiber.offset) + kill_fiber.extent;
  }
  return false;
}

FiberKey TransformFiberOnBackedge(MemorySpace space, size_t domain, const FiberKey& f, const ForStmt* loop,
                                  const AffineDomainBoundsMap& bounds) {
  if (f.kind == IterationFiber::Kind::Wildcard || f.frame != loop) return f;
  if (f.kind == IterationFiber::Kind::CompletedPast) return f;
  if (f.kind == IterationFiber::Kind::ModularSlot && f.modulus > 0) {
    FiberKey next = f;
    next.offset = PositiveMod128(__int128(f.offset) - f.coeff, f.modulus);
    return next;
  }
  if (f.kind == IterationFiber::Kind::AffineSlice && f.coeff != 0) {
    const auto bit = bounds.find({space, domain, loop});
    if (bit == bounds.end()) return FiberKey::Wildcard();
    const __int128 next_off = __int128(f.offset) - f.coeff;
    if (f.coeff > 0) {
      if (next_off + f.extent <= bit->second.first) {
        return {IterationFiber::Kind::CompletedPast, loop, 0, 0, 0, 0, 0, f.basis};
      }
    } else {
      if (next_off >= bit->second.second) {
        return {IterationFiber::Kind::CompletedPast, loop, 0, 0, 0, 0, 0, f.basis};
      }
    }
    if (next_off < std::numeric_limits<int64_t>::min() || next_off > std::numeric_limits<int64_t>::max() ||
        next_off + f.extent > std::numeric_limits<int64_t>::max()) {
      return FiberKey::Wildcard();
    }
    FiberKey next = f;
    next.offset = static_cast<int64_t>(next_off);
    return next;
  }
  return FiberKey::Wildcard();
}

FiberKey TransformFiberOnExit(const FiberKey& f, const ForStmt* loop) {
  if (f.kind != IterationFiber::Kind::Wildcard && f.frame == loop) {
    return FiberKey::Wildcard();
  }
  return f;
}

PipeMask LaneToMask(PipeType pipe) {
  const auto index = static_cast<size_t>(pipe);
  INTERNAL_CHECK(index < kAllPipeLaneCount) << "Internal error: invalid pipeline lane";
  return PipeMask{1} << index;
}

void AppendPipeMaskToSet(PipeMask mask, std::set<PipeType>& out) {
  for (size_t index = 0; index < kAllPipeLaneCount; ++index) {
    if ((mask & (PipeMask{1} << index)) != 0) out.insert(static_cast<PipeType>(index));
  }
}

size_t PopCount(PipeMask mask) {
  size_t count = 0;
  for (; mask != 0; mask &= (mask - 1)) ++count;
  return count;
}

// Pass 1 monotone state: tracks live (not cleanly overwritten) read/write pipes per
// (MemorySpace, domain, FiberKey). Domain 0 represents unknown/wildcard storage within `space`.
class LiveStorageState {
 public:
  struct DomainLanes {
    PipeMask reads = 0;
    PipeMask writes = 0;
    bool operator==(const DomainLanes& other) const { return reads == other.reads && writes == other.writes; }
    bool operator!=(const DomainLanes& other) const { return !(*this == other); }
  };

  using Key = std::tuple<MemorySpace, size_t, FiberKey>;

  bool MergeFrom(const LiveStorageState& other) {
    bool changed = false;
    for (const auto& [key, lanes] : other.entries_) {
      auto& dst = entries_[key];
      const DomainLanes before = dst;
      dst.reads |= lanes.reads;
      dst.writes |= lanes.writes;
      if (dst != before) changed = true;
    }
    return changed;
  }

  size_t KillDomain(MemorySpace space, size_t domain, const FiberKey& kill_fiber = FiberKey::Wildcard()) {
    if (domain == 0) return 0;  // Wildcard domain 0 cannot be safely killed.
    size_t killed = 0;
    for (auto it = entries_.lower_bound({space, domain, FiberKey{}});
         it != entries_.end() && std::get<0>(it->first) == space && std::get<1>(it->first) == domain;) {
      if (MustCoverFiber(kill_fiber, std::get<2>(it->first))) {
        killed += PopCount(it->second.reads) + PopCount(it->second.writes);
        it = entries_.erase(it);
      } else {
        ++it;
      }
    }
    return killed;
  }

  size_t CountDomain(MemorySpace space, size_t domain,
                     const FiberKey& kill_fiber = FiberKey::Wildcard()) const {
    if (domain == 0) return 0;
    size_t count = 0;
    for (auto it = entries_.lower_bound({space, domain, FiberKey{}});
         it != entries_.end() && std::get<0>(it->first) == space && std::get<1>(it->first) == domain; ++it) {
      if (MustCoverFiber(kill_fiber, std::get<2>(it->first))) {
        count += PopCount(it->second.reads) + PopCount(it->second.writes);
      }
    }
    return count;
  }

  void AddEffects(const ConservativeAccessSet& added) {
    for (const auto& fact : added) {
      const auto& [space, pipe, write, domain] = fact;
      auto& dst = entries_[{space, domain, FiberKey::Wildcard()}];
      const PipeMask bit = LaneToMask(pipe);
      if (write) {
        dst.writes |= bit;
      } else {
        dst.reads |= bit;
      }
    }
  }

  void AddEffects(const std::vector<SiteEffectFiber>& added) {
    for (const auto& eff : added) {
      auto& dst = entries_[{eff.space, eff.domain, eff.fiber}];
      const PipeMask bit = LaneToMask(eff.pipe);
      if (eff.write) {
        dst.writes |= bit;
      } else {
        dst.reads |= bit;
      }
    }
  }

  void ApplyLoopBackedge(const ForStmt* loop, const AffineDomainBoundsMap& bounds) {
    std::map<Key, DomainLanes> next;
    for (const auto& [key, lanes] : entries_) {
      const auto& [space, domain, fiber] = key;
      const FiberKey nf = TransformFiberOnBackedge(space, domain, fiber, loop, bounds);
      auto& dst = next[{space, domain, nf}];
      dst.reads |= lanes.reads;
      dst.writes |= lanes.writes;
    }
    entries_ = std::move(next);
  }

  void ApplyLoopExit(const ForStmt* loop) {
    std::map<Key, DomainLanes> next;
    for (const auto& [key, lanes] : entries_) {
      const auto& [space, domain, fiber] = key;
      const FiberKey nf = TransformFiberOnExit(fiber, loop);
      auto& dst = next[{space, domain, nf}];
      dst.reads |= lanes.reads;
      dst.writes |= lanes.writes;
    }
    entries_ = std::move(next);
  }

  std::set<PipeType> QueryConflicts(const ConservativeAccessFact& consumer,
                                    ConservativeSyncCost& cost) const {
    const auto& [space, destination, write, domain] = consumer;
    return QueryConflicts(SiteEffectFiber{space, destination, write, domain, FiberKey::Wildcard()}, cost);
  }

  std::set<PipeType> QueryConflicts(const SiteEffectFiber& consumer, ConservativeSyncCost& cost) const {
    ++cost.conflict_queries;
    std::set<PipeType> result;
    const auto collect = [&](const DomainLanes& lanes) {
      AppendPipeMaskToSet(lanes.writes, result);
      if (consumer.write) AppendPipeMaskToSet(lanes.reads, result);
    };
    if (consumer.domain == 0) {
      for (auto it = entries_.lower_bound({consumer.space, 0, FiberKey{}});
           it != entries_.end() && std::get<0>(it->first) == consumer.space; ++it) {
        collect(it->second);
      }
    } else {
      for (const size_t candidate : {size_t{0}, consumer.domain}) {
        for (auto it = entries_.lower_bound({consumer.space, candidate, FiberKey{}});
             it != entries_.end() && std::get<0>(it->first) == consumer.space &&
             std::get<1>(it->first) == candidate;
             ++it) {
          if (candidate == 0 || FibersConflict(std::get<2>(it->first), consumer.fiber)) {
            collect(it->second);
          }
        }
      }
    }
    return result;
  }

  ConservativeAccessSet ToAccessSet() const {
    ConservativeAccessSet result;
    for (const auto& [key, lanes] : entries_) {
      const auto& [space, domain, fiber] = key;
      (void)fiber;
      for (size_t index = 0; index < kAllPipeLaneCount; ++index) {
        const PipeMask bit = PipeMask{1} << index;
        const auto pipe = static_cast<PipeType>(index);
        if ((lanes.reads & bit) != 0) result.emplace(space, pipe, false, domain);
        if ((lanes.writes & bit) != 0) result.emplace(space, pipe, true, domain);
      }
    }
    return result;
  }

  size_t FactCount() const {
    size_t total = 0;
    for (const auto& [key, lanes] : entries_) {
      (void)key;
      total += PopCount(lanes.reads) + PopCount(lanes.writes);
    }
    return total;
  }

  bool operator!=(const LiveStorageState& other) const { return entries_ != other.entries_; }

 private:
  std::map<Key, DomainLanes> entries_;
};

// Pass 2 sparse delivery-debt lattice (Draft Section 6.4.2-6.4.3).
// Tracks pending completion masks per (space, domain, write, origin_pipe, fiber). A set bit `d`
// indicates that the event on `origin_pipe` is not yet guaranteed ordered before pipe `d`.
// Absent entries carry zero pending debts.
struct DebtKey {
  MemorySpace space;
  size_t domain;
  bool write;
  PipeType src;
  FiberKey fiber;

  bool operator<(const DebtKey& other) const {
    return std::tie(space, domain, write, src, fiber) <
           std::tie(other.space, other.domain, other.write, other.src, other.fiber);
  }
  bool operator==(const DebtKey& other) const {
    return space == other.space && domain == other.domain && write == other.write && src == other.src &&
           fiber == other.fiber;
  }
};

class DebtState {
 public:
  bool MergeFrom(const DebtState& other) {
    bool changed = false;
    for (const auto& [key, mask] : other.pending_) {
      auto& dst = pending_[key];
      const PipeMask merged = dst | mask;
      if (merged != dst) {
        dst = merged;
        changed = true;
      }
    }
    return changed;
  }

  // Batched 1-way relay from all sources in `sources_mask` to `dst` in one pass.
  void DeliverSources(PipeMask sources_mask, PipeType dst, const DebtState* excluded = nullptr,
                      const DebtState* capture = nullptr) {
    const PipeMask dst_mask = PipeToMask(dst);
    if (sources_mask == 0 || dst_mask == 0) return;
    for (auto it = pending_.begin(); it != pending_.end();) {
      auto pending = it->second;
      if (capture) {
        const auto recorded = capture->pending_.find(it->first);
        // An absent key had no outstanding obligations at the record.
        // Any new issue of the key since the record is in excluded.
        pending = recorded == capture->pending_.end() ? 0 : recorded->second;
      }
      const PipeMask ready = (~pending & kAllPhysicalPipesMask) | PipeToMask(it->first.src);
      if ((ready & sources_mask) != 0 && (!excluded || !excluded->pending_.count(it->first))) {
        it->second &= ~dst_mask;
      }
      if (it->second == 0) {
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Apply a precomputed sequential relay matrix in one pass over `pending_`.
  void ApplyServiceMatrix(const std::array<PipeMask, kPhysicalPipeCount>& matrix) {
    for (auto it = pending_.begin(); it != pending_.end();) {
      const PipeMask ready = (~it->second & kAllPhysicalPipesMask) | PipeToMask(it->first.src);
      PipeMask cleared = 0;
      for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
        if ((ready & (PipeMask{1} << p)) != 0) cleared |= matrix[p];
      }
      it->second &= ~cleared;
      if (it->second == 0) {
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void ApplySummary(const CompletionSummary& summary) {
    for (auto it = pending_.begin(); it != pending_.end();) {
      PipeMask next = it->second;
      for (size_t mid = 0; mid < kPhysicalPipeCount; ++mid) {
        const auto mid_pipe = static_cast<PipeType>(mid);
        const PipeMask mid_bit = PipeMask{1} << mid;
        const bool mid_ready = (it->first.src == mid_pipe) || ((it->second & mid_bit) == 0);
        if (mid_ready) {
          // Strict cross-pipe relays (`q != mid`) propagate already-established readiness.
          const PipeMask strict_relay = summary.relay[mid] & ~mid_bit;
          next &= ~strict_relay;
        }
      }
      if (next == 0) {
        it = pending_.erase(it);
      } else {
        it->second = next;
        ++it;
      }
    }
  }

  void KillDomain(MemorySpace space, size_t domain, const FiberKey& kill_fiber = FiberKey::Wildcard()) {
    if (domain == 0) return;
    for (auto it = pending_.lower_bound({space, domain, false, PipeType::MTE1, FiberKey{}});
         it != pending_.end() && it->first.space == space && it->first.domain == domain;) {
      if (MustCoverFiber(kill_fiber, it->first.fiber)) {
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void AddEffects(const ConservativeAccessSet& added, const CompletionSummary* summary = nullptr) {
    for (const auto& fact : added) {
      const auto& [space, pipe, write, domain] = fact;
      AddOneEffect(space, pipe, write, domain, FiberKey::Wildcard(), summary);
    }
  }

  void AddEffects(const std::vector<SiteEffectFiber>& added, const CompletionSummary* summary = nullptr) {
    for (const auto& eff : added) {
      AddOneEffect(eff.space, eff.pipe, eff.write, eff.domain, eff.fiber, summary);
    }
  }

  void ApplyLoopBackedge(const ForStmt* loop, const AffineDomainBoundsMap& bounds) {
    std::map<DebtKey, PipeMask> next;
    for (const auto& [key, mask] : pending_) {
      DebtKey nk = key;
      nk.fiber = TransformFiberOnBackedge(key.space, key.domain, key.fiber, loop, bounds);
      next[nk] |= mask;
    }
    pending_ = std::move(next);
  }

  void ApplyLoopExit(const ForStmt* loop) {
    std::map<DebtKey, PipeMask> next;
    for (const auto& [key, mask] : pending_) {
      DebtKey nk = key;
      nk.fiber = TransformFiberOnExit(key.fiber, loop);
      next[nk] |= mask;
    }
    pending_ = std::move(next);
  }

  // Returns true iff every pending debt from `source` that conflicts with `site_effects`
  // is already guaranteed complete at `probe`.
  bool CoversConflicts(PipeType source, PipeType probe,
                       const std::vector<SiteEffectFiber>& site_effects) const {
    const PipeMask probe_mask = PipeToMask(probe);
    if (probe_mask == 0) return false;
    for (const auto& [key, mask] : pending_) {
      if (key.src != source) continue;
      if (!ConflictsAny(key, site_effects)) continue;
      if ((mask & probe_mask) != 0) return false;
    }
    return true;
  }

  std::vector<PipeType> Select(const std::vector<PipeType>& required, PipeType destination,
                               const std::vector<SiteEffectFiber>& site_effects) const {
    const size_t count = required.size();
    std::vector<bool> base(count);
    std::vector<std::vector<bool>> coverage(count, std::vector<bool>(count));
    for (size_t requirement = 0; requirement < count; ++requirement) {
      base[requirement] = CoversConflicts(required[requirement], destination, site_effects);
      for (size_t candidate = 0; candidate < count; ++candidate) {
        coverage[candidate][requirement] =
            required[candidate] == required[requirement] ||
            CoversConflicts(required[requirement], required[candidate], site_effects);
      }
    }
    std::vector<PipeType> selected;
    for (const auto candidate : SelectConservativeCompletionCover(base, coverage)) {
      selected.push_back(required[candidate]);
    }
    return selected;
  }

  bool operator!=(const DebtState& other) const { return pending_ != other.pending_; }

 private:
  void AddOneEffect(MemorySpace space, PipeType pipe, bool write, size_t domain, const FiberKey& fiber,
                    const CompletionSummary* summary) {
    if (pipe == PipeType::ALL) return;
    const PipeMask pipe_bit = PipeToMask(pipe);
    PipeMask initial = kAllPhysicalPipesMask;
    if (summary != nullptr) {
      const size_t idx = static_cast<size_t>(pipe);
      if (idx < kPhysicalPipeCount) {
        // Keep self-pipe bit unless intra-pipe ordering is guaranteed; clear strict safe destinations.
        initial &= ~(summary->post_write_safe[idx] & ~pipe_bit);
      }
    }
    if (initial != 0) {
      pending_[{space, domain, write, pipe, fiber}] |= initial;
    }
  }

  static bool ConflictsAny(const DebtKey& key, const std::vector<SiteEffectFiber>& site_effects) {
    for (const auto& eff : site_effects) {
      if (eff.space != key.space) continue;
      if (!key.write && !eff.write) continue;
      if (key.domain == 0 || eff.domain == 0 ||
          (key.domain == eff.domain && FibersConflict(key.fiber, eff.fiber))) {
        return true;
      }
    }
    return false;
  }

  std::map<DebtKey, PipeMask> pending_;
};

struct CfgVertex {
  enum class Kind { Nop, Site, Child, BranchMerge, LoopBackedge, LoopExit };
  Kind kind = Kind::Nop;
  size_t index = 0;
  std::vector<size_t> preds;
  std::vector<size_t> succs;
  const ForStmt* loop = nullptr;
};

struct End {
  std::optional<size_t> yield;
  bool returned = false;
  bool Terminal() const { return yield.has_value() || returned; }
};

class Builder {
 public:
  Builder(const FunctionPtr& function, std::string target,
          const std::vector<std::shared_ptr<const NativeSyncChild>>& retained,
          const std::vector<std::shared_ptr<const EventfulSyncChild>>& eventful,
          bool retain_incoming_diagnostics, bool enable_refresh_kill, const SyncParameterBounds& parameters,
          const std::optional<std::pair<int64_t, int64_t>>& blocks, bool enable_loop_prefixes,
          const ImportedLocalSync* imported, bool enable_precise_captures)
      : function_(function),
        target_(std::move(target)),
        trip_prover_(function, parameters, blocks),
        resolver_(function, &trip_prover_, /*run_scalar_layout_check=*/true, target_),
        retain_incoming_diagnostics_(retain_incoming_diagnostics),
        enable_refresh_kill_(enable_refresh_kill),
        enable_loop_prefixes_(enable_loop_prefixes),
        imported_(imported),
        enable_precise_captures_(enable_precise_captures) {
    const auto* context = PassContext::Current();
    disjoint_gm_ = context && context->GetAutoSyncDisjointGM();
    scopes_.emplace_back();
    for (const auto& parameter : function->params_) {
      const auto& type = parameter->GetType();
      CHECK_SPAN(
          As<ScalarType>(type) || As<TensorType>(type) || As<BufferType>(type) || As<MultiBufferType>(type),
          parameter->span_)
          << "Conservative synchronization: parameter requires an invocation ownership/readiness contract";
      Bind(parameter);
      if (const auto tensor = As<TensorType>(type)) {
        const auto memory = tensor->memref_.value_or(nullptr);
        gm_roots_.emplace(parameter.get(), memory ? ExprPtr(memory->base_) : ExprPtr(parameter));
      }
    }
    for (const auto& child : retained) {
      CHECK(child) << "Conservative synchronization: missing retained child certificate";
      child->ValidateFor(function, target_);
      CHECK(child_index_.emplace(child->Scope().get(), children.size()).second)
          << "Conservative synchronization: duplicate retained child scope";
      children.push_back({child->Scope(), {}, {}, {}, {}, {}, CompletionSummary::Identity()});
    }
    for (const auto& child : eventful) {
      CHECK(child) << "Conservative synchronization: missing eventful child certificate";
      child->ValidateFor(function, target_);
      const size_t child_idx = children.size();
      if (child->IsSubblock()) {
        window_slices_by_seq_[child->WindowSeq().get()].push_back(
            {child->WindowBegin(), child->WindowEnd(), child_idx});
        children.push_back({child->Scope(),
                            {},
                            {},
                            {},
                            child,
                            {},
                            CompletionSummary::Identity(),
                            false,
                            child->WindowSeq(),
                            child->WindowBegin(),
                            child->WindowEnd(),
                            std::nullopt});
      } else {
        CHECK(child_index_.emplace(child->Scope().get(), child_idx).second)
            << "Conservative synchronization: duplicate retained child scope";
        children.push_back({child->Scope(), {}, {}, {}, child, {}, CompletionSummary::Identity()});
      }
    }
  }

  void RegisterInlineSubblocks(const std::vector<InlineExactSubBlock>& inline_subblocks) {
    for (const auto& block : inline_subblocks) {
      const size_t child_idx = children.size();
      window_slices_by_seq_[block.window_seq.get()].push_back(
          {block.window_begin, block.window_end, child_idx});
      children.push_back({nullptr,
                          {},
                          {},
                          {},
                          nullptr,
                          {},
                          block.completion,
                          false,
                          block.window_seq,
                          block.window_begin,
                          block.window_end,
                          block});
    }
    for (auto& [seq, slices] : window_slices_by_seq_) {
      std::sort(slices.begin(), slices.end(),
                [](const SliceWindow& a, const SliceWindow& b) { return a.begin < b.begin; });
      for (size_t i = 1; i < slices.size(); ++i) {
        CHECK(slices[i - 1].end <= slices[i].begin)
            << "Conservative synchronization: overlapping sub-block slices";
      }
    }
  }

  static CompletionSummary ComputeRecipeCompletionSummary(const StmtPtr& recipe, FunctionType func_type,
                                                          const std::string& target,
                                                          const PipeMask base_active,
                                                          const ForStmt* child_scope,
                                                          const SyncControlRangeProver* trip_prover) {
    CompletionSummary base = CompletionSummary::Identity();
    base.has_bypass_path = true;
    base.active = base_active;
    if (!recipe) return base;

    std::vector<CompletionVertex> vertices;
    std::vector<std::pair<size_t, size_t>> edges;
    const size_t entry_v = vertices.size();
    vertices.push_back(CompletionVertex{});
    std::unordered_map<const Expr*, PipeMask> local_tokens;
    PipeMask written_pipes = 0;

    auto find_pipe_arg = [](const CallPtr& call, const char* key) -> std::optional<PipeType> {
      for (const auto& [name, val] : call->kwargs_) {
        if (name != key) continue;
        return static_cast<PipeType>(AnyCast<int>(val, name));
      }
      for (const auto& [name, val] : call->attrs_) {
        if (name != key) continue;
        return static_cast<PipeType>(AnyCast<int>(val, name));
      }
      return std::nullopt;
    };

    auto new_vertex = [&](CompletionVertex vertex) {
      const size_t id = vertices.size();
      vertices.push_back(std::move(vertex));
      return id;
    };

    auto emit_payload = [&](const CallPtr& call, size_t cur) -> size_t {
      if (!call || GetLogicalSyncKind(call->op_) != LogicalSyncKind::None ||
          backend::QualifyBufferSyncNonPayload(call, target))
        return cur;
      const auto op = backend::QualifyBufferSyncOperation(call, func_type, target);
      const PipeMask mask = PipeToMask(op.Implementation().Pipe());
      if (mask == 0) return cur;
      written_pipes |= mask;
      for (auto& [tok, tok_mask] : local_tokens) tok_mask &= ~mask;
      CompletionVertex op_v;
      op_v.write_pipes = mask;
      const size_t v = new_vertex(std::move(op_v));
      edges.emplace_back(cur, v);
      return v;
    };

    auto is_const_unit_trip = [](const ForStmt* loop) -> bool {
      if (!loop || loop->kind_ != ForKind::Sequential) return false;
      const auto start = As<ConstInt>(loop->start_);
      const auto stop = As<ConstInt>(loop->stop_);
      const auto step = As<ConstInt>(loop->step_);
      return start && stop && step && step->value_ > 0 && stop->value_ > start->value_ &&
             (__int128(stop->value_) - start->value_) <= step->value_;
    };

    std::function<size_t(const StmtPtr&, size_t)> build = [&](const StmtPtr& stmt, size_t cur) -> size_t {
      if (!stmt) return cur;
      if (const auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) cur = build(child, cur);
        return cur;
      }
      if (const auto branch = As<IfStmt>(stmt)) {
        const auto saved_tokens = local_tokens;
        const PipeMask saved_written = written_pipes;
        written_pipes = 0;
        const size_t then_exit = build(branch->then_body_, cur);
        const auto then_tokens = local_tokens;
        const PipeMask then_written = written_pipes;
        local_tokens = saved_tokens;
        written_pipes = 0;
        const size_t else_exit = branch->else_body_ ? build(*branch->else_body_, cur) : cur;
        const PipeMask else_written = written_pipes;
        written_pipes = saved_written | then_written | else_written;
        for (auto it = local_tokens.begin(); it != local_tokens.end();) {
          const auto found = then_tokens.find(it->first);
          if (found == then_tokens.end()) {
            it = local_tokens.erase(it);
          } else {
            it->second &= found->second;
            ++it;
          }
        }
        const size_t merge = new_vertex(CompletionVertex{});
        edges.emplace_back(then_exit, merge);
        edges.emplace_back(else_exit, merge);
        return merge;
      }
      if (const auto loop = As<ForStmt>(stmt)) {
        const bool is_child_root = loop->HasAttr(kClosedSyncChildAttr);
        const bool unit_const = is_const_unit_trip(loop.get());
        const bool positive_trip =
            unit_const ||
            (trip_prover && (is_child_root ? (child_scope && trip_prover->IsPositiveTrip(child_scope))
                                           : trip_prover->IsPositiveTripCompatible(loop.get())));
        const bool exact_one_trip =
            is_child_root &&
            (unit_const || (trip_prover && child_scope && trip_prover->IsPositiveTrip(child_scope) &&
                            trip_prover->IsAtMostOneTrip(child_scope)));
        if (exact_one_trip) {
          return build(loop->body_, cur);
        }
        const auto saved_tokens = local_tokens;
        const PipeMask outer_written = written_pipes;
        written_pipes = 0;
        const size_t hdr = new_vertex(CompletionVertex{});
        edges.emplace_back(cur, hdr);
        local_tokens.clear();
        const size_t body_exit = build(loop->body_, hdr);
        edges.emplace_back(body_exit, hdr);
        const PipeMask loop_written = written_pipes;
        local_tokens = saved_tokens;
        for (auto& [tok, tok_mask] : local_tokens) tok_mask &= ~loop_written;
        written_pipes = outer_written | loop_written;
        return positive_trip ? body_exit : hdr;
      }
      if (const auto loop = As<WhileStmt>(stmt)) {
        const auto saved_tokens = local_tokens;
        const PipeMask outer_written = written_pipes;
        written_pipes = 0;
        const size_t hdr = new_vertex(CompletionVertex{});
        edges.emplace_back(cur, hdr);
        local_tokens.clear();
        const size_t body_exit = build(loop->body_, hdr);
        edges.emplace_back(body_exit, hdr);
        const PipeMask loop_written = written_pipes;
        local_tokens = saved_tokens;
        for (auto& [tok, tok_mask] : local_tokens) tok_mask &= ~loop_written;
        written_pipes = outer_written | loop_written;
        return hdr;
      }
      if (const auto assign = As<AssignStmt>(stmt)) {
        if (As<SyncTokenType>(assign->var_->GetType())) {
          if (const auto rhs = AsVarLike(assign->value_)) {
            const auto found = local_tokens.find(rhs.get());
            local_tokens[assign->var_.get()] = found == local_tokens.end() ? 0 : found->second;
          } else if (const auto call = As<Call>(assign->value_)) {
            if (IsOp(call, "sync.record")) {
              if (const auto pipe = find_pipe_arg(call, "pipe")) {
                local_tokens[assign->var_.get()] = PipeToMask(*pipe);
              }
            } else if (IsOp(call, "sync.completed")) {
              local_tokens[assign->var_.get()] = 0;
            } else if (IsOp(call, "sync.join")) {
              PipeMask mask = 0;
              bool all_known = !call->args_.empty();
              for (const auto& arg : call->args_) {
                const auto v = AsVarLike(arg);
                const auto found = v ? local_tokens.find(v.get()) : local_tokens.end();
                if (found == local_tokens.end()) {
                  all_known = false;
                  break;
                }
                mask |= found->second;
              }
              local_tokens[assign->var_.get()] = all_known ? mask : 0;
            } else {
              local_tokens.erase(assign->var_.get());
            }
          } else {
            local_tokens.erase(assign->var_.get());
          }
          return cur;
        }
        return emit_payload(As<Call>(assign->value_), cur);
      }
      if (const auto eval = As<EvalStmt>(stmt)) {
        if (const auto call = As<Call>(eval->expr_)) {
          if (IsOp(call, "sync.wait") && !call->args_.empty()) {
            const auto v = AsVarLike(call->args_[0]);
            if (const auto found = v ? local_tokens.find(v.get()) : local_tokens.end();
                found != local_tokens.end() && found->second != 0) {
              if (const auto dst = find_pipe_arg(call, "dst_pipe")) {
                CompletionVertex sync_v;
                for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
                  if ((found->second & (PipeMask{1} << p)) != 0) {
                    sync_v.pre_syncs.push_back(
                        CompletionSyncEdge{static_cast<PipeType>(p), *dst, /*bidirectional=*/false});
                  }
                }
                const size_t node = new_vertex(std::move(sync_v));
                edges.emplace_back(cur, node);
                return node;
              }
            }
            return cur;
          }
          return emit_payload(call, cur);
        }
      }
      return cur;
    };

    const size_t tail = build(recipe, entry_v);
    const size_t exit_v = new_vertex(CompletionVertex{});
    edges.emplace_back(tail, exit_v);
    auto summary = SolveCompletionTransfer(vertices, edges, entry_v, exit_v).summary;
    const PipeMask added_active = base.active & ~summary.active;
    for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
      if ((added_active & (PipeMask{1} << p)) != 0) {
        summary.post_write_safe[p] = PipeMask{1} << p;
      }
    }
    summary.active |= base.active;
    return summary;
  }

  CompletionSummary ComputeEventfulCompletionSummary(const ConservativeSyncChildEntry& entry) const {
    PipeMask base_active = 0;
    for (const auto& fact : entry.effects) {
      const PipeType pipe = std::get<1>(fact);
      if (pipe != PipeType::ALL) base_active |= PipeToMask(pipe);
    }
    const StmtPtr recipe = entry.inline_subblock
                               ? entry.inline_subblock->spliced_seq
                               : (entry.eventful ? entry.eventful->LogicalRecipe() : StmtPtr{});
    return ComputeRecipeCompletionSummary(recipe, function_->func_type_, target_, base_active,
                                          entry.scope.get(), &trip_prover_);
  }

  void Build() {
    Read(function_->body_);
    RefineStorage();
    const auto range_work = trip_prover_.Work();
    cost.range_indexed_definitions = range_work.indexed_definitions;
    cost.range_footprint_queries = range_work.footprint_queries;
    cost.range_fallback_definitions = range_work.fallback_definitions;
    for (size_t idx = 0; idx < children.size(); ++idx) {
      auto& child = children[idx];
      if (child.IsSubblockSlice()) {
        CHECK(seen_slice_children_.count(idx) != 0)
            << "Retained sub-block child is outside the original semantic input";
        for (size_t k = child.window_begin; k < child.window_end; ++k) {
          CollectChildEffects(child.window_seq->stmts_[k], child.effects);
        }
      } else {
        CHECK(seen_children_.count(child.scope.get()) != 0)
            << "Retained child is outside the original semantic input";
        CollectChildEffects(child.scope, child.effects);
      }
      child.completion = ComputeEventfulCompletionSummary(child);
    }
  }

  std::vector<ConservativeSyncSite> PrimitiveSites() {
    std::vector<ConservativeSyncSite> result;
    for (size_t index = 0; index < sites.size(); ++index)
      if (!inside_child_[index]) result.push_back(std::move(sites[index]));
    return result;
  }

  ConservativeAccessSet Solve() {
    std::vector<CfgVertex> cfg;
    const size_t cfg_entry = 0;
    cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
    const size_t body_exit = BuildSubCfg(function_->body_, cfg_entry, cfg);
    const size_t cfg_exit = cfg.size();
    cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
    AddCfgEdge(cfg, body_exit, cfg_exit);

    std::vector<size_t> rpo;
    std::vector<size_t> rpo_pos;
    ComputeRpo(cfg, cfg_entry, rpo, rpo_pos);

    bool has_backedge = false;
    for (const size_t u : rpo) {
      for (const size_t v : cfg[u].succs) {
        if (rpo_pos[v] <= rpo_pos[u]) {
          has_backedge = true;
          break;
        }
      }
      if (has_backedge) break;
    }

    if (!has_backedge) {
      auto exit = SolveAcyclic(cfg, rpo, cfg_exit);
      TryPreciseCaptures(cfg, rpo, rpo_pos, cfg_entry);
      return exit;
    }

    std::vector<LiveStorageState> live_in, live_out;
    SolveLiveStorage(cfg, rpo, rpo_pos, cfg_entry, live_in, live_out);
    ExtractSourcePipeServices(cfg, rpo, live_in, live_out);
    const auto before_selection = cost;
    SelectAndCertifyDebts(cfg, rpo, rpo_pos, cfg_entry);
    TryPreciseCaptures(cfg, rpo, rpo_pos, cfg_entry);
    if (enable_loop_prefixes_) {
      TryProducerPrefix(cfg, rpo, rpo_pos, cfg_entry, before_selection);
      TryLoopEntrySplit();
    }
    if (enable_precise_captures_ && !loop_capture_sources_.empty()) {
      TryLoopEntrySplit(/*precise_captures=*/true);
    }

    return retain_incoming_diagnostics_ ? live_out[cfg_exit].ToAccessSet() : ConservativeAccessSet{};
  }

  bool CertifyEarlierPrefixes(const ConservativeSyncAnalysis& baseline,
                              const std::vector<EarlierPeerPrefix>& moves) {
    if (sites.size() != baseline.Sites().size()) {
      return false;
    }
    for (size_t i = 0; i < sites.size(); ++i) {
      if (sites[i].cut != baseline.Sites()[i].cut) {
        return false;
      }
    }
    // Recover the bounded first-visit CFG used by the original proof. Only
    // phases that actually survived into the fixed schedule are relevant.
    NominatePreciseCaptures(function_->body_);
    sites = baseline.Sites();
    rechecking_prefixes_ = true;
    LoopEntryPhases phases{{}, std::vector<std::optional<size_t>>(sites.size()), {}};
    CollectLoopEntryPhases(function_->body_, nullptr, phases);
    const auto guarded = [](const auto& site) { return site.guarded_capture.has_value(); };
    if (static_cast<size_t>(std::count_if(sites.begin(), sites.end(), guarded)) != phases.heads.size()) {
      return false;
    }
    for (const auto& [loop, head] : phases.heads) {
      prefix_phase_loops_.emplace(loop->loop_var_.get(), std::make_pair(loop.get(), head.site));
    }
    for (const size_t site : phases.first_visit_copies) {
      sites.push_back(sites[site]);
      site_effect_fibers_.push_back(site_effect_fibers_[site]);
      site_refresh_fibers_.push_back(site_refresh_fibers_[site]);
    }
    for (const auto& [index, capture] : first_prefix_deliveries_) {
      auto& pipes = sites[index].source_pipes;
      const auto source = std::find(pipes.begin(), pipes.end(), sites[capture.record_site].pipe);
      if (source == pipes.end()) {
        return false;
      }
      pipes.erase(source);
    }
    for (const auto& move : moves) {
      if (!InstallEarlierPrefixProof(move, phases)) {
        return false;
      }
    }
    return CheckFixedPrefixDebts(phases);
  }

  std::vector<ConservativeSyncSite> sites;
  std::vector<ConservativeSyncChildEntry> children;
  ConservativeSyncCost cost;
  std::vector<ConservativeStorageDomain> domains;
  bool disjoint_gm_ = false;

 private:
  static void AddCfgEdge(std::vector<CfgVertex>& cfg, size_t from, size_t to) {
    auto& succs = cfg[from].succs;
    if (std::find(succs.begin(), succs.end(), to) == succs.end()) {
      succs.push_back(to);
      cfg[to].preds.push_back(from);
    }
  }

  struct LoopEntryPhases {
    struct Head {
      size_t site;
      PipeMask body_pipes;
    };
    std::map<ForStmtPtr, Head> heads;
    std::vector<std::optional<size_t>> first_sites;  // Original site index to first-visit analysis site.
    std::vector<size_t> first_visit_copies;          // Original indices in the order they will be copied.
  };

  struct FirstPrefixCapture {
    size_t record_site;
    size_t wait_site;
    ExprPtr condition;
    DebtState displaced_effects;
  };

  struct EarlierPrefixDelivery {
    PipeType source;
    DebtState excluded;
    std::vector<size_t> capture_path;  // Empty for legacy sibling and latch moves.
  };

  bool InstallEarlierPrefixProof(const EarlierPeerPrefix& move, const LoopEntryPhases& phases) {
    const auto found = site_index_.find(move.consumer);
    if (found == site_index_.end()) {
      return false;
    }
    const auto consumer = found->second;
    if (!std::binary_search(sites[consumer].source_pipes.begin(), sites[consumer].source_pipes.end(),
                            move.source)) {
      return false;
    }
    DebtState displaced;
    std::vector<size_t> path;
    const auto record = site_index_.find(move.reader_record);
    if (move.reader_record &&
        (record == site_index_.end() || sites[record->second].pipe != move.source || move.backedge_iv)) {
      return false;
    }
    for (const auto* statement : move.displaced) {
      const auto site = site_index_.find(statement);
      if (site == site_index_.end() || (!move.reader_record && sites[site->second].pipe != move.source)) {
        return false;
      }
      displaced.AddEffects(site_effect_fibers_[site->second]);
      if (move.reader_record) {
        path.push_back(site->second);
      }
    }
    if (move.reader_record) {
      path.push_back(record->second);
    }
    const auto first = phases.first_sites[consumer];
    if (move.backedge_iv) {
      const auto loop = prefix_phase_loops_.find(move.backedge_iv);
      if (loop == prefix_phase_loops_.end() || loop->second.second != consumer) {
        return false;
      }
      displaced.ApplyLoopBackedge(loop->second.first, affine_domain_bounds_);
    } else if (first) {
      std::vector<size_t> first_path;
      for (const auto index : path) {
        if (!phases.first_sites[index]) {
          return false;
        }
        first_path.push_back(*phases.first_sites[index]);
      }
      earlier_deliveries_.emplace(*first,
                                  EarlierPrefixDelivery{move.source, displaced, std::move(first_path)});
    }
    return earlier_deliveries_
        .emplace(consumer, EarlierPrefixDelivery{move.source, std::move(displaced), std::move(path)})
        .second;
  }

  bool MapReaderRecordVertices(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo) {
    for (const auto vertex : rpo) {
      if (cfg[vertex].kind != CfgVertex::Kind::Site) {
        continue;
      }
      const auto index = cfg[vertex].index;
      const auto delivery = earlier_deliveries_.find(index);
      if (delivery == earlier_deliveries_.end() || delivery->second.capture_path.empty()) {
        continue;
      }
      // Each wait is paired with its matching CFG occurrence of the record.
      // Follow the exact same-sequence payload cuts, with no intervening merge,
      // loop edge or opaque child changing the lifetime of the snapshot.
      auto record = vertex;
      for (const auto previous : delivery->second.capture_path) {
        const auto& predecessors = cfg[record].preds;
        if (predecessors.size() != 1) {
          return false;
        }
        record = predecessors.front();
        if (cfg[record].kind != CfgVertex::Kind::Site || cfg[record].index != previous) {
          return false;
        }
      }
      if (!reader_record_vertices_.emplace(index, record).second) {
        return false;
      }
      reader_record_dependents_[record].push_back(vertex);
    }
    return true;
  }

  bool CheckFixedPrefixDebts(const LoopEntryPhases& phases) {
    std::vector<CfgVertex> cfg{{CfgVertex::Kind::Nop, 0, {}, {}}};
    BuildSubCfg(function_->body_, 0, cfg, &phases);
    std::vector<size_t> rpo, rpo_pos;
    ComputeRpo(cfg, 0, rpo, rpo_pos);
    if (!MapReaderRecordVertices(cfg, rpo)) {
      return false;
    }
    std::vector<DebtState> in, out;
    std::vector<bool> visited;
    SolveDebtFixedPoint(/*use_selected=*/true, cfg, rpo, rpo_pos, in, out, visited);
    for (const size_t vertex : rpo) {
      if (cfg[vertex].kind != CfgVertex::Kind::Site) {
        continue;
      }
      const auto index = cfg[vertex].index;
      const auto& site = sites[index];
      auto known = std::move(in[vertex]);
      ApplySelectedDeliveries(known, index, site.source_pipes, &out);
      for (const auto required : site.required_source_pipes) {
        if (!known.CoversConflicts(required, site.pipe, site_effect_fibers_[index])) {
          return false;
        }
      }
    }
    return true;
  }

  size_t BuildSubCfg(const StmtPtr& statement, size_t entry, std::vector<CfgVertex>& cfg,
                     const LoopEntryPhases* phases = nullptr, bool first_visit = false) const {
    if (const auto child = child_index_.find(statement.get()); child != child_index_.end()) {
      const size_t node = cfg.size();
      cfg.push_back({CfgVertex::Kind::Child, child->second, {}, {}});
      AddCfgEdge(cfg, entry, node);
      return node;
    }
    if (const auto site = site_index_.find(statement.get()); site != site_index_.end()) {
      const size_t node = cfg.size();
      const auto index = first_visit ? phases->first_sites[site->second].value() : site->second;
      cfg.push_back({CfgVertex::Kind::Site, index, {}, {}});
      AddCfgEdge(cfg, entry, node);
      return node;
    }
    if (const auto seq = As<SeqStmts>(statement)) {
      size_t cur = entry;
      const auto wit = window_slices_by_seq_.find(seq.get());
      if (wit == window_slices_by_seq_.end() || wit->second.empty()) {
        for (const auto& child : seq->stmts_) {
          cur = BuildSubCfg(child, cur, cfg, phases, first_visit);
        }
        return cur;
      }
      const auto& slices = wit->second;
      size_t slice_idx = 0;
      size_t i = 0;
      while (i < seq->stmts_.size()) {
        if (slice_idx < slices.size() && i == slices[slice_idx].begin) {
          const size_t node = cfg.size();
          cfg.push_back({CfgVertex::Kind::Child, slices[slice_idx].child_idx, {}, {}});
          AddCfgEdge(cfg, cur, node);
          cur = node;
          i = slices[slice_idx].end;
          ++slice_idx;
        } else {
          cur = BuildSubCfg(seq->stmts_[i], cur, cfg, phases, first_visit);
          ++i;
        }
      }
      return cur;
    }
    if (const auto branch = As<IfStmt>(statement)) {
      const size_t then_exit = BuildSubCfg(branch->then_body_, entry, cfg, phases, first_visit);
      const size_t else_exit =
          branch->else_body_ ? BuildSubCfg(*branch->else_body_, entry, cfg, phases, first_visit) : entry;
      const size_t merge = cfg.size();
      cfg.push_back({CfgVertex::Kind::BranchMerge, 0, {}, {}});
      AddCfgEdge(cfg, then_exit, merge);
      AddCfgEdge(cfg, else_exit, merge);
      return merge;
    }
    if (const auto loop = As<ForStmt>(statement)) {
      const bool positive = trip_prover_.IsPositiveTrip(loop.get());
      const bool at_most_one = trip_prover_.IsAtMostOneTrip(loop.get());
      const bool has_fibers = loops_with_fibers_.count(loop.get()) != 0;
      auto wrap_exit = [&](size_t exit_node) -> size_t {
        if (!has_fibers) return exit_node;
        const size_t exit_v = cfg.size();
        cfg.push_back({CfgVertex::Kind::LoopExit, 0, {}, {}, loop.get()});
        AddCfgEdge(cfg, exit_node, exit_v);
        return exit_v;
      };
      if (phases && phases->heads.count(loop) != 0) {
        // Only the first body uses distinct analysis sites. Both backedges
        // enter the steady-state header, and original IR emits each site once.
        const size_t first_exit = BuildSubCfg(loop->body_, entry, cfg, phases, true);
        const size_t steady_header = cfg.size();
        cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
        const size_t steady_exit = BuildSubCfg(loop->body_, steady_header, cfg);
        for (const size_t body_exit : {first_exit, steady_exit}) {
          const size_t backedge = cfg.size();
          cfg.push_back({CfgVertex::Kind::LoopBackedge, 0, {}, {}, loop.get()});
          AddCfgEdge(cfg, body_exit, backedge);
          AddCfgEdge(cfg, backedge, steady_header);
        }
        const size_t merge = cfg.size();
        cfg.push_back({CfgVertex::Kind::BranchMerge, 0, {}, {}});
        if (!positive) AddCfgEdge(cfg, entry, merge);
        AddCfgEdge(cfg, first_exit, merge);
        AddCfgEdge(cfg, steady_exit, merge);
        return wrap_exit(merge);
      }
      if (!As<ConstInt>(loop->start_) && at_most_one) {
        if (positive) {
          return wrap_exit(BuildSubCfg(loop->body_, entry, cfg, phases));
        }
        const size_t body_exit = BuildSubCfg(loop->body_, entry, cfg, phases);
        const size_t merge = cfg.size();
        cfg.push_back({CfgVertex::Kind::BranchMerge, 0, {}, {}});
        AddCfgEdge(cfg, entry, merge);
        AddCfgEdge(cfg, body_exit, merge);
        return wrap_exit(merge);
      }
      if (IsSpmdAtMostOneTripFor(loop, ssa_defs_)) {
        const size_t body_exit = BuildSubCfg(loop->body_, entry, cfg, phases);
        const size_t merge = cfg.size();
        cfg.push_back({CfgVertex::Kind::BranchMerge, 0, {}, {}});
        AddCfgEdge(cfg, entry, merge);
        AddCfgEdge(cfg, body_exit, merge);
        return wrap_exit(merge);
      }
      const size_t header = cfg.size();
      cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
      AddCfgEdge(cfg, entry, header);
      const size_t body_exit = BuildSubCfg(loop->body_, header, cfg, phases);
      if (has_fibers) {
        const size_t backedge = cfg.size();
        cfg.push_back({CfgVertex::Kind::LoopBackedge, 0, {}, {}, loop.get()});
        AddCfgEdge(cfg, body_exit, backedge);
        AddCfgEdge(cfg, backedge, header);
      } else {
        AddCfgEdge(cfg, body_exit, header);
      }
      return wrap_exit(positive ? body_exit : header);
    }
    if (const auto loop = As<WhileStmt>(statement)) {
      const size_t header = cfg.size();
      cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
      AddCfgEdge(cfg, entry, header);
      const size_t body_exit = BuildSubCfg(loop->body_, header, cfg, phases);
      AddCfgEdge(cfg, body_exit, header);
      return header;
    }
    return entry;
  }

  static void ComputeRpo(const std::vector<CfgVertex>& cfg, size_t entry, std::vector<size_t>& rpo,
                         std::vector<size_t>& rpo_pos) {
    const size_t n = cfg.size();
    std::vector<bool> seen(n, false);
    std::vector<size_t> post_order;
    post_order.reserve(n);
    std::vector<std::pair<size_t, size_t>> stack;
    seen[entry] = true;
    stack.emplace_back(entry, 0);
    while (!stack.empty()) {
      auto& [u, idx] = stack.back();
      if (idx < cfg[u].succs.size()) {
        const size_t v = cfg[u].succs[idx++];
        if (!seen[v]) {
          seen[v] = true;
          stack.emplace_back(v, 0);
        }
      } else {
        post_order.push_back(u);
        stack.pop_back();
      }
    }
    rpo.assign(post_order.rbegin(), post_order.rend());
    rpo_pos.assign(n, n);
    for (size_t i = 0; i < rpo.size(); ++i) rpo_pos[rpo[i]] = i;
  }

  LiveStorageState TransferLiveStorage(const CfgVertex& vertex, LiveStorageState state) const {
    if (vertex.kind == CfgVertex::Kind::Site) {
      if (enable_refresh_kill_) {
        for (const auto& [space, domain, kill_fiber] : site_refresh_fibers_[vertex.index]) {
          state.KillDomain(space, domain, kill_fiber);
        }
      }
      state.AddEffects(site_effect_fibers_[vertex.index]);
    } else if (vertex.kind == CfgVertex::Kind::Child) {
      state.AddEffects(children[vertex.index].effects);
    } else if (vertex.kind == CfgVertex::Kind::LoopBackedge && vertex.loop) {
      state.ApplyLoopBackedge(vertex.loop, affine_domain_bounds_);
    } else if (vertex.kind == CfgVertex::Kind::LoopExit && vertex.loop) {
      state.ApplyLoopExit(vertex.loop);
    }
    return state;
  }

  std::vector<uint64_t> ComputeReachingChildDirs(const std::vector<CfgVertex>& cfg) const {
    const size_t n = cfg.size();
    std::vector<uint64_t> reaching_child_dirs(n, 0);
    std::queue<size_t> q;
    for (size_t w = 0; w < n; ++w) {
      if (cfg[w].kind != CfgVertex::Kind::Child) continue;
      const auto& w_entry = children[cfg[w].index];
      // InlineExactSubBlock (`eventful == nullptr`) uses parent IDs, never a private child palette.
      if (!w_entry.eventful) continue;
      uint64_t mask = 0;
      for (const auto& [producer, consumer] : w_entry.eventful->ResourceDirections()) {
        mask |= SyncDirectionBit(producer, consumer);
      }
      if (mask == 0) continue;
      for (const size_t nxt : cfg[w].succs) {
        const uint64_t before = reaching_child_dirs[nxt];
        reaching_child_dirs[nxt] |= mask;
        if (reaching_child_dirs[nxt] != before) q.push(nxt);
      }
    }
    while (!q.empty()) {
      const size_t x = q.front();
      q.pop();
      const uint64_t mask = reaching_child_dirs[x];
      for (const size_t nxt : cfg[x].succs) {
        const uint64_t before = reaching_child_dirs[nxt];
        reaching_child_dirs[nxt] |= mask;
        if (reaching_child_dirs[nxt] != before) q.push(nxt);
      }
    }
    return reaching_child_dirs;
  }

  static bool CfgNodeReachesSelf(const std::vector<CfgVertex>& cfg, size_t target) {
    std::vector<bool> visited(cfg.size(), false);
    std::queue<size_t> q;
    for (const size_t nxt : cfg[target].succs) {
      if (nxt == target) return true;
      if (!visited[nxt]) {
        visited[nxt] = true;
        q.push(nxt);
      }
    }
    while (!q.empty()) {
      const size_t u = q.front();
      q.pop();
      for (const size_t nxt : cfg[u].succs) {
        if (nxt == target) return true;
        if (!visited[nxt]) {
          visited[nxt] = true;
          q.push(nxt);
        }
      }
    }
    return false;
  }

  static uint64_t CollectRearmServices(const ConservativeSyncChildEntry& entry, uint64_t reaching_mask,
                                       std::set<std::pair<PipeType, PipeType>>& services) {
    if (!entry.eventful) return 0;
    uint64_t conflict = 0;
    for (const auto& [producer, consumer] : entry.eventful->ResourceDirections()) {
      const uint64_t bit = SyncDirectionBit(producer, consumer);
      if ((reaching_mask & bit) != 0) {
        // The previous child's consuming wait must complete before this
        // child can publish into the same directional palette.
        services.emplace(consumer, producer);
        conflict |= bit;
      }
    }
    return conflict;
  }

  ConservativeAccessSet SolveAcyclic(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                                     size_t cfg_exit) {
    const size_t n = cfg.size();
    const std::vector<uint64_t> reaching_child_dirs = ComputeReachingChildDirs(cfg);

    // Pass 1: Fused live-storage transfer and conflict extraction with immediate
    // predecessor state release so straight-line sequences hold O(1) active maps.
    std::vector<LiveStorageState> live_out(n);
    std::vector<size_t> rem_succs(n);
    for (size_t u = 0; u < n; ++u) rem_succs[u] = cfg[u].succs.size();
    ConservativeAccessSet tracked_facts;
    for (const size_t u : rpo) {
      const auto& v = cfg[u];
      if (v.kind == CfgVertex::Kind::BranchMerge && !v.preds.empty()) {
        cost.control_snapshot_facts += live_out[v.preds[0]].FactCount();
        for (size_t i = 1; i < v.preds.size(); ++i) {
          cost.control_merge_facts += live_out[v.preds[i]].FactCount();
        }
      }
      LiveStorageState in_u;
      if (v.preds.size() == 1) {
        const size_t p = v.preds[0];
        if (--rem_succs[p] == 0) {
          in_u = std::move(live_out[p]);
        } else {
          in_u = live_out[p];
        }
      } else {
        for (const size_t p : v.preds) {
          in_u.MergeFrom(live_out[p]);
          if (--rem_succs[p] == 0) live_out[p] = LiveStorageState{};
        }
      }
      if (v.kind == CfgVertex::Kind::Child) {
        auto& entry = children[v.index];
        for (const auto& fact : entry.effects) {
          if (tracked_facts.insert(fact).second) ++cost.conflict_summary_facts;
        }
        if (retain_incoming_diagnostics_) {
          entry.incoming = in_u.ToAccessSet();
          cost.incoming_diagnostic_facts += entry.incoming.size();
        }
        std::set<std::pair<PipeType, PipeType>> services;
        for (const auto& effect : entry.effects) {
          for (const auto source : in_u.QueryConflicts(effect, cost)) {
            services.emplace(source, std::get<1>(effect));
          }
        }
        const uint64_t rearm_conflicts = CollectRearmServices(entry, reaching_child_dirs[u], services);
        entry.first_visit = (rearm_conflicts == 0);
        entry.services.assign(services.begin(), services.end());
        entry.required_services = entry.services;
        cost.baseline_services += services.size();
        cost.services += services.size();
        std::array<PipeMask, kPhysicalPipeCount> matrix{};
        for (const auto& [src, dst] : entry.services) {
          const PipeMask sm = PipeToMask(src);
          const PipeMask dm = PipeToMask(dst);
          if (sm == 0 || dm == 0) continue;
          for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
            const PipeMask p_bit = PipeMask{1} << p;
            if (((p_bit | matrix[p]) & sm) != 0) matrix[p] |= dm;
          }
        }
        if (child_service_matrices_.size() <= v.index) child_service_matrices_.resize(v.index + 1);
        child_service_matrices_[v.index] = matrix;
      } else if (v.kind == CfgVertex::Kind::Site) {
        auto& site = sites[v.index];
        for (const auto& fact : site.effects) {
          if (tracked_facts.insert(fact).second) ++cost.conflict_summary_facts;
        }
        if (retain_incoming_diagnostics_) {
          site.incoming = in_u.ToAccessSet();
          cost.incoming_diagnostic_facts += site.incoming.size();
        }
        if (enable_refresh_kill_) {
          for (const auto& [space, domain, kill_fiber] : site_refresh_fibers_[v.index]) {
            cost.killed_facts += in_u.CountDomain(space, domain, kill_fiber);
          }
        }
        std::set<PipeType> sources;
        for (const auto& eff : site_effect_fibers_[v.index]) {
          const auto conflicting = in_u.QueryConflicts(eff, cost);
          sources.insert(conflicting.begin(), conflicting.end());
        }
        site.required_source_pipes.assign(sources.begin(), sources.end());
        cost.baseline_services += sources.size();
      }
      ++cost.dataflow_visits;
      live_out[u] = TransferLiveStorage(v, std::move(in_u));
      if (u != cfg_exit && rem_succs[u] == 0) live_out[u] = LiveStorageState{};
    }
    ConservativeAccessSet exit_set =
        retain_incoming_diagnostics_ ? live_out[cfg_exit].ToAccessSet() : ConservativeAccessSet{};
    live_out.clear();
    live_out.shrink_to_fit();

    // Pass 2: Fused debt selection and certification on the DAG with immediate
    // predecessor state release.
    std::vector<DebtState> debt_out(n);
    for (size_t u = 0; u < n; ++u) rem_succs[u] = cfg[u].succs.size();
    for (const size_t u : rpo) {
      const auto& v = cfg[u];
      DebtState cur_in;
      if (v.preds.size() == 1) {
        const size_t p = v.preds[0];
        if (--rem_succs[p] == 0) {
          cur_in = std::move(debt_out[p]);
        } else {
          cur_in = debt_out[p];
        }
      } else {
        for (const size_t p : v.preds) {
          cur_in.MergeFrom(debt_out[p]);
          if (--rem_succs[p] == 0) debt_out[p] = DebtState{};
        }
      }
      ++cost.certificate_visits;
      if (v.kind == CfgVertex::Kind::Child) {
        const auto& entry_child = children[v.index];
        INTERNAL_CHECK(entry_child.services == entry_child.required_services)
            << "Internal error: protected child/rearming services changed";
        debt_out[u] = TransferDebt(v, std::move(cur_in), /*use_selected=*/true);
      } else if (v.kind == CfgVertex::Kind::Site) {
        auto& site = sites[v.index];
        const auto destination = site.pipe;
        const auto& effs = site_effect_fibers_[v.index];
        site.source_pipes = cur_in.Select(site.required_source_pipes, destination, effs);
        PipeMask sources_mask = 0;
        for (const auto source : site.source_pipes) sources_mask |= PipeToMask(source);
        cur_in.DeliverSources(sources_mask, destination);
        std::vector<PipeType> missing;
        for (const auto req : site.required_source_pipes) {
          if (!cur_in.CoversConflicts(req, destination, effs)) missing.push_back(req);
        }
        if (!missing.empty()) {
          std::set<PipeType> unioned(site.source_pipes.begin(), site.source_pipes.end());
          unioned.insert(missing.begin(), missing.end());
          site.source_pipes.clear();
          PipeMask extra_mask = 0;
          for (const auto req : site.required_source_pipes) {
            if (unioned.count(req)) {
              site.source_pipes.push_back(req);
              extra_mask |= PipeToMask(req);
            }
          }
          cur_in.DeliverSources(extra_mask, destination);
        }
        size_t next = 0;
        for (const auto source : site.required_source_pipes) {
          if (next < site.source_pipes.size() && site.source_pipes[next] == source) ++next;
        }
        INTERNAL_CHECK(next == site.source_pipes.size())
            << "Internal error: conservative service selection added or reordered a service";
        for (const auto source : site.required_source_pipes) {
          INTERNAL_CHECK(cur_in.CoversConflicts(source, destination, effs))
              << "Internal error: conservative service selection omits a payload prerequisite";
        }
        const size_t omitted = site.required_source_pipes.size() - site.source_pipes.size();
        cost.certificate_omissions += omitted;
        cost.suppressed_services += omitted;
        cost.services += site.source_pipes.size();
        if (enable_refresh_kill_) {
          for (const auto& [space, domain, kill_fiber] : site_refresh_fibers_[v.index]) {
            cur_in.KillDomain(space, domain, kill_fiber);
          }
        }
        cur_in.AddEffects(effs);
        debt_out[u] = std::move(cur_in);
      } else {
        debt_out[u] = TransferDebt(v, std::move(cur_in), /*use_selected=*/true);
      }
      if (rem_succs[u] == 0) debt_out[u] = DebtState{};
    }
    INTERNAL_CHECK(cost.certificate_omissions == cost.suppressed_services)
        << "Internal error: conservative service certificate omitted a proposed deletion";
    return exit_set;
  }

  void SolveLiveStorage(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                        const std::vector<size_t>& rpo_pos, size_t entry,
                        std::vector<LiveStorageState>& live_in, std::vector<LiveStorageState>& live_out) {
    const size_t n = cfg.size();
    live_in.assign(n, LiveStorageState{});
    live_out.assign(n, LiveStorageState{});
    std::vector<bool> visited(n, false);
    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>,
                        std::greater<std::pair<size_t, size_t>>>
        worklist;
    std::vector<bool> in_queue(n, false);
    auto push_vertex = [&](size_t v) {
      if (!in_queue[v]) {
        worklist.emplace(rpo_pos[v], v);
        in_queue[v] = true;
      }
    };

    auto eval = [&](size_t u) {
      LiveStorageState in_u;
      for (const size_t p : cfg[u].preds) {
        if (visited[p]) in_u.MergeFrom(live_out[p]);
      }
      if (visited[u] && !(in_u != live_in[u])) return;
      live_in[u] = in_u;
      const LiveStorageState out_u = TransferLiveStorage(cfg[u], in_u);
      ++cost.dataflow_visits;
      const bool out_changed = !visited[u] || (out_u != live_out[u]);
      visited[u] = true;
      live_out[u] = out_u;
      if (out_changed) {
        for (const size_t v : cfg[u].succs) push_vertex(v);
      }
    };

    for (const size_t u : rpo) {
      (void)entry;
      eval(u);
    }
    for (const size_t u : rpo) {
      for (const size_t v : cfg[u].succs) {
        if (rpo_pos[v] <= rpo_pos[u]) push_vertex(v);
      }
    }
    while (!worklist.empty()) {
      const size_t u = worklist.top().second;
      worklist.pop();
      in_queue[u] = false;
      eval(u);
    }
  }

  void ExtractSourcePipeServices(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                                 const std::vector<LiveStorageState>& live_in,
                                 const std::vector<LiveStorageState>& live_out) {
    const std::vector<uint64_t> reaching_child_dirs = ComputeReachingChildDirs(cfg);
    ConservativeAccessSet tracked_facts;
    for (const size_t u : rpo) {
      const auto& v = cfg[u];
      if (v.kind == CfgVertex::Kind::BranchMerge && !v.preds.empty()) {
        cost.control_snapshot_facts += live_out[v.preds[0]].FactCount();
        for (size_t i = 1; i < v.preds.size(); ++i) {
          cost.control_merge_facts += live_out[v.preds[i]].FactCount();
        }
        continue;
      }
      if (v.kind == CfgVertex::Kind::Child) {
        auto& entry = children[v.index];
        for (const auto& fact : entry.effects) {
          if (tracked_facts.insert(fact).second) ++cost.conflict_summary_facts;
        }
        if (retain_incoming_diagnostics_) {
          entry.incoming = live_in[u].ToAccessSet();
          cost.incoming_diagnostic_facts += entry.incoming.size();
        }
        std::set<std::pair<PipeType, PipeType>> services;
        for (const auto& effect : entry.effects) {
          for (const auto source : live_in[u].QueryConflicts(effect, cost)) {
            services.emplace(source, std::get<1>(effect));
          }
        }
        const uint64_t rearm_conflicts = CollectRearmServices(entry, reaching_child_dirs[u], services);
        entry.first_visit = (rearm_conflicts == 0) && !CfgNodeReachesSelf(cfg, u);
        entry.services.assign(services.begin(), services.end());
        entry.required_services = entry.services;
        cost.baseline_services += services.size();
        cost.services += services.size();
        std::array<PipeMask, kPhysicalPipeCount> matrix{};
        for (const auto& [src, dst] : entry.services) {
          const PipeMask sm = PipeToMask(src);
          const PipeMask dm = PipeToMask(dst);
          if (sm == 0 || dm == 0) continue;
          for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
            const PipeMask p_bit = PipeMask{1} << p;
            if (((p_bit | matrix[p]) & sm) != 0) matrix[p] |= dm;
          }
        }
        if (child_service_matrices_.size() <= v.index) child_service_matrices_.resize(v.index + 1);
        child_service_matrices_[v.index] = matrix;
        continue;
      }
      if (v.kind == CfgVertex::Kind::Site) {
        auto& site = sites[v.index];
        for (const auto& fact : site.effects) {
          if (tracked_facts.insert(fact).second) ++cost.conflict_summary_facts;
        }
        if (retain_incoming_diagnostics_) {
          site.incoming = live_in[u].ToAccessSet();
          cost.incoming_diagnostic_facts += site.incoming.size();
        }
        if (enable_refresh_kill_) {
          for (const auto& [space, domain, kill_fiber] : site_refresh_fibers_[v.index]) {
            cost.killed_facts += live_in[u].CountDomain(space, domain, kill_fiber);
          }
        }
        std::set<PipeType> sources;
        for (const auto& eff : site_effect_fibers_[v.index]) {
          const auto conflicting = live_in[u].QueryConflicts(eff, cost);
          sources.insert(conflicting.begin(), conflicting.end());
        }
        site.required_source_pipes.assign(sources.begin(), sources.end());
        cost.baseline_services += sources.size();
      }
    }
  }

  DebtState TransferDebt(const CfgVertex& vertex, DebtState state, bool use_selected,
                         const std::vector<DebtState>* debt_out = nullptr) const {
    if (vertex.kind == CfgVertex::Kind::Child) {
      const auto& entry = children[vertex.index];
      state.ApplyServiceMatrix(child_service_matrices_[vertex.index]);
      state.ApplySummary(entry.completion);
      state.AddEffects(entry.effects, &entry.completion);
    } else if (vertex.kind == CfgVertex::Kind::Site) {
      const auto& site = sites[vertex.index];
      const auto destination = site.pipe;
      ApplySelectedDeliveries(state, vertex.index,
                              use_selected ? site.source_pipes : site.required_source_pipes, debt_out);
      if (site.precise_capture) {
        state.DeliverSources(PipeToMask(destination), site.precise_capture->destination);
      }
      if (enable_refresh_kill_) {
        for (const auto& [space, domain, kill_fiber] : site_refresh_fibers_[vertex.index]) {
          state.KillDomain(space, domain, kill_fiber);
        }
      }
      state.AddEffects(site_effect_fibers_[vertex.index]);
      if (site.post_destination) {
        state.DeliverSources(PipeToMask(destination), *site.post_destination);
      }
    } else if (vertex.kind == CfgVertex::Kind::LoopBackedge && vertex.loop) {
      state.ApplyLoopBackedge(vertex.loop, affine_domain_bounds_);
    } else if (vertex.kind == CfgVertex::Kind::LoopExit && vertex.loop) {
      state.ApplyLoopExit(vertex.loop);
    }
    return state;
  }

  void RunDebtWorklist(bool use_selected, const std::vector<CfgVertex>& cfg,
                       const std::vector<size_t>& rpo_pos, std::vector<size_t> initial_dirty,
                       std::vector<DebtState>& debt_in, std::vector<DebtState>& debt_out,
                       std::vector<bool>& visited) const {
    const size_t n = cfg.size();
    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>,
                        std::greater<std::pair<size_t, size_t>>>
        worklist;
    std::vector<bool> in_queue(n, false);
    const auto enqueue = [&](size_t u) {
      if (!in_queue[u]) {
        worklist.emplace(rpo_pos[u], u);
        in_queue[u] = true;
      }
    };
    for (const size_t u : initial_dirty) {
      enqueue(u);
    }
    while (!worklist.empty()) {
      const size_t u = worklist.top().second;
      worklist.pop();
      in_queue[u] = false;
      DebtState in_u;
      for (const size_t p : cfg[u].preds) {
        if (visited[p]) in_u.MergeFrom(debt_out[p]);
      }
      const DebtState out_u = TransferDebt(cfg[u], in_u, use_selected, &debt_out);
      const bool in_changed = !visited[u] || (in_u != debt_in[u]);
      const bool out_changed = !visited[u] || (out_u != debt_out[u]);
      if (!in_changed && !out_changed) continue;
      visited[u] = true;
      debt_in[u] = std::move(in_u);
      debt_out[u] = std::move(out_u);
      if (out_changed) {
        for (const size_t v : cfg[u].succs) {
          enqueue(v);
        }
        // Intervening transfers can absorb a change at the record without
        // changing the consumer's ordinary predecessor. Refresh its separate
        // snapshot dependency as well.
        if (const auto captures = reader_record_dependents_.find(u);
            captures != reader_record_dependents_.end()) {
          for (const auto v : captures->second) {
            enqueue(v);
          }
        }
      }
    }
  }

  void SolveDebtFixedPoint(bool use_selected, const std::vector<CfgVertex>& cfg,
                           const std::vector<size_t>& rpo, const std::vector<size_t>& rpo_pos,
                           std::vector<DebtState>& debt_in, std::vector<DebtState>& debt_out,
                           std::vector<bool>& visited) const {
    const size_t n = cfg.size();
    debt_in.assign(n, DebtState{});
    debt_out.assign(n, DebtState{});
    visited.assign(n, false);

    std::vector<size_t> back_targets;
    for (const size_t u : rpo) {
      DebtState in_u;
      for (const size_t p : cfg[u].preds) {
        if (visited[p]) in_u.MergeFrom(debt_out[p]);
      }
      debt_in[u] = in_u;
      const DebtState out_u = TransferDebt(cfg[u], std::move(in_u), use_selected, &debt_out);
      visited[u] = true;
      debt_out[u] = out_u;
      for (const size_t v : cfg[u].succs) {
        if (rpo_pos[v] <= rpo_pos[u]) back_targets.push_back(v);
      }
    }
    RunDebtWorklist(use_selected, cfg, rpo_pos, std::move(back_targets), debt_in, debt_out, visited);
  }

  void SelectAndCertifyDebts(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                             const std::vector<size_t>& rpo_pos, size_t entry) {
    (void)entry;
    std::vector<DebtState> base_in, base_out;
    std::vector<bool> base_visited;
    SolveDebtFixedPoint(/*use_selected=*/false, cfg, rpo, rpo_pos, base_in, base_out, base_visited);

    // Greedy forward RPO selection, pulling back-edge baselines so loop-carried
    // demands are visible at loop headers prior to certification.
    std::vector<DebtState> rpo_out(cfg.size());
    std::vector<bool> rpo_done(cfg.size(), false);
    for (const size_t u : rpo) {
      DebtState cur_in;
      for (const size_t p : cfg[u].preds) {
        cur_in.MergeFrom(rpo_done[p] ? rpo_out[p] : base_out[p]);
      }
      if (cfg[u].kind == CfgVertex::Kind::Site) {
        auto& site = sites[cfg[u].index];
        const auto destination = site.pipe;
        ApplyFirstPrefixCapture(cur_in, cfg[u].index);
        site.source_pipes =
            cur_in.Select(site.required_source_pipes, destination, site_effect_fibers_[cfg[u].index]);
      }
      rpo_out[u] = TransferDebt(cfg[u], std::move(cur_in), /*use_selected=*/true);
      rpo_done[u] = true;
    }

    // Closed-loop certification + incremental monotone restoration until full CFG fixed point holds.
    std::vector<DebtState> sel_in, sel_out;
    std::vector<bool> sel_visited;
    SolveDebtFixedPoint(/*use_selected=*/true, cfg, rpo, rpo_pos, sel_in, sel_out, sel_visited);
    std::vector<size_t> dirty = rpo;
    for (;;) {
      std::vector<size_t> changed_vertices;
      for (const size_t u : dirty) {
        if (cfg[u].kind != CfgVertex::Kind::Site) continue;
        auto& site = sites[cfg[u].index];
        const auto destination = site.pipe;
        DebtState after_services = sel_in[u];
        ApplyFirstPrefixCapture(after_services, cfg[u].index);
        PipeMask sources_mask = 0;
        for (const auto source : site.source_pipes) sources_mask |= PipeToMask(source);
        after_services.DeliverSources(sources_mask, destination);
        std::vector<PipeType> missing;
        for (const auto req : site.required_source_pipes) {
          if (!after_services.CoversConflicts(req, destination, site_effect_fibers_[cfg[u].index]))
            missing.push_back(req);
        }
        if (!missing.empty()) {
          std::set<PipeType> unioned(site.source_pipes.begin(), site.source_pipes.end());
          unioned.insert(missing.begin(), missing.end());
          site.source_pipes.clear();
          for (const auto req : site.required_source_pipes) {
            if (unioned.count(req)) site.source_pipes.push_back(req);
          }
          changed_vertices.push_back(u);
        }
      }
      if (!changed_vertices.empty()) {
        RunDebtWorklist(/*use_selected=*/true, cfg, rpo_pos, changed_vertices, sel_in, sel_out, sel_visited);
        dirty = rpo;
        continue;
      }
      for (const size_t u : rpo) {
        ++cost.certificate_visits;
        if (cfg[u].kind == CfgVertex::Kind::Child) {
          const auto& entry_child = children[cfg[u].index];
          INTERNAL_CHECK(entry_child.services == entry_child.required_services)
              << "Internal error: protected child/rearming services changed";
        } else if (cfg[u].kind == CfgVertex::Kind::Site) {
          const auto& selected = sites[cfg[u].index];
          size_t next = 0;
          const auto destination = selected.pipe;
          DebtState known = sel_in[u];
          ApplyFirstPrefixCapture(known, cfg[u].index);
          PipeMask sources_mask = 0;
          for (const auto source : selected.required_source_pipes) {
            if (next < selected.source_pipes.size() && selected.source_pipes[next] == source) {
              sources_mask |= PipeToMask(source);
              ++next;
            }
          }
          known.DeliverSources(sources_mask, destination);
          INTERNAL_CHECK(next == selected.source_pipes.size())
              << "Internal error: conservative service selection added or reordered a service";
          for (const auto source : selected.required_source_pipes) {
            INTERNAL_CHECK(known.CoversConflicts(source, destination, site_effect_fibers_[cfg[u].index]))
                << "Internal error: conservative service selection omits a payload prerequisite";
          }
          const size_t omitted = selected.required_source_pipes.size() - selected.source_pipes.size();
          cost.certificate_omissions += omitted;
          cost.suppressed_services += omitted;
          cost.services += selected.source_pipes.size();
        }
      }
      INTERNAL_CHECK(cost.certificate_omissions == cost.suppressed_services)
          << "Internal error: conservative service certificate omitted a proposed deletion";
      break;
    }
  }

  struct ProducerPrefix {
    size_t producer, consumer;
    std::vector<bool> can_relay_destination;
  };

  void ApplyFirstPrefixCapture(DebtState& incoming, size_t site) const {
    const auto found = first_prefix_deliveries_.find(site);
    if (found == first_prefix_deliveries_.end()) return;
    const auto& capture = found->second;
    incoming.DeliverSources(PipeToMask(sites[capture.record_site].pipe), sites[site].pipe,
                            &capture.displaced_effects);
  }

  void ApplySelectedDeliveries(DebtState& state, size_t index, const std::vector<PipeType>& pipes,
                               const std::vector<DebtState>* debt_out = nullptr) const {
    ApplyFirstPrefixCapture(state, index);
    const auto destination = sites[index].pipe;
    PipeMask mask = 0;
    for (const auto source : pipes) {
      mask |= PipeToMask(source);
    }
    if (const auto move = earlier_deliveries_.find(index); move != earlier_deliveries_.end()) {
      const auto early = PipeToMask(move->second.source);
      const auto record = reader_record_vertices_.find(index);
      const DebtState* snapshot = nullptr;
      if (record != reader_record_vertices_.end()) {
        INTERNAL_CHECK(debt_out && record->second < debt_out->size())
            << "Internal error: reader release has no record debt";
        snapshot = &(*debt_out)[record->second];
      }
      state.DeliverSources(mask & early, destination, &move->second.excluded, snapshot);
      mask &= ~early;
    }
    state.DeliverSources(mask, destination);
  }

  bool NominateCapture(const std::vector<size_t>& source_run, const std::vector<size_t>& target_run) {
    const auto consumer = target_run.front();
    const auto src = sites[source_run.back()].pipe;
    const auto dst = sites[consumer].pipe;
    if (src == dst || src == PipeType::S || dst == PipeType::S ||
        !std::binary_search(sites[consumer].source_pipes.begin(), sites[consumer].source_pipes.end(), src)) {
      return false;
    }
    // Find the earliest record before a producer that can remain in flight.
    // A producer may acquire from the deferred destination. Place the record
    // after that service so it cannot observe a delivery before its actual wait.
    const auto find_record = [&](const LiveStorageState& accesses) -> std::optional<size_t> {
      std::optional<size_t> record;
      for (auto it = source_run.rbegin(); it != source_run.rend(); ++it) {
        const auto producer = *it;
        const auto& effects = site_effect_fibers_[producer];
        if (std::any_of(effects.begin(), effects.end(), [&](const SiteEffectFiber& effect) {
              return !accesses.QueryConflicts(effect, cost).empty();
            })) {
          break;
        }
        record = producer;
        const auto& incoming = sites[producer].required_source_pipes;
        if (std::binary_search(incoming.begin(), incoming.end(), dst)) break;
      }
      return record;
    };
    LiveStorageState target_accesses;
    for (const auto target : target_run) target_accesses.AddEffects(site_effect_fibers_[target]);
    auto record = find_record(target_accesses);
    // Prefer a single capture for the entire destination run. When the tail
    // needs the full source prefix, a second service costs at least a SET and
    // WAIT. Consider that trade only when more destination payloads than
    // those two commands can issue before a late producer completes.
    constexpr size_t kSplitPrefixPayloads = 3;
    if (!record && target_run.size() > kSplitPrefixPayloads) {
      LiveStorageState early_accesses;
      for (size_t i = 0; i < kSplitPrefixPayloads; ++i) {
        early_accesses.AddEffects(site_effect_fibers_[target_run[i]]);
      }
      record = find_record(early_accesses);
    }
    if (!record) return false;
    sites[*record].precise_capture = ConservativePrefixCapture{sites[consumer].cut, dst};
    return true;
  }

  bool PrefixMetadata(const StmtPtr& statement, bool allow_peer_wait = false) const {
    if (const auto call = transform_utils::GetCallFromStmt(statement)) {
      // On this imported path an authored wait is wait_flag_dev, which
      // blocks S and leaves both asynchronous queues untouched.
      return (allow_peer_wait && IsOp(call, "system.sync_wait")) ||
             backend::QualifyBufferSyncNonPayload(call, target_).has_value();
    }
    return static_cast<bool>(As<AssignStmt>(statement));
  }

  // Advance a record over an independently placed source-pipe suffix. Descriptors
  // can stay between that payload and the consumer. Other native calls,
  // especially authored peer commands, and changes of control scope end the
  // straight-line window.
  size_t NominatePreciseCaptures(const StmtPtr& statement) {
    if (const auto seq = As<SeqStmts>(statement)) {
      size_t count = 0;
      std::vector<size_t> source_run, target_run;
      const auto flush = [&]() {
        if (!source_run.empty() && !target_run.empty()) count += NominateCapture(source_run, target_run);
      };
      for (const auto& child : seq->stmts_) {
        if (const auto found = site_index_.find(child.get()); found != site_index_.end()) {
          const auto current = found->second;
          if (imported_->calls.at(sites[current].operation.get()).HasOpaqueProtocol()) {
            flush();
            source_run.clear();
            target_run.clear();
            continue;
          }
          if (!target_run.empty() && sites[target_run.back()].pipe != sites[current].pipe) {
            flush();
            source_run.clear();
            source_run.swap(target_run);
          }
          target_run.push_back(current);
        } else if (PrefixMetadata(child)) {
          continue;
        } else {
          flush();
          if (const auto loop = As<ForStmt>(child); loop && !target_run.empty()) {
            loop_capture_sources_.emplace(loop.get(), target_run);
          }
          source_run.clear();
          target_run.clear();
          count += NominatePreciseCaptures(child);
        }
      }
      flush();
      return count;
    }
    if (const auto branch = As<IfStmt>(statement)) {
      return NominatePreciseCaptures(branch->then_body_) +
             (branch->else_body_ ? NominatePreciseCaptures(*branch->else_body_) : 0);
    }
    if (const auto loop = As<ForStmt>(statement)) return NominatePreciseCaptures(loop->body_);
    if (const auto loop = As<WhileStmt>(statement)) return NominatePreciseCaptures(loop->body_);
    return 0;
  }

  void TryPreciseCaptures(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                          const std::vector<size_t>& rpo_pos, size_t entry) {
    if (!enable_precise_captures_) return;
    const size_t captures = NominatePreciseCaptures(function_->body_);
    if (captures == 0) return;
    cost.services = 0;
    cost.certificate_omissions = 0;
    cost.suppressed_services = 0;
    // Between the record and wait no commands read or issue on the destination
    // pipe. Credit delivery before the displaced suffix so its newly issued
    // effects keep their debts.
    // Rerunning the fixed point restores required later services that had
    // relied on the incidental completion of the displaced payload.
    SelectAndCertifyDebts(cfg, rpo, rpo_pos, entry);
    cost.services += captures;
  }

  StmtPtr FirstPrefixOperation(const StmtPtr& body) const {
    if (const auto seq = As<SeqStmts>(body)) {
      for (const auto& statement : seq->stmts_) {
        if (!PrefixMetadata(statement, true)) return statement;
      }
      return nullptr;
    }
    return PrefixMetadata(body, true) ? nullptr : body;
  }

  size_t GuardAvailableAfter(const ExprPtr& guard) const {
    size_t latest = 0;
    CheckSyncScalarReadiness(guard, [&](const VarPtr& var) {
      const auto found = definition_order_.find(var.get());
      latest = std::max(
          latest, found == definition_order_.end() ? std::numeric_limits<size_t>::max() : found->second);
      return true;
    });
    return latest;
  }

  static bool IsPrefixBoundValue(const ExprPtr& expression) {
    if (AsVarLike(expression) || As<ConstInt>(expression)) return true;
    // An affine bound over already evaluated scalars may stay inline.
    // Never duplicate division, a memory read, or any other operation
    // originally evaluated only on a nonempty loop path.
    const auto add = As<Add>(expression);
    return add && (AsVarLike(add->left_) || As<ConstInt>(add->left_)) &&
           (AsVarLike(add->right_) || As<ConstInt>(add->right_));
  }

  // Recognize a preheader source suffix and the first operation in an
  // innermost loop. A one-sided leading IV test can only suppress the first
  // consumer. Its exact first-visit predicate guards the early pair.
  std::optional<FirstPrefixCapture> FindFirstPrefixCapture(const ForStmtPtr& loop) {
    const auto source = loop_capture_sources_.find(loop.get());
    if (source == loop_capture_sources_.end() || !CanDistinguishLoopEntry(loop) ||
        !IsPrefixBoundValue(loop->stop_)) {
      return std::nullopt;
    }
    auto head = FirstPrefixOperation(loop->body_);
    auto guard = MakeLt(loop->start_, loop->stop_, loop->span_);
    if (const auto branch = As<IfStmt>(head)) {
      const auto test = As<Lt>(branch->condition_);
      if (!test || test->left_ != loop->loop_var_ ||
          (!AsVarLike(test->right_) && !As<ConstInt>(test->right_))) {
        return std::nullopt;
      }
      guard = MakeAnd(guard, MakeLt(loop->start_, test->right_, test->span_), loop->span_);
      head = FirstPrefixOperation(branch->then_body_);
    }
    const auto consumer = site_index_.find(head.get());
    if (consumer == site_index_.end() ||
        imported_->calls.at(sites[consumer->second].operation.get()).HasOpaqueProtocol()) {
      return std::nullopt;
    }
    const auto src = sites[source->second.back()].pipe;
    const auto dst = sites[consumer->second].pipe;
    const auto& services = sites[consumer->second].source_pipes;
    if (src == dst || src == PipeType::S || dst == PipeType::S ||
        !std::binary_search(services.begin(), services.end(), src)) {
      return std::nullopt;
    }
    const auto guard_ready = GuardAvailableAfter(guard);
    auto record = source->second.rend();
    for (auto it = source->second.rbegin(); it != source->second.rend(); ++it) {
      const auto producer = *it;
      if (issue_order_.at(producer) <= guard_ready) break;
      DebtState added;
      added.AddEffects(site_effect_fibers_[producer]);
      if (!added.CoversConflicts(src, dst, site_effect_fibers_[consumer->second])) break;
      record = it;
      // Effects from another origin can become ready on the source pipe
      // through a new wait. The record can follow that wait at this cut.
      const auto& incoming = sites[producer].required_source_pipes;
      const bool has_cross_pipe_wait =
          std::any_of(incoming.begin(), incoming.end(), [src](PipeType pipe) { return pipe != src; });
      if (has_cross_pipe_wait) {
        break;
      }
    }
    if (record == source->second.rend()) return std::nullopt;
    DebtState displaced;
    for (auto it = source->second.rbegin(); it != std::next(record); ++it) {
      displaced.AddEffects(site_effect_fibers_[*it]);
    }
    return FirstPrefixCapture{*record, consumer->second, guard, std::move(displaced)};
  }

  struct PathGuard {
    ExprPtr condition;
    bool positive = true;

    bool operator==(const PathGuard& other) const {
      if (positive != other.positive) return false;
      if (condition == other.condition) return true;
      if (!condition || !other.condition) return false;
      try {
        return structural_equal(condition, other.condition, false);
      } catch (const StructuralComparisonUnsupported&) {
        return false;
      }
    }
  };

  // Keep one common loop occurrence frame and one layer of mutually
  // exclusive paths. The fixed-point proof itself still includes every path
  // and every entry/backedge. This path index only qualifies where placing a
  // WAIT on an otherwise idle destination could allow more overlap.
  bool CollectProducerPaths(const StmtPtr& statement, const PathGuard& guard,
                            std::vector<std::optional<PathGuard>>& paths) const {
    if (const auto seq = As<SeqStmts>(statement)) {
      for (const auto& child : seq->stmts_) {
        if (!CollectProducerPaths(child, guard, paths)) return false;
      }
      return true;
    }
    if (const auto branch = As<IfStmt>(statement)) {
      return !guard.condition && branch->return_vars_.empty() &&
             CollectProducerPaths(branch->then_body_, {branch->condition_, true}, paths) &&
             (!branch->else_body_ ||
              CollectProducerPaths(*branch->else_body_, {branch->condition_, false}, paths));
    }
    if (As<ForStmt>(statement) || As<WhileStmt>(statement)) return false;
    if (const auto found = site_index_.find(statement.get()); found != site_index_.end()) {
      paths[found->second] = guard;
    }
    return true;
  }

  ForStmtPtr FindSingleTopLevelLoop() const {
    // Nominate one direct, lexically ordered loop seam. Each caller
    // separately qualifies the loop body and any surrounding effects.
    const auto seq = As<SeqStmts>(function_->body_);
    if (!seq) return As<ForStmt>(function_->body_);
    ForStmtPtr loop;
    for (const auto& statement : seq->stmts_) {
      if (const auto candidate = As<ForStmt>(statement)) {
        if (loop) return nullptr;
        loop = candidate;
      } else if (As<IfStmt>(statement) || As<WhileStmt>(statement)) {
        return nullptr;
      }
    }
    return loop;
  }

  std::optional<ProducerPrefix> NominateProducerPrefix() {
    if (!children.empty()) return std::nullopt;
    const auto loop = FindSingleTopLevelLoop();
    if (!loop || loop->kind_ != ForKind::Sequential) return std::nullopt;
    std::vector<std::optional<PathGuard>> paths(sites.size());
    if (!CollectProducerPaths(loop->body_, {}, paths)) return std::nullopt;
    std::array<std::vector<size_t>, kPhysicalPipeCount> sites_by_pipe;
    for (size_t i = 0; i < sites.size(); ++i) {
      const auto pipe = static_cast<size_t>(sites[i].pipe);
      // Enclosing payloads and cross-pipe macros need entry/exit evidence
      // beyond this sealed-loop family.
      if (!paths[i] || pipe >= kPhysicalPipeCount) return std::nullopt;
      sites_by_pipe[pipe].push_back(i);
    }
    for (const auto& targets : sites_by_pipe) {
      if (targets.size() != 1) continue;
      const auto consumer = targets.front();
      const auto destination = sites[consumer].pipe;
      // Descriptor evaluation and scalar control may issue on S without a
      // Buffer payload site. It cannot be shown idle from this inventory.
      if (destination == PipeType::S) continue;
      // Reuse the same conservative wildcard/domain conflict index. The
      // footprint of a single native payload is bounded by its fixed effect
      // operands; only the number of original sites grows with IR size.
      LiveStorageState target_accesses, placed_target_accesses;
      target_accesses.AddEffects(sites[consumer].effects);
      ConservativeAccessSet placed;
      for (const auto& effect : sites[consumer].effects) {
        const auto& [space, pipe, write, domain] = effect;
        if (domain != 0 && space != MemorySpace::DDR) placed.insert(effect);
      }
      placed_target_accesses.AddEffects(placed);
      auto conflict = [&](const ConservativeSyncSite& site, bool require_placed) {
        const auto& accesses = require_placed ? placed_target_accesses : target_accesses;
        for (const auto& effect : site.effects) {
          const auto& [space, pipe, write, domain] = effect;
          if (require_placed && (domain == 0 || space == MemorySpace::DDR)) continue;
          if (!accesses.QueryConflicts(effect, cost).empty()) return true;
        }
        return false;
      };
      std::map<std::pair<const Expr*, bool>, bool> matching_guards;
      const auto matches_consumer = [&](const PathGuard& producer_guard) {
        const auto key = std::make_pair(producer_guard.condition.get(), producer_guard.positive);
        const auto [found, inserted] = matching_guards.try_emplace(key, false);
        if (inserted) found->second = producer_guard == *paths[consumer];
        return found->second;
      };
      for (const auto source : sites[consumer].source_pipes) {
        const auto pipe = static_cast<size_t>(source);
        if (pipe >= kPhysicalPipeCount || source == destination) continue;
        const auto& producers = sites_by_pipe[pipe];
        bool later_compatible = false;
        for (auto it = producers.rbegin(); it != producers.rend() && *it > consumer; ++it) {
          const auto i = *it;
          // No later conflicting producer may be skipped, even with a
          // different guard or an alias that has no matching SSA handle.
          if (conflict(sites[i], false)) {
            if (later_compatible && matches_consumer(*paths[i]) && conflict(sites[i], true)) {
              ProducerPrefix result{i, consumer, std::vector<bool>(sites.size())};
              result.can_relay_destination[consumer] = true;
              for (size_t j = consumer + 1; j <= i; ++j) {
                result.can_relay_destination[j] = !paths[consumer]->condition || matches_consumer(*paths[j]);
              }
              return result;
            }
            break;
          }
          if (!paths[i]->condition || matches_consumer(*paths[i])) later_compatible = true;
        }
      }
    }
    return std::nullopt;
  }

  void TryProducerPrefix(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                         const std::vector<size_t>& rpo_pos, size_t entry,
                         const ConservativeSyncCost& before_selection) {
    const auto candidate = NominateProducerPrefix();
    if (!candidate) return;
    // Native event claims can describe secondary queues. Leave those kernels
    // to the original allocator and its complete native reservation model.
    try {
      for (const auto& site : sites) {
        if (!site.buffer_operation->Implementation().Claims().empty()) return;
      }
    } catch (const pypto::ValueError&) {
      return;
    }
    const auto& [producer, consumer, can_relay_destination] = *candidate;
    const auto source = sites[producer].pipe;
    const auto destination = sites[consumer].pipe;
    const auto baseline_cost = cost;
    std::vector<std::vector<PipeType>> baseline;
    baseline.reserve(sites.size());
    for (const auto& site : sites) baseline.push_back(site.source_pipes);
    sites[producer].post_destination = destination;
    cost = before_selection;
    cost.conflict_queries = baseline_cost.conflict_queries;
    SelectAndCertifyDebts(cfg, rpo, rpo_pos, entry);
    const auto replay_visits = cost.certificate_visits - before_selection.certificate_visits;

    // A relay reading the destination pipe can pick up the eager WAIT. This
    // is covered by the original reuser's dependency only after the reuser
    // executes and before the new publication, under a path that executes
    // the reuser. In particular, a relay at the next visit's prefix or in
    // a mismatched branch must not see the eager edge.
    const auto has_source = [](const std::vector<PipeType>& pipes, PipeType pipe) {
      return std::find(pipes.begin(), pipes.end(), pipe) != pipes.end();
    };
    bool accepted = !has_source(sites[consumer].source_pipes, source);
    for (size_t i = 0; accepted && i < sites.size(); ++i) {
      for (const auto pipe : sites[i].source_pipes) {
        if (!has_source(baseline[i], pipe) || (pipe == destination && !can_relay_destination[i])) {
          accepted = false;
          break;
        }
      }
    }
    if (accepted) {
      ++cost.services;
      --cost.suppressed_services;
      --cost.certificate_omissions;
      cost.certificate_visits += baseline_cost.certificate_visits - before_selection.certificate_visits;
    } else {
      sites[producer].post_destination.reset();
      cost = baseline_cost;
      cost.certificate_visits += replay_visits;
      for (size_t i = 0; i < sites.size(); ++i) sites[i].source_pipes = std::move(baseline[i]);
    }
  }

  bool CanDistinguishLoopEntry(const ForStmtPtr& loop) const {
    if (!loop || loop->kind_ != ForKind::Sequential ||
        loop->hardware_loop_policy_ != HardwareLoopPolicy::Auto || trip_prover_.IsAtMostOneTrip(loop.get()) ||
        GetScalarDtype(loop->loop_var_) != DataType::INDEX ||
        GetScalarDtype(loop->start_) != DataType::INDEX || GetScalarDtype(loop->stop_) != DataType::INDEX ||
        (!As<ConstInt>(loop->start_) && !AsVarLike(loop->start_)))
      return false;
    const auto step = As<ConstInt>(loop->step_);
    if (!step || step->value_ <= 0) return false;
    if (step->value_ == 1) return true;
    const auto stop = trip_prover_.LookupAtCut(loop.get(), loop->stop_);
    // Each executing IV is at most stop - 1; its increment must be representable.
    // Therefore the original start is visited at most once, even for zero trips.
    return stop && __int128(stop->hi) - 1 + step->value_ <= std::numeric_limits<int64_t>::max();
  }

  static PipeMask EntryPrefixPipes(const std::vector<ConservativeSyncSite>& original, size_t head,
                                   const std::vector<PipeType>& entry) {
    // S also executes the original scalar/metadata/loop control, which does
    // not have payload sites. The head's incoming services occupy its
    // destination and their recorded sources before its first payload.
    PipeMask occupied = PipeToMask(PipeType::S) | PipeToMask(original[head].pipe);
    for (const auto source : entry) occupied |= PipeToMask(source);
    try {
      for (size_t i = 0; i < head; ++i) {
        const auto& site = original[i];
        const auto& native = site.buffer_operation->Implementation();
        occupied |= PipeToMask(native.Pipe());
        for (const auto source : site.source_pipes) occupied |= PipeToMask(source);
        // The resource catalog enumerates secondary event queues. Leave
        // opaque implicit effects or unqualified secondary ownership guarded.
        if (!native.ImplicitEffects().empty()) return kAllPhysicalPipesMask;
        for (const auto& claim : native.Claims()) {
          if (claim.pool.kind != pypto::sync::ResourceKind::PipelineEvent) return kAllPhysicalPipesMask;
          occupied |= PipeToMask(claim.pool.src) | PipeToMask(claim.pool.dst);
        }
      }
    } catch (const pypto::ValueError&) {
      return kAllPhysicalPipesMask;
    }
    return occupied;
  }

  void CollectLoopEntrySites(const StmtPtr& statement, bool conditional,
                             std::vector<std::pair<size_t, bool>>& body_sites) const {
    if (const auto seq = As<SeqStmts>(statement)) {
      for (const auto& child : seq->stmts_) {
        CollectLoopEntrySites(child, conditional, body_sites);
      }
    } else if (const auto branch = As<IfStmt>(statement)) {
      CollectLoopEntrySites(branch->then_body_, true, body_sites);
      if (branch->else_body_) {
        CollectLoopEntrySites(*branch->else_body_, true, body_sites);
      }
    } else if (const auto found = site_index_.find(statement.get()); found != site_index_.end()) {
      body_sites.emplace_back(found->second, conditional);
    }
  }

  void AddLoopEntryPhase(const ForStmtPtr& loop, const ForStmtPtr& top_level, LoopEntryPhases& phases) {
    if (!CanDistinguishLoopEntry(loop)) {
      return;
    }
    std::vector<std::pair<size_t, bool>> body_sites;
    CollectLoopEntrySites(loop->body_, false, body_sites);
    // The first payload must issue on every nonempty invocation. Moving its
    // wait to the entry then crosses only scalar and metadata operations.
    if (body_sites.empty()) {
      return;
    }
    if (enable_precise_captures_) {
      if (auto capture = FindFirstPrefixCapture(loop)) {
        if (rechecking_prefixes_) {
          const auto& guarded = sites[capture->record_site].guarded_capture;
          const auto& entry = sites[capture->wait_site].loop_entry;
          if (!guarded || !entry || entry->loop != loop ||
              guarded->destination != sites[capture->wait_site].pipe ||
              !structural_equal(guarded->condition, capture->condition)) {
            return;
          }
        }
        SaveLoopEntryPhase(body_sites, phases);
        phases.heads.emplace(loop, LoopEntryPhases::Head{capture->wait_site, 0});
        first_prefix_deliveries_.emplace(phases.first_sites[capture->wait_site].value(), std::move(*capture));
      }
      return;
    }
    if (body_sites.front().second) return;
    const size_t head = body_sites.front().first;
    const auto& site = sites[head];
    if (site.source_pipes.empty() || site.pipe == PipeType::S) {
      return;
    }
    PipeMask body_pipes = 0;
    for (const auto& [index, conditional] : body_sites) {
      body_pipes |= PipeToMask(sites[index].pipe);
    }
    // A source that issues no payload in the loop may still relay another
    // pipe's completion. The phase dataflow below certifies the full relay.
    if (loop != top_level &&
        std::all_of(site.source_pipes.begin(), site.source_pipes.end(),
                    [&](PipeType source) { return (body_pipes & PipeToMask(source)) != 0; })) {
      return;
    }
    phases.heads.emplace(loop, LoopEntryPhases::Head{head, body_pipes});
    SaveLoopEntryPhase(body_sites, phases);
  }

  void SaveLoopEntryPhase(const std::vector<std::pair<size_t, bool>>& body_sites,
                          LoopEntryPhases& phases) const {
    for (const auto& [index, conditional] : body_sites) {
      phases.first_sites[index] = sites.size() + phases.first_visit_copies.size();
      phases.first_visit_copies.push_back(index);
    }
  }

  // Split only loops with no loop in their body. Each original site then
  // has at most two CFG occurrences, even under enclosing loops and branches.
  bool CollectLoopEntryPhases(const StmtPtr& statement, const ForStmtPtr& top_level,
                              LoopEntryPhases& phases) {
    if (const auto seq = As<SeqStmts>(statement)) {
      bool contains_loop = false;
      for (const auto& child : seq->stmts_) {
        contains_loop |= CollectLoopEntryPhases(child, top_level, phases);
      }
      return contains_loop;
    }
    if (const auto branch = As<IfStmt>(statement)) {
      const bool then_has_loop = CollectLoopEntryPhases(branch->then_body_, top_level, phases);
      const bool else_has_loop =
          branch->else_body_ && CollectLoopEntryPhases(*branch->else_body_, top_level, phases);
      return then_has_loop || else_has_loop;
    }
    if (const auto loop = As<ForStmt>(statement)) {
      if (!CollectLoopEntryPhases(loop->body_, top_level, phases)) {
        AddLoopEntryPhase(loop, top_level, phases);
      }
      return true;
    }
    if (const auto loop = As<WhileStmt>(statement)) {
      CollectLoopEntryPhases(loop->body_, top_level, phases);
      return true;
    }
    return false;
  }

  void TryLoopEntrySplit(bool precise_captures = false) {
    if (!children.empty() || sites.empty()) {
      return;
    }
    // Producer-side captures have an independent proof and are never replayed
    // here. In particular the phase split does not delay a previously moved edge.
    const auto has_capture = [](const auto& site) { return site.post_destination.has_value(); };
    if (std::any_of(sites.begin(), sites.end(), has_capture)) {
      return;
    }

    const auto top_level = FindSingleTopLevelLoop();
    const size_t original_count = sites.size();
    LoopEntryPhases phases{{}, std::vector<std::optional<size_t>>(original_count), {}};
    CollectLoopEntryPhases(function_->body_, top_level, phases);
    if (phases.heads.empty()) {
      return;
    }

    auto baseline_sites = sites;
    const auto baseline_cost = cost;
    for (const size_t index : phases.first_visit_copies) {
      sites.push_back(baseline_sites[index]);
      const auto effects = site_effect_fibers_[index];
      const auto refreshed = site_refresh_fibers_[index];
      site_effect_fibers_.push_back(effects);
      site_refresh_fibers_.push_back(refreshed);
    }
    cost = {};
    std::vector<CfgVertex> cfg{{CfgVertex::Kind::Nop, 0, {}, {}}};
    BuildSubCfg(function_->body_, 0, cfg, &phases);
    std::vector<size_t> rpo, rpo_pos;
    ComputeRpo(cfg, 0, rpo, rpo_pos);
    std::vector<LiveStorageState> live_in, live_out;
    SolveLiveStorage(cfg, rpo, rpo_pos, 0, live_in, live_out);
    ExtractSourcePipeServices(cfg, rpo, live_in, live_out);
    SelectAndCertifyDebts(cfg, rpo, rpo_pos, 0);

    const auto subset = [](const auto& a, const auto& b) {
      return std::includes(b.begin(), b.end(), a.begin(), a.end());
    };
    // The replay certifies all phases together. Retaining original services
    // at any unmodified cut must cover the replay on every visit to that cut.
    bool accepted = true;
    for (size_t i = 0; accepted && i < original_count; ++i) {
      const auto& original = baseline_sites[i].source_pipes;
      accepted =
          subset(sites[i].source_pipes, original) &&
          (!phases.first_sites[i] || subset(sites[phases.first_sites[i].value()].source_pipes, original));
    }
    if (precise_captures) {
      for (const auto& [entry, capture] : first_prefix_deliveries_) {
        const auto& incoming = sites[entry].source_pipes;
        accepted &= !std::binary_search(incoming.begin(), incoming.end(), sites[capture.record_site].pipe);
      }
    }
    const auto replay_cost = cost;
    cost = baseline_cost;
    cost.dataflow_visits += replay_cost.dataflow_visits;
    cost.certificate_visits += replay_cost.certificate_visits;
    cost.conflict_queries += replay_cost.conflict_queries;
    cost.control_snapshot_facts += replay_cost.control_snapshot_facts;
    cost.control_merge_facts += replay_cost.control_merge_facts;
    if (accepted) {
      for (const auto& [loop, head] : phases.heads) {
        if (precise_captures) {
          const auto& capture = first_prefix_deliveries_.at(phases.first_sites[head.site].value());
          InstallFirstPrefixCapture(loop, capture, baseline_sites);
        } else {
          InstallLoopEntrySplit(loop, head, loop == top_level, phases, baseline_sites);
        }
      }
    }
    first_prefix_deliveries_.clear();
    sites = std::move(baseline_sites);
    site_effect_fibers_.resize(original_count);
    site_refresh_fibers_.resize(original_count);
  }

  void InstallFirstPrefixCapture(const ForStmtPtr& loop, const FirstPrefixCapture& capture,
                                 std::vector<ConservativeSyncSite>& original) {
    auto& record = original[capture.record_site];
    auto& receiver = original[capture.wait_site];
    record.guarded_capture = ConservativeGuardedCapture{capture.condition, receiver.pipe};
    // Retain the original service on all later trips, even when the current
    // analysis finds that only some of those trips require fresh completion.
    receiver.loop_entry = ConservativeLoopEntryServices{loop, false, {}, {record.pipe}, {}};
    ++cost.services;
  }

  void InstallLoopEntrySplit(const ForStmtPtr& loop, const LoopEntryPhases::Head& head, bool top_level,
                             const LoopEntryPhases& phases,
                             std::vector<ConservativeSyncSite>& baseline_sites) {
    const auto& entry = sites[phases.first_sites[head.site].value()].source_pipes;
    const auto& backedge = sites[head.site].source_pipes;
    auto& baseline = baseline_sites[head.site];
    const bool guard_nonempty = !trip_prover_.IsPositiveTrip(loop.get());
    std::vector<PipeType> entry_only;
    std::set_difference(entry.begin(), entry.end(), backedge.begin(), backedge.end(),
                        std::back_inserter(entry_only));
    if (!top_level) {
      // Preserve the regular services at their unconditional cuts. A
      // first-visit guard can hide a relay that closes physical event reuse.
      entry_only.erase(
          std::remove_if(entry_only.begin(), entry_only.end(),
                         [&](PipeType pipe) { return (head.body_pipes & PipeToMask(pipe)) != 0; }),
          entry_only.end());
      if (!entry_only.empty()) {
        baseline.loop_entry =
            ConservativeLoopEntryServices{loop, guard_nonempty, std::move(entry_only), {}, {}};
      }
      return;
    }
    if (entry_only.empty() || backedge.size() >= baseline.source_pipes.size()) {
      return;
    }

    std::vector<PipeType> selected, backedge_only;
    std::set_union(entry.begin(), entry.end(), backedge.begin(), backedge.end(),
                   std::back_inserter(selected));
    std::set_difference(backedge.begin(), backedge.end(), entry.begin(), entry.end(),
                        std::back_inserter(backedge_only));

    const auto occupied_entry = EntryPrefixPipes(baseline_sites, head.site, entry);
    std::vector<PipeType> empty_entry;
    const auto captures_empty_prefix = [&](PipeType source) {
      return (occupied_entry & PipeToMask(source)) == 0;
    };
    std::copy_if(backedge_only.begin(), backedge_only.end(), std::back_inserter(empty_entry),
                 captures_empty_prefix);
    // This initial consuming pair cannot order any earlier payload. Keeping
    // its original static cut on all visits can close the loop's ID reuse
    // proof without synthesizing a reverse acknowledgment.
    backedge_only.erase(std::remove_if(backedge_only.begin(), backedge_only.end(), captures_empty_prefix),
                        backedge_only.end());

    const auto removed = baseline.source_pipes.size() - selected.size();
    cost.services -= removed;
    cost.suppressed_services += removed;
    cost.certificate_omissions += removed;
    baseline.source_pipes = std::move(selected);
    baseline.loop_entry = ConservativeLoopEntryServices{loop, guard_nonempty, std::move(entry_only),
                                                        std::move(backedge_only), std::move(empty_entry)};
  }

  void RefineStorage() {
    std::vector<std::vector<RawEffect>> raw(sites.size());
    for (size_t site = 0; site < sites.size(); ++site) {
      if (imported_) {
        const auto& cut = imported_->calls.at(sites[site].operation.get());
        for (const auto& effect : cut.effects) {
          ExprPtr root;
          if (effect.gm_formal && !imported_->gm_domains.empty()) {
            root = imported_->gm_domains.at(*effect.gm_formal);
          }
          raw[site].push_back({effect.space, *cut.pipe, effect.write, effect.enclosure, root});
        }
        continue;
      }
      const auto& operation = *sites[site].buffer_operation;
      const auto& call = operation.Operation();
      const auto pipe = operation.Implementation().Pipe();
      const auto append = [&](MemorySpace space, BufferAccess mode,
                              std::optional<backend::BufferSyncByteWindow> enclosure, ExprPtr root,
                              bool candidate_full_overwrite, std::optional<StridedByteFootprint> vec_lattice,
                              std::optional<MatrixBoxFootprint> matrix_box, IterationFiber fiber) {
        if (mode == BufferAccess::Read || mode == BufferAccess::ReadWrite) {
          raw[site].push_back({space, pipe, false, enclosure, root, false, vec_lattice, matrix_box, fiber});
        }
        if (mode == BufferAccess::Write || mode == BufferAccess::ReadWrite) {
          raw[site].push_back({space, pipe, true, enclosure, root, candidate_full_overwrite,
                               std::move(vec_lattice), std::move(matrix_box), std::move(fiber)});
        }
      };
      for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
        const auto mode = operation.ArgumentEffects()[index].data;
        if (mode == BufferAccess::None) continue;
        const auto& operand = call->args_[index];
        const auto space = Space(operand);
        if (space != MemorySpace::DDR) ++cost.enclosure_lookups;
        auto resolved = resolver_.ResolveOperand(sites[site].cut.get(), call, index, mode, space,
                                                 enable_refresh_kill_, disjoint_gm_);
        if (space == MemorySpace::DDR && disjoint_gm_ && !resolved.gm_root) {
          if (const auto found = gm_roots_.find(operand.get()); found != gm_roots_.end()) {
            resolved.gm_root = found->second;
          }
        }
        append(space, mode, resolved.byte_hull, std::move(resolved.gm_root),
               resolved.candidate_full_overwrite, std::move(resolved.vec_lattice),
               std::move(resolved.matrix_box), std::move(resolved.fiber));
      }
      for (const auto& effect : operation.Implementation().ImplicitEffects()) {
        std::optional<backend::BufferSyncByteWindow> enclosure;
        std::optional<StridedByteFootprint> vec_lattice;
        if (effect.space != MemorySpace::DDR && effect.begin >= 0 && effect.end > effect.begin) {
          enclosure = backend::BufferSyncByteWindow{effect.begin, effect.end};
          if (effect.space == MemorySpace::Vec) {
            vec_lattice = StridedByteFootprint{effect.begin, effect.end - effect.begin, 1,
                                               effect.end - effect.begin, false};
          }
        }
        append(effect.space, effect.data, enclosure, nullptr, false, std::move(vec_lattice), std::nullopt,
               IterationFiber::Wildcard());
      }
    }
    StorageClasses classes;
    classes.Build(raw, resolver_.Roots());

    // A history group can expire old affine offsets only if all of its moving
    // accesses share an origin, unit, and coordinate mapping. The physical
    // overlap classes can merge incompatible matrix layouts or slot partitions.
    // Such a group uses wildcards, so an offset from one view cannot expire
    // accesses that still overlap bytes touched by another view.
    struct FiberGroup {
      bool valid = true;
      IterationFiber coordinates;
      int64_t min_b = std::numeric_limits<int64_t>::max();
      int64_t max_e = std::numeric_limits<int64_t>::min();
    };
    std::map<AffineDomainFrameKey, FiberGroup> groups;
    for (size_t site = 0; site < sites.size(); ++site) {
      for (size_t ei = 0; ei < raw[site].size(); ++ei) {
        const auto& effect = raw[site][ei];
        const size_t domain = classes.Domain(site, ei);
        if (domain == 0 || effect.fiber.kind == IterationFiber::Kind::Wildcard || !effect.fiber.frame) {
          continue;
        }
        auto& g = groups[{effect.space, domain, effect.fiber.frame}];
        if (!g.valid) continue;
        if (g.coordinates.kind == IterationFiber::Kind::Wildcard) {
          g.coordinates = effect.fiber;
          g.valid = g.coordinates.coeff != 0 && g.coordinates.modulus <= kMaxModularFiberModulus;
        }
        g.valid = g.valid && g.coordinates.HasComparableCoordinates(effect.fiber);
        if (g.valid && g.coordinates.kind == IterationFiber::Kind::AffineSlice) {
          const __int128 end = __int128(effect.fiber.offset) + effect.fiber.extent;
          if (end > std::numeric_limits<int64_t>::max()) {
            g.valid = false;
          } else {
            g.min_b = std::min(g.min_b, effect.fiber.offset);
            g.max_e = std::max(g.max_e, static_cast<int64_t>(end));
          }
        }
      }
    }
    affine_domain_bounds_.clear();
    loops_with_fibers_.clear();
    for (auto& [key, g] : groups) {
      if (!g.valid) continue;
      if (g.coordinates.kind == IterationFiber::Kind::AffineSlice) {
        const auto coeff = g.coordinates.coeff;
        const __int128 abs_c = coeff > 0 ? coeff : -__int128(coeff);
        const __int128 span_elems = __int128(g.max_e) - 1 - g.min_b;
        const __int128 s_trips = span_elems < 0 ? 1 : (span_elems / abs_c + 1);
        if (s_trips > kMaxAffineFiberSpan) {
          g.valid = false;
          continue;
        }
        affine_domain_bounds_[key] = {g.min_b, g.max_e};
      }
      loops_with_fibers_.insert(std::get<2>(key));
    }

    site_effect_fibers_.assign(sites.size(), {});
    site_refresh_fibers_.assign(sites.size(), {});
    for (size_t site = 0; site < sites.size(); ++site) {
      auto& effects = sites[site].effects;
      effects.clear();
      std::set<std::pair<MemorySpace, size_t>> refreshed;
      std::set<std::tuple<MemorySpace, size_t, FiberKey>> refreshed_fibers;
      for (size_t ei = 0; ei < raw[site].size(); ++ei) {
        const auto& effect = raw[site][ei];
        const size_t domain = classes.Domain(site, ei);
        effects.emplace(effect.space, effect.pipe, effect.write, domain);
        FiberKey fkey = FiberKey::Wildcard();
        if (domain > 0 && effect.fiber.kind != IterationFiber::Kind::Wildcard && effect.fiber.frame) {
          const auto git = groups.find({effect.space, domain, effect.fiber.frame});
          if (git != groups.end() && git->second.valid) {
            fkey = FiberKey::From(effect.fiber);
          }
        }
        site_effect_fibers_[site].push_back({effect.space, effect.pipe, effect.write, domain, fkey});
        if (effect.candidate_full_overwrite && domain > 0 && classes.HasUniformGeometry(domain)) {
          refreshed.emplace(effect.space, domain);
          refreshed_fibers.emplace(effect.space, domain, fkey);
        }
      }
      sites[site].refreshed_domains.assign(refreshed.begin(), refreshed.end());
      site_refresh_fibers_[site].assign(refreshed_fibers.begin(), refreshed_fibers.end());
    }
    domains = std::move(classes.domains);
    cost.storage_domains = domains.size();
  }

  void CollectChildEffects(const StmtPtr& statement, ConservativeAccessSet& out) const {
    if (!statement) return;
    if (const auto site = site_index_.find(statement.get()); site != site_index_.end()) {
      Union(out, sites[site->second].effects);
    } else if (const auto sequence = As<SeqStmts>(statement)) {
      for (const auto& child : sequence->stmts_) CollectChildEffects(child, out);
    } else if (const auto branch = As<IfStmt>(statement)) {
      CollectChildEffects(branch->then_body_, out);
      if (branch->else_body_) CollectChildEffects(*branch->else_body_, out);
    } else if (const auto loop = As<ForStmt>(statement)) {
      CollectChildEffects(loop->body_, out);
    } else if (const auto loop = As<WhileStmt>(statement)) {
      CollectChildEffects(loop->body_, out);
    }
  }

  void Bind(const VarPtr& variable) {
    CHECK_SPAN(ready_.insert(variable.get()).second, variable->span_)
        << "Conservative synchronization: a ready SSA definition is redefined";
    scopes_.back().push_back(variable.get());
    if (enable_precise_captures_) definition_order_.insert_or_assign(variable.get(), cost.statements);
  }
  void PopScope() {
    for (const auto* variable : scopes_.back()) ready_.erase(variable);
    scopes_.pop_back();
  }

  void Ready(const ExprPtr& expression) {
    cost.expression_visits += CheckSyncScalarReadiness(
        expression, [this](const VarPtr& variable) { return ready_.count(variable.get()) != 0; });
  }

  void Payload(const StmtPtr& cut, const CallPtr& call, const VarPtr& result) {
    CHECK_SPAN(call->attrs_.empty(), call->span_)
        << "Conservative synchronization: operation attributes require an execution contract";
    for (const auto& argument : call->args_) Ready(argument);
    if (imported_) {
      const auto found = imported_->calls.find(call.get());
      CHECK_SPAN(found != imported_->calls.end() && found->second.statement == cut.get(), call->span_)
          << "Conservative synchronization: missing original imported local cut";
      const auto& local = found->second;
      if (local.pipe) {
        const auto inserted = site_index_.emplace(cut.get(), sites.size()).second;
        CHECK_SPAN(inserted, cut->span_)
            << "Conservative synchronization: shared statement identity has ambiguous insertion cuts";
        sites.push_back({cut,
                         call,
                         *local.pipe,
                         std::nullopt,
                         {},
                         {},
                         {},
                         {},
                         {},
                         std::nullopt,
                         std::nullopt,
                         std::nullopt,
                         std::nullopt});
        issue_order_.push_back(cost.statements);
        inside_child_.push_back(false);
      }
      if (result) Bind(result);
      return;
    }
    if (auto descriptor = backend::QualifyBufferSyncNonPayload(call, target_)) {
      const auto behavior = descriptor->Result().behavior;
      CHECK_SPAN(static_cast<bool>(result) == (behavior != BufferResultBehavior::None), call->span_)
          << "Conservative synchronization: descriptor result must have an original SSA binding";
      if (result) {
        Bind(result);
        const auto& ownership = descriptor->Result();
        if (ownership.behavior == BufferResultBehavior::Alias && ownership.alias_arg) {
          const auto found = gm_roots_.find(call->args_[*ownership.alias_arg].get());
          if (found != gm_roots_.end()) gm_roots_.emplace(result.get(), found->second);
        }
      }
      return;
    }
    auto operation = backend::QualifyBufferSyncOperation(call, function_->func_type_, target_);
    resolver_.RequireVecCopyPlacement(operation);
    resolver_.RequireWorkspaceSeparation(operation);
    const bool ready_result = operation.Implementation().ProducesReadyScalar();
    CHECK_SPAN((!result && operation.Result().behavior == BufferResultBehavior::None) ||
                   (result && ready_result && operation.Result().behavior == BufferResultBehavior::Value),
               call->span_)
        << "Conservative synchronization: payload-produced scalar/handle readiness is unavailable";
    ConservativeAccessSet effects;
    for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
      const auto effect = operation.ArgumentEffects()[index].data;
      if (effect == BufferAccess::None) continue;
      const auto space = Space(call->args_[index]);
      const auto pipe = operation.Implementation().Pipe();
      if (effect == BufferAccess::Read || effect == BufferAccess::ReadWrite)
        effects.emplace(space, pipe, false, 0);
      if (effect == BufferAccess::Write || effect == BufferAccess::ReadWrite)
        effects.emplace(space, pipe, true, 0);
    }
    for (const auto& effect : operation.Implementation().ImplicitEffects()) {
      const auto pipe = operation.Implementation().Pipe();
      if (effect.data == BufferAccess::Read || effect.data == BufferAccess::ReadWrite)
        effects.emplace(effect.space, pipe, false, 0);
      if (effect.data == BufferAccess::Write || effect.data == BufferAccess::ReadWrite)
        effects.emplace(effect.space, pipe, true, 0);
    }
    CHECK_SPAN(!effects.empty(), call->span_)
        << "Conservative synchronization: command-only payload requires a protocol contract";
    const auto inserted = site_index_.emplace(cut.get(), sites.size()).second;
    CHECK_SPAN(inserted, cut->span_)
        << "Conservative synchronization: shared statement identity has ambiguous insertion cuts";
    sites.push_back({cut,
                     call,
                     operation.Implementation().Pipe(),
                     std::move(operation),
                     std::move(effects),
                     {},
                     {},
                     {},
                     {},
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt});
    inside_child_.push_back(native_child_depth_ != 0);
    // The ordinary service protects the scalar instruction's storage read.
    // Only its selected blocking native return then makes the original SSA
    // available to following addresses/control; it is never a pure query.
    if (ready_result) Bind(result);
  }

  End Read(const StmtPtr& statement) {
    if (child_index_.count(statement.get())) seen_children_.insert(statement.get());
    ++cost.statements;
    if (const auto seq = As<SeqStmts>(statement)) {
      End end;
      const auto wit = window_slices_by_seq_.find(seq.get());
      if (wit == window_slices_by_seq_.end() || wit->second.empty()) {
        for (const auto& child : seq->stmts_) {
          CHECK_SPAN(!end.Terminal(), child->span_)
              << "Conservative synchronization: statements after a terminal cut are unavailable";
          end = Read(child);
        }
        return end;
      }
      const auto& slices = wit->second;
      size_t slice_idx = 0;
      for (size_t i = 0; i < seq->stmts_.size(); ++i) {
        CHECK_SPAN(!end.Terminal(), seq->stmts_[i]->span_)
            << "Conservative synchronization: statements after a terminal cut are unavailable";
        const bool entering_slice = slice_idx < slices.size() && i == slices[slice_idx].begin;
        if (entering_slice) {
          CHECK(native_child_depth_ == 0)
              << "Conservative synchronization: retained child scopes must not overlap";
          seen_slice_children_.insert(slices[slice_idx].child_idx);
          ++native_child_depth_;
        }
        end = Read(seq->stmts_[i]);
        if (slice_idx < slices.size() && i + 1 == slices[slice_idx].end) {
          --native_child_depth_;
          ++slice_idx;
        }
      }
      return end;
    }
    if (const auto assignment = As<AssignStmt>(statement)) {
      ssa_defs_[assignment->var_.get()] = assignment->value_;
      if (const auto call = As<Call>(assignment->value_)) {
        Payload(statement, call, assignment->var_);
      } else {
        Ready(assignment->value_);
        Bind(assignment->var_);
        if (const auto found = gm_roots_.find(assignment->value_.get()); found != gm_roots_.end())
          gm_roots_.emplace(assignment->var_.get(), found->second);
      }
      return {};
    }
    if (const auto eval = As<EvalStmt>(statement)) {
      if (const auto call = As<Call>(eval->expr_))
        Payload(statement, call, nullptr);
      else
        Ready(eval->expr_);
      return {};
    }
    if (const auto branch = As<IfStmt>(statement)) {
      Ready(branch->condition_);
      ++control_depth_;
      scopes_.emplace_back();
      const auto then_end = Read(branch->then_body_);
      PopScope();
      End else_end;
      if (branch->else_body_) {
        scopes_.emplace_back();
        else_end = Read(*branch->else_body_);
        PopScope();
      }
      --control_depth_;
      CHECK_SPAN((!then_end.returned && !else_end.returned) ||
                     (imported_ && imported_->terminal_branch == branch.get()),
                 statement->span_)
          << "Conservative synchronization: nested invocation exits are unavailable";
      const auto count = branch->return_vars_.size();
      CHECK_SPAN(then_end.yield.value_or(0) == count && else_end.yield.value_or(0) == count, statement->span_)
          << "Conservative synchronization: branch result/yield arity mismatch";
      for (const auto& variable : branch->return_vars_) Bind(variable);
      return {};
    }
    if (const auto loop = As<ForStmt>(statement)) {
      CHECK_SPAN(loop->kind_ != ForKind::Parallel, statement->span_)
          << "Conservative synchronization: concurrent loop execution requires another domain contract";
      Ready(loop->start_);
      Ready(loop->stop_);
      Ready(loop->step_);
      ReadLoop(statement, loop->body_, loop->iter_args_, loop->return_vars_, loop->loop_var_, nullptr);
      return {};
    }
    if (const auto loop = As<WhileStmt>(statement)) {
      ReadLoop(statement, loop->body_, loop->iter_args_, loop->return_vars_, nullptr, loop->condition_);
      return {};
    }
    if (const auto yield = As<YieldStmt>(statement)) {
      for (const auto& value : yield->value_) Ready(value);
      return {yield->value_.size(), false};
    }
    if (const auto ret = As<ReturnStmt>(statement)) {
      CHECK_SPAN(control_depth_ == 0 || (control_depth_ == 1 && imported_ && imported_->terminal_branch),
                 statement->span_)
          << "Conservative synchronization: nested invocation exits are unavailable";
      for (const auto& value : ret->value_) Ready(value);
      return {std::nullopt, true};
    }
    CHECK_SPAN(false, statement->span_)
        << "Conservative synchronization: unsupported structured control or command statement: "
        << statement->TypeName();
    return {};
  }

  void ReadLoop(const StmtPtr& loop, const StmtPtr& body, const std::vector<IterArgPtr>& arguments,
                const std::vector<VarPtr>& results, const VarPtr& index, const ExprPtr& condition) {
    CHECK_SPAN(arguments.size() == results.size(), loop->span_)
        << "Conservative synchronization: loop result/initializer arity mismatch";
    for (const auto& argument : arguments) Ready(argument->initValue_);
    scopes_.emplace_back();
    if (index) Bind(index);
    for (const auto& argument : arguments) Bind(argument);
    if (condition) Ready(condition);
    const bool child = child_index_.count(loop.get()) != 0;
    CHECK(!child || native_child_depth_ == 0)
        << "Conservative synchronization: retained child scopes must not overlap";
    if (child) ++native_child_depth_;
    ++control_depth_;
    const auto end = Read(body);
    --control_depth_;
    if (child) --native_child_depth_;
    CHECK_SPAN(!end.returned && end.yield.value_or(0) == arguments.size(), body->span_)
        << "Conservative synchronization: loop backedge lacks matching ready yields";
    PopScope();
    for (const auto& result : results) Bind(result);
  }

  FunctionPtr function_;
  std::string target_;
  SyncControlRangeProver trip_prover_;
  SyncFootprintResolver resolver_;
  bool retain_incoming_diagnostics_;
  bool enable_refresh_kill_;
  bool enable_loop_prefixes_;
  const ImportedLocalSync* imported_;
  bool enable_precise_captures_;
  std::vector<size_t> issue_order_;
  std::unordered_map<const Var*, size_t> definition_order_;
  std::unordered_map<const ForStmt*, std::vector<size_t>> loop_capture_sources_;
  std::unordered_map<size_t, FirstPrefixCapture> first_prefix_deliveries_;
  bool rechecking_prefixes_ = false;
  std::unordered_map<const Var*, std::pair<const ForStmt*, size_t>> prefix_phase_loops_;
  std::unordered_map<size_t, EarlierPrefixDelivery> earlier_deliveries_;
  std::unordered_map<size_t, size_t> reader_record_vertices_;
  std::unordered_map<size_t, std::vector<size_t>> reader_record_dependents_;
  std::unordered_map<const Expr*, ExprPtr> gm_roots_;
  std::unordered_map<const Var*, ExprPtr> ssa_defs_;
  std::vector<std::vector<const Var*>> scopes_;
  std::unordered_set<const Var*> ready_;
  size_t control_depth_ = 0;
  size_t native_child_depth_ = 0;
  std::vector<bool> inside_child_;
  std::unordered_map<const Stmt*, size_t> child_index_;
  std::unordered_set<const Stmt*> seen_children_;
  std::vector<std::array<PipeMask, kPhysicalPipeCount>> child_service_matrices_;
  struct SliceWindow {
    size_t begin = 0;
    size_t end = 0;
    size_t child_idx = 0;
  };
  std::unordered_map<const SeqStmts*, std::vector<SliceWindow>> window_slices_by_seq_;
  std::unordered_set<size_t> seen_slice_children_;
  std::unordered_map<const Stmt*, size_t> site_index_;
  std::vector<std::vector<SiteEffectFiber>> site_effect_fibers_;
  std::vector<std::vector<std::tuple<MemorySpace, size_t, FiberKey>>> site_refresh_fibers_;
  AffineDomainBoundsMap affine_domain_bounds_;
  std::unordered_set<const ForStmt*> loops_with_fibers_;
};
}  // namespace

CompletionSummary SummarizeStraightLineLogicalCompletion(const StmtPtr& body, FunctionType func_type,
                                                         const std::string& target) {
  return Builder::ComputeRecipeCompletionSummary(body, func_type, target, 0, nullptr, nullptr);
}

bool CertifyEarlierPeerPrefixes(const ConservativeSyncAnalysis& baseline, const ImportedLocalSync& imported,
                                const std::string& target, const SyncParameterBounds& parameters,
                                const std::vector<EarlierPeerPrefix>& moves) {
  Builder builder(baseline.Function(), target, {}, {}, /*retain_incoming_diagnostics=*/false,
                  /*enable_refresh_kill=*/false, parameters, std::nullopt, /*enable_loop_prefixes=*/false,
                  &imported, /*enable_precise_captures=*/true);
  builder.Build();
  return builder.CertifyEarlierPrefixes(baseline, moves);
}

ConservativeSyncAnalysis AnalyzeConservativeSync(
    const FunctionPtr& function, const std::string& target,
    const std::vector<std::shared_ptr<const NativeSyncChild>>& children,
    const std::vector<std::shared_ptr<const EventfulSyncChild>>& eventful_children,
    bool retain_incoming_diagnostics, bool automatic_children, bool enable_refresh_kill,
    const SyncParameterBounds& parameters, const std::optional<std::pair<int64_t, int64_t>>& blocks,
    bool enable_loop_prefixes, const ImportedLocalSync* imported, bool enable_precise_captures) {
  CHECK(function && function->ir_stage_ == FunctionIRStage::Buffer)
      << "Conservative synchronization requires Buffer IR";
  CHECK(function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV)
      << "Conservative synchronization requires an owned local AIC/AIV domain";
  backend::RequireLocalSyncExecutionContract(target);
  RequireBufferRuntimeValid(function);
  CHECK(!automatic_children || (children.empty() && eventful_children.empty()))
      << "Automatic child selection cannot replace explicitly supplied child capabilities";
  CHECK(!imported || (imported->owner == function && !automatic_children && children.empty() &&
                      eventful_children.empty() && !enable_refresh_kill && !enable_loop_prefixes))
      << "Imported local synchronization requires its original owner without derived child or overwrite "
         "proofs";
  INTERNAL_CHECK(!enable_precise_captures || imported)
      << "Internal error: precise prefix captures require qualified imported effects";
  AutomaticSyncChildren selection;
  std::vector<std::shared_ptr<const EventfulSyncChild>> selected = eventful_children;
  if (automatic_children) {
    selection = PrepareAutomaticSyncChildren(function, target, parameters, blocks);
    for (const auto& attempt : selection.attempts)
      if (attempt.child) selected.push_back(attempt.child);
  }
  Builder builder(function, target, children, selected, retain_incoming_diagnostics, enable_refresh_kill,
                  parameters, blocks, enable_loop_prefixes, imported, enable_precise_captures);
  if (automatic_children && !selection.inline_subblocks.empty()) {
    builder.RegisterInlineSubblocks(selection.inline_subblocks);
  }
  builder.Build();
  auto exit = builder.Solve();
  builder.cost.retained_facts = exit.size();
  for (const auto& site : builder.sites)
    builder.cost.retained_facts += site.effects.size() + site.incoming.size();
  for (const auto& child : builder.children)
    builder.cost.retained_facts += child.effects.size() + child.incoming.size();
  builder.cost.parent_links = builder.cost.statements ? builder.cost.statements - 1 : 0;
  return ConservativeSyncAnalysis(function, builder.PrimitiveSites(), std::move(exit), builder.cost,
                                  std::move(builder.children), std::move(builder.domains),
                                  retain_incoming_diagnostics, builder.disjoint_gm_, enable_refresh_kill,
                                  std::move(selection));
}
}  // namespace pypto::ir::sync
