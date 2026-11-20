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

#include "pypto/backend/common/buffer_sync_contract.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/buffer_axis_recipes.h"
#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_indexed_recipes.h"
#include "pypto/backend/common/buffer_scalar_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/backend/common/pto_native_ops.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"

namespace pypto::backend {
static_assert(!std::is_default_constructible_v<BufferSyncImplementation>);
static_assert(!std::is_default_constructible_v<BufferSyncOperation>);
const char* BufferNativeOperation(const ir::CallPtr& call) {
  using ir::IsOp;
  if (IsOp(call, "buffer.load")) return "pto.tload";
  if (IsOp(call, "buffer.store")) return "pto.tstore";
  if (IsOp(call, "buffer.read")) return "pto.tgetval";
  if (IsOp(call, "buffer.write")) return "pto.tsetval";
  if (IsOp(call, "buffer.copy")) return "pto.tmov";
  if (IsOp(call, "buffer.extract")) return "pto.textract";
  if (IsOp(call, "buffer.insert")) return "pto.tinsert";
  if (IsOp(call, "buffer.transpose")) return GetPtoNativeOpSpec("tile.transpose").native_op;
  if (IsOp(call, "buffer.matmul")) return "pto.tmatmul";
  if (IsOp(call, "buffer.matmul_acc")) return "pto.tmatmul.acc";
  if (IsOp(call, "buffer.ci") || IsOp(call, "buffer.ci_workspace")) return "pto.tci";
  if (auto r = FindBufferElementwiseRecipe(call->op_->name_)) return r->native_op;
  if (auto r = FindBufferIndexedRecipe(call->op_->name_)) return r->native_op;
  if (auto r = FindBufferAxisRecipe(call->op_->name_)) return r->native_op;
  if (auto r = FindBufferConversionRecipe(call->op_->name_)) return r->native_op;
  return nullptr;
}

const LocalSyncExecutionContract& RequireLocalSyncExecutionContract(const std::string& target) {
  CHECK(target == "a2a3") << "Unsupported physical target: native variants are qualified only for a2a3";
  // Existing local-pipeline contract: scalar prefixes execute serially. ALL
  // completes asynchronous payloads at invocation exit; it does not consume flags.
  // A2/A3 local event semantics: publication observes completion of the issued
  // source prefix; acquisition consumes the matching flag before its queue can
  // advance. This is the backend semantic binding used by direct handoffs and
  // ordered acknowledgment protocols, not a device/wrapper qualification result.
  // CANN SetFlag/WaitFlag contract (including consuming flag clear):
  // https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0181.html
  static const LocalSyncExecutionContract contract{true, "system.bar_all", true, true};
  return contract;
}

BufferSyncImplementation SelectBufferSyncImplementation(const ir::CallPtr& call, ir::FunctionType core,
                                                        const std::string& target) {
  using ir::As;
  using ir::BufferType;
  using ir::FunctionType;
  using ir::IsOp;
  using ir::MemorySpace;
  using ir::PipeType;
  CHECK(core == FunctionType::AIC || core == FunctionType::AIV)
      << "Payload pipe is unavailable on the owning synchronization core";
  const auto name = BufferNativeOperation(call);
  CHECK(name) << "Unavailable instruction contract in synchronization import: " << call->op_->name_;
  std::string macro(name + 4);
  std::transform(macro.begin(), macro.end(), macro.begin(),
                 [](unsigned char c) { return c == '.' ? '_' : std::toupper(c); });
  BufferSyncImplementation result{PipeType::ALL, "pto::" + macro, core, target};
  if (IsOp(call, "buffer.read") || IsOp(call, "buffer.write")) {
    CHECK(core == FunctionType::AIV && target == "a2a3")
        << "Blocking local scalar access is qualified only for A2/A3 AIV Vec storage";
    // TGETVAL/TSETVAL select Tile::GetValue/SetValue: synchronous typed UB
    // dereferences, without GM DCache accesses, asynchronous scratch or flags.
    // Operand/coordinate validation belongs to the registered Buffer recipe.
    result.pipe_ = PipeType::S;
    result.produces_ready_scalar_ = IsOp(call, "buffer.read");
    result.audited_signal_free_variant_ = true;
  } else if (IsOp(call, "buffer.load") || IsOp(call, "buffer.store")) {
    const bool load = IsOp(call, "buffer.load");
    const auto gm_tensor = As<ir::TensorType>(call->args_[load ? 0 : 3]->GetType());
    if (SupportsBufferScalarTransfer(call->args_, load)) {
      CHECK(core == FunctionType::AIV && target == "a2a3")
          << "Scalar point Buffer DMA requires qualified A2/A3 AIV execution";
      // An ND unit partition in either supported rank gives nBurst=1 and
      // lenBurst=4 in the selected A2/A3 b32 path. TLOAD may write a local
      // alignment tail. TSTORE transfers only the four destination bytes.
      // Conservative import covers the full local carrier.
      result.audited_signal_free_variant_ = true;
    } else if (gm_tensor && gm_tensor->shape_.size() == 1) {
      CHECK(core == FunctionType::AIV && target == "a2a3" &&
            SupportsBufferVector1DTransfer(call->args_, load))
          << "Rank-one Buffer DMA requires a qualified A2/A3 AIV vector 1D transfer contract";
    }
    const auto local = As<BufferType>(call->args_[load ? 3 : 0]->GetType());
    CHECK(local && (local->memory_space_ == MemorySpace::Vec ||
                    local->memory_space_ == (load ? MemorySpace::Mat : MemorySpace::Acc)))
        << "Only GM/Mat, Acc/GM and GM/Vec DMA contracts are qualified";
    result.pipe_ = load                                       ? PipeType::MTE2
                   : local->memory_space_ == MemorySpace::Vec ? PipeType::MTE3
                                                              : PipeType::FIX;
    if (!load && call->GetKwarg<int>("atomic", 0) != 0) {
      const auto tensor = As<ir::TensorType>(call->args_[3]->GetType());
      CHECK(target == "a2a3" && tensor &&
            SupportsBufferStoreMode(call->GetKwarg<int>("atomic", 0), local->memory_space_, local->dtype_,
                                    tensor->dtype_))
          << "Atomic store requires the qualified A2/A3 FP32 Acc writeback";
      // pto-isa 5487a6b0 TStore.hpp: the default two-operand Acc overload
      // brackets TStoreAcc with SetAtomicAdd/SetAtomicNone, without scratch or
      // numbered events. Native configuration applies to the subsequent issued
      // transfer; this does not complete prior GM accesses or other workers.
      // https://asc.gitcode.com/api/SIMD-API/c_api/atomic/key_features.html
      result.audited_signal_free_variant_ = true;
    }
  } else if (IsOp(call, "buffer.copy") || IsOp(call, "buffer.extract")) {
    const bool extract = IsOp(call, "buffer.extract");
    const auto source = As<BufferType>(call->args_[0]->GetType());
    const auto destination = As<BufferType>(call->args_[extract ? 3 : 1]->GetType());
    if (source && destination && source->memory_space_ == MemorySpace::Vec &&
        destination->memory_space_ == MemorySpace::Vec) {
      result.pipe_ = PipeType::V;
      if (!extract) {
        const auto bytes = DenseBufferBytes(source);
        constexpr uint64_t kBlockBytes = 32;
        CHECK(target == "a2a3" && core == FunctionType::AIV && IsAlignedDenseVecBufferGeometry(source) &&
              source->shape_ == source->valid_shape_ && bytes &&
              *bytes / kBlockBytes <= std::numeric_limits<uint16_t>::max())
            << "Vec copy needs full static dense A2/A3 AIV block-burst geometry";
        // The registered Vec copy has identical descriptors. A2/A3 TMovToVec
        // selects one copy_ubuf_to_ubuf burst on PIPE_V, with no implicit
        // storage or numbered events; the importer proves actual placement.
        result.audited_signal_free_variant_ = true;
      }
    } else {
      CHECK(source && destination && source->memory_space_ == MemorySpace::Mat &&
            (destination->memory_space_ == MemorySpace::Left ||
             destination->memory_space_ == MemorySpace::Right))
          << "Only Mat to operand-buffer or Vec to Vec movement is qualified";
      result.pipe_ = PipeType::MTE1;
      result.mat_to_operand_extract_ = extract;
    }
  } else if (IsOp(call, "buffer.insert")) {
    const auto source = As<BufferType>(call->args_[0]->GetType());
    const auto destination = As<BufferType>(call->args_[3]->GetType());
    CHECK(source && destination && source->memory_space_ == MemorySpace::Acc &&
          destination->memory_space_ == MemorySpace::Mat)
        << "Only Acc to Mat FIXPIPE insertion is qualified";
    result.pipe_ = PipeType::FIX;
    result.acc_to_mat_insert_ = true;
  } else if (IsOp(call, "buffer.matmul") || IsOp(call, "buffer.matmul_acc")) {
    result.pipe_ = PipeType::M;
    const auto lhs = As<BufferType>(call->args_[0]->GetType());
    const auto rhs = As<BufferType>(call->args_[1]->GetType());
    const auto dst = As<BufferType>(call->args_[2]->GetType());
    // Existing Mmad size qualification, not a C->I ordering fact. The analysis
    // still proves identical physical accumulator storage and no interference.
    // https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html
    result.accumulator_size_eligible_ =
        target == "a2a3" && lhs && rhs && dst && lhs->memory_space_ == MemorySpace::Left &&
        rhs->memory_space_ == MemorySpace::Right && dst->memory_space_ == MemorySpace::Acc &&
        lhs->shape_[0] > 0 && lhs->shape_[0] <= 4095 && lhs->shape_[1] > 0 && lhs->shape_[1] <= 4095 &&
        rhs->shape_[1] > 0 && rhs->shape_[1] <= 4095 && lhs->shape_[1] == rhs->shape_[0] &&
        dst->shape_[0] == lhs->shape_[0] && dst->shape_[1] == rhs->shape_[1] &&
        (lhs->shape_[0] / 16) * (rhs->shape_[1] / 16) >= 10;
  } else {
    if (const auto conversion = FindBufferConversionRecipe(call->op_->name_))
      CHECK(conversion->kind == BufferConversionKind::Cast ||
            conversion->kind == BufferConversionKind::FillPad)
          << "Synchronization conversion import requires a qualified AIV cast or fillpad";
    result.pipe_ = PipeType::V;
  }
  if (IsOp(call, "buffer.ci_workspace")) {
    CHECK(target == "a2a3" && core == FunctionType::AIV)
        << "Buffer sequence generation requires A2/A3 AIV explicit workspace form";
    // pto-isa a2a3/TCI.hpp: the explicit-workspace TCI variant executes pure
    // vector instructions on explicit tmp without EVENT_ID/TMP_UB_OFFSET.
    result.audited_signal_free_variant_ = true;
  } else if (IsOp(call, "buffer.ci")) {
    CHECK(false) << "Synchronization import requires explicit workspace buffer.ci_workspace";
  }
  if (IsOp(call, "buffer.cast")) {
    const auto source = As<BufferType>(call->args_[0]->GetType());
    const auto target_type = As<BufferType>(call->args_[1]->GetType());
    if (source && target_type &&
        ((source->dtype_ == DataType::BF16 && target_type->dtype_ == DataType::FP32) ||
         (source->dtype_ == DataType::FP32 && target_type->dtype_ == DataType::BF16))) {
      // pto-isa 5487a6b0 TCvt.hpp: selected direct BF16/FP32 head/tail loops
      // consume destination valid rows/columns and explicit source/destination
      // strides, including runtime-row descriptors. Integer scratch/event
      // branches are not selected. Retain existing conservative catalog claims.
      CHECK(target == "a2a3") << "Direct BF16/FP32 conversion requires a qualified native implementation";
    }
  }
  if (IsOp(call, "buffer.fillpad")) {
    CHECK(target == "a2a3" && core == FunctionType::AIV)
        << "FillPad requires the qualified A2/A3 AIV implementation";
    // The 2/4-byte Vec branch of A2/A3 TFillPad.hpp uses copies and vector
    // duplication on explicit operands. The one-byte helper with a private
    // V-to-S event is excluded by the registered dtype contract.
    result.vec_fillpad_non_byte_ = true;
  }
  if (IsOp(call, "buffer.rsqrt") || IsOp(call, "buffer.rsqrt_workspace")) {
    CHECK(target == "a2a3") << "Rsqrt requires a qualified native implementation";
    // pto-isa 5487a6b0 TUnaryOp.hpp: the admitted two-operand default
    // TRSQRT_IMPL invokes TUNARY_IMPL<RsqrtOp> / vrsqrt, and the explicit
    // high-precision TRSQRT_IMPL uses vector_dup/vsqrt/vmul/vdiv on explicit tmp.
    // Neither uses hidden temporary or numbered-event operations.
    result.audited_signal_free_variant_ = true;
  }
  if (const auto row = FindBufferAxisRecipe(call->op_->name_); row && row->reduction && row->axis == 1) {
    CHECK(target == "a2a3") << "FP32 row reduction requires a qualified native implementation";
    // pto-isa 5487a6b0 TRowSum/TRowMax dispatch FP32 to the generic vector
    // reduction helper, including runtime-row descriptors. Scratch is explicit;
    // the integer-only EVENT_ID0 scalar reduction branch is not selected.
    // Inactive output rows are untouched, never an implicit zero-padding fact.
    // Keep the catalog's existing conservative integer-variant reservation;
    // selecting a narrower resource declaration is a separate contract change.
  }
  if (IsOp(call, "buffer.col_expand_mul")) {
    CHECK(target == "a2a3") << "Column broadcast requires a qualified native implementation";
    // FP32 row-major TColExpandMul selects NormMode for fixed full columns
    // and CountMode for dynamic valid metadata. Both issue VMUL over explicit
    // inputs/output without temporary storage or numbered events. CountMode
    // sets the column count mask, loops active rows and restores mask state.
    // Runtime metadata relations and a positive-row guard are checked at the
    // function boundary by BufferRuntimeValidAnalysis; -1 fields prove none.
    // pto-isa 5487a6b0: a2a3/TColExpandMul.hpp and TColExpandBinOp.hpp.
    result.audited_signal_free_variant_ = true;
  }
  if (const auto axis = FindBufferAxisRecipe(call->op_->name_); axis && !axis->reduction && axis->axis == 1) {
    CHECK(target == "a2a3" && (IsOp(call, "buffer.row_expand_add") || IsOp(call, "buffer.row_expand_sub") ||
                               IsOp(call, "buffer.row_expand_mul") || IsOp(call, "buffer.row_expand_div") ||
                               IsOp(call, "buffer.row_expand_max") || IsOp(call, "buffer.row_expand_min")))
        << "Row broadcast implicit effects require a qualified native implementation";
    // Registry validation selects FP32 Mode1 (a contiguous column carrier),
    // with no explicit temporary operand. Runtime-row CountMode has the same
    // vbrcb staging: ceil(rows/8) carrier groups are read, bounded by the
    // aligned physical column allocation. The enclosing access contract must
    // include those rounded reads, not claim an exact active-row footprint. The pinned A2/A3 TRowExpandBin
    // helper fills and reads TMP_UB_OFFSET. Its small-row paths use <=32 256-byte repeats; large-row paths
    // reuse <=30 repeats, within the same 8KiB range. This is an enclosure of native scratch, not its exact
    // used-byte footprint. Internal fill->use barriers do not discharge the next call's overwrite. pto-isa
    // 5487a6b0: common/constants.hpp and npu/a2a3/TRowExpandBinOp.hpp.
    result.implicit_effects_.push_back({MemorySpace::Vec, 184 * 1024, 192 * 1024, ir::BufferAccess::ReadWrite,
                                        "a2a3-row-broadcast-mode1-implicit-scratch", false});
    // These six selected wrappers and their shared Mode1 helper contain
    // PIPE_V barriers, but no SET/WAIT or compiler-allocated event operations.
    // This positive source audit, not absence from the signal catalog, supplies
    // the no-numbered-event premise for ADD/SUB/MUL/DIV/MAX/MIN.
    result.audited_signal_free_variant_ = true;
  }
  const bool vector = core == FunctionType::AIV;
  CHECK(vector ? (result.pipe_ == PipeType::MTE2 || result.pipe_ == PipeType::MTE3 ||
                  result.pipe_ == PipeType::V ||
                  (result.pipe_ == PipeType::S && (IsOp(call, "buffer.read") || IsOp(call, "buffer.write"))))
               : (result.pipe_ == PipeType::MTE2 || result.pipe_ == PipeType::MTE1 ||
                  result.pipe_ == PipeType::M || result.pipe_ == PipeType::FIX))
      << "Payload pipe is unavailable on the owning synchronization core";
  return result;
}

void RequireBufferSyncWorkspaceSeparation(
    const BufferSyncOperation& operation,
    const std::function<std::optional<BufferSyncByteWindow>(const ir::ExprPtr&)>& window) {
  const auto& call = operation.Operation();
  for (const auto& implicit : operation.Implementation().ImplicitEffects()) {
    for (size_t arg = 0; arg < call->args_.size(); ++arg) {
      if (operation.ArgumentEffects()[arg].data == ir::BufferAccess::None) continue;
      const auto& operand = call->args_[arg];
      const auto buffer = ir::As<ir::BufferType>(operand->GetType());
      const auto space = buffer ? buffer->memory_space_ : ir::MemorySpace::DDR;
      if (space != implicit.space) continue;
      const auto bound = window(operand);
      CHECK_SPAN(bound && bound->begin >= 0 && bound->end > bound->begin &&
                     (bound->end <= implicit.begin || implicit.end <= bound->begin),
                 call->span_)
          << "Native implicit workspace separation is unproved for an explicit operand";
    }
  }
}

BufferSyncOperation QualifyBufferSyncOperation(const ir::CallPtr& call, ir::FunctionType core,
                                               const std::string& target) {
  auto& registry = ir::OpRegistry::GetInstance();
  registry.ValidateBufferCall(call);
  auto implementation = SelectBufferSyncImplementation(call, core, target);
  const auto& entry = registry.GetEntry(call->op_->name_);
  std::vector<ir::BufferArgEffect> effects;
  effects.reserve(call->args_.size());
  for (size_t arg = 0; arg < call->args_.size(); ++arg)
    effects.push_back(entry.GetBufferArgEffect(arg, call->kwargs_));
  return BufferSyncOperation(call, std::move(implementation), std::move(effects), entry.GetBufferResultSpec(),
                             BufferSyncOperationKind::Payload);
}

std::optional<BufferSyncOperation> QualifyBufferSyncNonPayload(const ir::CallPtr& call,
                                                               const std::string& target) {
  using ir::IsOp;
  const bool query = IsOp(call, "buffer.get_block_idx") || IsOp(call, "buffer.get_block_num") ||
                     IsOp(call, "buffer.get_subblock_idx");
  const bool allocation = IsOp(call, "buffer.alloc") || IsOp(call, "buffer.alloc_multi");
  const bool alias =
      IsOp(call, "buffer.subview") || IsOp(call, "buffer.reshape") || IsOp(call, "buffer.slot");
  const bool setter = IsOp(call, "buffer.set_validshape");
  if (!query && !allocation && !alias && !setter) return std::nullopt;
  RequireLocalSyncExecutionContract(target);
  auto& registry = ir::OpRegistry::GetInstance();
  registry.ValidateBufferCall(call);
  CHECK(call->kwargs_.empty() && call->attrs_.empty())
      << "Descriptor/scalar synchronization contract requires the existing attribute-free native recipe";
  const auto& entry = registry.GetEntry(call->op_->name_);
  std::vector<ir::BufferArgEffect> effects;
  for (size_t arg = 0; arg < call->args_.size(); ++arg) {
    const auto effect = entry.GetBufferArgEffect(arg, call->kwargs_);
    INTERNAL_CHECK(effect.data == ir::BufferAccess::None)
        << "Internal error: descriptor/scalar recipe unexpectedly has payload effects";
    effects.push_back(effect);
  }
  const auto result = entry.GetBufferResultSpec();
  const auto expected = query        ? ir::BufferResultBehavior::Value
                        : allocation ? ir::BufferResultBehavior::Allocate
                        : alias      ? ir::BufferResultBehavior::Alias
                                     : ir::BufferResultBehavior::None;
  INTERNAL_CHECK(result.behavior == expected)
      << "Internal error: selected descriptor/scalar recipe disagrees with registered result behavior";
  return BufferSyncOperation(
      call, std::nullopt, std::move(effects), result,
      query ? BufferSyncOperationKind::ReadyScalar : BufferSyncOperationKind::Descriptor);
}

std::vector<sync::IsaResourceClaim> BufferSyncImplementation::Claims() const {
  RequireLocalSyncExecutionContract(target_);
  const auto declaration = GetBackendInstance(BackendType::Ascend910)->FindIsaMacroResources(macro_);
  if (audited_signal_free_variant_) {
    CHECK(!declaration || declaration->claims.empty())
        << "Audited signal-free native variant conflicts with its resource declaration";
    return {};
  }
  std::vector<sync::IsaResourceClaim> result;
  if (!declaration) return result;  // Only the selected, audited variants above.
  for (const auto& claim : declaration->claims) {
    using sync::CoreScope;
    using sync::IsaIdSource;
    using sync::IsaResourceLifetime;
    using sync::IsaSyncMode;
    using sync::ResourceKind;
    if (std::find(claim.modes.begin(), claim.modes.end(), IsaSyncMode::Manual) == claim.modes.end()) continue;
    // This exclusion is authorized by the selected Mat->operand extract or
    // Acc->Mat FIXPIPE insert variant, never by the macro's name alone. The Vec
    // TEXTRACT / TINSERT helpers remain reserved when executed on AIV Vec.
    if ((mat_to_operand_extract_ || acc_to_mat_insert_) && claim.pool.core == CoreScope::AIV) continue;
    // The macro catalog includes Handle32BAlignedPad_Byte. Suppress only
    // that helper's known manual claim for the selected non-byte variant.
    const sync::PoolKey byte_pad_pool{ResourceKind::PipelineEvent, CoreScope::AIV, ir::PipeType::V,
                                      ir::PipeType::S, 0};
    if (vec_fillpad_non_byte_ && claim.pool == byte_pad_pool && claim.source == IsaIdSource::Fixed &&
        claim.lifetime == IsaResourceLifetime::Invocation && claim.ids.size() == 1 &&
        claim.ids.front() == 0) {
      continue;
    }
    CHECK(claim.pool.kind == ResourceKind::PipelineEvent &&
          claim.pool.core == (core_ == ir::FunctionType::AIV ? CoreScope::AIV : CoreScope::AIC) &&
          claim.source == IsaIdSource::Fixed && claim.lifetime == IsaResourceLifetime::Invocation)
        << "Unavailable native flag owner/variant binding: " << macro_;
    result.push_back(claim);
  }
  return result;
}
}  // namespace pypto::backend
