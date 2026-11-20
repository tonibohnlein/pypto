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

#ifndef SRC_IR_TRANSFORMS_BUFFER_IR_ADMISSION_H_
#define SRC_IR_TRANSFORMS_BUFFER_IR_ADMISSION_H_

#include <optional>
#include <string>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"

namespace pypto::ir {
// Diagnostic taxonomy for the existing selector, not a second capability or
// effects registry. Unknown support is a refusal; no exception is intercepted.
enum class BufferAdmissionProperty {
  DeviceHelper,
  Disabled,
  Backend,
  Target,
  ParameterABI,
  ReturnABI,
  TypeLayout,
  TileShape,
  ValidShape,
  ExpressionForm,
  StatementForm,
  ScalarArithmetic,
  OperationRecipe,
  AtomicStore,
  TransferOptions,
  WindowGeometry,
  TransferTypes,
  MatrixRecipe,
  RegionType,
  RegionValidState,
  RegionYield,
  LoopCarry,
  LoopStep,
  TransferParameter,
  SlotGeometry,
  CanonicalSlots,
  DynamicValidRecipe,
  KernelShape,
  PeerTransport,
  PeerCompilationUnit,
};
inline const char* BufferAdmissionPropertyName(BufferAdmissionProperty property) {
  switch (property) {
    case BufferAdmissionProperty::DeviceHelper:
      return "device_helper_abi";
    case BufferAdmissionProperty::Disabled:
      return "buffer_ir_disabled";
    case BufferAdmissionProperty::Backend:
      return "configured_backend";
    case BufferAdmissionProperty::Target:
      return "target_architecture";
    case BufferAdmissionProperty::ParameterABI:
      return "parameter_abi";
    case BufferAdmissionProperty::ReturnABI:
      return "return_abi";
    case BufferAdmissionProperty::TypeLayout:
      return "type_layout";
    case BufferAdmissionProperty::TileShape:
      return "static_tile_shape";
    case BufferAdmissionProperty::ValidShape:
      return "valid_shape";
    case BufferAdmissionProperty::ExpressionForm:
      return "expression_form";
    case BufferAdmissionProperty::StatementForm:
      return "statement_form";
    case BufferAdmissionProperty::ScalarArithmetic:
      return "scalar_arithmetic";
    case BufferAdmissionProperty::OperationRecipe:
      return "operation_recipe";
    case BufferAdmissionProperty::AtomicStore:
      return "atomic_store_recipe";
    case BufferAdmissionProperty::TransferOptions:
      return "transfer_options";
    case BufferAdmissionProperty::WindowGeometry:
      return "window_geometry";
    case BufferAdmissionProperty::TransferTypes:
      return "transfer_types";
    case BufferAdmissionProperty::MatrixRecipe:
      return "matrix_recipe";
    case BufferAdmissionProperty::RegionType:
      return "region_type";
    case BufferAdmissionProperty::RegionValidState:
      return "region_valid_state";
    case BufferAdmissionProperty::RegionYield:
      return "region_yield";
    case BufferAdmissionProperty::LoopCarry:
      return "loop_carry";
    case BufferAdmissionProperty::LoopStep:
      return "positive_loop_step";
    case BufferAdmissionProperty::TransferParameter:
      return "transfer_parameter";
    case BufferAdmissionProperty::SlotGeometry:
      return "slot_geometry";
    case BufferAdmissionProperty::CanonicalSlots:
      return "canonical_slot_index";
    case BufferAdmissionProperty::DynamicValidRecipe:
      return "dynamic_valid_recipe";
    case BufferAdmissionProperty::KernelShape:
      return "load_or_fill_and_store";
    case BufferAdmissionProperty::PeerTransport:
      return "delegated_peer_transport";
    case BufferAdmissionProperty::PeerCompilationUnit:
      return "delegated_peer_compilation_unit";
  }
  INTERNAL_UNREACHABLE << "Internal error: unknown Buffer admission property";
}

class BufferAdmission {
 public:
  // Borrowed context exists only during the synchronous visitor. Save the first
  // failed predicate immediately: reporting at visitor return would blame an
  // enclosing branch for a failure in a nested operation.
  class Context {
   public:
    Context(const IRNode*& current, const IRNode* node) : current_(current), previous_(current) {
      current_ = node;
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() { current_ = previous_; }

   private:
    const IRNode*& current_;
    const IRNode* previous_;
  };
  Context At(const IRNode* node) { return Context(current_, node); }
  bool supported() const { return !property_; }
  void Require(bool condition, BufferAdmissionProperty property) {
    if (condition || property_) {
      return;
    }
    property_ = property;
    if (current_) {
      const auto* call =
          current_->GetKind() == ObjectKind::Call ? static_cast<const Call*>(current_) : nullptr;
      operation_ = call ? call->op_->name_ : current_->TypeName();
      location_ = current_->span_.is_valid() ? current_->span_.to_string() : "";
    }
  }
  void AppendTo(Function& function) const {
    if (!property_) {
      return;
    }
    function.attrs_.emplace_back(kAttrBufferAdmissionVersion, int{1});
    function.attrs_.emplace_back(kAttrBufferAdmissionProperty,
                                 std::string(BufferAdmissionPropertyName(*property_)));
    function.attrs_.emplace_back(kAttrBufferAdmissionStage, std::string("SelectBufferIRFunctions"));
    function.attrs_.emplace_back(kAttrBufferAdmissionOperation, operation_);
    function.attrs_.emplace_back(kAttrBufferAdmissionLocation, location_);
  }

 private:
  const IRNode* current_ = nullptr;
  std::optional<BufferAdmissionProperty> property_;
  std::string operation_, location_;
};
}  // namespace pypto::ir
#endif  // SRC_IR_TRANSFORMS_BUFFER_IR_ADMISSION_H_
