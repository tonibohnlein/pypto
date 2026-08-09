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

#ifndef SRC_IR_TRANSFORMS_AUTO_TILE_CUBE_GRAPH_H_
#define SRC_IR_TRANSFORMS_AUTO_TILE_CUBE_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {

/** One admitted tensor-level matmul in source topological order. */
struct CubeMatmulNode {
  AssignStmtPtr stmt;
  CallPtr call;
  VarPtr lhs;
  VarPtr rhs;
  VarPtr output;
  int64_t lhs_producer = -1;
  int64_t rhs_producer = -1;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  DataType operand_dtype = DataType::FP32;
  DataType accumulator_dtype = DataType::FP32;
  DataType storage_dtype = DataType::FP32;
  bool is_sink = false;
};

/** Admitted homogeneous tensor-level matmul DAG and output contract. */
struct CubeGraph {
  FunctionPtr function;
  std::vector<CubeMatmulNode> matmuls;
  size_t sink = std::numeric_limits<size_t>::max();

  // Sink aliases retained for the single-request planner and emitter. They are
  // populated from ``matmuls[sink]`` after admission.
  AssignStmtPtr matmul_stmt;
  CallPtr matmul_call;
  VarPtr lhs;
  VarPtr rhs;
  VarPtr output;
  VarPtr explicit_output_buffer;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  DataType operand_dtype = DataType::FP32;
  DataType accumulator_dtype = DataType::FP32;
  DataType storage_dtype = DataType::FP32;
};

/** Fail-closed result of admitting one tensor-level function to cube AutoTile. */
struct CubeAdmissionResult {
  bool supported = false;
  CubeGraph graph;
  std::string reason;
  std::exception_ptr failure;
};

/** Return true when the marked function contains any tensor-level cube op. */
[[nodiscard]] bool ContainsCubeOperation(const FunctionPtr& function);

/**
 * Admit a homogeneous cube-only tensor DAG.
 *
 * Every compute statement must be a static rank-2, non-transposed
 * ``tensor.matmul``. The returned sink and every transitive producer are
 * scheduled as one kernel; unsupported functions are not partially tiled.
 */
[[nodiscard]] CubeAdmissionResult AdmitCubeGraph(const FunctionPtr& function, const ProgramPtr& program);

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_AUTO_TILE_CUBE_GRAPH_H_
