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

/**
 * @file common.h
 * @brief Common macros, constants, and utility definitions
 *
 * This header provides commonly used macros and constants that are shared
 * across the PyPTO codebase, including:
 * - Compiler hints and attributes
 * - Utility macros for code generation
 * - Build configuration constants
 * - nanobind module configuration
 */

#ifndef PYPTO_CORE_COMMON_H_
#define PYPTO_CORE_COMMON_H_

#include <cstdint>

namespace pypto {

// ============================================================================
// Version Information
// ============================================================================

#define PYPTO_VERSION_MAJOR 0
#define PYPTO_VERSION_MINOR 1
#define PYPTO_VERSION_PATCH 0

// ============================================================================
// IR Constants
// ============================================================================

// Dynamic dimension constant for tensor/tile shapes
// Use -1 to represent dimensions that are unknown at compile time
constexpr int64_t kDynamicDim = -1;

// ============================================================================
// Aligned Arithmetic
// ============================================================================
//
// Round-to-multiple helpers used wherever a tile/buffer extent must satisfy a
// hardware granule (e.g. the cube's 16-element fractal, the vector's 32-byte DMA
// block). `constexpr` and check-free by design: callers pass a backend-constant
// alignment (always >= 1), so there is no runtime input to validate.

/// Largest multiple of `a` that is <= x. Precondition: a >= 1.
constexpr int64_t AlignDown(int64_t x, int64_t a) { return (x / a) * a; }

/// Smallest multiple of `a` that is >= x. Precondition: a >= 1.
constexpr int64_t AlignUp(int64_t x, int64_t a) { return ((x + a - 1) / a) * a; }

/// Ceiling of a / b. Precondition: b >= 1.
constexpr int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Compile-time checks (exact-multiple, ragged, already-aligned, unit granule).
static_assert(AlignUp(66, 8) == 72 && AlignUp(64, 8) == 64 && AlignUp(0, 8) == 0, "AlignUp");
static_assert(AlignDown(66, 8) == 64 && AlignDown(64, 8) == 64, "AlignDown");
static_assert(CeilDiv(66, 8) == 9 && CeilDiv(64, 8) == 8 && CeilDiv(1, 1) == 1, "CeilDiv");

// ============================================================================
// nanobind Module Configuration
// ============================================================================

// Default docstring for the nanobind module
#define PYPTO_NANOBIND_MODULE_DOC "PyPTO core library"

// ============================================================================
// Compiler Hints and Attributes
// ============================================================================

#define PYPTO_ALWAYS_INLINE __attribute__((always_inline))
#define PYPTO_UNUSED __attribute__((unused))
#define PYPTO_STR_CONCAT_IMPL(__x, __y) __x##__y
#define PYPTO_STR_CONCAT(__x, __y) PYPTO_STR_CONCAT_IMPL(__x, __y)
}  // namespace pypto
#endif  // PYPTO_CORE_COMMON_H_
