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

#ifndef PYPTO_CODEGEN_GM_PIPE_LAYOUT_EVIDENCE_H_
#define PYPTO_CODEGEN_GM_PIPE_LAYOUT_EVIDENCE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/program.h"

namespace pypto::codegen::gm_pipe {
using PipeKey = std::pair<int, int>;

struct SetupEvidence {
  ir::FunctionPtr function;
  ir::StmtPtr cut;
  ir::CallPtr call;
};

struct LayoutResult {
  std::map<PipeKey, int64_t> offsets;
  int64_t bytes = 0;
  std::string error;
  int64_t Elements() const { return bytes / 4 + (bytes % 4 != 0); }
};

class WorkspaceContext;

struct LayoutWork {
  size_t statements = 0;
  size_t setup_entries = 0;
  size_t setup_decodes = 0;
  size_t setup_lookups = 0;
  size_t call_edges = 0;
  size_t layout_requests = 0;
  size_t layout_cache_hits = 0;
  size_t functions_visited = 0;
  size_t setup_incidences = 0;
  size_t layouts_rejected = 0;
  size_t binding_requests = 0;
  size_t binding_cache_hits = 0;
  size_t forwarding_edges = 0;
  size_t setup_validations = 0;
  size_t bindings_rejected = 0;
};

// Immutable source facts plus attempt-owned query memo. Retains the exact
// Program and original setup cuts; no native readiness or ownership claim.
// Use one instance per code generation context. Results, including refusals,
// remain immutable and valid while the original Program is retained.
class LayoutEvidence {
 public:
  explicit LayoutEvidence(ir::ProgramPtr program);
  ~LayoutEvidence();
  LayoutEvidence(const LayoutEvidence&) = delete;
  LayoutEvidence& operator=(const LayoutEvidence&) = delete;
  std::shared_ptr<const LayoutResult> WholeProgram();
  std::shared_ptr<const LayoutResult> ReachableFrom(const ir::FunctionPtr& root);
  // Bind the original workspace formal and every contributing call edge.
  // Failed bindings are cached; malformed IR exceptions remain exceptions.
  std::shared_ptr<const WorkspaceContext> BindRoot(const ir::FunctionPtr& root);
  // Direct codegen without a launch owner requires identical independently
  // bound entry layouts and native slot geometry. This qualifies codegen of
  // bundled definitions, not shared launch ownership or peer synchronization.
  // It never silently unions incompatible workspaces.
  std::shared_ptr<const WorkspaceContext> InferCodegenContext();
  // Includes ambiguous or malformed occurrences of the managed formal role.
  // Absence permits legacy spelling only, never a managed layout certificate.
  bool HasWorkspaceFormal() const;
  const SetupEvidence* SetupAt(const ir::FunctionPtr& function, const ir::StmtPtr& cut) const;
  const LayoutWork& Work() const;
  const ir::ProgramPtr& Owner() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};
// Factory-only capability. Its source Program and original identities are
// retained independently of the query cache, avoiding owner/cache cycles.
class WorkspaceContext {
 public:
  ~WorkspaceContext();
  const LayoutResult& Layout() const;
  const ir::FunctionPtr& Root() const;
  // A reduced module may reuse original Function objects, never merely names.
  std::string ValidateProgram(const ir::ProgramPtr& program) const;

 private:
  friend class LayoutEvidence;
  struct Data;
  explicit WorkspaceContext(std::shared_ptr<const Data> data);
  std::shared_ptr<const Data> data_;
};
}  // namespace pypto::codegen::gm_pipe
#endif  // PYPTO_CODEGEN_GM_PIPE_LAYOUT_EVIDENCE_H_
