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

// Benchmark the native solvers on a pre-layout Program exported by export_auto_layout_input.py.
// The extraction, IR writeback, warmup and serialization are outside the timed region.
// Standalone CMake builds configured with BUILD_TESTING=ON expose this target.
// source .claude/skills/testing/load-env.sh
// cmake --build build --target auto_layout_solver --parallel "$PYPTO_BUILD_JOBS"
// build/tests/ut/cpp/auto_layout_solver <pre-layout-program.msgpack>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "pypto/backend/common/backend_config.h"
#include "pypto/core/error.h"
#include "pypto/ir/serialization/deserializer.h"
#include "pypto/ir/transforms/auto_layout/extractor.h"
#include "pypto/ir/transforms/auto_layout/solver.h"
#include "pypto/ir/transforms/auto_layout/writer.h"

namespace al = pypto::ir::auto_layout;

int main(int argc, char** argv) {
  using pypto::backend::BackendType;
  BackendType target = BackendType::Ascend910;
  bool dump_costs = false;
  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--dump-costs") {
      dump_costs = true;
    } else if (flag == "--backend" && i + 1 < argc) {
      const std::string arch = argv[++i];
      if (arch != "a2a3" && arch != "a5") {
        std::cerr << "Unknown backend: " << arch << '\n';
        return 2;
      }
      target = arch == "a5" ? BackendType::Ascend950 : BackendType::Ascend910;
    } else {
      std::cerr << "Unknown or incomplete argument: " << flag << '\n';
      return 2;
    }
  }
  if (argc < 2) {
    std::cerr
        << "Usage: auto_layout_solver <pre-layout-program.msgpack> [--backend a2a3|a5] [--dump-costs]\n";
    return 2;
  }
  const auto program = std::dynamic_pointer_cast<const pypto::ir::Program>(
      pypto::ir::serialization::DeserializeFromFile(argv[1]));
  if (!program) throw pypto::ValueError("Expected serialized Program");
  pypto::backend::BackendConfig::SetBackendType(target);
  const auto problem = al::ExtractLayoutInstance(program);
  if (dump_costs) {
    std::cout << "backend\top_id\top_name\tinputs\toutput\tinvocations\tcycles_per_invocation\n";
    for (const auto& op : problem.operations) {
      for (const auto& config : op.configs) {
        std::cout << pypto::backend::BackendTypeToString(target) << '\t' << op.op_id << '\t' << op.op_name
                  << '\t';
        for (size_t i = 0; i < config.input_layouts.size(); ++i) {
          if (i > 0) std::cout << ',';
          std::cout << config.input_layouts[i].ShortName();
        }
        std::cout << '\t' << config.output_layout.ShortName() << '\t' << op.invocations << '\t'
                  << std::setprecision(10) << config.op_cost_cycles << '\n';
      }
    }
    return 0;
  }
  const al::IndexedProblem indexed(problem);
  const auto movable = std::count_if(indexed.Ops().begin(), indexed.Ops().end(),
                                     [](const auto& op) { return op.costs.size() > 1; });
  size_t max_scope = 0;
  for (const auto& value : indexed.Values()) {
    size_t active = 0;
    for (auto u : value.participants) {
      if (indexed.Ops()[u].costs.size() > 1) ++active;
    }
    max_scope = std::max(max_scope, active);
  }

  al::LayoutSolverConfig cfg;
  cfg.collect_treewidth_stats = false;
  const std::string path = argv[1];
  const std::string filename = path.substr(path.find_last_of('/') + 1);
  const auto extension = filename.find_last_of('.');
  const std::string name = extension == 0 ? filename : filename.substr(0, extension);
  const std::array<std::string, 4> modes = {"dp", "greedy", "greedy_fm", "auto"};
  std::cout << "case\tops\tmovable\tmax_factor_scope\tmode\trepeat\tms\tmodeled_"
               "cycles\tevaluations\ttermination\twriteback\n";
  for (int round = 0; round <= 5; ++round) {
    for (size_t i = 0; i < modes.size(); ++i) {
      const auto& mode = modes[(i + round) % modes.size()];
      al::LayoutSolverResult result;
      std::string rejected;
      const auto start = std::chrono::steady_clock::now();
      try {
        if (mode == "dp") {
          result = al::SolveTreewidthDP(problem, cfg);
        } else if (mode == "greedy") {
          result = al::SolveGreedy(problem, std::nullopt, cfg);
        } else if (mode == "greedy_fm") {
          result = al::SolveGreedyFM(problem, cfg);
        } else {
          result = al::SolveAutoLayout(problem, std::nullopt, cfg);
        }
      } catch (const pypto::ValueError& e) {
        const std::string reason(e.what());
        const bool dp_limit =
            reason.find("exceeds max_") != std::string::npos ||
            reason.find("exceeded max_evaluations=") != std::string::npos ||
            reason.find("exceeding the min-fill decomposition limit of ") != std::string::npos;
        if (mode != "dp" || !dp_limit) throw;
        rejected = reason.substr(0, reason.find('\n'));
      }
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      if (!round) continue;
      std::cout << name << '\t' << indexed.NumOps() << '\t' << movable << '\t' << max_scope << '\t' << mode
                << '\t' << round << '\t' << std::setprecision(10) << ms << '\t';
      if (rejected.empty()) {
        if (round == 1 && result.writeback_issues.empty()) {
          al::ApplyLayoutAssignment(program, result.WritebackAssignment());
        }
        std::cout << std::setprecision(17) << result.TotalCostCycles() << '\t' << result.stats.evaluated_moves
                  << '\t' << result.stats.termination_reason << '\t'
                  << (result.writeback_issues.empty() ? "ok" : result.writeback_issues.front()) << '\n';
      } else {
        std::cout << "NA\tNA\t" << rejected << "\tNA\n";
      }
    }
  }
}
