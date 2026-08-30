# DSA-RP 设备实验记录

最后汇总日期：2026-08-30。

## 用途

本记录是 DSA-RP 设备证据的长期索引。它同时保留失败的基础设施 campaign 与正面结果，
避免后续分析静默复用已被推翻的数据。每一行列出的 archive 是 source of record；正式结论
必须从其中的表格重新读取，而不能只依赖本摘要。百分比沿用 archive 的符号约定，通常是
candidate/baseline 减一，因此负值表示更快。

不同 campaign 使用了不同 corpus 与 measurement stratum。standalone kernel、single-submit
driver、per-task swimlane 与 whole-parent 结果不能合并。capacity sensitivity row 也不是独立
workload。

## 实际执行设备测量的 campaign

| Campaign archive | 设备证据 | 主要结果与当前解释 |
| ---------------- | -------- | ------------------ |
| `dsa-rp-final-pr-bounded-final.tar.gz` | PR system-test lane | 原 PR 在 `test_dyn_orch_paged_attention` 上失败：DSA-RP 强制要求 top-level `SeqStmts`。按规则未运行性能。 |
| `dsa-rp-final-pr-bounded-ef581ae7-final.tar.gz` | 两条 lane 共 39 个 test；有界 timing | compile precondition 修复后 39 passed、2 skipped；timing 无结论。 |
| `dsa-rp-pr-large-kernel-overnight-ef581ae7-final.tar.gz` | 60 个 timed standalone endpoint；双设备确认 | 六个 program 上 13 个确认 win、零确认 regression；最大 `q_lora_rmsnorm` 约 28.7%。参数重建限制 coverage，因此是探索性证据。 |
| `dsa-rp-paper-three-arm-broad-screen-final.tar.gz` | 20 个确认 kernel，四台设备 | DSA-RP 在每台设备上比 geometry first-fit 快 16-19%。DSA-RP 对 Cypress 没有复现稳定排序。 |
| `dsa-rp-paper-81-four-capacity-screen-final.tar.gz` | 19 个 timed capacity cell，仅四个 kernel 覆盖四种容量 | 仅为 pilot；一个 tight Cypress cell 数值分歧，capture/reconstruction failure 主导 coverage。 |
| `dsa-rp-paper-81-four-capacity-parent-partial.tar.gz` | 40 个 parent cell 有终态 | 七个 arm-specific golden failure 暴露 unsafe placement class；在 correctness 修复前不可用于 paper timing。 |
| `dsa-rp-paper-correctness-then-broad-native-final.tar.gz` | seeded reproducer 与 placement ablation | 隔离 MTP/HCA deterministic failure；最初的 missing-handoff 假设后来撤回。 |
| `dsa-rp-paper-partial-overlap-sync-causal-v2.tar.gz` | 精确 overlap ladder | 证明单条 instruction 内 staggered source/destination overlap 会破坏数据；问题属于 DSA lifetime/alias construction，而非 InsertSync。 |
| `dsa-rp-current-corpus-device-screen-fced3c67-final.tar.gz` | 106 个 native target-only parent cell | 一个确认 regression 与四个 correctness failure；后者最终定位为 loop-return lifetime root 缺失。 |
| `dsa-rp-softmax-pool-correctness-causal-final.tar.gz` | relation ladder 与 maximal-barrier control | 证明非零 offset containment 会出错、同 base containment 通过；最大同步也不能修复，根因是 lifetime construction。 |
| `dsa-rp-paper-canary-v054-9b59f244-final.tar.gz` | 21 个 comparable endpoint，两台设备 | 第一次 lifetime 修复关闭 MTP failure；`prefill_c4_softmax_pool` 仍失败，促成 chained loop-return 修复。 |
| `dsa-rp-loop-return-softmax-canary-48053d242-final.tar.gz` | 四 arm、两设备共 24 个成功 `prefill_c4` run | 修复后所有 arm bit-identical；第二个 parent 因 stock heap-ring deadlock 被排除。 |
| `dsa-rp-paper-broad-stock-gated-536fed244-final.tar.gz` | 315 个 correct target-only parent cell，1,185 个 comparison | 六个确认 effect，但单 kernel 改动被 parent runtime 稀释；主要贡献是 stock-first funnel 与紧凑 archive 规则。 |
| `dsa-rp-current-minimal-drivers-db5a6dcf-final.tar.gz` | 三个 direct driver 加一个 parent control | direct RMSNorm/HC-post 中 penalty-aware placement 比 geometry 快约 8-21%；DSA-RP 与 Cypress 在 noise 内。 |
| `dsa-rp-full-corpus-parent-dispatch-screen-final.tar.gz` | 19 个 target 覆盖四容量 | DSA-RP 对 geometry 有九个确认 win，Cypress 有十四个；DSA-RP 没有系统性胜过 Cypress。 |
| `dsa-rp-dedicated-driver-corpus-886b52614-final.tar.gz` | 64 个 problem-capacity cell，四 arm | DSA-RP 在 30 个 cell 最快、Cypress 在 19 个、geometry variant 在 15 个。所有至少 10% effect 均复现；这是 development dataset v1。 |
| `dsa-rp-weighted-dag-device-validation-2e027d131-final.tar.gz` | 48 个 logical cell，9,920 sample | 首个完整 weighted-DAG test 未胜过 unit reuse cost；unit cost 在 device-decided ordering 上 14/14，DAG 有两个 false-confident error。 |
| `dsa-rp-loop-aware-model-prospective-0820ab418-final.tar.gz` | 两设备 51,840 launch | 12 个 Cypress-vs-DSA-RP cell 均未通过 preregistered effect gate；frozen sync feature 无法区分有用 near-miss。 |
| `dsa-rp-hc-post-exposed-wait-final.tar.gz` | native HC-post，双设备 | MTP regression 未复现；作为 null 的 DsPark 出现 repeatable DSA-RP slowdown，但 swimlane 没有暴露 intra-kernel wait。 |
| `dsa-rp-dspark-sync-address-ablation-final.tar.gz` | 五个合法 endpoint，双设备 | DsPark slowdown 再次未复现；恢复的 PTO handshake 编译成相同 device binary，只能作为 noise control。 |
| `dsa-rp-driver-first-corpus-verification-0fe01d2e-final.tar.gz` | 20 workload、320 logical cell、419 correctness run | 20 个 workload 在四容量、四逻辑 policy 下全部通过，无 placement-correctness failure；没有 timing。 |
| `dsa-rp-driver-first-timing-8d8e76df-final.tar.gz` | 19/20 primary workload，两种 device domain | tight selection 产生大量 physical null；仅 `dspark/rmsnorm.py` 确认 DSA-RP 比 Cypress 快约 2.6-3.6%。native-map control 证明 instrument 可解析 8-33% effect。 |
| `dsa-rp-replay-fixed-prospective-holdout-9b05800db-final.tar.gz` | 八个 frozen workload、九个 target、13,440 sample | DSA-RP 在五个 target 上确认胜过 Cypress，且从未确认更慢；幅度约 3.2-6.1%。这是最强的 prospective policy 证据。 |
| `dsa-rp-four-candidate-physical-penalty-aeba32c70-final.tar.gz` | 四个新 workload、6,400 sample | `kv_score_proj_c128` 确认 DSA-RP 比 Cypress 快约 2.3-2.5%，尽管 sync site 更多；Gumbel 的大 unit-cost gap 是 latency null。 |
| `dsa-rp-kv-gumbel-legal-ablations-2bdc441b0-final.tar.gz` | 五个 relation family、address control、6,400 sample | Gumbel `(2,39)` 是 beneficial causal relation：加入 overlap 删除一个 barrier 并提速 2.1-2.35%。`(38,42)` 约 +1.9%，`(3,38)` 约 +0.4%，`(38,79)` 为 null；KV `(8,22)` 非 causal。 |

## 在有效 timing 前按规则停止的 campaign

| Campaign archive | 停止原因 | 长期结论 |
| ---------------- | -------- | -------- |
| `dsa-rp-paper-canary-9b59f244-final.tar.gz` | frozen PyPTO/PTOAS 在 level 3 不兼容 | 确认 explicit-`tmp` toolchain mismatch；没有 placement 结论。 |
| `dsa-rp-standalone-four-arm-pilot30-ec4151192-final.tar.gz` | parent completion snapshot 不是 per-dispatch golden | 所有 endpoint 相互一致，证明是 reference reconstruction defect；无 timing。 |
| `dsa-rp-standalone-harness-canary-final.tar.gz` | runtime capture 不是 task-adjacent | 证明 `topk_select` 单 task 只写一行，而 completion snapshot 包含后续 task；因此转向 dedicated driver。 |
| `dsa-rp-four-arm-capacity-corpus-f8ff87b04-final.tar.gz` | 少于 20 个 model-eligible standalone case | 证明 compile-only export 缺 dispatch ABI，并纠正 static loop 被误排除的问题。 |
| `dsa-rp-prospective-opportunity-holdout-fd228dbc1-final.tar.gz` | 八个 survivor 中只有三个有 opportunity capacity | 证明十一个 proposed driver 数学不可行，而非 canonical-greedy failure；没有打开 latency。 |
| `dsa-rp-prospective-driver-validation-021354ee-final.tar.gz` | 排除 development overlap 后只剩五个 workload | 发现 replay checker 把 late-eliminated allocation 当成 missing；随后修复。 |
| `dsa-rp-critical-path-penalty-holdout-f9a7e00f-final.tar.gz` | PTOAS divergence 与 control-flow rejection | 导出 7,712 条 realized reuse；未冻结 prediction。 |
| `dsa-rp-pto-isa-duration-calibration-39cfb942d-final.tar.gz` | 仅三个 model-v0 problem 且 Perf-Sim 无法构建 | 精确 duration coverage 最高 14.75%，并暴露 operation metadata 缺失。 |
| `dsa-rp-pto-isa-calibration-canary-7cf5960ea-final.tar.gz` | pinned Perf-Sim source 无法编译 | 修复 type、transfer work size 与 matmul join；记录 scalar constant gap。 |
| `dsa-rp-weighted-dag-calibration-5d30202c-final.tar.gz` | unsupported signature 阻止完整 score | Perf-Sim 与 product-faithful exporter 首次端到端运行；weighted-DAG 未计算。 |
| `dsa-rp-weighted-dag-device-validation-final.tar.gz` | 75 个 node 中 34 个缺 exact/pinned duration | `topk_select_inactive` 是首个完整 existence proof；未使用设备。 |

## 当前证据边界

prospective 八 workload holdout 支持“结构化 penalty-aware DSA-RP 能在真实 kernel 上胜过
Cypress”的结论，但尚未验证当前 analytical penalty model。合法 ablation 解释了原因：
pair cost 依赖上下文且可以为负，exposed latency 取决于完整 post-InsertSync graph，而不只是
relation 或 barrier 数量。下一次 model evaluation 必须在打开 held-out timing 前冻结带符号的
post-InsertSync marginal。
