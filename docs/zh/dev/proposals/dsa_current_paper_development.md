# 当前论文开发集

2026-09-07 的 host-only 汇总；没有新 placement 或设备计时。这是回顾性开发
证据（retrospective development evidence），不是 prospective 验证。
[英文文档](../../../en/dev/proposals/dsa_current_paper_development.md) 为准；
英文目录中的机器表格是唯一数据副本。

## 延迟引导 placement search：实现状态

下方历史表比较四个算法，**不含延迟引导 placement 的设备计时**。
现在独立研究工具 `python -m pypto.tools.dsa_latency_planner` 可以生成这种
placement：从显式指定的完整合法 seed 开始，将 allocation class 贪心移动到
对齐的地址边界。每个 candidate 先验证 DSA hard constraint，再对完整物理
reuse edge 的并集评分：

```text
score(P) = LP(non-reusing SSA/pipe graph + E_reuse(P), w)
           - LP(non-reusing SSA/pipe graph, w)
```

工具复用现有 complete-placement oracle 及其支持的 loop expansion，不运行
InsertSync、不读取设备 timing，也不依赖 penalty-candidate catalog 枚举物理
reuse。缓存完整 map 的分数，最后绕过缓存复核。这是有预算的 greedy relocation，
**不是** C++ canonical greedy，也未实现增量 longest-path 更新；不保证全局最优
或遍历所有地址。Pool 与 structured pipeline member 保持不变；colocation class
整体移动，并显式报告预算耗尽。

```bash
PYTHONPATH=python python -m pypto.tools.dsa_latency_planner \
  --problem problem.json --seed-solution geometry.solution.json \
  --objective latency --schedule schedule.jsonl --graph research-graph.txt \
  --model duration-model.json --max-evaluations 128 \
  --output-root build/latency-search
```

新目录包含 `solution.json` 和 `search.json`，记录输入 hash、接受的移动、预算、
分数及 duration evidence class。`--objective structural` 使用**相同搜索**和
原有 weighted reuse sum。诊断 control 必须采用相同 seed 和预算；与 structural
canonical greedy 比较时，搜索算法和目标都发生变化，不能声称只隔离了 cost model。

延迟目标要求完整 non-fallback coverage 和单一静态分数。Captured runtime branch
profile、未解析 dynamic score、缺失 access provenance、输入漂移和不支持的几何
约束均 fail closed。Pinned approximation 仍明确标记，不能称为 calibrated
signature。发布完整 parent replay map 前，必须逐个处理所有函数。

三个 host integration canary 使用统一诊断权重 16 cycles、每次搜索 32 个 score
evaluation，结果如下：

| Function | Structural-search objective | Latency-search penalty (cycles) |
| --- | ---: | ---: |
| `build_bias` | 13 → 6 | 461 → 0 |
| `mtp_hidden_norm_quant` | 30 → 29 | 1017 → 168 |
| `split_pre_post` | 30 → 24 | 214 → 174 |

它们采用既有 export 与 geometry seed，不是新冻结的 19-workload 五算法 panel。
搜索均耗尽预算。这证明实际 placement selection，不证明设备加速或完成权重校准。
Host-only 复现工具为 `tests/tools/run_dsa_latency_planner_audit.py`，只读结构性
artifact。新 map 仍需设备 correctness 与平衡 timing 验证。

### 五算法 release 准备

`tests/tools/prepare_dsa_five_arm_release.py` 不读取 timing，生成固定 panel 的全部
95 个 host slot。保留四个 baseline map，复核历史 digest，用独立 pair-scan checker
验证 native 与 selected capacity。全局同步权重固定为 16 cycles，不拟合；每个 child
采用 geometry-FF seed、最多 128 次 score evaluation。保留相同搜索的 structural
control。只有所有 child 均完整可评分时，才发布 latency parent map，不使用 fallback。

在 planner commit `ae16f3a93`，**5/19 workload** 有完整五算法 map：
`dspark_o_lora_quant`、`mtp_dequant`、`mtp_hidden_norm_quant`、`build_bias`、
`split_pre_post`。其余 14 个明确排除原因包括 child graph 缺失、不支持的 branch/
dynamic-loop score、`tpush` join 或 `tcmp` duration 缺失。Blocked parent 内个别
function 可评分，不等于 parent 可评分。这比之前的 per-function bound 分析严格，
不表示失去 device measurability。

Packet 携带 graph input，固定 research PTOAS
`062d4b16f27f7a6baef91b5d6cfdcf6fe5f2f26f`。评分这些 graph 不需要在 device host
构建 exporter。可选 thin Git bundle 以 `9d72b90ff49f67749ede982b21b30d98508028fb`
为 prerequisite，包含该 commit，不含未提交修改。Product compiler 仍为 official
v0.57。外部 task 必须报告 partial coverage，不能声称 19 个全部成功。

发布使用 file allowlist（总计 128 MiB、单文件 25 MiB）、精确 manifest、sidecar，
并对全新解压目录验证。使用发布的 tooling checkout：

```bash
PYTHONPATH=python python tests/tools/prepare_dsa_five_arm_release.py verify PACKET
```

该命令验证 95 个 slot、每个发布 child map 的两种 capacity profile、native fingerprint、
map identity 和 capacity-overflow negative control；不替代设备 replay/comparability/
correctness 检查。

## 数据与计数

- [主表](../../../en/dev/proposals/data/current-paper-development/paper-primary.tsv)：
  四个逻辑算法、每个 workload 的两个实际设备编号、完整 map 和计量范围。
- [容量 manifest](../../../en/dev/proposals/data/current-paper-development/primary-manifest.json)
  与 [选择审计](../../../en/dev/proposals/data/current-paper-development/capacity-selection.tsv)。
- [全部比较](../../../en/dev/proposals/data/current-paper-development/pairwise.tsv)
  与 [敏感性数据](../../../en/dev/proposals/data/current-paper-development/sensitivity.tsv)。
- [逐项预测](../../../en/dev/proposals/data/current-paper-development/predictor-evaluation.tsv)、
  [匹配覆盖率汇总](../../../en/dev/proposals/data/current-paper-development/predictor-matched-summary.tsv)
  和 [全部覆盖率汇总](../../../en/dev/proposals/data/current-paper-development/predictor-summary.tsv)。
- [hidden-norm](../../../en/dev/proposals/data/current-paper-development/hidden-norm-case-study.tsv)
  与 [扩展候选审计](../../../en/dev/proposals/data/current-paper-development/reserved-candidate-audit.tsv)。

32 个已测配置中有 4 个内在不确定性配置被排除。28 个确定性配置包含 25 个
语义 target，但只有 **19 个独立 driver/参数 workload**。同一 parent 的不同
target 标签不等于独立 kernel 计时。主表为 **19 workload / 38 设备行**；
其他 9 个配置仅作容量敏感性分析。没有因为某算法慢而丢弃测量。

其中 13 个 driver 只有一个函数和一个 DSA instance，6 个是多函数 parent。
归档 harness 测的是整个 driver 的 device window，不是 per-dispatch latency。
两种模式分别标为 `SINGLE_FUNCTION_DRIVER_WINDOW` 和 `PARENT_PROGRAM_WINDOW`；
不除以 dispatch 数，也不混合汇总。不同 workload 使用不同设备对。
完整 map 覆盖整个 driver，但容量收紧只作用于记录的 target pool，其他 pool
保持 native；`half` 不表示所有 child pool 同时减半。

## 选择与统计

仅在已经测量的配置中，优先 Cypress 存在带 penalty 的复用、三个主要算法
完整 map 不同、且 DSA-RP 的完整 map unit objective 更低的配置。依次最大化
objective 差和复用关系差异，最后用更小容量与 artifact ID 决胜。
没有 opportunity 时仍保留 control。选择器拒绝 latency 字段；其他配置不删除。
这是结构选择但仍是回顾性分析，并且有利 unit objective 的选择会影响后续准确率。

报告值为三个 launch median 的 median，而不是原报告合并 60 samples 的 median。
每个 launch 有 20 个设备样本；bootstrap 以 launch 为单位。相同完整 map
共享计时；设备不合并。两个设备均达到 2% 且 95% launch-bootstrap 区间不含零，
才记为有方向。仅三个 launch、每设备固定 arm 顺序，因此这些是探索性判断，
不是经过多重比较校正的强确认结论。

四个主 workload 的 DSA-RP 快于 Cypress：`idx_qr_proj_dequant`、
`mtp_hidden_norm_quant`、MTP `hc_post`、`split_pre_post`。
12 个 small/null，3 个未确认或设备不一致。退化和 null 在完整表中保留。

## 模型比较

仅 12/19 个主 workload 有计量范围匹配的单函数 DAG bound；6 个 parent
需要图组合，另有 1 个单函数不可评分。可评分不等于完整 invocation latency
模型；duration 含 pinned approximation，不全是 calibrated signature。

在同样 12 个 workload、全局权重 8–64 cycles 下：

| 预测量 | 方向正确 | 方向错误 | 漏掉方向（tie） | small/null 上 tie | small/null 上严格排序 | 设备未确认 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unit reuse | 4 | 0 | 0 | 2 | 4 | 2 |
| 完整 placement DAG bound | 3 | 0 | 1 | 3 | 3 | 2 |
| 物理区间相交对数 | 1 | 2 | 1 | 1 | 5 | 2 |
| Pool peak 之和 | 0 | 2 | 2 | 4 | 2 | 2 |

权重 96–256 仍漏掉 hidden norm，并在另一个 small/null 上给出严格排序。
本次不拟合或选择权重。严格排序没有校准成百分比，不能自动称为大幅度错误。
后两项只是 Cypress-inspired proxy：原始 allocation 相交数不等于 contracted
search-node alias 数，而且缺 relaxed-edge 数；不能称为完整 Cypress objective。
当前 DAG 未优于 unit penalties，四个方向一致样本也不足以可靠做留一校准。

## 有限 hidden-norm case study

half 下 DSA-RP/Cypress 在设备 0/1 为 −8.44%/−7.82%，unit cost 从 10 降至 4。
两者 base LP 都是 3589 cycles，权重 8–256 的 DAG penalty 全为零。
Cypress/DSA-RP 的 distance-zero reuse edge 数为 38/42，正距离 recurrence
为 12/4；barrier/set/wait site 为 25/20/20 和 27/22/22。

共同 44-node base 的 636 个零距离边中有 604 个 control 边。Duration 来源：
1 calibrated signature、14 analytical、2 shape approximation、27 pinned Perf-Sim
approximation。控制顺序 envelope 和未完成的有限循环组合是具体线索，不是已证实
的失配原因。更多权重或更少同步 site 都不能直接解释此结果。
**在此停止**：不拟合该 kernel 特有常数，不让因果调查阻止扩展与论文撰写。

## Prospective 扩展

12 个 reserved row 实际只有 6 个来源 family：cache write(1)、build bias(4)、
merge/rope pack(4)、KV matmul(1)、KV RMS/rope(1)、Q-rope preparation(1)。
build bias 与 Q-rope 已有开发集近亲；带编号副本未证明为新 workload。

隔离原始 helper，使用确定性输入和直接 Torch reference；记录 source body、
shape、tiling、dtype 和语义 problem identity。名字不同但实现/shape 相同的副本
必须合并，相关变体按 family 分组。必要时增加新 family 以达到 8–12 个新可运行
workload；补充候选必须在计时前登记，不能看性能后替换。null 不是正确性失败。
设备正确性与 timing-blind 容量 freeze 在计时前完成。仍用现有四个算法；
模型不支持不得删除设备可测项，缺少预测必须显式报告。

## 复现

```bash
python tests/tools/consolidate_dsa_paper_development.py \
  --analysis-root build/dsa-paper-current-analysis \
  --output-root docs/en/dev/proposals/data/current-paper-development
```

源归档为 `dsa-rp-rebased-corpus-device-continuation-3eabcfd22-final.tar.gz`，SHA-256
`a797b5b0d93fd963980197df8559c740e2ca05c4251e500dde39728e17e969e7`。
工具核验原表哈希与预测 seal，并记录源记录哈希、endpoint pins。
紧凑表格是派生证据，不代替原始归档和 launch 记录。
