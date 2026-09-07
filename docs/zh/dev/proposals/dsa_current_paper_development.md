# 当前论文开发集

2026-09-07 的 host-only 汇总；没有新 placement 或设备计时。这是回顾性开发
证据（retrospective development evidence），不是 prospective 验证。
[英文文档](../../../en/dev/proposals/dsa_current_paper_development.md) 为准；
英文目录中的机器表格是唯一数据副本。

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
