# Reuse weight 回顾验证，2026-09-05

后续 [v6 图审计](dsa_cache_gate_graph_audit_v6.md) 发现 cache-write 缺少 RAW 路径。
下文 resolved graph inputs 不代表完整正确的 latency graph；这些数字保留为历史模型输出。

本轮目标是论文实验数据，不是修改 PyPTO planner，也没有新增设备计时。
在本轮读取已有 timing 表前，对固定全局 grid
`0,8,16,24,32,48,64,96,128,160,192,256` 重复生成了逐字节一致的预测。
这些 timing 已是已知 development 数据，不能称为 prospective 或盲测验证。

## 结果

51 个历史 endpoint、14 个 problem-capacity cell 的 graph 输入现已可解析。
补充的 fp32 `tmax`、scalar `tcmps`（包括 NE 内部反转）、`tdiv` 和 `tsel`
来自 pinned lowering，标为 `pinned_analytical_model`，不是实测精确 signature。
假设完整 valid tile 和独立空队列；select 使用逐行独立执行的上包络近似。
DAG/loop 分数仅是该假设模型内的界，不是硬件延迟界或完整 invocation 预测。

在 weight 8 和 16，两台设备上同向且均至少 2% 的五个 DSA-RP 胜出中，四个
得到正确严格排序。`kv_and_cache_write/tight` 实测快 3.05%/3.93%，但全部 weight
均预测平局，recurrence bound 也无法区分；unit cost 则预测方向正确。
两个低于 1% 的 null 在 8/16 仍预测平局；24 及以上开始错误区分 Gumbel。
这里的幅度筛选不是独立的置信区间检验。

Gate 从 `f9a7e00f` 加诊断性 access/loop provenance 重建，五个问题的 solver-visible
字段全部匹配。新增 metadata 会改变 fingerprint，因此保留历史 placement，并使用
原有 placed PTO。60 个 child endpoint（五个函数、四档容量、三种物理策略）中，36 个
可评分；`gate_aic`/`gate_aiv` 仍受 mixed-function composition 和最终 trace join 阻塞。
native 的 `ffn_norm` 在 weight 8/16 分别倾向 Cypress 8/17 cycles，但这不是整个 Gate
parent 的延迟预测，不能忽略其余 child 或从 parent 时间反推 child 时间。

**结论：**8–16 cycle 区间尚未通过混合方向验证，不能据此集成 incremental planner。
下一步研究 cache-write 平局和完整 Gate composition。

## 数据与复现

- [逐 cell 分数与设备比值](../../../en/dev/proposals/data/dsa_host_cell_scores_v5.tsv)
- [全局 weight 敏感性](../../../en/dev/proposals/data/dsa_host_weight_grid_v5.tsv)
- [Gate child 分数与 recurrence 差值](../../../en/dev/proposals/data/dsa_gate_partial_scores_v5.tsv)
- [来源、证据分类、阻塞项与哈希](../../../en/dev/proposals/data/dsa_host_weight_validation_v5.json)

`tests/tools/score_frozen_dsa_graph_manifest.py --help` 描述串行 host-only scorer。
它使用 exact placed PTO、replay solution、原问题和 metadata-only export，拒绝
solver-visible 字段变化或不完整 join，并明确不认证完整 invocation model。
focused duration/graph tests：234 通过。冻结 compiler 使用单 worker、2500 MiB/no-swap
cgroup 限制构建，未在 `/tmp` 中构建。
