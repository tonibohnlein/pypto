# Cache-write 与 Gate 图审计，2026-09-05

这是[权重验证 v5](dsa_host_weight_validation_v5.md) 的 host-only 后续。
没有新设备计时、输入调优、solver 修改或 planner 集成，仍是开发集证据。

## Cache-write

kv_and_cache_write/tight 的既有 DSA-RP 优势为 3.05%/3.93%，不是另一个约 7% 的
rope/cache-write 案例。独立检查复现了全部 24 个 native LP 分数，但基础图缺少：

| 逻辑 allocation | 生产 access / node | 消费 access / node |
| --- | --- | --- |
| 5 | 24 / 4，trowmax | 27 / 6，tmax |
| 9 | 29 / 8，tdiv | 33 / 10，trowexpandmul |

证据来自 allocation_accesses_v1 的唯一 dominating writer 和完整读范围覆盖，
不是地址相等。reshape view 被降为不同 alloc_tile SSA value，getAliasRoot 无法从这些
值恢复逻辑关系。仅在标为诊断的图中补 RAW 后，baseline 从 548 变为 854 model cycles。
权重 8、16 仍然平局；到固定 grid 的 96 才分离为 Cypress 59、DSA-RP 30，不能因此
为这个案例选择 96。还需一个全局权重通过混合方向数据验证。

另有 84 条保守 control edge，将 14 个 prelude operation 全部连到六个 loop-body
operation，表示 loop 入口前完成而不只是指令顺序。诊断未删除这些边，动态 trip count、
逐次 cache-row predicate 与 pipe 语义也尚未完整组合。确认的是图一致性缺陷，
不是设备加速的因果解释；只补 RAW 尚不能救活 8–16 区间。

## Gate

importer 改为按指定函数的 raw-PTO operation/provenance 唯一匹配 final trace，
保留 virtualElse placeholder，并拒绝重复 ID、不完整计数、歧义和可检测的交错。
PTOAS 的 function-name 选项在不改动 mixed module 的情况下选择定义函数，不存在则
报错。这避免替换 peer 为 stub，但不等于 mixed execution composition。

重新分析 60 个 child endpoint：ffn_norm、route_sort、x_norm_quant 共 36 个可评分，
其余 24 个仍不完整：gate_aic 的 access 75 是没有 OpPipeInterface 的 tpush_to_aiv，
被基础图跳过，product lowering 后则为 PIPE_FIX 的 tpush。product v0.57 没有
mlir-disable-threading 选项；首次带该参数的失败输出已被取代，后续损坏或歧义的
mixed AIV debug trace 仍被拒绝。

orchestration 包含 8-block ffn_norm、x_norm_quant、空 pre-route task、16-block
mixed {AIC,AIV,AIV} gate、1-block route_sort 和 phase-fence dummy。必须组合 tensor
依赖、pipe transport、block 数和 runtime overlap，不能简单相加 child LP 或取
max(AIC,AIV) 作为 parent latency。

## 验证与下一步

[数据、分数、blocker 与 hash](../../../en/dev/proposals/data/dsa_cache_gate_graph_audit_v6.json)。
audit_dsa_cache_write_graph.py 提供 RAW witness 与 LP 独立核对；
score_frozen_dsa_graph_manifest.py 的 product-cache 选项在核对 manifest、assembler
和 source hash 后复用 product log。工作数据在 build/dsa_cache_gate_audit_v6。

241 个 focused Python test 和三个 PTOAS FileCheck 命令通过。analysis executable
单 worker、2 GiB/zero-swap 构建；分析与测试每个 job 串行、1.2–1.5 GiB cap，
没有 /tmp 大型产物。

下一步保留 logical view identity、明确 control/pipe boundary 语义，并在 frontend
pipe lowering 后导出 Gate、核对 product equivalence。通过后只用既有计时重跑原 grid。
当前结果不支持集成 incremental planner。
