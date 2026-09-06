# 逻辑分配身份与结构化边界修复 v7

本文对应[英文完整记录](../../../en/dev/proposals/dsa_logical_graph_repair_v7.md)。
仅进行 host 分析，没有新 placement 或设备计时；已有数据属于开发集，不能称为新的前瞻验证。

## 实现

- 从 `allocation_accesses_v1` 恢复同一逻辑分配跨 reshape 的零延迟 RAW/WAR/WAW
  边；不通过物理地址相同或 penalty candidate 推断身份。PTOAS 在加载 duration 和
  placement sidecar 前验证并加入这些基础边。
- 可选 `semantic-boundaries` 用真实条件/循环边界依赖、pipe 顺序、跨 region 内存
  依赖和 yield SSA 依赖替换 `scf.for`/`scf.if` 周围的全连接完成屏障。
  相斥分支不建立同迭代顺序边；未知边界保留保守处理，正距离递归边仍保留。
- research recognizer 补充 `tile.tpush_to_aiv` 的 L0→UB 源读取。
  Gate access 75 读取 1,024 字节 accumulator；五个子问题的 solver 可见字段均未变化。
- 仅对分析副本运行既有 frontend pipe lowering，保留原始 product 输入与证据。

未知子范围默认拒绝。显式保守模式以整个逻辑 allocation 包络代替未知范围，逐项记录，
不能称为精确恢复。已解析 duration 也分别标记实测校准、解析公式和固定 Perf-Sim 近似。
配对分析可能产生二次规模工作/输出；它是研究模式，不是默认产品优化。

## 结果与限制

历史 14 个已测 cell 的 51 个 endpoint 得到数值子图评分：43 个使用源范围，8 个使用
保守包络。Gate 的 60 个 child endpoint 全部评分，其中 12 个使用保守范围。
14 个历史基础图和 20 个 Gate child/capacity 基础图都验证跨 arm 不变。

Gate 的命名双槽通道恢复 AIC 75→AIV 57/92，以及 pop→free 57→59、92→94。
原始 orchestration 恢复 `ffn_norm → x_norm_quant`、`ffn_norm → mixed gate`、
`mixed gate → route_sort` 的 tensor 依赖。保留 `{AIC,AIV,AIV}` 组和分支上下文，
不把 child bound 相加冒充 parent latency。

有限循环、运行时 lane/core 展开和显式 fence 尚未完整组合。
所有结果仍标记 `invocation_model_complete=false`。

固定 8–16 cycle 权重下，五个双设备超过 2% 的已测方向中，三个正确、一个 tie、
一个反向；cache-write 的已知胜出仍预测为 tie，hc_post 仍反向。
64–256 下是四对一错，但不能据此追选权重。Gumbel near-null 仍有非零预测。
Gate 尚无完整 parent 预测，混合方向科学门槛未通过，不推进增量 planner 或新计时。

完整表格、哈希、重现工具和 280 个 Python 测试及六个 FileCheck 结果见英文记录。
本轮最多两个串行评分 worker，各限 1.2 GiB；构建一个 worker、2 GiB、禁用 swap。
