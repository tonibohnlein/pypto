# AutoFuse 混合 Cube/Vector 调度契约

**状态：** 可构建的 `C->V` 增量以及第一个精确的
`C,C->V->C` dense-SwiGLU 增量已在 `PYPTO_AUTOFUSE_MIXED=1` 后实现。
显式 dual-AIV FIFO lane bridge 按函数限定作用域、感知 pipe descriptor，并且字节精确：
它只改写目标 AIV 函数，给仅使用 pipe 的 AIV 函数增加一个私有 runtime lane
参数，并根据每个 pipe 端点自身的 tile 形状和 dtype 派生 entry offset。frontend
logical ID 用于验证 IR 连线；PTOAS 可以重编号生成的 `TPipe` template ID，因此
backend 根据保留的方向和 slot 几何进行绑定，而不假设数值相等。

通用 `[M,N] + [1,N]` 修正为 `tile.col_expand_add` 后，silicon 已完成单向 C2V
epilogue 闭环（连续 50/50 次通过，无漂移）。dense SwiGLU 尚未完成 silicon
闭环：旧 lowering 把两个 C2V projection channel 和一个 V2C activation reply
合并成单个双向 FIFO，而 primitive 诊断表明该合并协议下的 split-lane completion
不安全。当前 host revision 修复了这一 model/emit 违约：版本化 solver descriptor
会一直传递到 `ExpandMixedKernel`，并生成三个独立 logical pipe，保留精确
frontend ID、字节宽度、slot 数、workspace 范围和 runtime lane offset。PTOAS
v0.55 在首次 sentinel 中把 C++ template ID 从 `0/1/2` 重编号为 `0/2/4`；backend
现在按方向和 slot 几何匹配 physical declaration，同一 descriptor 若需要不同
offset 则 fail closed。host 结构与 tensor replay 测试均通过，但修复后的三 pipe
协议仍需新的 910B sentinel。主机测试还验证了
tensor 级等价性以及完整的
`ExpandMixedKernel -> SkewCrossCorePipeline -> AutoTileMatmulL0` 结构。各引擎内部
的工作继续以同构 vector/cube 契约为准；本文只定义引擎边界处新增的契约。

## 1. 硬件与执行模型

Ascend 910B 没有直接的 UB 到 Mat/L1 通路。AIC 与 AIV 之间的张量由生产者写
入 GM，再由消费者从 GM 读取，因此融合不会消除跨引擎流量。收益来自单次启动，
以及连续流水项使用不同 GM FIFO 槽时的引擎重叠。

逻辑资源是 24 个组，每组包含一个 cube lane 与两个 vector lane：

```text
group g = AIC g + AIV 2*g, 2*g+1
```

空间工作分给各组，vector 阶段的行再分给两个 AIV lane。每个 mixed 解同时描述
组网格以及组内流水项循环；仅有多个全局 tile 并不能证明同一组存在可重叠的后继项。

## 2. MixedSchedulePlan

候选无关的最大同引擎 stage 与跨引擎 transfer 在子图创建时只分析一次。获胜或
强制配置再轻量地派生 `MixedSchedulePlan`，避免把大计划存入代价缓存。计划记录：

- 算法种类、stage 成员和 transfer 方向；
- 空间分区、活动组数、split-K；
- 同构 stage 视图：cube GM 到 L1 K window 与 vector `VectorStreamPlan`；
- 流水轴、chunk、项数、stage 数与 skew 深度；
- FIFO 的 pipe/bundle ID、有效形状、槽大小、槽数和预留字节；
- AIV 的行/列切分及 lane 数；
- 独立派生的 `model_overlap_granted` 与 `overlap_implementable`。

dense-SwiGLU 计划还记录输入、中间和输出维度，feature-chunk 循环，gate/up 的 K
window、down feed window，以及跨 feature chunk 存活的 FP32 down accumulator。
这些是 mixed 组合事实，不是对同构 tiling 的第二份实现。

获胜计划通过私有、版本化 FIFO descriptor 跨越 tensor-to-tile 边界。每个逻辑
crossing 对应一个 record 和一个独立单向 physical queue。`ExpandMixedKernel`
先校验方向、形状、字节数与顺序，再把该 record 的 ID 写到
`tpush`/`tpop`/`tfree`。first-matmul 与 accumulate 两个互斥 branch 对同一
activation 的重复使用共享 reply ID；gate/up 这类形状相同但 SSA source 不同的
张量不会合并。expansion 完成后会删除该私有 descriptor。

## 3. 流水模式与忠实性

当前安全 skew 支持“一个有序 push bundle 后接一个回复 pop”。bundle 可以包含
SwiGLU 的 gate/up 两个独立 pipe ID，但所有 push 必须位于第一个 pop 之前，
且 op 顺序与 ID 都属于协议。pop 后再次
push 表示第二次往返，必须降级为串行，防止改变 FIFO 顺序。
若 reply 位于 conditional 内，两个 branch 必须具有完全相同的 FIFO op 与 ID；
首个 matmul/后续 `matmul_acc` 因而算作一个逻辑 pop。任何 path-dependent
conditional 协议都降级为串行。

`V` 与 `C` 表示最大同构 stage，而不是单个算子或物理 core。因此相连的
`C->V->V->C` 源图会折叠为三 stage 的 `C->V->C` 协议；两个 vector 算子仍属于
同一个逻辑 vector stage。910B 每组的两个物理 AIV core 由
`MixedVectorSplit` 单独描述，它们执行该逻辑 stage 的空间分片。相反，两个独立
vector 分支若向 matmul 返回两个不同 tensor，就是两个逻辑 vector stage，需要
双回复 bundle；当前 skew pass 不支持，production AutoFuse 会切分该图。

候选无关拓扑显式记录 `MixedCrossCoreProtocol`：`OneWay`、
`SingleRoundTripBundle` 或 `Unsupported`。bundle 协议记录 producer stage、peer
stage、sink stage，以及 producer/reply bundle 中精确的 transfer 索引。只有该
descriptor 与 `SkewCrossCorePipeline` 兼容时，模型才允许单次往返重叠；emitter
在构造 mixed scope 前再次校验同一 descriptor。识别协议本身不等于允许计价：
generic cost 当前只接受一个 producer 与一个 reply；更大的 bundle 必须由 dense
SwiGLU 这样的精确算法提供全部 stage-local cost 和跨 stage lifetime。

代价模型只有在 emitter 能构造同样的项循环、FIFO 深度和 skew 时才可使用
跨引擎 `max`。串行 prologue、drain 和 ragged tail 始终加法计价。每个 crossing
张量都支付一次 GM 写和一次 GM 读；FIFO 槽必须覆盖同时存活的项。

## 4. 复用同构模型

`Ascend910BMixed` 不重新拟合 cube 或 vector 工作。各 stage 使用相同的 grounded
同构原语：

- cube 的 MAC、extract、GM 到 L1 和 L1 到 L0 项；
- vector 的 primitive、流量、UB lifetime 和阶段局部重叠项。

mixed 包装只增加跨界流量、FIFO 容量、跨引擎 wavefront，以及生命周期跨越
stage 的状态。纯 cube/vector 组直接交给同构模型。

## 5. Dense-SwiGLU 第一增量

精确支持的子图为：

```text
gate = matmul(x, w_gate)       C
up   = matmul(x, w_up)         C
act  = swiglu(gate, up)        V
out  = matmul(act, w_down)     C
```

activation 必须是精确的
`neg -> exp -> scalar_add(1) -> recip -> mul -> mul -> cast` 源链。cube 输入为
BF16/FP16，gate/up 输出为 FP32，activation tile 降为低精度，down 输出为 FP32。
任何中间投影都不得逃逸。

最终输出的 M/N tile 分给最多 24 个组。组内 feature 循环：

1. 使用计划的同构 cube K window 计算 gate 和 up；
2. 将两个 tile 作为有序双 push bundle 发送；
3. 在两个 AIV lane 上执行物化的同构 vector 计划；
4. 返回一个 activation tile；
5. 首个 chunk 初始化 down matmul，后续 chunk 累加到同一个 FP32 accumulator；
6. 最后一个 chunk 后仅 drain 一次最终输出。

AutoFuse 只在 `UP_DOWN` mixed scope 中发射 tensor 级 matmul/vector IR。
`ExpandMixedKernel` 创建 `tpush`/`tpop`/`tfree`，`SkewCrossCorePipeline` 实现单次
往返 wavefront，`AutoTileMatmulL0` 独占所有 L0 M/N/K tile 与 buffer 选择。
AutoFuse 不发射原始跨核指令，也不附加 L0 plan。

A2A3 codegen 使用 runtime subblock 参数显式分离两个 AIV lane 的 FIFO entry，
因为 simpler MIX dispatch 不会设置 native hardware subblock register。当前
bridge 由 split FIFO op 驱动，而不是由 tensor 索引驱动：已有 subblock 参数时
直接复用，否则只给选中的 AIV PTOAS 函数增加 wrapper 私有参数。frontend ID
验证 endpoint 连线；重编号后的 physical `TPipe` 按
`(direction, slot_size, slot_num)` 匹配，并根据该 descriptor 获得独立
consumer/producer offset。同一 descriptor 只有在所需 offset 完全相同时才允许
重复，否则 fail closed；分组输出中的 AIC sibling 函数不参与该重写。
planned path 支持多个不同静态 size/dtype 的 pipe；dynamic/ragged transfer 仍会
fail closed，直到 launch path 提供等价的 native subblock ID 与动态端点契约。
在 910B 上，如果 vector writer 同时消费 MemRef-less `tpop_from_aic`，MemoryReuse
还必须让其输出与加载的 broadcast tile 使用不同 buffer；该决定不能随 IR dump
是否创建 `PassContext` 而变化。

down accumulator 是唯一的拓扑专用 cube 包装：若对每个 feature chunk 独立重放
完整 `CubeSchedulePlan`，会错误地在每个 chunk 后 drain 到 GM。

## 6. 未完成项

- 对更多串行组与更少流水组进行枚举或解析选择；
- 支持 mixed pointwise strip streaming 和带状态的 P2/P4；
- 在不复制同构搜索的前提下泛化 stage-local plan 视图；
- 支持对称 `V->C`；
- 支持完整多次往返 skew，并实现具有 key-chunk 循环和 `(m,l,O)` 状态的
  FlashAttention；
- 在完成 PTOAS physical-ID descriptor binding 的三 independent-pipe dense SwiGLU
  lowering 上重跑 910B 数值 sentinel，
  通过后再完成流量、重叠和排序验证。

mixed fusion 在 M1-M10 的计划/发射结构测试和芯片验证完成前默认保持关闭。
