# AutoFuse 混合 Cube/Vector 调度契约

**状态：** 可构建的 `C->V` 增量、按能力准入的物化 `C->V->C` 增量，以及
第一个精确的 `C,C->V->C` dense-SwiGLU 增量已在
`PYPTO_AUTOFUSE_MIXED=1` 后实现。
显式 dual-AIV FIFO lane bridge 按函数限定作用域、感知 pipe descriptor，并且字节精确：
它只改写目标 AIV 函数，给仅使用 pipe 的 AIV 函数增加一个私有 runtime lane
参数，并根据每个 pipe 端点自身的 tile 形状和 dtype 派生 entry offset。frontend
logical ID 用于验证 IR 连线；PTOAS 可以重编号生成的 `TPipe` template ID，因此
backend 根据保留的方向和 slot 几何进行绑定，而不假设数值相等。

通用 `[M,N] + [1,N]` 修正为 `tile.col_expand_add` 后，silicon 已完成单向 C2V
epilogue 闭环（连续 50/50 次通过，无漂移）。dense SwiGLU 也已在 PyPTO
`67df6fb6` 与 PTOAS v0.55 上完成 silicon 闭环。旧 lowering 把两个 C2V projection
channel 和一个 V2C activation reply 合并成单个双向 FIFO；修复后的协议保留三个
独立 logical pipe 及其精确字节宽度、slot 数、workspace 范围和 runtime lane offset。
PTOAS 把 C++ template ID 从 `0/1/2` 重编号为 `0/2/4`，因此 backend 按方向和 slot
几何绑定 physical declaration，而不依赖数值相等。生产 kernel 在两台 910B2 上
累计通过 200 多次 launch，包括每台设备 F=128 强制 descriptor 的连续 50/50 次，
以及此前间歇失败的 F=112/F=144 natural plan。独立 tagged 三 pipe primitive 已完成
结构验证，但尚未在 device 上执行。mixed mode 继续默认关闭，等待流量、重叠和排序
grounding，而不是等待正确性修复。主机测试还验证了
tensor 级等价性以及完整的
`ExpandMixedKernel -> SkewCrossCorePipeline -> AutoTileMatmulL0` 结构。各引擎内部
的工作继续以同构 vector/cube 契约为准；本文只定义引擎边界处新增的契约。

通用单次往返增量目前只完成 host 闭环，尚未完成 silicon 闭环。首个支持面是两个
默认方向的 FP32 matmul，中间连接按行局部、物化的 vector DAG；准入由拓扑、形状、
dtype、容量和单次往返协议决定，不识别 QK、softmax、PV、attention 或其他命名算法。

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

## 2. 一个统一的子图计划

mixed planner 不会先独立规划多个同构 stage，再把这些 kernel 拼装在一起。一个
候选是一个连通 tensor DAG 和一个统一 group schedule：

1. 为 group 的边界输出选择一个共同空间网格；
2. 从输出 region 反向传播经过 pointwise、reduction 和 matmul 边，得到每个 tensor
   的请求 region；
3. 按一个确定性的拓扑/pebbling 顺序执行 region DAG；
4. 将输入与中间值的生命周期延长到最后一次使用，并检查组合后的 L1、L0 与 UB
   工作集；
5. 对同一 region 的计算、边界流量、crossing 流量及串行/重叠阶段计价；
6. 发射完全相同的网格、顺序、region、生命周期、FIFO crossing 与循环。

下文的最大同引擎 stage 只是统一解内部的执行元数据，用于选择同构代价公式和
PyPTO pipeline 工具；它们不是具有独立网格、单独搜索的 kernel。stage-local
homogeneous plan 是固定 group 候选的派生视图。

例如在 `C->V->C` 中，每个最终 `[M_tile,N_tile]` 输出 region 可能需要第一个 matmul
生成 `[M_tile,S]` crossing。若 `N` 被切分，该 crossing 的重算/重载次数与存活形状
都由反向 region 传播决定，而不是由独立最优的 QK 或 softmax 网格决定。

## 3. MixedSchedulePlan

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

获胜计划通过公开、类型化的 `pl.cross_core_pipe(...)` 调度项跨越 tensor-to-tile
边界；这些调度项在 IR 中由版本化 carrier 承载。每个逻辑 crossing 对应一个
record 和一个独立单向 physical queue。`ExpandMixedKernel`
先校验方向、形状、字节数与顺序，再把该 record 的 ID 写到
`tpush`/`tpop`/`tfree`。first-matmul 与 accumulate 两个互斥 branch 对同一
activation 的重复使用共享 reply ID；gate/up 这类形状相同但 SSA source 不同的
张量不会合并。expansion 完成后会删除该 IR carrier。

## 4. 流水模式与忠实性

当前安全 skew 支持“一个有序 push bundle 后接一个回复 pop”。bundle 可以包含
SwiGLU 的 gate/up 两个独立 pipe ID，但所有 push 必须位于第一个 pop 之前，
且 op 顺序与 ID 都属于协议。pop 后再次
push 表示第二次往返。analytic 模式可以保留串行 descriptor，但 compiler 模式会
拒绝该 group；发射一个串行近似会违背所选统一计划并悄悄丢失目标流水。
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

## 5. 复用同构模型

`Ascend910BMixed` 不重新拟合 cube 或 vector 工作。各 stage 使用相同的 grounded
同构原语：

- cube 的 MAC、extract、GM 到 L1 和 L1 到 L0 项；
- vector 的 primitive、流量、UB lifetime 和阶段局部重叠项。

mixed 包装只增加跨界流量、FIFO 容量、跨引擎 wavefront，以及生命周期跨越
stage 的状态。纯 cube/vector 组直接交给同构模型。

## 6. Dense-SwiGLU 第一增量

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

## 7. 未完成项

- 对更多串行组与更少流水组进行枚举或解析选择；
- 支持 mixed pointwise strip streaming 和带状态的 P2/P4；
- 已为物化的单次往返完成 host 实现：在不复制同构搜索的前提下泛化 stage-local
  plan 视图；generic emitter 消费统一计划中的拓扑顺序、传播后的 transfer region、
  生命周期、FIFO record 与 pipeline-item 循环。`C->V->C` 依据能力和协议准入，
  而不是识别 QK、softmax、PV、attention 或其他命名算法；latest-PTOAS silicon
  正确性与重叠 grounding 仍待完成；
- 通用单次往返目前会物化 vector frame；嵌入式 online P4 必须先显式表示阶段局部
  状态、流量和 pipeline-item 轴，才能忠实计价和发射；
- 支持对称 `V->C`；
- 在准入 `C->V->C->V` 之前先支持完整多次往返 skew；之后的 attention 计划可以
  使用 key-chunk 循环和 `(m,l,O)` 状态，但仍是普通 op-DAG 计划，而不是命名 emitter；
- 完成三 independent-pipe dense SwiGLU 的流量、重叠和排序验证；其生产 kernel
  数值契约已经 silicon-closed，独立 tagged primitive 仍只有结构证据。

mixed fusion 在 M1-M10 的计划/发射结构测试和芯片验证完成前默认保持关闭。
