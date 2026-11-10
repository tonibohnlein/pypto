# RealizeSync（实验性）

此可选 Pass 在存储降低之后按完整内核尝试 PyPTO 同步。
**两种模式都保持 PTOAS 自动同步启用。** 默认编译仍将同步交给 PTOAS。

```python
from pypto import passes

# Explicit caller promise: distinct GM arguments denote disjoint allocations.
with passes.PassContext([], enable_pypto_autosync=True,
                        autosync_disjoint_gm=True):
    compiled = kernel.compile(a, b, c, config=config)

# Turn PyPTO's pass off; PTOAS still handles synchronization.
with passes.PassContext([], enable_pypto_autosync=False):
    baseline = kernel.compile(a, b, c, config=config)
```

PyPTO 仅对已完成物理地址分配的 Buffer IR、已限定的 AIC 矩阵乘与 AIV Vec 片段和
manual a2a3 宏变体调用现有访问导入器、紧凑最小需求分析与物理实现。
如果地址分配、GM 不相交、目标、访问、控制、端点或资源证明不满足要求，
将**原始内核主体**留给 PTOAS。这不保证 PTOAS 支持所有输入；下游错误仍会传播。
内部编译器错误不会转换为回退。保留分析结果的研究 API 仍严格报告资格失败。

成功尝试会发布完整主体和布尔属性 `pypto_autosync=True`。PTOAS 对每个含显式
SET/WAIT 或 record/wait 事件的函数独立跳过插入，因此同一模块可同时包含
PyPTO 已同步函数和委托函数。**仅有屏障不会触发跳过。** PyPTO 的空计划或
仅含屏障的候选会被丢弃，并委托原始主体；不会伪造标志事件。失败尝试不会发布
部分同步。用户已有的同步仍遵循 PTOAS 原有跳过规则；此开关不会补全或独立验证它。

启用后，`report/autosync.json` 列出各尝试内核的所选路线（`pypto` 或 `ptoas`）
和回退原因。这是选择元数据，不证明 PTOAS 已插入命令或编译成功。
外部源码、内置模板和运行时绑定函数保留外部所有权，不纳入报告。
不要修改已拥有的主体或伪造属性：与物理地址一样，它是可信编译器输出，
并非任意用户修改的证明检查器。对已拥有计划再次运行此 Pass 会拒绝。

开关与不相交契约属于 JIT 缓存标识；转储和性能分析上下文保留它们。
`PassManager` 构造后不能更改二者。分配算法不变；此接口不提供不安全的无同步模式。

限定的[物理构造](../ir/15-physical-demand-realization.md)建立模型载荷顺序、
匹配和合法方向复用。它假定调用入口通知状态为空并关闭通知；它**不证明**
函数返回时最后的异步载荷已完成。原生宏语义和运行时调用完成需要单独的设备验证。

## 有限 Vec 覆盖

AIV 路径复用现有需求分析、插入位置和标志分配器，支持静态、完整有效的二维 Vec
load/store，以及共享 Buffer 配方目录中的逐元素操作。常量标量操作数可用；任意标量
生产计算仍不被接纳。共享 Buffer 验证器检查各指令的原地别名契约。

同一路径支持共享的 `row_max`、`row_sum`、`row_expand_sub` 和 `row_expand_div`
配方以及完整有效的连续列向量。归约读取源，写入目标及覆盖整个已声明 scratch
缓冲区的保守范围；广播读取两个输入。最小需求与选择器针对这些建模效果，不声称
精确恢复原生 scratch 字节访问。现有验证器拒绝非法 scratch 重叠，资源目录在分配前
保留原生 V/S 事件 ID。一般列主序矩阵和部分有效描述符仍不支持。

原始 `examples/intermediate/02_softmax.py` 现可选择 PyPTO autosync 并通过 PTOAS
生成 C++。主机检查覆盖建模访问、covers、查询、选择器、匹配、物理复用，以及具有
scratch 复用和计数器重置的小型重复流水线。测试循环次数从 4 增至 1,000,000 时，
保留表示大小不变。原始 RMSNorm 和 LayerNorm 示例仍委托给 PTOAS：其完整函数尚未
被选为 Buffer IR。这是更早的降低边界，而非 autosync 分配失败。新增案例尚无设备验证。

已分配的 Vec 根与静态 subview 使用统一物理字节域。嵌套 subview 保留原始行距；
连续 reshape 保持字节范围。不同 SSA 名称或元素宽度不会使重叠分配变得不相交。
连续访问用一个区间，非连续窗口每物理行用一个区间；成本取决于静态窗口行数而非
循环次数。单一行带中的静态列区间通过有序索引跳过无关存储原子；`class_tests`
统计成员关系检查，`class_lookup_tests` 单独统计二分查找比较。重复区域仍须满足
现有成员关系与每次访问刷新条件。

动态 valid 更新、运行时窗口偏移、旋转槽选择和非连续 reshape 缺少所需导入与就绪
证明，仍委托给 PTOAS。GM 视图与步长仍遵循现有导入器、发射器契约，不把有间隔的
GM 视图当作连续视图，也不从 SSA 身份推导不相交。静态局部视图不代表一般动态元数据支持。

同 MTE2 需求可以有正确的最小需求分析，但没有已限定的局部屏障机制，此时明确拒绝
可执行导出。研究物理结果的 `native_reserved_ids` 单独列出原生宏固定事件占用；
分配器从相应 AIC/AIV 方向池排除这些占用，分配算法与资源容量不变。

主机回归覆盖原始加法/乘法示例、重叠分配、嵌套行距、原地操作、零次与重复访问，
独立检查字节访问、实际 covers、四种事件查询、边界选择及因果复用。PTOAS C++ 生成
单独检查，不代表新增设备验证。

## 原生累加器保护与标量屏障

在命名的 manual `a2a3` 目标契约下，对同一物理 L0C 累加器的连续 Mmad 操作，若
`floor(M/16) * floor(N/16) >= 10`，可省略其存储同步。导入器检查完整有效形状、
Left/Right/Acc 操作数与 `[1, 4095]` 范围内的匹配维度，实现
[Mmad 同步优化](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html)。
这不是整个矩阵流水或 FIX 流水的豁免。

物理访问和边界选择器保持不变。提供者证书仅在有限归约与跨区域归约前过滤受保护的
存储见证，不过滤显式先决条件。稀疏计数器证明中间的矩阵操作都在同一累加器上满足
条件，且没有其他访问触及该累加器；通过循环次数计算而不展开迭代。FIX 读取会中断
保护。重新初始化不接受入向保护；同址同布局别名共享物理身份。此规则不会添加
completion-before-start 事件边。原有前沿扫描仍有效：原生启动顺序沿受保护写入链
传递入向义务，原生完成顺序将出向义务传递到后续消费者；只读访问的 WAR 义务仍明确保留。

支持首个矩阵位置的初始化/累加选择。位于其他矩阵操作之后的条件初始化仍采用保守
处理，因为其仅启动阶段有效的需求需要更丰富的族活动契约。未知目标不获得豁免；
物理降低拒绝与保留分析契约不一致的目标。最小需求与顺序相等的声明针对所选硬件及
效果契约，不代表利用了全部原生优化。

禁止 `PIPE_S` 屏障；物理资格检查与发射仅允许 M/V 局部屏障。数值服务可使用显式
提供的标量原生顺序边，但仍保留跨流水就绪义务。标量载荷导入尚不支持，因此继续
委托处理，而不是构造虚假的标量屏障。
[PipeBarrier 规范](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0271.html)
说明标量同步由硬件保证，标量屏障会引发硬件错误。主机回归验证这些契约下的图性质，
不代表新增设备验证。

## 索引 Buffer 契约

显式启用的导入器复用 `buffer.gather` 和 `buffer.scatter` 配方，只接受已放置、
完整有效且稠密的 AIV 缓冲区。索引必须是目标局部缓冲区内有效的物理元素偏移；
这是操作的调用方契约，并非静态范围证明，也不支持任意指针。
Gather 的源读取与 scratch 写入使用整个缓冲区的保守包络。
Scatter 先清空整个目标，因此在共享操作契约下目标写入是精确的。
保留诊断逐访问标注 `exact` 或 `enclosure`；最小需求和边界选择器描述所选模型。
保守模型的 covers 不一定是具体执行的最小 covers，但其闭包覆盖具体访问冲突。

现有原生资源目录保留手动模式 V/S 辅助标志，分配算法及 PTOAS 委派保持不变。
缺少放置、部分有效范围或非法写别名仍以事务方式拒绝。
PR #3333 的带填充行主序归约结果尚不满足此完整有效 Buffer 契约；
集成该 API 不会把填充区域静默视为有效数据。

## 就绪的调用坐标

Pass 在分析内核之前从编译程序的直接 Call/Submit 调用点收集有界 INDEX 坐标，
合并所有调用点，保留原始形参位置，对未知范围保持未证明。导入器支持各 GM 域的
共同参数平移，并检查算术和分配边界；数值分析不会特化为单个 tile 偏移。
存储查询需要对应的原始坐标绑定。
参见[参数化 prefill 合同和检查点](../ir/16-parameterized-prefill-checkpoint.md)。
嵌套 Qwen 组合及动态 GM 范围资格仍待实现。
