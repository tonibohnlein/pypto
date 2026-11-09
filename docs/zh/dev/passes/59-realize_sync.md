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

PyPTO 仅对已完成物理地址分配的 Buffer IR、已限定的 AIC 矩阵乘片段和
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
