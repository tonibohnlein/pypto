# 外部 AutoFuse 前端与 PyPTO 源码生成

## 状态

本文记录未来研究方向，并非当前已支持的 PyPTO API。现有编译器内 AutoFuse 实现仍是成本模型、
调度方案契约和内核生成行为的参考实现。

## 目标

把 AutoFuse 和 AutoTile 作为外部的源码到源码调度系统运行：

```text
PyTorch/Hugging Face 模块 + 配置 + 形状约束
        |
        v
torch.export / FX 图捕获
        |
        v
Torch 到调度器 DAG 的适配器
        |
        v
Fusebox AutoFuse + AutoTile 规划
        |
        v
PyPTO DSL 源码生成器
        |
        v
普通 PyPTO 编译器与运行时
```

生成的 PyPTO 源码已经包含选定的融合边界、网格、区域、拓扑顺序、物理 tile、循环、流水线、
跨核 FIFO 和有效形状处理。PyPTO 只验证并降低这个显式调度，不再次运行 AutoFuse 或 AutoTile。

## 组件边界

### Torch/Hugging Face 捕获适配器

使用 `torch.export` 或 FX，而不是翻译任意 Python 源码。适配器导入：

- tensor 算子、数据类型、布局、别名、修改和输出；
- 符号形状约束和代表性输入元数据；
- 解析模型常量所需的模块配置；
- 对不支持的控制流、自定义算子和数据相关访问的显式不透明节点。

Hugging Face 模型代码通常是 PyTorch `nn.Module` 加配置和权重。它描述模型语义，但不一定包含
分页 KV-cache 布局、量化或运行时任务依赖等部署决策。这些决策必须已经存在于捕获图中，或作为
显式前端配置提供；适配器不能猜测。

### Fusebox 调度器核心

Fusebox 应独立于 Torch 和 PyPTO 的实现类型。它的输入是现有 tensor DAG，只增加与前端无关的
形状约束和不透明边界元数据。

对于每个连通候选组，调度器保留统一设计：

1. 选择融合或切分边界；
2. 给选定组分配一个公共网格；
3. 从输出区域反向传播 tensor 区域；
4. 选择合法的拓扑/pebbling 顺序；
5. 统计边界值和生成值的生命周期；
6. 检查 UB/L1/L0 和跨核 FIFO 可行性；
7. 计算计算、传输、drain 和可实现重叠的成本；
8. 返回完整、可生成代码的方案描述符。

调度器不创建完整程序的硬件调度。它生成良好的内核并保留依赖图；PyPTO 运行时调度器负责启动
就绪内核并重叠独立的 AIC/AIV 工作。

### PyPTO 源码后端

后端只消费选中的方案描述符并生成普通、可读的 PyPTO DSL。根据方案，它会生成 `pl.spmd`、
静态物理 tile 形状、运行时 `valid_shape`、`pl.range`、`pl.pipeline`、tensor view、显式 GM
边界，以及受支持的 `tpush`/`tpop`/`tfree` 跨核传输。

后端不能重新规划。每个生成的循环、生命周期、传输和 FIFO 都必须能追溯到方案字段。除源码外，
还应发布现有调度报告和伪代码，让用户了解实现被选择的原因。

## 动态形状

PyPTO 已支持一个 extent-polymorphic 产物：运行时维度控制 tensor view、循环边界、偏移和有效
形状，而硬件物理 tile 保持静态。外部调度器应保留这个基线。

### 类型 1：动态 extent，静态物理 chunk

第一类支持的动态形状保持所有决定调度的量为静态：

```text
运行时：  M、循环次数、偏移、最后一个有效 extent
编译时：  CHUNK、物理 tile 形状、网格策略、流水深度、内存分配
```

对固定的 `CHUNK`，Fusebox 像静态问题一样规划循环体。动态轴只改变静态区域的副本数量和最后
一个副本的逻辑大小：

```python
m = pl.tensor.dim(x, 0)
for m0 in pl.range(0, m, CHUNK):
    valid_m = pl.min(CHUNK, m - m0)
    x_tile = pl.slice(x, [CHUNK, D], [m0, 0], valid_shape=[valid_m, D])
    # 对物理 [CHUNK, D] 的静态规划 DAG。
```

初始准入要求：

- 动态维是外部/自由轴，各 chunk 相互独立；
- 所有物理 tile extent 和内存占用都是编译时常量；
- 组内区域传播是仿射的，并保持 chunk 边界；
- tail 能表示为同一物理 frame 内的裁剪逻辑区域；
- 没有数据相关地址或控制决策改变每个 chunk 的 DAG。

容量按完整物理 chunk 检查。对于运行时 extent `M`，规划器得到 `ceildiv(M, CHUNK)` 个逻辑区域，
并用现有静态区域模型计算其 wave；最后区域使用普通 clamped/tail 成本。如果这些区域实际是并发
硬件 work unit，就不能简单乘以单 chunk 延迟。

同一契约支持两种模式：

1. **程序员固定 chunk。** 把 `CHUNK` 作为调度约束导入，只优化其内部的组。
2. **调度器选择 chunk。** 枚举一个小的合法静态集合，拒绝容量不可行的值，并针对代表性 `M`、
   有界区间或给定形状分布比较总成本。

两种模式生成的产物都保持 extent-polymorphic。只有当不同形状区间的最佳物理 chunk、网格或
流水线明显变化时，才需要多个变体。

动态 reduction 轴不属于第一类：它需要显式循环携带状态，例如 cube accumulator、online-softmax
状态或 Welford tuple。每个 chunk 仍然是静态的，但递推需要独立的模型/生成契约。

`pypto-lib` 中已有的例子：

- `models/deepseek_v4_pro/rmsnorm.py`：动态 token extent `T_DYN`，静态 `T_TILE=8` 和
  `D_TILE=128`。支持的 decode/prefill 配置要求整除，因此没有 tail。
- `models/deepseek_v4_pro/hc_head.py`：动态 token extent、静态 `LINEAR_T_TILE`，并把
  `t_rows = min(LINEAR_T_TILE, t_dim - t0)` 作为输入 slice 的 `valid_shape`。这是标准的完整
  chunk 加运行时 tail 形式。
- `models/qwen3_14b/rms_lm_head.py`：动态 batch 行、静态 `BATCH_TILE`，并在输出写回前把
  `lm_valid_rows = min(BATCH_TILE, batch - b0)` 应用于 cube 结果。
- `models/deepseek_v4_pro/hc_post.py`：动态 token 数决定 SPMD work-unit 数，每个 work item 使用
  静态 token/data tile；prefill 变体保护不完整的最后一个 tile。

只有当另一有界形状区间使用不同物理 tile、网格或流水线会明显更好时，才生成多个方案变体。
外部工具随后生成这些变体和一个小型 host dispatcher。它不会为每个请求编译一个变体；在语义和
成本尚未建模前，TopK 或分页 gather 等数据相关操作仍是不透明融合边界。

## 概念 API

下面只说明职责边界，不是最终 Python API 提案：

```python
exported = torch.export.export(module, example_args, dynamic_shapes=shape_constraints)
problem = torch_frontend.to_fusebox(exported, model_config=config)
solution = fusebox.plan(problem, target="ascend910b")

source, report = pypto_backend.generate(solution)
compiled = pypto_compile(source)
```

`fusebox.plan` 负责分组和 tiling。`pypto_backend.generate` 确定性地序列化方案，
`pypto_compile` 则使用普通编译器路径。

## 初始验证阶梯

1. 往返一个静态 vector RMSNorm 图，并将生成的 PyPTO 与当前 AutoTile 方案和设备结果比较。
2. 往返一个静态 cube matmul，并比较方案字段、生成 PTO 和数值。
3. 在没有 attention 识别器的情况下往返通用 `QK -> vector softmax DAG -> PV` 图。
4. 将不支持的节点保留为显式切分，并验证跨越每个边界的值。
5. 使用静态物理 tile 和运行时有效形状增加一个有界动态外轴。
6. 只有设备证据表明单一动态方案明显次优时，才增加方案变体。
7. 在 Ascend A5 上评估 DeepSeek V4 Pro 前，先移植目标参数和能力准入。

每一级都要求与 PyTorch 比较图语义、测试方案到源码的契约、测试 PyPTO 解析/编译，并先完成设备
正确性再做性能排序。

## 非目标

- 直接翻译任意 Python 控制流；
- 识别模型名或硬编码 FlashAttention/SwiGLU 算法；
- 代替模型作者选择量化精度；
- 替换 PyPTO 编译器、验证器、PTOAS 或运行时调度器；
- 静默近似不支持的别名、修改、view 或数据相关访问。
