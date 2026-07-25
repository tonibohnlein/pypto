# AutoFuse 参考内核对等性

## 范围

本文记录公开 PTO 仓库中的纯向量/纯 Cube 算法、AutoFuse 当前可以实现的算法，以及参考对比能够
和不能证明的内容。混合 Cube/向量程序仅用于防止错误的等价比较。

审计版本：

- PTO-ISA `0c112d61f41342bd0867ce1080c29f1590d72484`；
- PTO-Kernels `a8675c5a30bb4792ccc5e5f096737d12e0dfb0cc`；
- MegaGDN-PTO `8b4ae6f9413976c598d7149b545ad003efc72164`；
- PTO-DSL `b10afbea191dcce6f718d1f1240d5fdc4fca990a`；
- PTOAS `8296984f3e89913ce07fd4696542c60c2737f053`。

PTOAS 主要是汇编器、优化器和指令一致性仓库；其中的样例不必然是调优后的性能参考。

## PyPTO 职责边界

AutoFuse 发射张量级 `spmd`/`pipeline` 作用域、slice/assemble 和
`tensor.matmul`/`tensor.matmul_acc`。标准 Default 流水线负责 outlining、tensor-to-tile
转换、复合算子降低、`AutoTileMatmulL0`、存储空间/布局推导、流水线降低、复用/分配、
依赖构建和 PTO 生成。

`tests/ut/ir/transforms/test_auto_fuse_pto_isa_reference.py` 执行完整流水线。它验证算法和
流水线类别对等，而不是源码或二进制完全相同。

## 当前可实现的参考

| 参考 | 状态 | 当前对比 |
| ---- | ---- | -------- |
| PTO-ISA add→ReLU→mul | 就绪 | 一个软件流水 AIV 内核；PyPTO 使用等价的 `TMAXS(x,0)`，而非 `TRELU`。 |
| PTO-Kernels `abs` | 就绪 | 一个 `TLOAD→TABS→TSTORE` AIV 数据流。 |
| PTO-Kernels FP16 SiLU/SwiGLU | 就绪 | 一个两级 AIV 内核，无中间 GM；PyPTO 使用 `TNEG`，参考使用 `TMULS(-1)`。 |
| PTO-Kernels 仿射 FP16 LayerNorm | 语义匹配、存在性能差距 | AutoFuse 发射两个 AIV 内核和一个 GM 边界；参考在单内核中重叠各阶段，并在 `TRSQRT` 后执行两次 Newton 修正。 |
| PTO-DSL GEGLU | 就绪 | 基于 exp 的 tanh 表达式降低为一个 AIV 内核。 |
| PTOAS `FFN/ffn_act.pto` | 就绪 | 截断三次门控和第二输入乘法降低为一个 AIV 内核，无中间 GM。 |
| PTO-ISA 1536³ FP16→FP32 GEMM | 就绪 | 相同的 GM→L1→L0A/L0B→Matrix→FIXPIPE 层级和 `TMATMUL_ACC`。 |
| PTO-ISA BF16 链式 GEMM | 就绪 | 生产者通过 L0C→L1 保持，仅根结果写入 GM。 |

聚焦主机测试通过 9/9。持久 a2a3 文件收集 62/124；新增参考标记用例仍需要首次双设备硅验证。

## 已覆盖但不是新参考

- PTOAS FFN FC1/FC2、GQA QK/SV 和 FlashAttention QK/SV 是已有 GEMM、ragged-K 和
  `FirstPartialThenAtomic` 测试覆盖的独立 matmul/split-K 阶段。
- PTO-DSL add/ReLU 和简单 matmul 与现有向量融合、GEMM 对比重复。
- PTOAS LReLU/PReLU 和原语样例主要验证专用指令；其语义可组合，但在没有专用降低前不适合
  用作内核性能基线。
- PTOAS GQA/FlashAttention “softmax” 是截断多项式，没有精确 row-max/row-sum 递推，
  因此不等价于 P4 精确 softmax。

## 能力和调度缺口

| 类别 | 缺口 |
| ---- | ---- |
| PTOAS 动态尾部 matmul | 已覆盖固定 ragged M/N/K；未覆盖运行时有效尺寸和动态工作映射。 |
| 转置 GEMM | Cube replay 当前仅接收默认操作数方向。 |
| GEMV/MX | 专用低精度/缩放指令缺少张量能力、角色、成本和发射描述符。 |
| PTO-DSL matmul swizzle | 支持基础 GEMM；自定义 L2/输出网格 swizzle 不是当前工作单元策略。 |
| PTO-DSL Sinkhorn K=4 | 需要矩阵在交替行/列循环阶段间驻留；静态切分组并不等价。 |
| 批量矩阵平方 | 需要批量/三维 matmul 和批索引工作映射。 |
| 三角求逆 | 需要递归/控制流 matmul 序列、三角结构、填充和原地更新。 |
| Scan/CSR/Hadamard/量化 | 需要 scan 状态、gather/shuffle、reshape/pad 或打包 INT4/INT8。 |
| 因果 Conv1D/GDN/KDA | 需要 stencil/循环状态、scan、动态 chunk 循环、mask、三角求解或混合引擎。 |

最高价值的缺失类别：

1. **TopK：**没有 sort/gather 能力、流式状态、值/索引双输出描述符或 merge 计划。
2. **Conv2D：**没有卷积/stencil 张量算子、halo/stride/dilation 传播、
   im2col 与 direct 选择或发射器。
3. **MoE 路由/分发/合并：**需要 TopK、prefix/scan 或 histogram 状态、gather/scatter、
   可变 expert batch、expert GEMM、combine，可能还需要通信。
4. **FlashAttention：**有意归为混合内核。QK→online-softmax→PV 需要 Cube/向量
   FIFO/流水线；向量 P4 只是一个组成部分。

在显式 capability/plan/emit 契约存在之前，系统测试不得声称这些类别与参考实现对等。

## 解释和设备验证义务

结构测试通过表示算法工作和流水线类别等价。仍存在以下差异：

1. 手写参考自行选择核数、tile 几何、ping/pong 布局、barrier，有时还有 swizzle；
   AutoFuse 将其中若干决策委托给标准 PyPTO pass。
2. 等价 opcode 的性能可能不同（`TNEG` 与 `TMULS(-1)`、`TMAXS` 与 `TRELU`）。
3. 仿射 LayerNorm 存在单内核与双内核的真实边界差距。
4. GEMM 对等止于存储层级；动态尾部、转置、GEMV/MX 和参考 swizzle 是独立工作。
5. 混合程序和不同近似算法不纳入纯引擎性能声明。

设备对比必须报告双方描述符、归一化流量、指令/流水线计数、数值容差和隔离 wall 分布，
不能把所有差异都归因于 AutoFuse 成本模型。
