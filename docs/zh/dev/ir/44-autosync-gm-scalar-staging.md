# GM 标量暂存：操作契约子门禁提案

状态：架构已获准实施，尚待实现与语义验收。Native 指令正确性仍是已有
selected-TLOAD 契约前提，不是新增的硬件证明。这扩展[架构计划](41-general-autosync-architecture-plan.md)
的 M2/M3，并复用已确认的[本地标量桥接](43-autosync-value-and-scalar-contracts.md)。
不宣称完整 Qwen 所有权改善或设备验证。

## 构造与语义

首个支持族是 A2/A3 AIV 内直接 SSA 赋值 `value = tensor.read(source, [index])`。
源为普通、紧凑、无填充 ND 的一维 INT32 GM 张量，原切点的索引已经就绪。
原操作的地址必须合法且不溢出；不能从基准配置、不同 SSA 名称或区间包络推断。
输入任务的发布和可见性契约仍须成立。

在完全相同的控制与执行切点，用现有 TLOAD 将一个 INT32 元素装入私有 Vec
载体，其容量为 `[1,8]`、有效形状为 `[1,1]`，然后以现有 `buffer.read([0,0])`
定义原标量 SSA。保留源 TensorType、ABI 秩、存储根、索引表达式和控制流。
不得把调用者参数伪装成 `[1,N]`，不得为了对齐而读取八个 GM 元素。

使用现有分配器保留 32 字节载体。仅首个元素有可观察值；其他通道不得被读取。
对 native 可能写入的对齐尾部，采用完整 32 字节载体包络，而不是声称只写四字节。
原 GM 读取仍为单个元素。MTE2 到 S 的服务保护 DMA 完成，随后本地阻塞读取
使原 SSA 就绪。多次访问的载体覆盖必须满足已有物理重用证明。

起初仅接收规范化的直接赋值。丢弃或嵌套读取、缺少原语句切点的有副作用 while
条件、写入、其他数据类型/内核/目标或不可用的布局保持拒绝。不移动条件内的读取，
不遗漏 while 最后一次 false 求值。跨核发布或轮询状态仍需要原协议可见性契约；
DMA 不会凭空建立远程发布保证。

## Native 依据

PTO-ISA 固定版本为 `5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f`。
[TLOAD 文档](https://github.com/hw-native-sys/pto-isa/blob/5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f/docs/isa/TLOAD.md)
支持 A2/A3 Vec INT32 同布局路径。该版本 `tload_common.hpp` 的
`TLoadGm2ubNd2nd` 对源形状 `[1,1,1,1,1]` 和有效 `[1,1]` 计算一个四字节 burst；
`TLoad.hpp` 的 `TLoadInstrGm2ubNative` 选择 `copy_gm_to_ubuf_align_b32`。

[官方 CANN 非对齐搬运参考](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0265.html)
明确支持 Atlas A2/A3 INT32 GM-to-UB；GlobalTensor 源无地址对齐要求，
LocalTensor 目标要求 32 字节对齐，burst 长度以字节计，四字节也在范围内。
因此合法 INT32 指针加整数元素偏移不要求 32 字节对齐。必须将该公开契约
绑定到固定 intrinsic；它不证明任意调用者地址有效。

源代码与非对齐搬运模型允许 UB 对齐尾部。始终保留并汇总完整 32 字节私有空间，
不依赖填充值，也不把 `PadValue::Null` 当作只写四字节的证明。
架构评审同意沿用普通加载已依赖的 normative selected-TLOAD 契约。
固定源代码选择四字节搬运，公开接口允许该形状。尚未独立复现 CCE intrinsic 与
公开 API 实现的对应关系：本地没有 CANN 头文件，官方源服务受到访问速率限制。
必须保留这一资格边界，不能把桥接当作硬件证明。DMA 完成到 GetValue 仍须由
现有本地事件契约建立。没有设备验证。

## 复用的接口与变更范围

共享 transfer/native recipe 提供一维 INT32 单点到二维私有载体的契约。
SelectBuffer 只构造完整不可变候选；普通整个函数的 admission 仍决定接收与否。
`memref_utils.h` 的 `CreateAllocStatement`/`InsertAllocsIntoBody` 在现有 placement 前
插入顶层载体分配。Buffer transfer 推导与 runtime-valid 分析把源 `[i]`/`[1]`
对应到目标 `[1,1]`。LowerTileToBuffer 保留原 SSA，复用本地标量桥接。
PTO codegen 保留一维 ABI，复用已有 tensor/partition-view、TLOAD 与 GetValue。

保守效果覆盖 DDR 源与完整 Vec 载体，保留别名策略、原根与 scalar readiness。
精确一维提取或 GM selector 缺失时必须显式不可用，不能变成空答案。
保守覆盖不等同于精确最小物理 demands。不新增后端、Buffer 模型、分配算法、
ID 池、admission 限制或 watchdog。关闭 autosync 时，必须验证现有下游所有者能
同步新 transfer，否则只在明确支持的请求内构造。

## 验收门禁

- 独立检查索引 0、1、7、8 与最后元素，长度为 1 的输入；污染相邻 GM 元素及载体填充。
- 检查零/单/多次访问、分支跳过、尾部、标量控制、counter reset 与原求值切点。
- 检查实际逻辑/物理顺序、匹配、无环、合法方向和消费先于重发布，包含父级命令。
- 检查两种别名策略、重叠视图、先前本地 GM 写入、缺失外部发布契约。
- 检查三种 planner、关闭 Buffer 的委托、非 32 字节 GM 偏移的真实 PTO/C++，
  并保留本地标量和 pitch 回归。
- 通过共享族门禁后再刷新已有原始 Qwen capture；新的首个拒绝不等于后续能力成立。

构造和保留状态应随静态读取点增长，而非 runtime 索引或循环次数。
分别计量输入恢复、暂存、逻辑协议、物理实现与生成。单点暂存新增 DMA 及握手，
因此不承诺设备提速。完整英文设计给出文件级责任和全部边界。
