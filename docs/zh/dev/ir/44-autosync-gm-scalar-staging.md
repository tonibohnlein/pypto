# GM 标量暂存：操作契约检查点

状态：有界实现已通过源码审查和首次一致的 53 项主机测试。格式化后一致构建、
修正后的操作门及固定整模型捕获也通过；最终 hooks 与接受后的提交发布待完成。Native 指令正确性仍是已有
selected-TLOAD 契约前提，不是新增的硬件证明。这扩展[架构计划](41-general-autosync-architecture-plan.md)
的 M2/M3，并复用已确认的[本地标量桥接](43-autosync-value-and-scalar-contracts.md)。
整模型主机捕获的局部所有权从 17 增至 19；不宣称完整 Qwen 所有权或设备验证。

## 紧凑二维张量的单点读取

`PassContext(enable_scalar_gm_dma=True)` 在 A2/A3 AIV 上为紧凑、无填充的
ND 一维或二维 INT32 和 FP32 张量选择单点 DMA，保留原始张量、坐标和执行切点。
二维读取在 `[row, col]` 处对原始张量划分范围 `[1, 1]`。列数须为静态，
父视图的行跨度须符合原生突发间隔字段的范围。行数可以是动态值。显式步长
须符合紧凑布局，GM 视图须全有效。

对于列数为一的紧凑张量，单点加载和存储在原 GM 指针上使用步长为
`[1, 1]` 的 ND 视图，以匹配行主序 Vec 载体的传输布局。同一张量的
列向量传输仍可使用其 DN 视图。

两种类型的单位分区都产生一次四字节 GM 突发传输。窗口只有一行时，原父视图
的行跨度仍参与起始地址计算。TLOAD 可能访问完整的 32 字节 UB 载体。保守
同步导入该包络，此路径不提供精确物理访问恢复。

A2/A3 的公开 DMA 契约支持两种四字节类型，并允许 GM 起始地址不按 32 字节
对齐。原生实现要求本地载体按 32 字节对齐，并保留原张量访问不越界的前提。
[CANN DataCopyPad 契约](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0265.html)

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

在 A2/A3 AIV 上，本地标量规则也适用于稠密行主序 FP32/INT32 Tile 及连续
FP32 列向量。每次访问均须证明有符号运行时坐标落在静态有效范围内。证明保留
原整数位宽及步进循环的实际取值区间，包括逆序扫描。GetValue 和 SetValue
接收 uint32 元素偏移，因此发射复用准入检查的布局步长。窄子视图在重解释后
仍保留继承的行间距。已完整填充的区域允许读取相应位置。同步规划使用本地
存储窗口包围单点访问，并将标量写入视为部分写入。

实现仅接收规范化的直接赋值。丢弃或嵌套读取、缺少原语句切点的有副作用 while
条件、写入、其他数据类型/内核/目标或不可用的布局保持拒绝。不移动条件内的读取，
不遗漏 while 最后一次 false 求值。跨核发布或轮询状态仍需要原协议可见性契约；
DMA 不会凭空建立远程发布保证。

## Native 依据

PTO-ISA 固定版本为 `5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f`。
[TLOAD 文档](https://github.com/hw-native-sys/pto-isa/blob/5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f/docs/isa/TLOAD.md)
支持 A2/A3 Vec INT32 同布局路径。该版本 `include/pto/common/arch/memory/tload_common.hpp` 的
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
ID 池、admission 限制或 watchdog。只有启用 PyPTO autosync 与 Buffer IR，并显式选择 A2/A3 AIV 的
`conservative_prefix` 时才构造候选。关闭任一开关时保留原 Functional 读取。
选择 PTOAS planner 时，已暂存的 Buffer 程序仍委托物理所有权；实际下游 C++ 路径已测试。
普通整函数准入拒绝时不发布部分候选。

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

## 已实现主机门与限制

首次一致门在原 120 秒看门狗内 **53 项通过，耗时 0.86 秒**：

```bash
timeout 120s python -m pytest -q --require-ptoas \
  tests/ut/ir/transforms/test_gm_scalar_staging.py
```

[回执](../../../en/dev/ir/44-autosync-gm-scalar-staging.json) 保留设计／原生依据并记录
实现文件、日志与发布状态。门覆盖三个已有 planner、长度 1 和 9、索引 0／1／7／8、
两种别名策略及符号一维长度。小执行保持同一函数，变化零／单／多次循环、跳过分支及
计数器重置。实际正／负标量值控制原分支及本地写入。独立字节／命令解释器区分四字节
GM 源与完整 32 字节 UB 写包络，检查结果值、覆盖、匹配、无环与因果物理复用。
may-alias 执行包含允许重叠的 GM 根，不从不同指针名推断分离。

三个 planner 均产生包含 TLOAD 与 GetValue 的真实 PTOAS C++。关闭 autosync 或
Buffer 时保留原 `tensor.read`；丢弃读取和 INT64 保留整函数委托。runtime 次数不会复制
载体或命令：夹具保留一个标量读点、含普通输入在内两个加载点，以及原两个循环。
源秩、存储根、实际索引和标量 SSA 不变。

传输仍是异步 DMA，加合格阻塞本地标量消费者，不是虚构单 pipe 标量操作。静态读取点
需要私有载体，可能新增握手或保守顺序。不因此建立最小需求、最小 ID 或设备加速。
此门不独立证明对等发布、原生缓存行为、任意有副作用 while 条件、INT64 中转或精确 GM
边界导出。原 M1–M6 与完整 Qwen 所有权仍未完成。

### 更广门禁与未修改整模型捕获

格式化后一致构建通过。去重的广泛回归记录 **3,132 通过、两个失败、四个警告，
430.90 秒**。两个失败只是过期 `rank-2` 诊断期望：一维 FP32 load/store 仍被新的
INT32 单点精确诊断拒绝。仅修正期望诊断及说明。新的完整操作模块与 GM 暂存门
**202 项通过，1.21 秒**。这些是分别的回执，不声称单次 3,134 项全绿。
文档冻结后再执行最终 hooks。

原两层、sequence 128、batch 1 的 Qwen 捕获生成全部 32 个叶 C++，包含时间 6.692 秒。
报告 47 个观察到的任务控制所有者、**19 个 PyPTO 叶、13 个委托叶**。共享配方新增
`token_embed_single` 和 `token_embed_group`；此前 17／15 保留为基线。该时间是一次
描述性主机测量，不是设备或单独分析时间。

首个剩余义务包括四个混合有效域 ADD 叶（`out_proj_aiv`、`out_proj_aiv_0`、
`rms_recip`、`rms_recip_0`）、`qkpv_plan` 的 `tensor.write`、`rope_kv_cache` 的
`tile.reshape`、`lm_head` 的 Acc 描述符，以及六个既有 GM pipe-buffer 类型尚未准入的
混合对等叶：`qk_pv_partial_plan_aic`、`qk_pv_partial_plan_aiv`、
`qk_pv_single_block_0_aic`、`qk_pv_single_block_0_aiv`、`qk_pv_single_block_aic`、
`qk_pv_single_block_aiv`。首个诊断不等于算法失败或不可行证明。已有对等协议仍保留
原所有者，不构造新的跨核同步。精确导出与原生资格仍独立。

| 已建立结果 | 剩余前提或限制 | 合适表述 |
| ---------- | -------------- | -------- |
| 直接 INT32 GM 读取经普通 DMA 与本地 S 载荷降低 | 选定 TLOAD、原访问合法性及发布前提 | 可复用共享降低配方，不是新硬件定理 |
| ABI、SSA、控制和保守载体包络通过有界检查 | 未支持读取仍委托，精确 GM 选择器不可用 | 显式策略下保守局部同步 |
| 三个 planner 生成 C++，PyPTO 实际命令检查使用已有池 | 无设备或统一最优性结论 | 合格家族的主机可行性 |
