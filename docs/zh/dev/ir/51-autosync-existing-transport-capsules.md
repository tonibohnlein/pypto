# 既有传输周围的本地同步

状态：**设计已获准实现，须满足下述结构证明和工作计费要求；尚无实现或验证声明**。本方案扩展 M4 对既有通信边界的
消费，保持已选 TPUSH/TPOP/TFREE 协议、peer 配对、ring epoch 和进展规则，
不构造新的跨核或跨 kernel 同步协议。

## 缺少的是本地证据接口

原始捕获中六个 leaf 仍委托给 PTOAS：`qk_pv_partial_plan_{aic,aiv}`、
`qk_pv_single_block_{aic,aiv}` 和 `qk_pv_single_block_0_{aic,aiv}`。
`qwen-pure-write-qualified` 的首个拒绝为 Var 的 `type_layout`，尚未进入本地
传输资格检查。这不能证明硬件不支持本地 autosync。现有所有者已经掌握工作区
大小、pipe 参数、本地保留区和 consumer 释放结构，但尚未向 Buffer 准入及本地
同步导出。

`__gm_pipe_buffer` 在 IR 中是单元素 FP32、四字节 MemRef 占位符；真实工作区
大小由 codegen 管理。不得把它当成四字节传输范围，也不能仅凭名字证明工作区。
证据必须绑定原注入参数身份及实际 pipe setup/call 所有者。

## 复用现有所有者

| 现有来源 | 可恢复事实与所需导出 |
| -------- | -------------------- |
| `cross_core_pipe.cpp` / `BuildAutomaticPipeSetup` | 方向、slot 大小/深度、本地保留和原配对 setup；保持手写参数，不用默认值替换 |
| 同文件 `RequiredConsumerDepth` | FIFO pop/free 顺序及支持的作用域局部闭合；导出借用/释放证据 |
| `tpop_tfree_finalizer.cpp` | view 别名和使用延长 pop 生命周期；保留原 pop、使用和逻辑释放位置 |
| `gm_pipe_layout.h`、`ComputeGMPipeWorkspaceElements`、`PrepareGMSlotBufferLayout` | 有效 slot 数、独立 rings、完整工作区及各 pipe 偏移；共享一次检查后的结果 |
| `pto_ops_crosscore.cpp` | 实际 split、dtype、transport valid shape、initializer 操作数及原生调用 |
| `isa_macro_resources_protocol.cpp` | pipe 所有的 ready/free 信号及特定变体的本地事件；按实际方向/目标选择，不混淆资源池 |
| 现有原生资格和保守效果服务 | 消费目标/core、效果、工作区隔离和本地命令证据，不增加平行效果注册表 |

捕获的双向 setup 使用 `slot_size=8192`、`slot_num=2` 和两个 ring，因此 GM
范围为 **32,768 字节**而非四字节；所示每个本地 consumer 保留区为 16,384
字节。这些是该 setup 的事实，不是通用常量。一般实现须使用受检算术、一致的
两端 setup、实际本地深度和所有独立 pipe 区域，复用共享布局规则。

## 能力归属于原选中传输

在现有选中原生实现证据旁增加工厂所有的本地传输能力，持有原 Program/function
配对及目标相关 setup；每个调用的贡献引用该所有者。调用名或外部属性不能伪造
能力。能力保留原 initializer、workspace 参数、reserve/import 身份、方向/ID、
split/dtype/深度和原生变体；完整 GM/local 包络和与普通 buffer 的重叠；原本地
输入就绪位置、pending transfer 和结果就绪；借用 slot 的使用区间及 rearm；
跨调用信号和本地事件的独立资源域；不变的 peer 命令和支持的进展前提。

首个实现应限定于几何和词法 setup 已建立的 A2/A3 GM-backed tile-data 变体。
未知 control payload、外部初始化、混合变体、逃逸 borrow/carry 或不支持的 split
保留精确拒绝；不能宣称其他所有形式都缺少硬件契约。

## 分开证明入口、释放和进展

已检查的本地源码快照为 `physical-sync-prototype/isa/hw-native-sys-pto-isa-f83d4b7`。
最终准入必须固定编译器实际选择的实现，并在该版本验证相同义务；此快照不是当前
runtime/toolchain 身份证明。

该 A2/A3 实现中，TPUSH 可能等待空闲 slot，然后通过 FIX/MTE3 TSTORE，并在相应
pipe 发布 ready。TPOP 等待 peer ready，绑定本地保留 slot，再通过 MTE2 TLOAD。
tile-data TPOP 可在 MTE2 发布 GM slot 的 free，而 tile-data TFREE 是空操作。
因此必须区分：

1. Push 实际读取前的本地 producer 完成，以及读取仍 pending 的状态。
2. Peer ready 只允许读取 GM ring，不能证明随后本地 MTE2 写已完成；consumer
   仍需现有 MTE2 到 consumer 的依赖。
3. GM ring 读取后的释放与本地 slot 对 consumer 保持存活是两个生命周期。
   不得把空操作 TFREE 当成本地完成指令；FIFO 证据和后续 slot 写共同约束复用。
4. Ready/free 信号跨调用存活，单个 push/pop 不等于 resource-empty。按实际变体
   绑定信号，不将所有替代规则同时预留，也不从本地 pipeline-event 池分配它们。
5. 本地服务不能移动 peer wait/publication/release，也不能引入必须等待更晚 peer
   命令才能完成的等待。每个 transport 周围增加 ALL 不是自动成立的进展证明。

若原生前提确实未解决，必须指出具体选中操作和位置，例如某 pop 的 free 是否在
GM read 后发布，或下次 local-ring 写是否能越过仍存活的 consumer。几何/FIFO
尚未导出属于实现工作，不应称为硬件前提不存在。保持原已接受 peer 协议，其配对
和进展继续由现有 module 级所有者负责。

## 构造、回滚和验收

从原 setup 一次构造共享 pipe/layout 所有者，在原命令位置导出本地贡献。Buffer
选择消费真实 ABI/保留 slot 证据，不能把占位符改称普通小 Tensor。Lowering 保持
原 push/pop/free 及结果别名；现有保守效果服务处理本地读写与依赖，当前 reservation
机制处理隐式资源。不增加 exact backend、访问枚举或通用协议求解器。

所有本地效果、生命周期、目标和资源义务成功后才发布。失败保留原源码及 peer
协议；局部成功不能抹去外部完成/transport owner，或宣称未限定 sibling 已被接管。

验收需要：单向/双向、多大小/深度、多 pipe 偏移及真实工作区检查；配置冲突、
不足空间、重叠、外来 owner 负例；push 输入、pop 输出、slot 复用依赖，含 view、
零/跳过/重复访问、嵌套 reset；独立比较全部原 peer 命令。对既有协议建立小型
双核模型，保留 credits、ready/free epoch、pending DMA、阻塞 wait 和最终 drain，
证明新增本地服务保持进展，并用破坏释放/就绪边的变体检验。还需资源碰撞、未知
变体/目标，以及 ALL 不消费远端通知的负例；公共选择到实际 PTOAS C++ 检查
工作区和原生调用，再捕获六个原 leaf，逐个报告新拒绝而非预设成功。

本提案未运行测试。六个传输导出是可继续实施的工作；设计或 C++ 成功都不能证明
设备行为或整个模型已被接管。

## 统一结构证明与实际工作计费

有限双核执行仅作为回归 oracle，不能替代验收证明。支持的 capsule 必须为所有
准入执行建立结构不变量：setup 支配所有使用；分支保持进入时的 borrow/credit
状态；循环对零次或任意次数访问恢复同一本地所有权不变量；别名使用始终位于
借用区间；新增本地依赖不能等待未来 peer 动作。原生 credit/epoch 仍由原协议
负责。分支、循环或别名前提失败时拒绝该 capsule，不猜测循环次数上界。

每个不可变原始 owner 只恢复一次 setup/layout/FIFO，逐调用证据使用保留索引，
不重新扫描全函数。分别记录不同源码语句/表达式、setup/layout 项、borrow/use
关联、逐调用查询和被拒绝尝试的工作。失败恢复同样保留精确义务，不能仅缓存
成功结果而让重复拒绝无限重复恢复。此局部计费不等于整个编译器线性时间声明。

首个证据组件可先恢复共享原 setup/layout，并明确返回尚未满足的原生、生命周期
或进展义务；全部选中变体证明进入生产消费前保持准入关闭。只供诊断的平行清单
不算本里程碑完成；最终布局和传输消费者必须共享保留的 owner。

## 独立设计门

架构及正确性审查接受本设计进入实现。设计接受不表示实现、可执行所有权或验证完成。

有界双核执行须补充对重复访问和计数器复位的结构性生命周期与进展证明，不能替代它。
成本须区分一次 setup/layout/FIFO 恢复、索引化逐调用贡献、effects 关联输出及失败尝试；
禁止逐调用反复扫描全函数或别名 DAG，也不枚举运行时槽与访问。
选定5487版 TPOP/TPUSH/FIFO 源码已固定于磁盘工件，完整选定变体及 TFREE 对应仍是准入门。
