# 既有传输周围的本地同步

状态：**共享生产布局及工作区上下文已安装，一致构建与上下文/集成测试通过，最终源码 hooks 通过；传输胶囊准入保持不变**。本方案扩展 M4 对既有通信边界的
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

已检查 TPUSH、TPOP、FIFO、TFREE 的版本为
`5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f`，固定于磁盘
`native-transport-5487` 工件。配套 JSON 记录主来源 URL、Git blob 和 SHA-256。
源码证据不等于编译器选中变体对应关系或运行时、设备行为证明。

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
选定 5487 版 TPOP/TPUSH/FIFO/TFREE 源码已固定于磁盘工件。tile-data TFREE
重载为空操作，GlobalData free 路径不同；完整选定变体对应关系仍是准入门。

## 首个生产基础组件（验证待执行）

`gm_pipe::LayoutEvidence` 保留精确不可变 Program、原 setup 语句及函数调用索引。
Orchestration 入口和 Graph 发射通过 `KernelRegistrations` 共享一个所有者；PTO
模块偏移消费同一所有者绑定的上下文。旧的两份布局扫描已删除，`gm_pipe_layout.h` 仍是唯一
ring、深度和范围公式来源。

每个实际请求的 setup 几何只解码一次。按原 root 身份缓存不可变成功或拒绝结果；
外来 root 和重复语句身份不能导出唯一原 cut。未使用的独立工作区大小接口已删除。
选中 setup 的非法 kwargs 保留原异常行为，不解码无关函数中未请求的几何。

仅扫描语句体，统一覆盖 Runtime、Hierarchy、SIMT 等具体作用域，避免两个布局
消费者漏掉不同的几何。这不授权任何对应上下文的原生传输。不展开表达式 DAG、
peer 调度或运行时访问。计数分别记录原语句出现、setup 解码/查询、调用边、闭包
函数/setup 关联、缓存命中和拒绝。不同 root 的重叠闭包仍可能重复访问索引并保留
独立结果映射，因此不声称全部查询的总工作线性。

新增测试覆盖身份、几何、失败缓存、递归调用、重复 cut、溢出、嵌套作用域及
8/32/128 个 leaf 的规模。现有公开 orchestration/PTO codegen 测试仍是生产回归门。
先前布局基础通过 168 项测试；新的上下文测试尚未运行。该组件只提供布局证据；原生就绪性、FIFO 生命周期、资源和
统一进展证明仍须完成，之后才能接受传输胶囊或新增 leaf 所有权。

## 布局上下文一致性子门（实现已安装，验证待执行）

共用公式不等于共用分配上下文。Orchestration 按实际 launch root 的可达闭包分配，
修复前 PTO 按整个输入 Program 排偏移。两个无关 root 的 `(id,dir)` 分别为 `(3,3)`
和 `(0,3)`、各需 2,048 字节时，前者可能仅分配 2,048 字节却使用 2,048 偏移。
正常 backend 已按 Group 分模块，不能未测试便宣称该例影响正常生产路径；直接
codegen、ungrouped peer 及独立工作区仍须验证。不能把所有工作区扩大为全模型范围。

已按接受设计在现有 `LayoutEvidence` 增加 `BindRoot(original_root)`，发放无公开构造器、无
可变字段的 `WorkspaceContext`。它保留原所有者、root、已验证工作区形参/调用
转发、原 setup cut，以及同时决定分配字节和偏移的 `LayoutResult`。单纯几何结果
不能充当上下文证书。

绑定须沿原 Call/Submit ABI 映射验证工作区实参身份；保留形参名字只标识角色，
不证明共享。缺失、歧义、单独分配、不支持的 alias/carry 或重复 cut 精确拒绝。
复用调用索引和 ABI 工具，不逐贡献重扫函数，也不从四字节占位符推断容量。
无 setup 的上下文无需工作区证明。

已安装的 API 与消费者改动：

1. 在 codegen 绑定公开现有布局所有者及不透明 `bind_root` 结果；检查计数仍来自
   同一所有者，不另建分析器或可信属性。
2. `_generate_single_chip` 为原 Program 建一个所有者。缩减 Group 成员模块或
   加入 ungrouped peer/helper 前，绑定实际 launch root；添加 peer 不扩大分配。
3. `GenerateOrchestration` 可接收共享所有者，既有 create-use 解析确定 root，
   使用绑定结果的字节数。入口和 Graph 共用该所有者。
4. `PTOCodegen::Generate`、`GenerateBundle` 和 `_generate_pto_bundle` 的正常及
   调试路径转发可选上下文。每个 setup 都须匹配原函数/cut 的 `SetupAt`、上下文
   成员关系和未变几何。缩减 Program 可保留原 Function；复制、修改或外来身份拒绝。
5. 直接 PTO API 未提供上下文时，仅当各含 GM setup 的 entry 自身具备有效且完全
   相同的键、偏移和范围时可推导。不同布局必须拒绝并要求原 root 上下文，不再
   默默合并。成员有意使用不同子集的 Group 必须提供显式上下文。
6. 句柄为进程内能力，不序列化为属性。反序列化或改写后须用完整原 Program/root
   重建。缺少该所有者的缩减调试模块须满足默认规则或拒绝，名字、字节数和手写
   metadata 不能恢复授权。

验证须配对实际 orchestration 分配和实际偏移：独立 Group 保持各自大小，多 pipe
Group 仍共用互不重叠区域；无关 root、外来所有者、改变 setup、缺失转发、独立
工作区、歧义 cut 拒绝。Graph、重复生成、bundle、调试及原 Program 序列化重建
使用同一规则。该子门不改变原生传输准入。

分别计费原索引、请求的上下文闭包、setup 成员验证、命中和拒绝。不同上下文可能
重叠，保留映射及验证成本按输出计费。此一致性修复已安装并冻结，等待构建和回归审查；尚未声明新测试通过。

实现将不可变 `LayoutSource` 与查询缓存分离，保留的上下文不会与缓存形成环。
每个相关 Call/Submit 出现都验证实际 ABI 转发，包括同一 callee 的多次调用。
限定为精确原形参身份，不推断 alias/carry。缩减模块通过不可变 Function 身份
封存其 body 和原 setup cut，无需再次扫描函数体。公开
`codegen.GMPipeLayoutEvidence` 是生产所有者，testing 名字仅为同一绑定的别名。
旧的独立大小接口已删除。

新回归将实际 orchestration 的 3,072/4,096 字节分配与各 Group 的实际发射偏移
配对，还覆盖默认/显式上下文、Call/Submit、失败缓存、歧义 cut、外来/反序列化
身份、新建绑定、所有者生命周期及重复 normal/bundle 生成。原生胶囊的就绪性、
重用、资源和进展准入仍关闭，不属于本批实现。

经审查保留既有 unmanaged 直接 codegen 拼写路径：调用没有显式上下文，且整个
发射 Program 都没有 managed 工作区形参。此路径仍做原有全模块数值几何检查，
不发射 GM 指针或 managed 偏移，也不发布工作区证书。它不声称 A2/A3 原生流水
能接受省略 GM 操作数；本地 PTOAS 源码要求该操作数，历史单元测试刻意在注入
前只验证拼写。任何 managed 角色形参（包括错误或歧义）、混合模式或显式上下文
仍走严格检查，`BindRoot` 拒绝保持不变。回归比较独立/合并的 unmanaged 函数体，
并检查数值错误、混合模式拒绝和 GM 偏移缺失。

## 最终工作树收口

最终一致构建通过。各通过门分别包含681例验证器/令牌擦除/布局、原10秒看门狗下
26例共享 DAG、274例 PTO 生成、151例原生/编排、96例所有权/运行时/后端及
三个 C++ 资源顺序测试。各门重叠，不能相加声称一次新的去重全绿测试。
配套 JSON 的 `final_validation` 固定源码、已加载二进制、Python、PTOAS、
模型、命令与日志。

原两层模型仍生成全部32个 C++ 叶：21个 PyPTO 所有者、11个委派，另观察到47个
已有任务控制所有者。32个 C++ 文件与已接受的标量写检查点逐字节相同。
八例实际命令审计包含不清空物理历史的重复调用；共享载体参考本身为全序，
因此零额外顺序不证明重叠改进或最少命令。不增加内核或传输准入、协议、ID预算或分配算法。

保留此前5014通过/一失败的原记录；它暴露的部分令牌擦除边界已有有效输入回归及
导入前精确拒绝。联合120秒看门狗和陈旧 API/元数据测试失败记录同样保留，
最终分模块测试继续使用120秒限额，不能将旧失败重标为全套全绿。最终 hooks 与
报告审查状态见机器可读检查点。

## A2/A3 消费者槽位检查

自动 pipe 配置与 AutoSync 委托使用同一项 FIFO 分析，从原始 pop 和 free 标记
统计尚未释放的槽位。自动配置统计其单一生成队列，委托按已声明的 pipe ID 分别
统计。受支持的循环和分支恢复入口队列状态，无需枚举执行
次数。委托报告保留原有 Buffer 准入原因，并将标记深度与消费者初始化引用的
reserve 声明比较。标记深度不能证明别名使用范围、局部 DMA 完成或对端进展。

在复制 no-split AIV 执行体之前，编译器会拒绝原始活动 lane 中能证明的槽位覆盖。
该证明要求单一支配的消费者初始化、无条件执行的全有效稠密 FP32 Vec pop 序列，
以及槽位回绕后对较早未释放值的直接、已注册的 tile 数据读取。形状未知、提前
退出或通过别名读取的情况不在此拒绝范围内。所有原生通信操作均保持原有顺序，
包括有效行数为零的 lane 的握手。

## A2/A3 原生 FIFO 端点的单核 AutoSync

跨核通信操作由程序员或上游变换指定，ready/credit 信号由原生 FIFO 实现。
AutoSync 保留通信序列，在每个执行核内规划存储依赖。

第 `k` 次接收的本地槽位起点是 `reserve + (k % local_depth) * slot_bytes`，
传输描述符限定字节范围。pop 在 MTE2 上写入这些字节，TMOV 在 MTE1 上读取，
向量算子在 V 上读取。TFREE 释放逻辑借用。后续 pop 覆盖该槽之前，仍须等
上一次物理读取完成。push 在选定流水线上读取本地源数据。这些访问与普通算子
共用依赖分析和本地事件分配器。规划器验证事件复用前已被消费，并在显式返回之前
或无返回值路径结束时清空流水线。末尾 AIV 子块分支的每个分支路径独立清空。

导入器将每个 FIFO 表示为轮转的物理存储。若顺序循环每次迭代接收 `p` 次，
第 `v` 次迭代从槽位 `(entry_ordinal + v * p) % local_depth` 开始。
退出循环时接收序号增加 `trip_count * p`。区域依赖分析将每个槽位的最后一次写入
及尚待完成的读取传递给下一区域，即使一次迭代只走过部分槽位，尾部接收也能等到
前面迭代中对应的读取完成。

当前支持直线序列、静态有界且每次接收次数固定的顺序循环，以及末尾按 AIV 子块
分支且两臂均可独立验证的端点。借用须在区域内释放。有返回值的函数要求每条
路径末尾都有显式返回。提前返回、嵌套循环和运行时确定边界的循环仍需进一步验证。
要求显式放置、conservative_prefix 策略及受支持的 FIFO 工作区。实际调用
须将按容量分配的工作区与普通 GM 张量分开。普通张量之间可以别名，本地分析
将这些张量视为同一保守存储域。

每个 PTOAS 编译单元只有一个本地同步插入开关。两个端点分别规划。若全部通过则
采用生成的本地同步并关闭 PTOAS 自动插入；否则保留原始函数并为整个单元开启
PTOAS 本地同步插入。回退原因分别标明哪个端点失败、哪个已验证但共享同一调用。
两种模式都保留程序员指定的跨核协议。

主机诊断 `testing.inspect_peer_fifo_protocols(program)` 可显式检查收发序列、
写入方数量和 ready/credit 进展。AutoSync 的所有权仅表示单核存储顺序已验证。
例如，仅编译的回归用例保留五次发送和四次接收，并验证两次本地 Mat 槽位覆盖。
这种通信序列在设备上可能阻塞。
