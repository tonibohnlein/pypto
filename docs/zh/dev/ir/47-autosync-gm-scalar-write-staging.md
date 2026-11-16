# 通过现有私有载体实现 GM 标量写入

状态：**设计在下述选中 TSTORE 契约限定下已获批准；实现和验证待完成**。
本方案扩展[GM 标量读暂存](44-autosync-gm-scalar-staging.md)与
[本地标量契约](43-autosync-value-and-scalar-contracts.md)，不增加后端、
公开 pass、分配器或外部可见性规则。
[机器可读检查点](../../../en/dev/ir/47-autosync-gm-scalar-write-staging.json)
列出待完成的门槛；当前没有新增测试或设备结果。

## 一个原始写入对应一次实际传输

初始范围为 A2/A3 AIV 上普通、紧密、无填充 ND 一维 INT32 GM 张量的
`tensor.write(destination, [index], value)`。原始索引与 INT32 值必须在该
语句位置就绪；保留原操作地址不越界且不回绕的前提，不能用基准测试尺寸
或指针名称代替证明。非默认 kwargs/属性、peer/distributed 张量、原子或
发布变体、其他类型/核心，以及没有现成语句位置的嵌套副作用调用均不准入。

```text
变换前，两种语句形式：
  tensor.write(destination, [index], value)
  updated = tensor.write(destination, [index], value)

变换后，在同一实际执行位置：
  local = tile.write(private_carrier, [0,0], original_value)
  tile.store(local, [original_index], original_destination)  # 仅一个 GM 元素
  updated = original_destination                           # 仅赋值形式
```

这里是语义伪代码，不是新增公开语法。常规 Tile/Buffer lowering 提供明确的
单元素传输范围。复用现有 32 字节 Vec 载体：INT32 物理形状 `[1,8]`、有效
形状 `[1,1]`、常规分配与放置、常量本地点访问。不得读取旧 GM 值，也不得
初始化八个 GM 元素。本地 S 写入定义唯一有效标量；MTE3 必须只写目标的
四个字节。其他七个载体元素不能被结果或副作用观察。

赋值结果按照 `ResultAliasedArgIndex` 与原目标别名，不是复制或新存储根。
保留原 SSA 名称/类型及后续循环、分支的别名使用。无结果形式也必须恰好
执行一次两项副作用。保持原 For/While 框架、跳过路径、重置和语句顺序。
载体可在常规分配位置分配，但写入/传输不能移出实际执行的循环或分支。

## 完成性与内存前提

现有 `buffer.write` 契约提供 S 上阻塞的 typed UB 写入。现有 prefix 服务
必须保证该写入先于 MTE3 读取，并保证 MTE3 读取先于同一载体的下一次写入；
还必须处理 GM 写入与后续重叠本地读写的依赖，保留调用结束的完成屏障。
SET/WAIT 配对闭合本身不等价于 payload 完成。重复访问只能在这些实际依赖
下复用私有载体，不展开运行时访问次数，也不创建新事件池。

准入前需要独立审查写方向的两个原生事实：

1. 选中的 A2/A3 INT32 Vec→GM TSTORE 支持有效形状 `[1,1]`、32 字节对齐
   UB 载体和元素对齐 GM 地址，包括不按 32 字节对齐的索引 1、7。GM 写入
   长度必须是四字节，不得对邻接 GM 元素做读改写或覆盖。
2. 现有 SetValue/UB 与 MTE3 同步契约使写入的标量对传输可见。本地读取的
   保守包络可以是 32 字节，但填充不能影响目标值或目标字节。扩大本地
   读取包络不能授权扩大 GM 写入。

本地固定快照已提供写方向构造证据：`tstore_common.hpp:366–394` 的
`TStoreUb2gmNd2nd` 对单元素 GM 视图产生 `nBurst=1`、
`lenBurst=1*sizeof(INT32)=4`；`TStore.hpp:19–28` 选择
`copy_ubuf_to_gm_align_b32`。JSON 记录两份快照哈希。这是在规范 TSTORE
契约下的字节数构造证据，不是对私有 CCE intrinsic 与公开非对齐复制 API
对应关系的独立复现。

其余事实必须绑定到 design44 使用的已固定 PTO-ISA 版本及实际选中实现。
GM→UB 的 TLOAD 证据不能替代反方向证据。编译和带毒化邻接内存的主机测试
验证构造，不能独立证明硬件缓存行为。缺少有依据的选中 TSTORE 契约时
仍不提供该能力。与 design44 相同，未独立验证的私有 CCE/公开 API 映射
作为现有规范 TSTORE 前提下的明确原生限定；它本身不要求拒绝已有依据的
选中契约。

本方案不推导 peer 初始化、原子性、跨核可见性、发布顺序、跨 kernel 就绪
或竞争确定性，也不将 peer 协议视为普通私有 GM 访问。保留选择前对同一 GM
根混用标量 D-cache 与 MTE3 写入的现有拒绝；支持该情形需要单独证明整个根
的通道替代。仅在后期 selection 暂存的增量保持此拒绝。
若实现要准入原本混合的根，同一不可变 stager 必须在
`UpgradeWrittenTensorParamDirections` 分类之前构造完整、已验证的候选，
由实际 Tile stores 分类通道并从原 GM 目标推导方向，不能仅将原
`tensor.write` 标成 MTE3。尚未暂存的标量访问或未证实的缓存/发布交互仍
拒绝。原函数必须保持为后续完整准入的回滚所有者；早期候选位置及回滚
生命周期需要在宣称支持混合根之前审查。本次批准的增量仅在后期 selection
暂存，不包含早期候选搬移。

## 按顺序扩展现有所有者

| 现有所有者 | 有界修改 |
| ---------- | -------- |
| `buffer_scalar_recipes.{h,cpp}` | 复用紧密一维几何，加入载体源、原始 GM 目标和单元素范围的写方向检查；不能交换 load 参数冒用读证明。 |
| `buffer_scalar_staging.{h,cpp}` | 当前读 stager 增加直接赋值和 Eval 写入；复用名称、分配及不可变语句遍历，保留目标别名，不修改不支持的候选。 |
| `SelectBufferIRFunctions` | 保持 enable-Buffer、enable-autosync、`conservative_prefix`、A2/A3 AIV 条件。只有完整常规准入成功才发布；否则返回原函数。 |
| Buffer 类型推导、Tile lowering、PTO Buffer codegen | 消费同一写几何证据，保留一维 ABI、原索引、根、输出方向及精确范围；覆盖 addressed 和 PTOAS 放置。 |
| `buffer_sync_contract.cpp` 与现有效果导入 | 保留 S 载体写、MTE3 载体读/GM 写、就绪和复用义务，使用现有操作注册表。 |
| `gm_access.cpp` 通道/方向分析 | 保留常规混合通道拒绝；早期候选必须完整、持有原回滚对象并按实际 payload 分类，不能放宽通道标签。 |
| DCE/结果别名服务 | 利用现有操作契约保留 Eval 副作用和赋值别名，不能当作未使用纯结果删除。 |

当前 `StageBufferScalarReads` 只处理读并跳过 Eval，原生和 codegen 的一维
传输检查仅接受 load。原 `tensor.write` 注册为标量通道，其 Functional
codegen 发出 `pto.store`。应扩展同一 stager（必要时改为涵盖双方向的名称），
保留原操作契约和默认关闭路径，不增加公开配置。

构造工作应按原始语句次数及不同表达式/类型节点计费，每个静态位置保留
一个载体及常数项操作。共享表达式 DAG 保持共享，运行时索引、访问次数和
重置次数不扩大表示。此界仅约束 stager，不是完整同步/放置编译界。

## 支持声明之前的验收门槛

1. 先固定 TSTORE 实现、字节数、对齐、本地读取包络和 S/MTE3 顺序证据，
   将原生限定与编译器测试分开记录。
2. 独立验证 Eval/赋值形式、索引 0/1/7/8/末元素、长度 1/9、INT32 极值、
   符号长度、两种别名策略及允许的重叠根。毒化邻接 GM 字节和载体填充，
   比较原始目标字节及未修改邻居，而非仅检查打印的范围。
3. 覆盖零/一次/多次访问、条件跳过、标量读改写递推、重复写、外层重置及
   While 最后一次假条件。保留根/索引/值身份和返回别名；嵌套或不支持的
   副作用必须事务式拒绝。
4. 独立检查实际内存顺序、配对、无环、非连续事件池、复用前消费和最终
   完成性。删除握手的破坏性变体必须使 oracle 失败。
5. 覆盖公共 Select/LowerBuffer、Insert/Optimize/LowerSync 及三种 planner
   的真实 PTOAS C++；仅显式策略启用。关闭 autosync/Buffer、exact-only、
   不支持的类型/元数据保持原准入行为，保留混合通道拒绝测试。
6. 不展开循环地统计载体/操作/命令，回归读及本地标量测试后再捕获原模型。
   新达到的诊断位置不意味着整个模型已被接管、最少需求或设备加速。
