# 已放置累加器的行裁剪：操作契约子门禁

状态：设计原则已认可；实现、独立源码评审和新测试尚待完成。这是
[架构计划](41-general-autosync-architecture-plan.md) 的通用 M2 元数据/导出修复，
在独立的 [GM 标量暂存门禁](44-autosync-gm-scalar-staging.md) 之后实施。
不改变累加器、tile 大小、分配算法或同步策略。

## 缺失接口与支持族

已有静态 Acc producer/carry 对应物理 `[M,N]`。其后的
`tile.set_validshape(acc, rows, N)` 只是同一完整字节窗口的存储别名，
供原切点的默认双操作数 TSTORE 观察前 `rows` 行。
该元数据证明不额外接收 atomic、量化、activation 或显式 phase 变体。静态 matmul、初始化与循环 carry 保持原样，
不把动态元数据传回 matmul，不复制载荷，也不把元数据操作当作完成事件。

首个族仅支持 A2/A3 AIC、已有支持 dtype/fractal/layout 的已放置 Acc。
producer/carry 已建立完整静态有效 `[M,N]`；别名保持全部列、零偏移、同一 dtype、
物理形状、容量与 `CompactMode::Null`。原就绪行表达式应在已有操作定义域/范围
契约下满足 `0 <= rows <= M`；native store 仅在严格正行数时执行。
保留编码字段范围、GM 目标边界和原别名策略。不能从容量推断有效值或 `rows=M`。

动态 Acc producer/carry、裁剪列、compact 打包、偏移/步幅/layout 改变、动态别名
逃逸或进入未知操作、以及无地址 placement 继续拒绝。范围、就绪或所有权证据缺失时，
在选择阶段保留原 Functional 函数；不得用基准形状或内核名称补充前提。
必须检查整个受影响物理别名组件中的未知元数据观察，包括旧静态源、属性和聚合引用；
仅检查 trim 结果的直接消费者不够，因为原 Tile setter 共享元数据。

## Native 与表示依据

PTO-ISA 固定版本 `5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f` 的
`tstore_common.hpp::TStoreAccNz2nd` 接收 runtime validRow/validCol，
并检查它们等于 GM partition 维度。compact 关闭时源 stride 始终为
`TileData::Rows`，不依赖裁剪后的行数。CompactNormal 会按有效行数重新计算 stride，
因此不纳入该族。`TStore.hpp` 要求有效行列严格为正，所以零行必须跳过原空 store。
已有 writer 的物理打包资格与 native 编码范围仍适用。

现有 `pto.set_validshape` 要求两个有效元数据维度均为动态。
不能以 `[-1,N]` 绕过这个接口。给 store 别名独立的 `[-1,-1]` 元数据对象，
在每个 use 设置真实 `(rows,N)`；证明仍必须保留列值恒等于原全宽。
同地址同类型的多个 runtime trim 可能共享一个动态 handle，因此每次使用都要通过
`RestoreDataOperands` 恢复原别名的真实范围。与静态 writer 独立不表示所有 runtime 别名
彼此独立。
两个 `-1` 不证明相等、范围或可用性。没有新增设备或一般动态 layout 资格。

## 实现责任与有效性

共享 matrix/metadata recipe 识别静态全 Acc writer 和仅用于 store 的行别名。
BufferValidAnalysis 与选择仅接收已放置支持族。复用
`StorageIndex::FinalizeTypedRoot`，在同一已证明地址保存独立静态 writer 描述符与
动态 store 描述符。MatrixDescriptor 与 use-site lowering 只为已确认的别名发出
双动态元数据，并在每次使用恢复 `(rows,N)`。runtime-valid verifier 与 codegen
检查就绪、全部列、正行 guard 与编码边界。

字节地址、容量与 layout 相同不表示描述符不别名；效果分析仍应看到它们共享存储。
静态 writer 与 runtime store 的 native 元数据分离；多个 runtime 别名可能共享
动态对象，必须逐 use 恢复，不能宣称它们全局独立。整个组件内未知共享元数据观察均拒绝。
GM partition 必须为 `[rows,N]`；静态源 pitch 仍对应已有累加器内容。同步继续负责 M/FIX 完成、真实存储需求、
事件重用和最终 drain；元数据 setter 不完成载荷。保守效果不变成精确 selector。
不改变 ID 池、硬件豁免、专家协议、默认 dispatch 或零内循环 barrier。

## 验收门禁

- 通用已放置 GEMM/Acc：零、单、部分、全部行，不同静态形状，多次访问与 reset。
- 独立检查原 store 的地址、值、guard 与静态 matmul 初始化/carry。
- trim 前后使用旧静态别名，交替多个 runtime trim，暴露错误的共享元数据。
- 负例：列非全宽、无效/未知范围、未就绪界、compact、未知消费者、动态 carry/view、
  逃逸及 PTOAS 无地址 planner；均应在提交 Buffer lowering 前拒绝。
- 两种别名策略、required-order 覆盖、实际 token/flag 顺序、匹配、无环和消费先于重发布。
- 真实 PTO/C++：partition 为 `(rows,N)`，native 两维均动态，而静态 writer 不变；保留最终 drain。
- 保留 compact K/replay、本地标量、runtime-valid 与 zero-inner-body 回归，随后刷新原始 Qwen census。

构造与保留大小随静态别名/use 和共享元数据 DAG 增长，不展开 runtime 行数或访问次数。
结果是已确认的元数据桥接，不是新的最小需求定理或设备提速结论。
