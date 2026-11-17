# 已放置累加器的行裁剪：操作契约子门禁

状态：实现与独立源码审查已接受；一致构建、139 项组合测试及原模型捕获通过。
随后双别名策略组合门 151 项通过，格式化后重新构建的 151 项复验、
三项 C++ 测试和最终 hooks 也通过。这是
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

## 实现检查点（2026-11-16）

新增 `BufferAccTrimAnalysis` 统一持有函数内只读能力，供选择、
`BufferValidAnalysis` 与 `MatrixDescriptor` 使用；放置后重新证明完整字节窗口相同。
初始观察者门比按组件检查更保守：函数中任意未知 Acc 元数据观察都会拒绝此族。
必须已有配置且当前 backend handler 为 A2/A3；AIC 类型本身不足。无地址 planner
保留 Functional。范围仅来自原就绪 SSA、Tensor 形状 ABI 的非负性、常量、min/max
和原子比较，不补入任意算术或 benchmark 界。原 For 坐标仅在原循环体中就绪，
不推断符号、次数或范围。另显式要求 native 字段 `M<=8192`、完整 `N<=4095`。

沿用 `FinalizeTypedRoot`，静态 writer/carry 不变，store 别名的两维均动态；
`RestoreDataOperands` 在每次 use 恢复 `(rows,N)`，包括 A/B/A 交替和再次使用原全矩形。
实际 Buffer 门独立验证已放置自有句柄、正且有界的行、全列与 GM partition 相等。
保守分析、真实 token lowering、所有权和代码生成共享此门。缺证是拒绝，绝不是空访问。

组合门 **139 项通过，55.46 s**，覆盖 Acc 桥、17 项证明上下文、静态 row descriptor、
runtime row/cast 与 GM scalar。通用物理形状为 `(16,32)`、`(32,64)`、`(64,128)`，
检查零／一／`M-1`／全行、重复访问和 reset、0/1/3 次完整静态 carry、多个 runtime 别名。
独立解释器核对载荷身份、原物理 packing 中声明的传输矩形、guard、匹配、无环及消费先于重发布；
实际非连续 ID `{2,5}` 不超池。已放置 planner 和 PyPTO 同步默认关闭、PTOAS 同步仍开启的路径
均生成真实 PTOAS 0.72 C++。不声称数值 matmul、设备或精确 burst footprint 验证；
保守全 Acc/DDR enclosure 仍保留。随后双别名策略组合门 **151 项在 105.16 s 通过**，仍使用 120 s watchdog。
两种策略使用相同且明确不重叠的具体 GM 基址，may-not-alias 明确要求不同根不重叠。

深共享 guard 测试使原 receipt 的 `ScheduleTreeSize` 树遍历触发原 **30 s** watchdog。
修复以表达式子树的展开出现次数为权重缓存，不改变原大小度量；语句／命令仍按出现计，
实际检查工作另计。小例与独立展开 census 相等；深度 28 不再展开路径，但保留大的展开计数。
溢出明确为 `cost_unproved`，可选完整候选无法证明预算时保留原 guarded assigned-cut 发射，
不把饱和值当作可容纳，也不抛 baseline 大小内部错误。加数在运算前提升位宽。
需求、ID、协议与预算均不变。Guard 证明迭代抽取不同原子合取，不把共享布尔树交给递归简化器；
范围缓存按证明上下文隔离，不声称跨任意上下文全局线性，也不展开行数或访问次数。

保留早期门：29 通过／3 个 fixture 失败、136 通过／1 个新反序列化身份与旧空执行参数映射
不匹配；修复未放宽 native 门。保留原 30 s 超时；另一个 **10 s** 原始 DAG 的 BufferIR
验证超时仍是通用 SSA/use-after-def 成本缺口，不能跨作用域盲缓存引用。含 flag 的最终完成
重写器通用表达式遍历也是独立后续，不是本次无 flag receipt 超时的原因，尚未宣称修复。

首次捕获把 timeout 选项传给 Python，未进入编译器，作为 runner 错误保留。修正后仍用
**120 s** watchdog，原双层、sequence128、batch1 捕获在 **6.7544 s（包含时间）** 内生成
全部 **32 C++** 叶：**20 个 PyPTO、12 个委托、47 个任务控制所有者**；新增 `lm_head`，
前一 GM 门为 19/13。预期所有权失败为空；单次主机测量不证明提速或完整模型正确。
剩余 12 叶为四个混合 valid ADD、`qkpv_plan` 的 `tensor.write`、`rope_kv_cache` reshape 族
及六个已有 mixed-peer 域。保持原 M1–M6 计划未全部完成、对等／跨 kernel 协议不变。

EN 报告与 JSON 保存可重现命令、源码／构建／工具哈希及历史失败。捕获时哈希明确属于
最终 hooks 之前的快照；格式化后另加最终源码 pin，不冒充同一构建。精确动态 GM 导出、
其他算术／就绪形式、更广调用者配置、native 与设备资格仍分开。

| 已建立结果 | 剩余前提或限制 | 合适论文表述 |
| ---------- | -------------- | ------------ |
| 静态 Acc writer 与 row-only store 别名共享原字节并恢复元数据 | A2/A3 native store、原访问有效性、全列和 packing | 通用 descriptor/export 桥 |
| 独立有界命令检查与原分配器通过 | 保守 root 顺序；无新最小需求／ID／burst 定理 | 主机模型同步可行性 |
| 展开大小权重与小 census 一致且不展开 DAG 路径 | 通用 BufferIR/SSA 原始 DAG 验证仍未资格确认 | 有界诊断和候选计数构造 |
| 原 Qwen 新增 `lm_head`，20/32 叶自有 | 12 叶及 M1–M6 未完成；无设备验证 | 增量生产覆盖，不是完整模型自动同步 |

最终格式化源文件、重新构建扩展及验证日志固定在 JSON 的
`final_formatted_snapshot`。重新构建后的独立门槛为 151 项通过（106.14 秒），
三项 C++ 资源/顺序测试和仓库 hooks 通过。保留早期捕获快照，不重标旧哈希。
