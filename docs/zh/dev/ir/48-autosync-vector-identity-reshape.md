# 连续向量恒等 reshape：有界设计门

状态：实现及测试审查已接受；最终4259例完整门、128例格式后测试、三个 C++ 测试
及全部 hooks 通过。此前20/12捕获是历史证据；写入和恒等视图联合捕获为21/11，
新增所有者归因于写入暂存。最终报告审查接受本检查点。此门在
[检查点 46](46-autosync-acc-row-trim.md) 后继续原 M2/M3；
[原计划](41-general-autosync-architecture-plan.md) 保留。不新增 buffer 模型、
native 几何、分配算法或同步协议。

## 缺失契约

现有 `SupportsLogicalContiguousVectorReshape` 支持合格 FP32 Vec 的
`[L,1] <-> [1,L]` 轴映射及对应活动前缀，但拒绝物理形状、layout、pitch、偏移、
dtype 与 valid 元数据全相同的同方向 reshape。因此含 runtime 元数据的函数会在
`BufferValidAnalysis` 受限。原 Qwen `rope_kv_cache` 的显式列结果恒等 reshape
暴露了该门。这是 admission／元数据缺口，不是新的需求分析类。

只在既有连续 FP32 向量几何内扩展共享谓词：保持 `[1,L]` 或 `[L,1]` 物理形状、
dtype、空间、方向、有效 layout/fractal/pad/compact、零 descriptor offset、
已证明的稠密物理 pitch、完整原字节窗口，以及两维实际 valid（含零与 runtime 前缀）。
复用 `LogicalContiguousVectorLength`：物理 L 仍为正的八倍数。
物理 `[1,1]` 仍不可用；L>=8 中活动长度为一并不是物理 L=1。
不新增任意二维恒等、dtype 重解释、移动窗口或 stride 几何。

## 所有者与正确性

保留原共享布尔谓词，在活动域 transfer 处区分恒等与轴交换。
`DomainIndex::Forward` 和 `OperandDomain` 对同方向恒等均保留原 prefix 轴；
真正方向交换仍走原映射。不根据某一 extent 恰为一重新推断恒等的轴。

几何谓词不证明 view 来源。选择保留此路的 `!IsWindow` 与 dense-pitch 检查；
`BufferValidAnalysis` 判断 runtime identity 不受限时须同样约束来源，
不能经旧静态 view fallback 意外接收动态窗口。原已支持静态窗口行为不变。

沿用 `StorageIndex`／`FinalizeTypedRoot` 与已有 relabel lowering，重查原放置窗口。
恒等不传输载荷，不新增 alias 分配、native 指令、flag 或完成切点。
每次 use 恢复原 SSA 元数据；别的 setter 不能抹去旧别名的活动前缀。
沿用原 planner 的元数据所有权限制，不声称新的无地址 reinterpret 导出。

证明仅为已接收域上的 descriptor 等价：活动元素的物理地址和值不变，prefix 轴不变。
缺少 pitch／来源证据的 view 不算恒等；payload 完成、GM 选择器和 event 复用仍由原服务证明。

## 实施与验收

1. 为现有共享连续向量谓词增加严格恒等分支，审计全部消费者，特别是前后活动域 transfer。
2. 一致使用带来源的 runtime admission；复用放置后 relabel 和 per-use 元数据服务，不发 copy。
3. 通用行／列例覆盖静态与 runtime、零／一／部分／全活动长度、跳过分支、其他 setter 后的
   旧别名、重复访问与 reset；多个物理 L，并保留物理 L=1 拒绝。
4. 独立检查声明访问与活动元数据、原字节／值对应、真实 token/flag 载荷序、匹配与复用。
   两别名策略仅使用各自合同允许布局；stride、offset、view、形状／元数据变化、dtype、
   未支持消费者应有明确拒绝。记录实际 planner 结果，不假定所有无地址路径可用。
5. 合格者生成 PTOAS C++，保留轴映射、活动域、标量、Acc 与 completion 回归，刷新同一
   原 Qwen 捕获。若仍不能接管整个叶，报告新的首个义务。

实现须再经审查才接受提交。保留表示随静态 descriptor/use 与证明上下文增长，
不展开访问次数或活动长度。原始 DAG 验证与最终完成重写器成本仍为独立后续。
此门不完成原 M1–M6 或全模型局部所有权。

## 实现与首门

现有共享谓词已加入严格恒等分支，保留原 FP32 长度／对齐、规范 pitch、零偏移和
非活动维固定为一的要求；真正方向交换规则不变。规范化器的私有 `IsVectorReshape`
与 `ReshapeDomain` 在 forward 和 operand-demand 两方向均显式保留恒等的 prefix 轴。
`BufferValidAnalysis` 对继承窗口或未知 pitch 保持受限；原 SelectBuffer 的 vector-relabel
来源门不改，不把旧静态 fallback 升级为动态元数据权限。无新的 selector hunk、公有操作、
后端或 CMake 源码。

首个模块 **22 项在 9.11 s 通过**；其后将 no-reshape 断言增强为同时检查 Assign 和 Eval
中的 Call。原 no-copy 检查已针对实际载荷。这是增强前的历史门，不冒充最终源码已复测。
通用例涵盖物理 L=8/16 的行／列恒等、runtime 零／一／部分／全部、静态前缀1/15/16、
其他 setter 后的旧别名、变化前缀、重复访问与 reset、跳过分支，以及相同不重叠布局上的两种
别名策略。PYPTO/DSA_RP 生成真实 PTOAS C++；矩阵恒等、继承窗口和物理 L=1 仍拒绝。
复用原 dtype/stride/offset 几何限制，不扩展。

Torch 生成结果与独立算术及输出 sentinel 比较；原字节／控制解释器独立检查模型所需顺序覆盖、
真实 token/flag 载荷序相等、匹配、无环与消费先于复用。这些是主机模型检查，不是 native／设备
数值资格，也不是新的精确 minimum-cover 结果；root 使用原 conservative-prefix 策略。
访问次数3/257/65537 时静态 Buffer 语句／载荷 inventory 不展开，恒等不应新增 copy 或 reshape Call。
此项不证明设备提速或任意证明上下文的全局复杂度。增强断言后的47写入、48恒等及既有GM读
组合门 **122项在11.26 s通过**，日志为
`build-py314/review-repairs-20261116/write-identity-qualified-gate.log`。

原两层Qwen的`qwen-write-identity-qualified/receipt.json`生成全部32个局部C++叶，
仍为 **20接管／12委托**、47个任务控制域，并未接管整个Rope叶。新的首个拒绝是
`prefill_fwd.py:781`的`tile.mul`：来自`[16,128]`分配的`[16,64]`切片与cos/sin切片相乘，
两输入声明的活动行分别为完整16和`group_tok`，尚未建立相等证明；
其后仍有来源pitch义务。这是活动元数据／窗口契约，而非已经修复的列向量标量乘法。后续仅标量写入捕获是另一快照。
源码子集仍明确标为最终格式化前pin；完整捕获回执保留实际共享源码／工具环境。
M1–M6 与独立验证／实现缺口保留。

后续标量写入组合捕获的21接管／11委托包含47的point-write修复，不能归为恒等admission新增整个叶。
共享语句重写后续见[检查点49](49-autosync-statement-rewrite.md)；通用raw-DAG验证成本仍分开。
最终格式化后组合测试结果尚待完成。

## 最终一致验证（2026-11-16）

最终源码通过独立审查。新的串行完整门禁4259例通过、零失败、七条警告，耗时560.27秒。
格式及类型检查修正后，一致构建、128例重点测试（11.32秒，原120秒看门狗）、
三个 C++ 测试与全部配置 hooks 均通过。配套 JSON 的 `final_verification`
分别固定脏源码、已加载扩展、Python、PTOAS、命令和日志哈希，而不是只记录提交号。

最终原模型捕获仍生成32个 C++ 叶函数：21个 PyPTO 所有者、11个委派，
47个已有任务控制所有者被观察到，严格名称检查通过。最终八例实际命令审计
包含不清空物理 ID 历史的重复调用；相对所选暂存字节参考零遗漏、零额外顺序。
共享载体的冲突已使该参考为全序，因此不证明重叠改进或最少命令。
报告47保留实际协议成本。

首次刷新脚本错误地按文件运行带包相对导入的模块，在编译前失败；改用 `-m`
后同一输入及看门狗通过，原失败记录保留。通用验证器 DAG 成本、本地传输契约、
混合有效域语义及原始 M1–M6 剩余义务仍独立跟踪；不声明设备验证或整模型本地所有权。
