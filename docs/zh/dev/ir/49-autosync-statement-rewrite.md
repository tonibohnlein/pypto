# 仅语句的同步重写

状态：设计及最终源码审查已接受；修正构建、原三种深度回归、4259例完整门、
128例格式后测试、三个 C++ 测试和全部 hooks 通过。最终报告审查接受本检查点。
这是[原 M1–M6 计划](41-general-autosync-architecture-plan.md)中有界的 R4 遍历修复，
与[向量恒等 admission](48-autosync-vector-identity-reshape.md)分开。

## 失败与所有者

`CompleteOwnedInvocation` 只在 return 切点插入已有资格的最终完成；closed-prefix emitter
只改写语句切点。两者原先继承普通 `IRMutator` 的容器 hook；覆盖 `VisitExpr` 无法阻止
共享表达式 DAG 的展开，因为 `IfStmt` 等 typed hook 直接分派条件表达式。
整个 Function 遍历还会重访与此变换无关的签名及元数据。

公有 owned-return 回归在深度0、4通过，在共享合取深度28触发原 **30 s watchdog**；
之前的内部隔离触发12 s限制。历史失败日志保留在
`build-py314/review-repairs-20261116/`。没有通过延长 watchdog 宣称修复。

## 共享实现与正确性

`sync_statement_rewrite.h` 的 `SyncStatementMutator` 仅供已资格化的同步所有者使用，
明确处理支持的语句容器并按 identity 保留表达式。If 只改写分支 body，条件与结果 binder 不变。
loop/scope 仅在 body 改变时浅拷贝具体子类，保留原 bounds、binder、attrs、control 与 scope
证据。Assign/Eval/Yield/Return 叶保持原样，除非派生所有者明确替换该切点。
未知容器不递归解释或接纳，输入资格仍由原 preflight 和各所有者的拒绝边界负责。

closed-prefix emitter 和最终完成包装器共用此 helper，浅拷贝 Function，仅重写 body，
保留签名、形参 identity 与类型。`RewriteSyncSequence` 沿用原语句 occurrence 遍历／flatten。
不缓存语句 occurrence：两个实际 occurrence 仍须分别处理；不访问或复制未变的标量／类型 DAG。

正确性来自边界：载荷语句、guard、frame 表达式和元数据的操作数及控制不变；派生 hook 在相同
切点插入相同已分配命令或完成。没有新的同步、分配、硬件豁免、native 操作或 caller 前提。
`sync.record`、immutable token、物理匹配及 event-ID 复用含义不变。

## 验证与剩余限制

`test_invocation_completion_rewrite.py` 在共享 guard 深度0/4/28运行公有 flag-bearing
`LowerSync`，检查原形参与 guard identity、实际消费通知及 return 前的 terminal ALL。
零行选择 false guard 跳过载荷：这检验重写与完成，不证明 Acc 初始化、数值 matmul，
也不执行展开布尔树的所有叶。

有界目标是这两个所有者的语句 occurrence 与新增语句工作；不宣称任意 pass 的全局复杂度。
通用 raw-DAG BufferIR/SSA 验证器和表达式解释器仍是独立成本边界。实际命令的匹配、无环、
消费先于再发布及载荷覆盖仍由独立解释器检查。

新的 point-planner 检查器复用既有索引测试 DAG，并非编译算法改动。其顺序参照是 staged
MTE2/S/MTE3、标量 SSA／控制 readiness、独立解码的字节冲突、私有32字节UB enclosure 与明确
invocation 完成义务，不是原 scalar-cache 实现等价、最小需求定理或 native／设备资格。

新构建／测试结果及最终源码／工具 pin 在门完成后追加。之前八例通过的 point-planner 回执保留
为历史证据；新增顺序诊断需要新运行。此检查点不完成 M1–M6，也不声称全模型由 PyPTO 接管。

## 最终格式化前的已接受观察

修正构建通过；原30 s watchdog内，深度0/4/28 **三项在0.07 s通过**，日志
`shared-body-rewrite-gate.log`。独立审查后的point-planner扩展通过七个正caller例与单独标注的零stress，
finalize9包含第二次调用。`qwen-point-plan-order-audit/receipt.json`中，单次调用参照与实际均有
**296065**个完成至开始顺序对，重复两次均为 **1185030**。八例相对此staged模型均为
**零缺失／零额外**。计数恰为`770*769/2`与`1540*1539/2`：共享staging carrier复用使此模型中
全部staged载荷对串行，因此不能从顺序相等声称重叠改善。这些是选定staged模型的主机诊断，不是最小需求或原scalar／native结论。
之后检查器为lint提取了helper但未改检查；最终格式化源码pin与新运行尚待完成。

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
