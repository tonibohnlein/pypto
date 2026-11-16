# 验证器表达式DAG遍历：有界设计门

状态：独立最终源码审查已接受，构建、有界复现和聚焦回归门通过；最终hooks尚待完成。继续[原M1–M6计划](41-general-autosync-architecture-plan.md)
的R4/M6；与[检查点49](49-autosync-statement-rewrite.md)仅语句重写分开。
49保留未变DAG，本门处理其结构验证成本。

## 复现与语义边界

原raw-guard隔离输出`constructed`、`enter_BufferIR_verify`后触发原10 s watchdog，日志
`build-py314/review-repairs-20261116/acc-guard-buffer-verify-isolation.log`。
深度28共享标量guard仅有线性个不同节点，此处不冒充新性能结果。

BufferIR自身递归表达式；它还组合SSA、严格词法use-after-def与赋值类型对称检查；
选择后的完整pipeline额外执行TypeCheck和ManualDeps。NoNestedCall是独立公有property，
并非BufferIR直接子检查。全局Expr指针seen集不安全：同一表达式可在定义前／后、兄弟分支、
普通操作数／属性元数据环境有不同合法性。不能跳过语句occurrence、定义、native调用、window及其放置检查。

## 构造与精确生命周期

新增仅供验证器使用的内部每次表达式use遍历工具，不改普通IRVisitor／IRMutator或生产dispatch。
RAII进入记录递归深度，深度0开启新cache，最后退出销毁该use的cache。key包括Expr identity及
客户端的语义检查context。相同use/context内共享子节点只访问一次；新顶层use重新检查。

表达式遍历不引入语句定义或控制scope，所以一次use/context内binding、window表及元数据状态稳定。
不得跨语句、分支、loop header、类型use复用缓存，不缓存语句。接收与错误category/span不变，
重复DAG边造成的相同诊断可合并。

限定客户端：

- BufferIR的VisitExpr包含普通／实际attribute/type/window context、直接Assign/Eval角色及当前call身份；
  每个原语句仍更新allocation/window/constant状态。
- SSA的VisitExpr只在单use缓存，不跨DefineVar、scope进出、赋值或loop结果；重复定义与cardinality照查。
- UseAfterDef的VisitExpr包含普通／visiting_type模式；严格及legacy scope转换不改。
- TypeCheck保留每函数callee/value来源和原valid/type检查，仅缓存一次use的表达式分解。
- ManualDeps在每个新语句/use重新检查Call/Submit，依赖字段合同不变。
- 赋值计数／类型对称visitor只需语句定义和类型，不遍历无关表达式；每个Assign occurrence仍检查。
- 独立NoNestedCall客户端保留父边禁令和SIMT while／声明式predicate豁免，模式不可漏入正常body。

CollectTypeVars是单请求的Var identity集合，可在请求内缓存不同表达式／类型节点，保持原metadata字段合同。
词法类型use的类型去重也只能在该use内；不改变一般VarDefUseCollector的语句occurrence列表或无关codegen。
本门不声称修复任意共享Tuple类型验证或通用pretty-printer复杂度。

## 有界错误诊断与验收

SSA原RecordError会自动PythonPrint整个函数，错误共享DAG仍可爆炸。现保留函数名、错误类别、原span
与简洁消息，已取消该验证器自动完整函数打印；显式打印不变，不新增formatter、不隐藏拒绝或延长watchdog。

用公有BufferIR及独立SSA/use-after-def检查深度0/4/28的合法与非法输入，保留原10 s限制，root串行运行。
覆盖共享表达式定义前后两种顺序；分支局部变量在本分支合法、兄弟或join后非法；嵌套loop binder/reset；
同表达式的type/attribute use和隐藏Buffer引用；同Call Expr用于不同语句、变化的window来源、重复赋值和效果；
先经豁免predicate／SIMT条件后仍须拒绝的nested call，以及词法window size、call切点、类型不匹配和malformed输入。

小树等价例比较接收结果与诊断类别/span；大合法DAG不展开，大非法DAG不自动打印展开函数。
记录每use/context成本，不宣称任意use数下全局线性。不新增caller、alias、资源或硬件前提。
源码审查、fresh门与source/tool pin通过后才能接受提交，M1–M6仍独立跟踪。

## 独立设计门

架构及正确性审查接受本设计进入实现。设计接受不表示实现、可执行所有权或验证完成。

## 已实现，尚待资格门

验证器专用expression_use_cache.h提供RAII单use/context缓存；BufferIR含attribute与direct-call结果角色，
use-after-def区分普通／type模式。赋值及局部value collector保留语句occurrence、不遍历标量DAG。
VisitTypeExprFieldsOnce仅显式用于一次词法type use；普通枚举器不改。CollectTypeVars按其返回集合语义
使用请求内distinct-node collector。SSA保留简洁诊断，不再自动展开整个函数。
新test_verifier_expression_dag.py覆盖公有深度28、完整selected pipeline、scope/reset、元数据、属性、
重复native语句／定义与nested-call模式。作者未运行测试，协调agent串行完成下列主机门；没有设备执行。

## 最终hooks前的fresh资格门

一致native构建通过；新模块在原10 s限制内 **26项在0.06 s通过**（verifier-dag-first.log）。
包含这26例的九模块门 **588项在1.45 s通过**（verifier-existing-gate.log）。完全相同的
metric_fixture28／公有BufferIR probe从原10 s超时exit124变为 **零诊断、0.0002418477 s**
（generic-verifier-post-repair.log）。这只是表示／证明接收检查，不证明Acc初始化或数值执行。

更广continued门为 **5014通过／1失败**、76警告、597.55 s（continued-combined-gate.log）。
唯一失败暴露原有部分token擦除边界，并非fresh五千项全绿。修复后受影响scoped-family模块
**97项在5.30 s通过**；这是独立运行，不合并伪称新的完整全绿门。

## 部分擦除拒绝与fixture修正

旧unsupported-wait-cut fixture保留Assign叶却让generic mutator重映射capture／loop绑定，
实际可能先因SSA失败；旧完整函数诊断偶然满足regex。现在仅改语句body，保留原条件、bounds、
carry／result身份和叶；原始及编辑后token程序均先通过已有SSA／BufferIR／logical验证，
然后检查未放宽的capability拒绝。

更强fixture发现structured_token_sync.cpp的Erase：普通If内部擦除token后，若候选仍非空，
返回原分支会复活引用已删外层carry的WAIT。现用单调结构擦除generation记录token Assign、WAIT、
pure capture、For token binder/result pair与Yield token值。非空候选若generation变化，
在载荷import前明确拒绝不支持的guarded logical-token cut。空token分支仍擦除，未改native分支保持原状。
两个仅completed token carry的合法Scalar／payload guarded loop覆盖binder／Yield路径。
这是保守拒绝边界修复，不是新planner、协议或同步语言；拒绝时原IR不变。

## 原模型检查点与pin

新qwen-verifier-layout/receipt.json仍为 **21接管／11委托局部叶**、47任务控制域、全部32个C++，
与上次捕获逐字节相同。9.59558 s捕获时间仅描述，不是编译提速测量；没有新增kernel覆盖或重叠结论。
回执pin当前源码子集（含擦除／fixture修复）、扩展与Python，parent为
41a0c33870f4f955b4a2e87c26e38c711785509f。二进制同时包含51第一层共享layout基础代码，
不代表51 transport context/capsule资格。最终hooks后可刷新pin。
未改生产dispatch、需求构造、分配算法、硬件豁免或专家peer协议。

复杂度结论限于每expression-use/context及语句occurrence。结构类型比较、shape验证、其他dim/type
诊断打印与任意shared-Tuple路径没有被全局修复。原M1–M6及native／visibility／transport义务仍保留。

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
