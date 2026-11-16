# 验证器表达式DAG遍历：有界设计门

状态：设计提议，尚未实现或新增测试结果。继续[原M1–M6计划](41-general-autosync-architecture-plan.md)
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

SSA原RecordError会自动PythonPrint整个函数，错误共享DAG仍可爆炸。保留函数名、错误类别、原span
与简洁消息，取消该验证器自动完整函数打印；显式打印不变，不新增formatter、不隐藏拒绝或延长watchdog。

用公有BufferIR及独立SSA/use-after-def检查深度0/4/28的合法与非法输入，保留原10 s限制，root串行运行。
覆盖共享表达式定义前后两种顺序；分支局部变量在本分支合法、兄弟或join后非法；嵌套loop binder/reset；
同表达式的type/attribute use和隐藏Buffer引用；同Call Expr用于不同语句、变化的window来源、重复赋值和效果；
先经豁免predicate／SIMT条件后仍须拒绝的nested call，以及词法window size、call切点、类型不匹配和malformed输入。

小树等价例比较接收结果与诊断类别/span；大合法DAG不展开，大非法DAG不自动打印展开函数。
记录每use/context成本，不宣称任意use数下全局线性。不新增caller、alias、资源或硬件前提。
源码审查、fresh门与source/tool pin通过后才能接受提交，M1–M6仍独立跟踪。

## 独立设计门

架构及正确性审查接受本设计进入实现。设计接受不表示实现、可执行所有权或验证完成。
