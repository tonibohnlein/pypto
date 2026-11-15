# 通用自动同步集成：架构与 Qwen 验收计划

状态：持续实施的操作、所有权和子区域门禁已审查接受，但原计划六个里程碑及
全模型所有权仍**部分完成**。缺失的原生或值域前提不得由单次捕获推断。
最新未修改双层 Qwen 两个策略均生成 32/32 个 C++ 叶：exact 接管 3 个，
conservative_prefix 接管 16 个、委托 16 个；另有 47 个既有任务控制所有者单独记录。
运行时包 256 项测试通过。52 模块广泛检查记录 2669 通过及 6 个过时夹具期望，
经过审查修正后相关模块 196 项通过。这是组合证据，不是单次 2675 项全绿结果。
参见后续门禁表及[只读子区域检查点](41-general-autosync-readonly-child.md)。
基线为 `b5597650a75b598645b1b1d3fe05bd3aab220f09`，扩展
[区域同步架构](39-regional-sync-architecture.md)。Qwen 端到端仍是验收目标，
不是内核专用规则来源。自动同步默认关闭，生产者策略默认 `exact`；显式选择
`conservative_prefix` 本身既不启用自动同步，也不建立所有权。英文版本为权威说明。

## 证据与范围

当前实现已有不可变分析结果（immutable analysis）、统一原生实现绑定、
受检查的端点解码、紧凑循环前沿、索引化存储划分和已接管调用的终止完成操作。
这些工作不应重新列为待实现事项。

历史全 Qwen 捕获使用未修改的 Qwen3-14B prefill，batch=1、sequence=128、
hidden/intermediate=5120/17408，以及一层和两层配置。库版本为
`b72c66d9df69ba5f11982b212530a255c050988d`。旧两层记录中 32 个叶函数有
2 个由 PyPTO 接管；24 个首先未通过 Buffer 准入，5 个缺少调用者坐标证据，
1 个缺少参数化活动分区支持。这不是当前 HEAD 的新结果。
首个门禁已使用现有 `qualify_qwen_whole_model.py` 更新记录；新的策略及严格指定域
所有权捕获单独记录如下，不重做广泛调查。

捕获包含标量规划、数据加载地址、scratch/行操作、原子输出和成对 AIC/AIV 通信。
仅有精确 GEMM 分析或局部 ALL 屏障不能证明整个模型的同步正确性。
首个拒绝原因也不能证明后续阶段可行。原生访问、可见性及设备行为需分别验证。
初始验收配置不意味着所有 prefill/decode、形状及目标均受支持。

## 复用现有 PyPTO 服务

| 职责 | 现有所有者 | 同步集成要求 |
| ---- | ---------- | ------------ |
| 结构化控制流 | `stmt.h`、访问器、If/For/While | 直接遍历；仅在算法需要时派生分析图 |
| 效果与结果所有权 | `op_registry.h` 的 BufferArgEffect、OpArgEffectSpec、ExecutionMemoryAccessEvidence、BufferResultBehavior | 区分数据/元数据、Allocate/Alias/Borrow，不建立第二套注册表 |
| 物理存储与视图 | root/view/valid/indexed analyses、存储放置 | 保留重叠、有效范围、槽和来源；不同 SSA 根不证明不相交 |
| 原生实现 | Buffer recipes、buffer_sync_contract | 与实际生成代码一致，覆盖 scratch、多流水线及资源占用 |
| 算术 | `arith::Analyzer` | 复用数值/符号/模约束和作用域，不新增通用求解器 |
| 调用与形状 | call_shape_binding、CollectSyncInvocationEvidence | 保留事实关系与作用域，区分 Call/Submit |
| 通信 | ExpandMixedKernel、cross_core_pipe、tpush/tpop | 保留通信所有权，补齐局部发布/获取/释放条件 |
| 同步实现 | token_sync、scoped_token_families、AssignedSyncPlan | 降低实际 token 程序，而非重新推导所需内存顺序 |

`StmtDependencyGraph` 是具有 InOut-use 前提的 SSA 依赖图，复合节点聚合子树。
它不能替代放置后的物理存储冲突分析。不引入新的公共 CFG、Buffer 方言或 Qwen 专用 IR。

## 目标架构及契约

```text
Existing structured IR + placed storage + selected backend implementation
  -> qualified effects / control / caller context
  -> exact regional providers OR explicitly selected conservative construction
  -> logical token program expressing the selected order
  -> OptimizeSync
  -> LowerSync: actual token and protocol order
  -> existing resource interfaces, assignment and emission
  -> qualified invocation exit and domain ownership
```

从 `ImportPlacedSync` 的数值准入之前抽取可复用语义恢复，并迁移原客户端。
`SyncStorage` 的数值矩形和 `SyncSite` 的单流水线模式仍是精确分析的规范化视图，
不是通用操作模型。共享输入引用原始 IR/存储，并按需携带经过证明的包围范围。
复合通信需使用协议适配器，不能伪装成单流水线载荷。

输入契约分别记录效果精度、物理身份/别名、原始控制与合法插入点、标量就绪性、
调用者参数证据，以及通信边界。分析结果独立提供 demands、queries、selectors、
endpoints。子区域另外提供逐流水线可能访问、封闭匹配、未完成载荷和资源所有权。

精确访问、可靠包围范围、未知操作语义、显式无访问互不等价。
`Write` 不表示覆盖整个类；可能写不能无证据地 KILL 旧义务。
元数据访问不自动变成载荷访问或已就绪标量。未知地址可以扩大到已知分配、候选集合、
存储空间或所有类，但扩大范围不能修复未知效果、未就绪控制或越界/溢出程序。
选择器必须区分已证明不存在与不可用；GM 导出失败不能破坏有效局部导出。

保留 finite、rotating、numerical 精度提供者，不强制所有请求构建完整选择器。
精确端点导出失败仍保留数学结果。覆盖策略可显式另行请求保守构造，并记录额外顺序。

对于保持实际 token 顺序的 LowerSync，确认协议的强化必须先进入逻辑程序：

```python
# Schematic token IR at a qualified consumer cut; not Python frontend syntax.
forward = sync.record(p)
sync.wait(forward, q)
returned = sync.record(q)
sync.wait(returned, p)
```

反向记录包含 q 当前前缀和刚导入的等待，可能强化载荷顺序。这是保守策略的显式选择，
不是分配器可暗中添加的边。匹配、进展及消费先于重发布仍需命令级证明。
载荷顺序相等不能单独授权删除无载荷的确认握手；未知循环次数需要结构性证明。

通用 scoped parser 保留原有载荷切点契约。新增的受限封闭协议路径支持上述四条命令
和同流水线两条命令，包括空前缀及 wait→record 转发；不伪造载荷点，也不扩展为
任意命令程序的通用解析。支持语法与证明记录在下方门禁清单。
公共六个 token 操作没有通用 ALL 操作。区域共同完成需先审查等价 token 协议/原生替换，
或另行提出 IR/RFC 修改。终止 drain 不是通用循环内完成能力，本计划不授权公共 IR 扩展。

## 执行域与所有权

纯标量程序不能为了绕过 Buffer 的 load/store 选择启发式而伪造 Buffer 操作。
其内存效果、可见性、原生顺序和退出条件仍需证明。无同步计划必须经过证明，
不能将未发现效果视为无效果。空计划/仅屏障计划需要明确的 PyPTO/PTOAS 所有权协议，
不能插入虚假 SET/WAIT 标记。

混合 AIC/AIV 程序保留既有通信协议。局部 token 不是 TaskId，也不是跨核完成 token。
通信契约应给出参与者、借用存储生命周期、可见性、局部合法插入点和资源占用。
证明涉及双方时必须成对资格化，部分接管不得破坏对端协议。
原子输出需要效果、范围和原生完成契约；局部同步不证明跨 worker 原子语义或无数据竞争。
既有合格 ID 策略、标量规则及已资格化的累加器豁免保持不变。

## 实施里程碑及依赖

每个门禁由架构/正确性审查与独立有界检查验收后，再开展依赖工作。
这是完整架构的增量交付，不以最小补丁为目标；扩展服务时迁移客户端并删除重复路径。

| 里程碑 | 实施内容 | 验收证据 |
| ------ | -------- | -------- |
| M1 共享资格化与执行域 | SelectBufferIRFunctions 从 Boolean 准入扩展为带操作/性质/阶段/位置的拒绝；连接现有效果、存储、后端与就绪证据；刷新 Qwen 记录 | 原有精确输出不变、全部域有归属、unknown/no-access 与元数据/载荷分离、重叠和 Call/Submit 测试 |
| M2 操作与存储契约 | 使用共享 recipe 完成 selection→lowering→effects→geometry/enclosure→native→emission；按 row/scratch、partial-valid/view、scalar/indexed、atomic 家族推进 | 非 Qwen 行归约、重叠视图、动态有效形状、控制标量、原子效果和负例，与对应 Qwen 叶函数一起检查 |
| M3 结构化保守分析与协议 | 原始控制上的 may-dataflow；序列组合、分支并集、循环有限单调不动点；初始无 KILL；真实前缀命令点和转发 | 跳过写、零/多次循环、重置、扇出、空前缀、命令死锁、最终 false 条件、标量控制及反向资源不足 |
| M4 子区域与既有通信边界 | 保留紧凑子程序、逐流水线效果、封闭匹配、未完成载荷、联合 palette 及本地重用证明；保留既有手工／原生／编译器跨核协议，不生成对端握手 | 不规则父区域、共享存储、跳过/重置、资源冲突；仅在契约已建立时检查既有通信的本地就绪、借用生命周期和资源共存；比较完整新增/删除前提关系 |
| M5 参数关系与精确组合 | 使用 CallShapeBindingPlan、arith::Analyzer、原始坐标服务保留已证明关系；对声明的符号形式扩展活动分区 | 未修改 projection/gate-up、缺失/非法关系、多调用者/Submit、窄化、零/短循环、covers/queries/selectors 与规模 |
| M6 全模型接管与验证 | 明确无 flag 所有权；保留 task/transport 所有者；严格检查全部要求由 PyPTO 负责的局部域；相同退出策略比较 | 配置、生成域、实际执行三个分母；任务边界、重复调用、最终 store；host/逻辑/物理/device 分开报告 |

M1 首先完成；M2 按操作家族供给 M3；M3 支持 M4；M5 在 M1 后是独立精度/覆盖轨道，
不是所有保守分析的前提；M6 汇合所有适用契约。普通委托仍是覆盖缺口，不是外部协议所有权。
M2 各家族独立验收，首个局部条件缓冲切片无需等待原子或跨核资格化全部完成。
M1/M2 的验收必须区分不可用同步范围和非法地址/有效范围/溢出；包围范围不能证明
程序内存安全，外部提供的安全前提必须明确记录。

M3 必须证明未知次数循环中的逻辑匹配身份、分支兼容的命令访问、确认协议的重置/重用及进展。
may-access 不动点不证明这些协议性质。不能直接把载荷间直接握手的分配前提套到纯命令协议；
新增充分证书必须通过既有资源接口单独论证，否则保留不可用结果。

例如 `offset = layer * 5120` 需要与分配行数的已证明关系及机器算术安全。
benchmark helper 设置 `rows = layers * 5120` 不是自动成立的调用契约。
`6*band` 与 `17*band` 必须保留同一 SSA 参数的相关性，不能退化为无关区间，
也不能通过枚举参数值或展开整个循环巢通过测试。

建议首个纵向切片为条件缓冲变换：通用测试含动态次数/有效范围、跳过 store、局部重用、
零访问及重置。原始 Qwen 候选为 `mlp_out_cast`：

```text
for hb = block; hb < 40; hb += 20:
    load a [128,128] FP32 tile
    if hb == 0: store a [1,8] slice to the completion tensor
    cast the full tile to BF16
    expose a [valid_tt,128] view
    store that view at [m0,hb*128]
```

先诊断精确的视图/有效性拒绝及标量契约；该见证尚未证明可准入。
不能特化 valid_tt 或次数到捕获值，通用家族才定义接口。
该切片验证效果→控制→逻辑协议→物理重用，不要求先解决原子/跨核语义；
但它们仍是整个 Qwen 的后续必要条件。

## 验证、复杂度与审查

独立从输入效果重建所需顺序，不依赖编译器 generator 自证。
验证访问/包围覆盖、声明精确时的实际 covers、四类事件查询、选择器不存在/不可用、
逻辑顺序、物理匹配/进展/重用及载荷控制保持。保留 compact K、replay、零 inner body、
标量/Cube 豁免和终止 drain 回归。有界执行不是未知参数的统一证明。

设 J 为 class/mode/pipe 事实宇宙。增量传播应计入 O((V+E)J) 传播工作及 O(VJ) 状态，
另外统计输入恢复、service 查找和协议工作。这不自动满足 O(N log N)。
广泛准入前选择可靠粗化策略或明确复杂度例外；不能截断工作后发布不完整结果。
精确几何/接口输出大小单独计费，循环次数不能被展开进证明或命令电路。

独立变化静态操作、类交集、控制深度、符号维数和循环次数；统计失败尝试、转换、
保留接口、各构造阶段、静态/执行命令、guard 工作和多余载荷顺序。
service pruning 必须联合验证，不是首个正确路径的前提。

以前 PTOAS 的 PrefixInput/PrefixServices/PrefixRecipe/child qualification 可提供
算法论证和负例；不能原样移植 MLIR 适配器、原生假设或暗中强化物理顺序。
每次审查记录复用所有者、删除重复代码、新证明、独立测试、规模和事务回滚。
效果、几何、就绪性、数学/导出、资源证据、所有权失败分别报告。
全模型承诺以这些契约为条件，有限成功样例不证明任意内核。

两位审查者在明确未知次数协议的身份/进展/重用证明，以及包围范围与地址有效性的区别后接受设计。
初始设计审查不认证尚未实现的算法、原生契约或 Qwen 准入。英文同目录的架构 JSON
固定该次源码审查与历史证据；下方另行记录后续实现。

## 实施门禁记录

[实施收据](../../../en/dev/ir/41-general-autosync-implementation.json) 保留历史 M1／静态
cast 证据，并补充源码和工具哈希、完整日志引用、32 个叶函数及另外 47 个执行域名称。
已接受的实施切片不等于六个原始里程碑全部完成；没有设备运行或全函数所有权声明。

| 里程碑 | 已交付范围与证据 | 剩余契约及负责所有者 |
| ------ | ---------------- | -------------------- |
| M1 | `71b83d34b`：共享静态操作资格化、首个类型化义务；历史 984 项门禁及 32 份 C++ 字节一致。 | 操作／就绪性生产者仍需标量、索引值可见性及不支持的执行语义；仅有效果还不够。 |
| M2 | `fb0727c53`、`76221dc46`、`eff99b7de`、`ed7590db9`、`c4b9c237e`：静态 cast、连续视图、scratch/MUL、按使用点槽位证明及默认 Rsqrt。`a111cd34a` 加入合格原子家族。 | 原生实现／几何所有者仍需其他布局、动态 cast、高精度 Rsqrt 及更广原子模式。局部完成不提供 GM 初始化、跨 worker 可见性或确定性归约。 |
| M3 | `ce4d593b5`：显式保守前缀生产策略、无 KILL 效果、实际闭合命令协议及物理降低。 | 接纳仍依赖已资格化的效果、就绪性和本地执行，不能通用恢复标量／传输效果。 |
| M4 | `ce4d593b5` 消费者加新测试的公开批量生产者 `prepare_native_sync_children`（`362716444`）。全公开生产者→Insert→Optimize→Lower→PTOAS 和独立命令 oracle 纳入 444 项门禁。 | 子区域／资源所有者仍需有事件匹配、联合调色板、重置及传输契约；原生、空资源切片不完成整个原始里程碑。 |
| M5 | `fad11f7f4`：检查作用域内数值调用者关联，68 项测试及新 Qwen C++ 一致。 | 调用／关系提供者仍需符号激活、形状关系、携带状态及外部 ABI 证明；不注入基准范围。 |
| M6 | 严格命名执行域工具、32 叶函数清单、十行紧凑性测量和六个有界命令／字节检查；exact 2 个及 prefix 11 个要求名称均通过。 | 2,437 项广泛门禁通过；21 个委托叶、47 个其他未验证执行域及空／仅屏障计划的下游所有权协议仍开放。 |

原子切片为默认 A2/A3 FP32 Acc→packed ND FP32 GM atomic add。共享调用级效果注册表
将目标建模为读写，保留已有搬运几何和原生配置／复位实现，不声明隐含 flags/scratch。
[固定原生审计](../../../en/dev/ir/41-general-autosync-atomic-audit.json)
明确将其与初始化、跨 worker 语义分开。不支持的变体不能伪装为普通写或空效果。

公开 M4 工厂使用当前上下文目标及显式 disjoint-GM 前提，对**整个原始、已放置函数**
只执行一次精确分析，再导出唯一且不重叠的原始循环作用域。失败不返回部分列表；
不分离子区域重新分析或自动搜索。不可变句柄保留数学所有者。导出仍拒绝子区域外部
存储效果、未提供的调用／启动前提及事件资源，保守父区域不继承根分析的 disjoint-GM
前提。参见[公开 API](../passes/60-logical_sync_pipeline.md#闭合的当前前缀事务)。
M5 在简化前检查原始机器宽度及中间值，定义绑定仅在原定义点有效，分支退出恢复事实；
后续条件不能使先前溢出合法，未知调用者仍污染数值范围。这不是符号被调用方关系 ABI。

## 当前资格化与复现

最新收据位于 `build-py314/general-autosync-20261115/m6-atomic/{exact,conservative_prefix}`。
身份为编译器基线 `c4b9c237ed1606fa29484504c93955c5c3002fec` **加已记录脏源码和二进制哈希**，
不是假定的干净提交。库／模型固定为 `b72c66d9df69ba5f11982b212530a255c050988d`；
未修改 Qwen3-14B prefill，双层、batch=1、sequence=128、维度 5120/17408。
机器收据保留 Python、匹配扩展、PTOAS 0.72 启动器／原生工具哈希、完整参数向量和环境。
完整模型输入和源码清单保留在已固定的原始收据中，不重复复制三份函数清单。

| 策略 | C++ 叶 | PyPTO 接管 | 委托 | 严格要求名称 | 包含时间（秒） | C++ 字节 |
| ---- | -----: | ---------: | ---: | -----------: | -------------: | -------: |
| exact | 32/32 | 2/32 | 30/32 | 2/2 | 6.5621 | 1,399,234 |
| conservative_prefix | 32/32 | 11/32 | 21/32 | 11/11 | 7.7847 | 1,440,504 |

上表仅指**最新双层捕获**；历史单／双层观察保留各自分母。C++ 大小比较同一组
32 个生成文件，不是运行成本。另 47 个执行域单独标为未验证，不是已证明的外部
所有者。单次包含时间不证明加速。字节为实际文件长度，已核对哈希；没有设备执行。
早期 pre-Rsqrt 命令未要求任何严格名称，只能作为历史清单。后续 Rsqrt 捕获的 2/10
命名门禁通过，但所有权没有增加：`final_rmsnorm` 首个拒绝从 Rsqrt 推进为
`tile.col_expand_mul`。原子资格化随后使 prefix 接管 `down_proj`；exact 仍拒绝未证明
的调用者坐标范围。

复现使用既有 `tests/ut/ir/transforms/qualify_qwen_whole_model.py`，传入 `--layers 2`
和 `--policy exact` 或 `--policy conservative_prefix`，以及收据里的库、工具路径和
**全部** `--require-function` 参数。exact 要求 `mlp_out_seed`、`silu`；prefix 要求
下表全部十一行 `pypto` 名称。省略名称会把严格资格化降为清单观察。

| Function | Exact owner | Prefix owner | Prefix first obligation | C++ bytes exact / prefix | Prefix SET / WAIT / barrier |
| -------- | ----------- | ------------ | ----------------------- | -----------------------: | --------------------------: |
| `attention_finalize_phase` | ptoas | ptoas | type_layout/Var | 15709 / 15709 | 0 / 0 / 0 |
| `down_proj` | ptoas | pypto | — | 23962 / 31408 | 104 / 104 / 27 |
| `final_rmsnorm` | ptoas | ptoas | operation_recipe/tile.col_expand_mul | 19185 / 19185 | 0 / 0 / 0 |
| `gate_up_proj` | ptoas | pypto | — | 30559 / 41172 | 144 / 144 / 37 |
| `kv_proj` | ptoas | pypto | — | 31191 / 42357 | 144 / 144 / 37 |
| `lm_head` | ptoas | ptoas | type_layout/Var | 22780 / 22780 | 0 / 0 / 0 |
| `mlp_out_cast` | ptoas | pypto | — | 12696 / 13466 | 16 / 16 / 5 |
| `mlp_out_cast_0` | ptoas | pypto | — | 12727 / 13497 | 16 / 16 / 5 |
| `mlp_out_seed` | pypto | pypto | — | 7371 / 7714 | 4 / 4 / 3 |
| `out_proj_aic` | ptoas | pypto | — | 19113 / 24565 | 72 / 72 / 19 |
| `out_proj_aiv` | ptoas | ptoas | operation_recipe/tile.cast | 13110 / 13110 | 0 / 0 / 0 |
| `out_proj_aiv_0` | ptoas | ptoas | operation_recipe/tile.cast | 13142 / 13142 | 0 / 0 / 0 |
| `post_rmsnorm` | ptoas | ptoas | type_layout/Var | 22611 / 22611 | 0 / 0 / 0 |
| `q_proj` | ptoas | pypto | — | 19104 / 24556 | 72 / 72 / 19 |
| `qk_pv_partial_merge` | ptoas | ptoas | type_layout/Var | 42305 / 42305 | 0 / 0 / 0 |
| `qk_pv_partial_plan_aic` | ptoas | ptoas | type_layout/Var | 208297 / 208297 | 0 / 0 / 0 |
| `qk_pv_partial_plan_aiv` | ptoas | ptoas | type_layout/Var | 208459 / 208459 | 0 / 0 / 0 |
| `qk_pv_single_block_0_aic` | ptoas | ptoas | type_layout/Var | 85114 / 85114 | 0 / 0 / 0 |
| `qk_pv_single_block_0_aiv` | ptoas | ptoas | type_layout/Var | 85276 / 85276 | 0 / 0 / 0 |
| `qk_pv_single_block_aic` | ptoas | ptoas | type_layout/Var | 85106 / 85106 | 0 / 0 / 0 |
| `qk_pv_single_block_aiv` | ptoas | ptoas | type_layout/Var | 85268 / 85268 | 0 / 0 / 0 |
| `qkpv_plan` | ptoas | ptoas | type_layout/Var | 139663 / 139663 | 0 / 0 / 0 |
| `rms_recip` | ptoas | ptoas | type_layout/Var | 15858 / 15858 | 0 / 0 / 0 |
| `rms_recip_0` | ptoas | ptoas | type_layout/Var | 15892 / 15892 | 0 / 0 / 0 |
| `rope_kv_cache` | ptoas | ptoas | type_layout/Var | 81374 / 81374 | 0 / 0 / 0 |
| `save_prefill_last_token_group` | ptoas | pypto | — | 8785 / 8910 | 4 / 4 / 3 |
| `save_prefill_last_token_single` | ptoas | pypto | — | 8396 / 8521 | 4 / 4 / 3 |
| `silu` | pypto | pypto | — | 14556 / 13564 | 40 / 40 / 11 |
| `token_embed_group` | ptoas | ptoas | type_layout/Var | 12043 / 12043 | 0 / 0 / 0 |
| `token_embed_single` | ptoas | ptoas | type_layout/Var | 10556 / 10556 | 0 / 0 / 0 |
| `x_gamma` | ptoas | ptoas | operation_recipe/tile.cast | 14497 / 14497 | 0 / 0 / 0 |
| `x_gamma_0` | ptoas | ptoas | operation_recipe/tile.cast | 14529 / 14529 | 0 / 0 / 0 |

**命令列不是整计划优化比较。** exact 与 prefix 接管不同叶集合（2 对 11），
发布的 PyPTO 阶段元数据不含其余 30 对 21 个委托叶中的 PTOAS 插入。委托行的零
因此**不表示** C++ 无同步。8/8/8 与 620/620/169 的 SET/WAIT/屏障总数只描述这两个
不同子集，不能证明完整同步的改善或退化。这是静态站点，非执行次数；粗包络可能
明显增加顺序／控制成本。机器表另保留原始和降低后的 For/If/While 数量。

本次每个 PyPTO 接管叶经过：共享效果资格化→所选 exact／conservative 策略→实际
逻辑命令程序→合格有向 ID 池内的受证明物理降低→C++。当前目标目录在策略和
原生占用扣除前有八个硬件 ID，公开捕获未要求六 ID 上限；受限池测试是另行证据。
这是所接纳本地协议的编译器证书，不是最小 ID 或设备验证。委托叶另报首个 PyPTO
义务，再使用 PTOAS 自己的分配池，其 C++ **不属于** PyPTO 路径证明。局部原子完成及所有权
标记均不证明 GM 初始化、跨 worker 可见性或确定性归约。

另行检查实际接管输出 `after-sync.msgpack`，两个 exact 叶与十一个 prefix 叶中插入
的字面量事件 ID 仅出现 0 和 1。已哈希的 `m6-atomic/literal-id-inventory.json` 保留
每个方向；不包含原生宏内部及委托插入。这不是六 ID 接纳上限，也不证明最小 ID 数。

失败检查点完整保留：第一轮 Rsqrt 21 失败／1,093 通过；随后夹具轮次 11 失败／
1,103 通过。修复后最终 24 项 Rsqrt 门禁通过。原子／公开生产者聚焦门禁 444 项通过，
耗时 8.56 秒。历史槽位门禁 1,791 项通过、四个继承警告。**最终广泛门禁：2,437 项
通过、四个警告，496.55 秒。** 收据固定完整测试清单和日志哈希。最终聚焦 CTest
（VF IR 契约、数值同步顺序）2/2 通过，总耗时 0.35 秒。`final-gate/toolchain.json`
固定 Python 可执行文件／版本／哈希、C++／CMake 版本、构建缓存／Ninja 哈希、匹配扩展、
测试日志及显式双 worker 构建命令；两个文件均由实施收据固定哈希。失败日志不被覆盖。

## 有效性前提及构造成本

证明依赖已资格化原生语义，而非仅因存在注册操作。
[A2/A3 流水完成契约](https://asc.gitcode.com/api/SIMD-API/c_api/sync/intra_core_sync/asc_sync_pipe.html)
支持 V、M、MTE1/2/3、FIX、ALL，不支持 S；
[2201 AIC/AIV 同步表](https://asc.gitcode.com/api/SIMD-API/c_api/sync/overview.html)
给出同流水屏障及有向通知／等待组合。这里使用既有 DMA/FIX 完成原语，不增加硬件
原语，也不提供设备验证。

1. 原始操作按存储类／流水产生可能读写事实。序列和分支并集保留义务；循环输入为
   IN union GEN(body)，覆盖零次和回边且不展开次数。无 KILL，可能写不擦除旧冲突。
2. 跨流水服务显式给出 `record(p), wait(q), record(q), wait(p)`；同流水使用已资格化
   两命令完成。物理降低解释该顺序，不暗增确认。
3. 前向消费先于返回发布，返回消费先于下次前向发布；反方向服务使用不同角色。
   序列、分支、顺序循环结构归纳证明闭合通知状态和重置，包括跳过／零次情况。
   每个方向至多两个合格 ID 足够，但池拒绝不是硬件最小容量定理。ALL 完成载荷，
   不消费未匹配 flags。
4. 原生子区域保留原所有者／目标／作用域、无 flag 生命周期和未完成可能效果。
   Acc 原生保护只免除合格危险，不伪造完整 completion→next-start 边；有事件或
   对端传输所有权需要另行证明。

设 N 为原始语句出现次数，C 为有限存储类／流水／读写事实宇宙，E 为实际就绪查询
DAG 访问，K 为静态服务。使用有序集合的实现保守计费为 O(N C log C + N C² + E)
工作和 O(N C) 事实存储，C 随目标固定。E 仅在每次查询内去重；请求控制路径按输出
深度计费，共享一个父索引。实现保留 O(K) 静态命令（本地两条，跨流水四条），不展开
运行次数。它不包含精确子提供者、诊断、映射／命名、验证及后续编译，不是整编译器
线性界。

串行紧凑性诊断覆盖 primitive/native-child 两类各 count=0/1/3/31/1023，十行完成；
count≤3 的六个实际命令／物理字节检查通过，大次数只检查静态构造。primitive 固定
六个站点、42 事实、15 服务、17 语句／数据流访问；native-child 固定六个原始站点加
一个子区域、94 事实、21 服务、20 语句及 17 数据流访问；非零子导出访问五个源节点和
一个汇总节点。这些计数不包含整个根精确分析或峰值 RSS；脚本和输出哈希在收据中。
独立 oracle 明确验证屏障差 `outer_visits * (1 - inner_count)`，零次导入可能增加入口
屏障，同时仍验证载荷顺序包含、匹配及因果复用，不声明普遍更少命令或加速。

## 剩余契约及论文含义

| 剩余契约 | 缺少的证明及负责所有者 |
| -------- | ---------------------- |
| 更广原子／并发 | 原生实现与执行域所有者仍需变体效果、GM 初值、外部可见性／顺序和确定性要求；当前局部 atomic-add 不提供这些。 |
| 标量／索引就绪 | 就绪性所有者需定义到使用的可见性和原始切点；存储包络不能使异步地址／控制值就绪。 |
| 布局／动态 cast | 选择、降低和原生契约所有者需对描述符、有效范围、精度、scratch、资源达成一致；缺失配方不是空访问。 |
| 有事件子区域／传输 | 子区域和传输导出需匹配封闭、未完成效果、联合调色板、重置与 acquire/release；不能用原生空资源导入代替。 |
| 符号关系 | 调用和精确接口所有者需保留受检查的调用者关系及作用域；基准值不建立外部 ABI 前提。 |
| 空／仅屏障所有权 | PyPTO 与下游需明确所有义务已完成；没有 flags 不能伪造所有权标记。 |

数学内核保留其限定结论。实际编译还支付恢复、转换、导出、协议构造、分配及生成控制
的成本；紧凑依赖分析不能单独证明整编译界。保守前缀显式选择更强顺序，LowerSync
保持输入的实际令牌顺序。部分覆盖指出具体未提供契约，不证明算法或硬件不可处理。
六个原始里程碑和全模型所有权在该检查点仍未全部完成。

## 后续 M2 检查点：列广播（2026-11-15）

现有行归约/广播配方显式表示广播方向。`col_expand_mul` 将 row-major `[1,N]`
向量复制到 `[M,N]` 矩阵各行；原有行扩展仍使用 col-major `[M,1]` 向量。
选择、降低、验证、原生发射和同步效果共用该配方。限定变体为 full-valid、对齐的
FP32 Vec，要求 `M <= 255`、`N <= 2040`、`N % 8 == 0`，避免原生 uint8
重复次数及块步长截断。允许矩阵与输出精确别名；现有存储验证器禁止输出与广播向量
重叠。此变体无隐藏 scratch 或 event。部分有效范围及其他布局仍不可用。

原生审计基于 pto-isa `5487a6b0` A2/A3：`TColExpandBinOp.hpp` SHA256 为
`00bf43def423bd0701b28ea78784b64ce988f2a7f2e00275f7e9f281c00855ab`，
`TColExpandMul.hpp` 为
`76a3eb7be75560f5ef909ecfad9c61d9a8ac839b6cf80d8a309f74a7f5d386b8`。
通用前端测试涵盖多种形状和重复调用，独立检查访问、覆盖边、查询、选择器、物理匹配与
复用及 PTOAS C++ 生成。未修改的 RMSNorm 示例现在由精确方案接管；LayerNorm
仍明确拒绝其缺失配方。这是主机模型验证，不代表设备验证或整个 Qwen 验收。

下一类型/布局检查点需要完整传输构造，不能只放宽描述符。单列 GM `[M,1]` 尚无
限定的 column-major 传输，也无精确的部分有效填充行传输。运行时有效列向量需要
描述符、配方及足迹联合证据；一维索引加载还需要标量内存可见性和就绪证据。
在这些义务成立前保留拒绝。带事件子区域组合及配对 AIC/AIV 传输仍是后续检查点，
本次操作配方不证明它们。

刷新后的两层 Qwen prefix 捕获生成 32/32 个 C++ 叶函数，拥有 12/32，新增
`final_rmsnorm` 并保留原有全部 11 个必需所有者。聚焦测试 80 个通过，扩展回归
198 个通过。上文 11/32 表保留为配方增加前基线。命令和哈希见英文目录的
`41-general-autosync-column-broadcast.json`。

## 后续已接受门禁与剩余契约（2026-11-15）

上文历史表保留原始快照。当前复现、工具及源码哈希见 runtime-axis、
module-ownership 和 readonly-child 配套收据。模型仍为库 b72c66d9、双层、
batch=1、sequence=128、原始维度 5120/17408。

| 里程碑 | 已通过审查交付 | 首个剩余义务 |
| ------ | -------------- | ------------ |
| M1 | 类型化首个拒绝、共享效应、原始 owner/frame、显式完整模块所有权 | 尚未验证的协议域不能由叶函数统计代替证明 |
| M2 | 列广播与传输、实际运行时元数据、FP32 行归约/轴别名、BF16/FP32 转换 | 混合有效范围 ADD 值语义、标量 GM 可见性及更多原生变体 |
| M3 | 既有 no-KILL 转移、前缀确认、联合剪枝及实际 token 降低 | 一般共同完成/任意命令协议需独立语法及进展证明 |
| M4 | 本地及只读 GM 完整调用、原始坐标、待定效应、入口复用和方向调色板 | 跨核借用的就绪/释放/所有权；缺失精确 GM 边界仍不可用 |
| M5 | 既有调用界和紧凑 K/replay；原始形状 ABI 可作就绪不透明坐标 | 更多符号关系、范围与可执行精确 GM 导出需真实证据 |
| M6 | 显式空/仅屏障/含标志完整模块、末尾 drain、缓存和重建检查 | 仍委托 16 叶，无全模型设备及普遍所有权资格 |

紧邻前一检查点 prefix 由 14 增至 16，新增 x_gamma、x_gamma_0。
exact 接管 final_rmsnorm、mlp_out_seed、silu。旧 11/32 与 2/32 均为历史结果。
精确导出失败不会静默变成保守方案。

新子区域适配器经显式请求，而非自动选择。原始 Qwen K 调用通过准备、逻辑
组合、六 ID 分配和 C++。每个独立有界调用检查 416 原始访问、284 个实际
保留 covers、692224 个事件查询及 86 个边界样本；GM 选择器明确不可用。
两个完整父执行均删除 2188 对前提且不新增。通用夹具也有包装新增顺序，
不能推广为统一支配。通知匹配、载荷完成和物理复用分别报告。

仍需操作/原生契约：满有效 carrier 与部分 matrix 不得静默收窄；归约未写
非活动行，不能假设零填充修复混合 ADD；标量同 pipe 原生顺序不证明外部
数据可见性；跨核 push/pop/free 的本地完成、对端释放及资源 epoch 不能用
本地 ALL 或前缀替代。没有调用方证据就保留拒绝。

保留初始失败、联合穷举超时及研究检查器内存拒绝。研究验证可以展开小执行，
生产 K/replay 表示不按 trip count 展开。分配器、ID 池、硬件豁免、专家协议
及零次内层屏障均未更改。原计划全部里程碑不得据此宣称完成。

## 范围澄清与 M1 控制域所有者（2026-11-15）

用户明确排除新的跨核同步构造。M4 保留既有手工／原生／编译器通信协议，
只处理有契约的本地就绪、借用生命周期和资源共存。不得用本地 ALL 声称对端
完成，不得生成或替换 tpush/tpop/tfree 的对端握手；缺失边界契约仍委托。

既有编排代码生成器现通过只读 task_domains 发布实际处理的入口、Graph
片段、Spmd/Group 包装。未调用包装不自动获得所有者；Call/Submit 仍走原有
ABI、launch、TaskId 和依赖验证。只有整个编译单元成功后，自动同步报告才
把这些事实与每个 IR 函数关联；失败不发布已验证的任务控制域。

任务控制所有者不等于载荷可见性、完成或对端协议证书，也不取代本地
PyPTO/PTOAS 所有权。默认分派、PTO 所有权清单和同步协议不变。配套任务域
收据记录源哈希、全模型统计和输出源码相同检查。

M1 最新固定捕获覆盖 79 个观察域：47 个既有任务控制所有者、16 个 PyPTO 本地
叶及 16 个委托叶。状态 task_control_owner_observed 只表明原有生成器及正确
身份／类型／产物被观察到，不证明任意手工包装的全部语义。32 个内核 C++ 及
编排 C++ 与相邻基线逐字节相同。147 项广泛测试通过，含 41 个既有警告。
两个 Graph 模拟编译器仅补上已经提交的 sync_ownership 关键字参数。
完整固定命令与哈希见英文目录的配套任务域 JSON 收据。局部同步覆盖没有增加。
