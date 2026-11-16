# 自动同步构造修复

本计划扩展原有 M1–M6，基线为 bd8000fca。每一阶段经过独立评审、验证并提交后，才开始依赖它的工作。

1. 使用现有语句重写服务（statement-only rewriting），保留共享表达式、元数据和原始身份；验证直接调用上下文及工作区所有权。
2. 将标量 GM 的本机实现选择与同步策略分离。DMA 暂存必须显式选择且默认关闭；未证明缓存可见性和就绪性时保留委托。
3. 在保守控制流中保留存储身份、物理视图、精确范围和可靠包络。不同 SSA 根本身不能证明不相交。
4. 在构造时只生成尚未满足的同步义务。分支使用必然成立的完成事实；循环需要不变量。保护控制、调用边界及资源重用，验证整个协议。
5. 用相同载荷实现、别名及退出契约重新验证精确分析与原模型；分别报告代码扩张、额外顺序、命令及构造成本。

第一阶段已通过语句重写和验证器修复的独立评审。新回归验证深度28共享签名表达式及输入/输出 IR；130项测试通过。没有共同分配所有者的独立 peer 定义继续拒绝，不能通过合并工作区绕过契约。

qkpv_plan 修复前静态8450对、执行2818对标志命令来自源展开、标量 DMA 暂存和重复保守服务。早期八个有界诊断将执行数量降至1282并保留暂存模型顺序和资源安全；这不是现已实现的统一算法或最小命令证明，生产结果见第5阶段。首先确认原有标量实现，而不是先优化环形暂存缓冲。没有设备验证。

## 第二阶段：显式标量实现

新增 `PassContext(enable_scalar_gm_dma=False)`，默认保留原标量 GM 操作，不随同步策略变化。显式启用 DMA 暂存与自动同步开关、分析策略独立；暂存本身不授予同步所有权。上下文复制、PassManager 契约与 JIT 缓存键保留该选项。

现有原始 PTOAS qkpv_plan 输出使用标量指针访问，退出含 PIPE_ALL、全缓存 dcci 和 dsb。输出代码不等于本机可见性证明；未建立该契约前保留委托。默认所有权数量可能下降，隐式 DMA 扩张不再算默认覆盖提升。新增开关/策略矩阵、缓存区分和上下文变更负例，原暂存回归显式启用该实现。

## 第三阶段：存储精度与成本

保守分析使用自身所有的存储域目录。复用本机布局索引的局部字节包络，通过半开区间扫描合并可能重叠的域，包含隐式暂存访问；相邻不重叠区间可区分。未知域0与同空间所有域冲突。显式 GM 不相交契约仅区分已建立的原始根；共享 MemRef 基址仍属于同根，未知循环携带或借用来源保持保守。

子区域与原始访问共用目录；构造完域后重建子树摘要，保持无 KILL 和确认协议。源管线查询采用索引，诊断改为 `storage_domain_no_kill`，输出标记 `qualified_storage_enclosures`，不声称精确选择器。

L个包络的扫描为 O(L log L)，完整数据流仍依赖域/模式/管线事实数 J。下面的增量修复消除直线代码每切点重建索引的成本；控制分支与子树摘要仍可能保留 O(VJ) 工作。这不是整个编译器 O(N log N) 保证，也不是循环次数展开。不得用任意截断或不完整结果隐藏成本。

成本评审要求修复直线代码：生产分析移动并增量更新同一事实/索引状态，不在每切点复制整个历史。完整输入快照仅在显式诊断时保留；未请求的快照标为不可用，而非空。分支复制/合并及循环摘要仍依赖 J 并单独计数。新增静态独立域测试要求生产保留状态及增量工作线性增长。

## 第四阶段：只构造尚未满足的存储服务

生产构造器区分可能访问历史与必然成立的源载荷完成事实。这些是编译期事实，不是运行时计数或私有分支历史。每切点用初始完成事实构造覆盖矩阵，通过顺序更新覆盖计数选择包含意义下最小的服务集合，然后保持原顺序生成令牌；此前尚无令牌或物理命令。

确认协议按非对称顺序更新：前向 WAIT 将源完成导入目的管线；返回 WAIT 将目的完成导入源管线。发布确认本身不证明目的管线已有载荷完成。仅保留的服务能建立事实。分支、循环和子区域边界重置知识；子区域入口及资源重用服务保持受保护。任意用户令牌与转发协议不受此选择器改写。

独立的整候选 AST 检查在最终载荷切点验证全部义务、原序子集及受保护服务；物理降低仍须证明匹配、容量与因果重用，不事后修补标志。生产覆盖计数函数与现有 C++ 数值测试共用，含穷举小矩阵。k条管线使用 O(k²) 覆盖单元，映射查询另计有序查找因子；不声称全局最少协议。未知循环次数依靠块入口无已知事实的不变量，而非有限测试替代证明。

## 第五阶段：匹配生产输入的最终验证

本次五阶段修复完成；原 M1–M6 全模型所有权目标仍未完成。权威[机器收据](../../../en/dev/ir/52-autosync-construction-repair.json)记录源码、工具和输入哈希、命令与本地证据。基线 `7fd98e54a` 与候选 `89a481d46` 捕获时均干净；使用库 `b72c66d9` 的未修改双层 Qwen3-14B、batch1/sequence128、A2/A3、may-alias、PYPTO 布局及显式 conservative-prefix。每捕获或审计保持120秒 watchdog；无设备执行。

| 标量实现 | 修复前后 PyPTO 叶所有者 | 委托叶 | C++ 叶 | 包含性编译时间（秒） |
| -------- | ----------------------- | ------ | ------ | -------------------- |
| 默认原生 | 18 → 18 | 14 | 32/32 | 6.661 → 6.659 |
| 显式 DMA staging | 21 → 21 | 11 | 32/32 | 9.972 → 12.175 |

另47个既有任务控制所有者单独观察，不是新合成同步。每实现模式内同步前序列化 IR 字节一致；同步后原始载荷/控制清单一致。19个原生模式叶及16个 staging 模式叶的 C++ 字节一致，其余叶同步清单发生变化，载荷/控制清单保持不变。单次 staging 编译时间增加，不能声称编译加速。

共同11个委托叶仍为4个混合 valid 的 ADD、1个需部分列描述符的 RoPE MUL，以及6个缺少局部 Buffer 契约的 peer-transport 叶。默认原生另外委托 qkpv_plan 和两个 embedding 叶；不再隐式 staging。原生标量缓存可见性、就绪、退出限定及 admission/export 支持仍缺失。跨核及内核间所有者保持原状。

| 显式 staging 的 qkpv_plan | 修复前 | 修复后 |
| ------------------------- | ------ | ------ |
| 静态 SET 数（WAIT 同数） | 8450 | 4610 |
| 静态 MTE3 同管线屏障 | 768 | 0 |
| 静态 MTE2 屏障 / 终止 ALL | 383 / 1 | 383 / 1 |
| 每调用执行标志对 | 2818 | 1538 |
| 每调用执行载荷 | 770 | 770 |
| 完成到开始的载荷前提对 | 296065 | 296065 |

8个实际命令审计使用 finalized token1、8、9、16、17、127、128、0；9执行两次调用，0是正长度调用契约外的结构压力例。两个输出程序都等于同一个独立 staged-byte/native-pipe 参考关系，因此新增及删除载荷前提均为0，不只是对数相等。两次调用比较1540个载荷的1,185,030对。字节值、分配窗口、守卫、匹配、无环、通知闭合、终止完成及消费先于再次发布均通过；各方向使用 `{0,1}` 的子集，仍在6-ID池内。

参考从实际字节访问提取冲突，不依赖编译器域或生成器；索引图与小规模展开图交叉验证。共享 carrier 已使该模型全序。SB_BATCH1 输出0..127，仅覆盖不同分支/就绪路径，不覆盖非恒等置换。宿主审计峰值 RSS806→657MiB，限制1024MiB。不能推断原生 DCache 等价、全局最少协议或设备性能。

13个原生模式叶及16个 staging 叶的静态命令减少，包含投影、归一化和 attention-finalize；整个关系等价结论仅限上述 qkpv_plan 有限执行。通用夹具另比较完整前提关系、控制/字节、跳过写入、for/while、空与重复访问、计数器重置和物理重用。用户提供的任意转发令牌不受改写。

### 成本与紧凑性

第三阶段 N=8、32、128独立组的生产保留事实57、225、897，增量冲突摘要25、97、385，查询32、128、512；显式诊断快照191、3071、49151，生产不保留它们。控制连接/子树仍有 O(VJ) 依赖。

候选 qkpv_plan 共享服务测量有2306载荷站点、1个已知局部域加 GM 通配域、3465保留事实、无输入诊断快照；6528个基线服务请求保留3072、抑制3456。服务含同管线请求，不等于标志对数。带 Python 诊断导出的分析0.042秒，含分析/选择/证书的逻辑构造0.085秒，含原生检查及导出的物理限定/分配/发射0.265秒。独立调用重复分析，不能相加；没有独立分配计时。源生成仅包含在整次编译时间中。

最终检查保留精确 covers、边界选择器、四种事件查询、紧凑重复、resident/rotating 存储、重置、实际令牌顺序及物理重用。最终8个有界子组通过692个不同 Python 测试及3个 C++ 目标。组合回归曾触及120秒，日志保留；子组分别完成，未增加 watchdog。旧 small-GEMM 测试新运行通过，不代表普遍 Buffer admission。

复现时加载 `.claude/skills/testing/load-env.sh`，显式设置1个构建/测试/代码生成 worker 及1个 BLAS/OpenMP线程。使用收据固定的 Linux Python、扩展和 PTOAS0.72 启动器；不要加载陈旧 macOS路径。`PYTHONPATH` 包含本仓库 python、tests/ut/ir、tests/ut/codegen及根目录。持久结果在 `build-py314/production-repair`，精确 pytest子组和哈希见收据。

```bash
timeout 120s python -m transforms.qualify_qwen_whole_model \
  --library "$PYPTO_LIB" --output "$NEW_CAPTURE" --layers 2 \
  --policy conservative_prefix --require-domain-owners
# A separate capture adds --scalar-gm-dma; never overwrite old receipts.
timeout 120s python -m transforms.qualify_qwen_point_planner \
  --capture "$STAGED_CAPTURE" --library "$PYPTO_LIB" --output "$NEW_AUDIT"
```

| 已建立结果 | 剩余前提或限制 | 论文合适表述 |
| ---------- | -------------- | ------------ |
| DAG安全重写、源所有的包络域和显式别名契约 | 未知访问为通配域；非精确物理选择器 | 对选定效应模型的可靠保守覆盖 |
| 生成令牌前选择服务并验证整候选 | 控制/子区域重置；非全局最少命令 | 现有协议内可证明的保守服务削减 |
| staged qkpv_plan 命令减少、有限载荷顺序等价 | staging参考已全序；原生标量仍未限定 | 有限实现验证，不声称更强精确或原生定理 |
| 精确提供者及用户令牌降低保留 | 通用§8与全模型所有权仍不完整 | 共用接口，区分精确与保守保证 |
| 六ID与 C++生成通过 | 仅宿主；单次编译增加，无设备性能结论 | 限定协议的物理可行性 |

## 后续修复计划：区域构造与生产契约

计划基线为 `a19c1ead6`。上文五阶段保留为已接受的有界结果，没有完成原 M1–M6，也未消除所有服务冗余。以下 P1–P5 均为**计划中**，不是已实现或已接受。英文版本为完整权威计划，继续现有[区域架构](39-regional-sync-architecture.md)。参考草稿 `7f2f244` 和 PTOAS Frontier `fe32dacf3`。

最新有限诊断在8个实际执行上将1538对减少到1282对，保持 staging 值、全序及重用；不构成统一生产变换。已复现普通 Tile alias 的类型不对称及7个旧 Buffer-codegen 测试期望。复用 Group 的 reciprocal-peer 集成还需完整新复现；已有拒绝测试不是其修复。

### P1：修复普通编译集成

修改现有 `buffer_active_domain_analysis.cpp`、`pto_backend.py`、`gm_pipe_layout_evidence.cpp` 和 Group ABI 服务。`alias = scaled` 必须随 RHS 更新类型及后续绑定，保持分配身份；不能处理时原子拒绝候选。发布前验证 assignment symmetry 等必要属性。

完整发射单元与 workspace context 必须来自同一个实际所有者。先用 autosync 关闭的默认管线、复用 Group、双向 GM transport 和真实 PTOAS 复现。不能任意合并独立分配，也不能删除合法公开入口或绕过验证器。验收覆盖 alias chain、负例、Group adapters/originals、独立布局及适用的三种 planner。修正 `test_buffer_codegen.py` 六个旧诊断期望，保留语义负例；将已支持 singleton-column 改为正例。

### P2：分离策略、别名契约与原生实现

复用 PassContext、原生契约及 qualifier。移除 `policy == exact` 决定 `autosync_disjoint_gm` 的耦合，增加显式 campaign 输入和收据。主要 PTOAS 可比检查对两个生产者都显式提供不同已建立 GM 根不相交的调用契约；另做 may-alias 检查。每例必须固定调用方/ABI 契约来源及其强制执行或显式承诺；仅为比较 PTOAS 不能证明不相交。缺少前提时保持 may-alias 或委托。相同根重叠、别名 view、局部存储复用仍保留；未知根不被 SSA 名字证明独立。

主矩阵保持 `enable_scalar_gm_dma=False`。审计原标量 load/store 的就绪、缓存可见性和退出契约；S→S 原生顺序不代表跨管线可见性。无法证明时委托并明确原因，不靠自动 staging 增加覆盖。显式 staging 仅为独立诊断。验收 exact/prefix × disjoint/may-alias × native/DMA，匹配载荷、别名及退出契约。

### P3：以保留区域作为组合单位

复用不可变区域分析、`retained_sync_child.*`、`conservative_sync_realization.cpp`、`token_sync.cpp`；先清点现有子区域选择与拒绝原因，不新建并行 dispatcher。

精确组合消费原始坐标、guards、四种事件查询和完整 writer/reader selectors，构造并联合归约 crossings。保守 prefix 组合消费闭合子命令、每管线访问、合法入口和 palette/rearming 证明，保持出口 pending effects。common-completion 必须使用独立 `(A,E,S,t)` 语义及限定命令，不能假装 prefix 摘要具有该能力。

不规则外层里的 K loop 应保留完整闭合调用，内部已有同步不再被逐载荷父服务覆盖。不能把跨迭代通知的开放 iteration 当闭合 child。参考 PTOAS ChainComposition、CompletionTransfer、CompletionPrefixChild、PrefixChildren；保留三个公共 pass 及实际令牌降低语义。精确导出失败保留数学结果，显式保守请求另标保证。

验收先通用嵌套/分支夹具，再原有 Qwen projection 和一个 vector 区域。验证精确 covers 或保守 coverage、selectors 的 unavailable、子内部及跨边界顺序、空/跳过/重复访问、坐标重置、匹配重用与 trip-count 紧凑性。缺失能力明确报告，不能因此添加内核专用识别器。

### P4：沿控制流保留已证明的交付事实

在 P3 后剩余保守区域中，扩展现有 producer 与整候选证书。may-access 使用并集；must-delivery 仅保留每个可达入口都成立的语义事实。无 else 的跳过路径不能忽略。不得比较分支局部 epoch 数字并误当同一工作。

先说明有限事实身份、transfer/kill、格高度和 worklist 成本；使用原始 occurrence/frame 或已证明的有限 class delivery。禁止运行时私有历史、分支赋值枚举、未知次数展开。循环必须处理入口、backedge 和 zero-trip；子区域仅通过显式 command-transfer 能力贡献完成事实，否则保持保守及受保护 rearming。

例：`complete(A); if(c) unrelated(); use(A)` 保留完成；若一臂新写 A，旧完成不能证明新写完成。新读 A 也会在后续覆盖前产生读完成义务；旧交付不能消除 WAR，事实必须区分访问模式和新发出的工作。两臂语义相同而 token 名不同的交付，必须有对应证明才可合并。生成 token 前选择服务，独立整候选检查失败则保留基线；控制、调用和物理重用独立保护，不重写用户令牌。

验收 adversarial joins/loops、跳过写、新读之后的覆盖、空/重复、子 pending effects、联合省略；比较完整前提关系及实际匹配重用。qkpv_plan 仅附加回归，1282不是硬编码目标或最优数。

### P5：质量验证与剩余义务

重跑现有内核及未修改双层模型，分离原生与 staging。逐叶记录区域提供者、边界、精度、所有权/拒绝、逻辑、分配、C++，保留所有委托分母。

分别统计 modeled minimum demands、额外载荷顺序、forward handoffs、ack/rearming、同管线/退出屏障、guards和静态 alternatives。有限执行比较关系包含。计时包括失败尝试、导出和证书，不能相加包含性时间；保留 watchdog 与静态/动态规模证据。计数大不自动授权新协议，先说明成本及可复用的实现前提。

验收 P1 默认关闭回归、compact K/replay/zero-body、标量与累加器原生豁免、用户令牌、6-ID、终止完成及 C++。输出命令/哈希/收据；宿主计数不代表设备速度。范围不含 allocator、tiling、跨核协议、ID池和 admission-limit 修改。

### 评审与提交循环

按 P1→P2→P3→P4→P5：固定输入和工具，说明不变量与接口，实施，独立验证，再作架构/正确性/成本评审；修复意见后仅提交已接受里程碑，再进入依赖步骤。明确证明缺失可结束有界调查，但不能标为已实现覆盖。正确性通过不等于质量/复杂度通过。遵守全机2个资源密集 worker 与磁盘持久工件规则。

### P1 实施检查点

普通 Tile alias 与已知调用结果共用重绑定服务，保留原物理存储及后续引用；分配不兼容时整体拒绝候选。默认关闭 autosync 的公开 Default 回归覆盖 alias chain 和三种 planner。

非分组 codegen 使用已有的独立入口推断绑定完整 peer 发射清单，并检查它与原根布局一致。相同布局只允许定义打包，不证明共享 launch 分配或 transport 完成；额外检查原生 slot 几何，避免相同总字节掩盖不同 ring。原根 context 对无关 setup 的拒绝仍保留。复用双向 Group 的真实 PTOAS 回归保留合法 originals 与 adapters，覆盖三种 planner。

Buffer-codegen 保留不支持几何的负例并更新诊断；已支持 packed singleton-column 改为正向发射检查。随各里程碑结束，续篇收据记录命令、源码/工具及最终评审。

P1 独立源码评审已接受所有权、正确性和成本边界。最终回归通过187个不同用例（组合186项及 workspace 重跑新增的原根 offset 负例），包含真实 PTOAS。命令和源码哈希见英文[续篇收据](../../../en/dev/ir/52-autosync-regional-repair.json)。

### P2 契约检查点

生产比较独立选择分析策略、别名承诺与标量实现。qualifier 默认 may-alias 和原标量；disjoint 必须提供非空调用契约工件并记录原字节哈希与来源。这只是显式输入假设，不是指针名字证明；同根 views、局部物理重叠及未知来源保持现有处理。

源码确认共享 `SelectBufferSyncImplementation` 仅将 `buffer.read/write` 限定为 A2/A3 AIV Vec 的 TGETVAL/TSETVAL，直接 Buffer codegen 使用相同注册配方与载荷切点。它不限定 cacheable GM 标量。旧 qkpv_plan 的原标量 C++ 含指针及 cache/exit 操作，但不能证明跨管线可见性和数据加载就绪。缺少这些契约或 Buffer admission 时继续委托；显式 DMA 独立报告 carrier 效应和成本。S→S 与累加器原生顺序规则保留。

P5 对两个策略同样显式使用 `--gm-alias-policy disjoint --gm-contract <caller-contract>`，或默认 may-alias。比较 PTOAS 不能证明调用承诺；不删除全部 GM 效应，也不改变默认 ABI。

### P3 区域构造检查点

显式保守生产请求一次发现互不重叠的整个 sequential leaf-For 调用，复用现有 eventful-child provider；公开 InsertSync 与 RealizeSync 共用选择，显式提供子能力保持原语义，精确策略不变。每 owner 仅建立一次发现/绑定及参数维度索引，逐尝试统计引用绑定、效应、分析与导出工作，不展开 trip count。

带类型尝试区分 effects/binding/analysis 与 executable export；后者失败仍在发布的分析中保留成功不可变数学结果。只读外部投影的 GM boundary 明确 unavailable，父区域保留所有原 GM 效应。子 token 配方、原始坐标与 guards 保留；入口确认与物理证明复用现有服务，出口 pending effects 不被删除。

逻辑导出就绪不等于物理证书。后续整计划匹配/palette/reuse 失败时带原因委托整个原内核，不自动丢弃某个子区域后重试。因此可选导入可能比 primitive prefix 少所有权，P5 必须报告。GM 写、未知控制和缺失 caller 证据继续拒绝。选择整个 leaf loop 不代表保留所有外层嵌套 K 管线。common-completion `(A,E,S,t)` 仍是独立缺失能力，不从 prefix 摘要推断。

P3 经独立评审接受；12项新增、75项既有子区域回归、串行构建及全部 hooks 通过。count=2/16/1024 时保留9个分析 occurrence、5 cells及2 demands。原完整模型 down_proj 的3个 leaf 候选有2个逻辑导出，首个明确拒绝缺失外围坐标 cb_inline1641__idx_v0；联合计划成功分配并经真实 PTOAS 生成 C++。不声称保留所有外层 K 区域，精确命令及源码/工具见续篇收据。
