# DSA 复用惩罚建模

## 状态

PyPTO 的复用惩罚 recognizer 仍是默认关闭的实验功能。必须分开理解：

1. 稳定的 DSA-RP 优化问题；
2. PyPTO 当前实现的 recognizer 与 promotion policy；
3. 用于判断哪些 candidate 应获得正权重的实验依据。

所有设备 campaign（包括 blocked 与已被替代的运行）都记录在
[DSA-RP 设备实验记录](dsa_device_experiment_ledger.md)中。

现有证据支持优化模型的 hard-constraint 部分，但最新合法 ablation 表明 relation 的实测
marginal 可以为负。当前非负 solver objective 可以保守地把这类 relation 截断为零，但
不能主动寻找 beneficial overlap。现有证据仍不支持在生产环境中启用当前 promotion policy。

## 稳定优化问题

每个物理内存空间都是独立的标准 DSA arena。容量和 correctness constraint 是硬约束。
稀疏 soft edge `e = (buffer_i, buffer_j, weight_e)` 在两个生命周期可复用 buffer 的
物理 byte range 重叠时激活：

```text
reuse_cost(p) =
    sum(weight_e for active overlapping edges e)
```

planner 在容量内寻找合法 placement，最小化 `reuse_cost`，并可用 peak 作为最终
tie-break。权重保持非负。没有 edge 或权重为零表示 compiler 没有性能损失证据，并不
表示 overlap 一定免费。

优化问题只包含 buffer pair 与 weight。candidate recognition、PTOAS 同步行为和权重
估计是 producer-side 建模问题。pipeline intent 的 hard constraint 及显式 soft
fallback 与本文的 access-hazard recognizer 相互独立。

## 当前 recognizer 实现

`DsaReusePenaltyRecognizer.QUADRATIC` 是唯一启用的研究模式。它默认关闭，并以覆盖率
优先。

### Candidate 生成

recognizer：

1. 把 semantic alias 规范化为物理 allocation identity；
2. 根据各 operator 的权威 `ArgEffect` 声明与 SSA result 收集执行期读写，包括
   tuple result、mutating operation、base allocation 和已知 byte range；
3. 将 access 映射到抽象 source/destination route 与执行资源；
4. 为每个 allocation 构造 terminal-access 和 initial-write frontier；
5. 扫描一个 address space 内所有 lifetime-compatible buffer pair；
6. 记录 distance-zero 与 distance-one WAR/WAW handoff，并保留 route、range、loop、
   control-flow 和 ordering provenance。

legacy recognizer population 仍位于
`metadata.recognized_reuse_candidate_records_v4`。扩展后的 v5 population 还包含
pipeline-serialization provenance，并由 `recognized_reuse_candidates_v5` 单独计数。
SSA 可达记录带有确定性的 region/statement `dag_path`，无序记录使用
`dag_path=none`。可用：

```bash
python -m pypto.tools.dsa_reuse_candidates PROBLEM.dsa.json
```

解析这些 policy 过滤前的记录。

### 当前 v5 edge 构造与权重 policy

当前 `cross_resource_completion_pair_v5` policy 在一个 pair 至少存在一条满足以下
条件的记录时构造 pair edge：

- cross-resource；
- full-allocation 且 access 集合完整；
- 不依赖 conservative initial anchor；
- 不是 same-operation alias-contract 问题。

SSA-ordered 与 loop-carried 记录仍然 eligible。设备实验否定了“SSA reachability
能够证明 completion”这一假设，并发现了高代价的 distance-one handoff。
same-resource、partial-view 和不确定记录仍只用于报告。同一 buffer pair 的多条
qualifying record 只构造一条 edge。`unit_v1` 为每条 edge 赋值 `1`。

该 policy 形成 additive、non-negative 的 `cross_pipe` 潜在同步义务模型，但尚不
判断同步义务是否扩展 consumer 的有效 completion frontier，或是否暴露在 critical
path 上。metadata 用
`reuse_penalty_completion_exposure_model=unmodeled_v1` 明确记录此限制。
recognizer 仍为默认关闭的实验功能。

当前实现会在构造 per-allocation access frontier 时使用 same-resource issue order
与 SSA reachability。same-resource ordering 是抽象 completion chain 假设；对于
cross-resource candidate，SSA reachability 仅作为 provenance 导出，不再作为
suppression rule。

## 证据与已否定规则

受控 placement 保持 DSA problem 与生成操作不变，只改变指定物理 overlap。现有结果
表明：

- 大部分合法 reuse 不改变同步；
- synchronization-group count 不能预测 latency；
- route class 不足以分类，同一路由可能 harmful、neutral 或被其他 release 覆盖；
- loop frequency 会放大暴露在 hot path 上的 handoff，但不能让已覆盖 handoff 变贵；
- 同一 consumer 的多个 predecessor 更接近“最新 predecessor”而不是简单求和；
- 受控 Gumbel ablation 现在支持负的 *relation marginal*：恢复 `(2,39)` 在两台设备上
  提速 2.1-2.35%。它同时删除一个静态 ELSE-arm barrier，但后续 branch-profiled validation
  表明，大部分收益来自该 barrier 根本不执行的 THEN block。因此 relation effect 是因果
  证据，barrier mechanism 还不是。

最新 exact ordered-pair study 给出了对旧 v4 policy 的两个重要反例：

| Pair 类型 | 结果 |
| --------- | ---- |
| SSA-ordered `V -> MTE2` WAR | overlap 增加一个匹配的 PTOAS handoff |
| unordered `M -> MTE1` WAR | overlap 删除冗余 handoff，但没有确认 latency 收益 |
| 另外四个匹配 pair | 同步不变 |

因此 v5：

- 将 SSA `dag_path` 保留为 provenance，而不是 suppression predicate；
- 纳入 distance-one candidate，而不是抑制所有 loop-carried handoff；
- 仍把所有已构造 edge 当作尚未校准的 unit obligation。

实验也说明，给所有 cross-resource obligation 相同的正性能权重仍然过于宽泛：

- 不能给所有改变同步的 pair 正权重；
- 非负 pair model 仍可作为保守近似：neutral 或 beneficial pair 可以不产生正 edge，
  但该近似无法偏好具有 beneficial synchronization interaction 的 placement。
- 多个 whole-kernel RP placement 在 UB 和 L1 case 上都能稳定提速，但插入的
  synchronization-group 数量无法对这些收益排序；
- clean pair ablation 可以重现明显 latency 变化，但其 synchronization summary
  delta 方向可能相反，因此 candidate recognition 目前领先于 mechanism attribution
  与 weight calibration；
- pair 的效果可能依赖周围 placement。pair isolation 必须保持 capacity，并统计构造
  过程中改变的每一条 overlap relation。

## Completion-frontier conjecture

必须区分三类 ordering：

| 层次 | 问题 |
| ---- | ---- |
| Logical ordering | 一个 PyPTO value-producing operation 是否可达另一个？ |
| Completion ordering | overwrite 前，之前的异步 access 是否已经释放复用 byte？ |
| Exposed delay | release point 延后是否改变 initiation interval 或 kernel latency？ |

对于生命周期可复用的不同逻辑 buffer `A` 与 `B`，令 `u` 为 `A` 被复用 subrange 的
最后相关 access，`v` 为 `B` 对同一 subrange 的首次 overwrite。reuse 会产生候选
物理 WAR/WAW handoff `u -> v`。

只有已有 **completion-carrying path** 能保证 `u` 在 `v` 前完成时，下一版 policy
才应 suppress candidate。例如显式 event/barrier/token，或 target 明确保证的 FIFO
completion；普通 SSA reachability 不够。v5 constructor 因此保留这类 candidate，
但其 unit weight 只是实验 surrogate，不是校准后的 latency claim。

只有当 `u` 扩展 `v` 已有的 completion-release frontier 时，才有理由产生正 edge。
定性 cost conjecture 是：

```text
dynamic frequency
    * max(0, completion(u) - latest unavoidable release of v)
```

这只是建模指南，并非 PyPTO 当前可计算的 cycle estimator。当前 v5 unit model
刻意停在这一步之前：它识别稀疏 pair obligation，但不把它们称为 production
performance cost。未来经校准的 producer 应默认使用零权重，并只给具有重复 harmful
证据的机制正权重。对于同一 consumer 的多个 candidate predecessor，应保留有依据的
dominant pair，而不是重复求和。OR group、hyperedge、负 marginal 的 solver 表达方式和
全局 event-budget term 继续推迟。

## Critical-path model v0

下一层模型使用 schedule graph，而不是再增加 recognizer filter。PTOAS diagnostic
exporter 记录 post-InsertSync operation node、每条 pipe 的 issue-order edge、
synchronization edge、loop membership 与已知 allocation size。
`pypto.tools.dsa_schedule_model` 为 operation 赋予 duration，并计算两个有向无环图：

- 只包含每条 pipe stream edge 的 baseline；
- 加入端点均为已表示 operation node 的 synchronization edge 后，得到折叠的
  operation-only graph。

版本 0 把两者最长路径之差报告为 `synchronization_exposure_cycles`，并相对 baseline
critical path 评估每条 synchronization edge。这与简单统计 synchronization group
不同：如果一条 edge 的延迟已经被 critical path 上的工作覆盖，它的 exposure 为零。

在 whole-function DAG 中，静态 loop 按 trip count 聚合。由于模型没有可靠的
multiplier，动态 loop 会直接失败。loop-carried synchronization edge 会从该无环 DAG
中排除，并由下文的 recurrence lower bound 单独处理。有效的 loop-marker endpoint 会作为
duration 为零的 node 保留在结构图中；只有确实未出现在导入图中的 endpoint 才会被排除并
报告。legacy trace import 还可能遗漏 barrier dependency node；此时，即使导入后的 graph
中看不到被排除的 edge，`latency_graph_complete` 也为 false。

exporter 暴露的是 SyncCodegen lowering 之前的 active Final-SyncIR record。这些 record
不是 synchronization instruction：codegen 可能合并相同的 set/wait operation 和相邻
barrier。因此模型将它们报告为 `pre_codegen_sync_record_summary`。candidate reuse 使用
另一个显式标为假设量的 `sync_endpoint_estimator_version`；其中 source 加 target 的执行
次数是未合并的 pressure feature，并非观测到的 instruction count。每个 placement arm
实际的 post-InsertSync instruction summary 必须从 lowering 后的 IR 中采集。

`loop_sync_ii_and_boundary_v1` 模型会在结构图中保留 duration 为零的 loop marker，
分别报告 loop-entry 与 loop-exit synchronization，并把每条已识别的 distance-one
loop-carried dependency 建模为 initiation interval 的 recurrence lower bound。它仍然只是
lower bound：不会声称得到 modulo schedule，也不包含有限 event slot 分配的影响。

实际的逐 arm instruction collector 读取由 lowering 后 post-InsertSync PTO 文件组成的
manifest。如果仍存在高层 `record_event`/`wait_event` operation，它会直接失败；它还要求
每个 case/capacity 都包含声明的完整 arm 与 function 集合，并统计 lowering 后 IR 中实际
存在的 synchronization instruction site。它还会按照带版本的
`event_key_lexical_and_innermost_backedge_v1` contract 推断候选 event lifecycle
transition。这些 transition 并非直接 emitted fact：event ID 可以复用，而且 lowering 后
IR 不再保留 Final-SyncIR group identity。因此推断结果与实际 instruction-site count 分开
报告。对于静态有界 loop，collector 还会用每个实际 site 乘以其外层 trip count，
报告估算的动态指令执行 (dynamic instruction execution)。当 synchronization site 的外层 loop
bound 为 dynamic 或无法解析，或者 site 位于无法解析的条件/控制流 region 中时，该估算就标记为
incomplete，而不会猜测。unstructured control flow 同样会使 function-level 估算变为 incomplete：

realized-placement scorer 会保留六个不同层次的证据，而不会把每个逻辑 buffer pair
都视为独立的硬件事件：

1. `unit_realized_cost` 统计被提升的逻辑 buffer-pair weight；
2. `canonical_physical_reuse_group_count` 把引用同一对完整物理 tile range（忽略顺序）
   的重复逻辑 pair 折叠；
3. `unique_induced_sync_edge_count` 再按照经过验证的 schedule-edge identity 折叠；
4. `estimated_sync_endpoint_executions` 把静态 loop trip count 应用于唯一的 source 和
   target endpoint；
5. `critical_path_realized_cost_cycles` 对每条已实现逻辑 relation 的 singleton
   critical-path extension 求和；
6. `complete_placement_critical_path_cycles` 把所有唯一的已实现 dependency edge
   取并集后一次性加入 reference DAG，并计算一次 longest-path extension。

物理 range key 包含两个完整的 placement range，而不只是它们的交集，因为不同 tile
布局可能具有相同交集。该规范化会删除重复 alias 证据，但不会推断未公开的硬件 bank
或 interleave mapping。penalty-model evaluator 会同时报告六个指标，从而可以通过设备
排序判断各 arm 最早在哪个抽象层出现区分。

第六个指标是 model v2 中不依赖 InsertSync 的 complete-placement score。它只从两个来源
重建不发生复用的 base graph：

- 每条 execution pipe 上固定的 issue order；
- 根据 operation `uses`/`defs` metadata 的 logical root 推导出的 RAW、WAR 与 WAW
  dependency。

基础图中原本存在的跨 pipe dependency 使用与布局新增 dependency 相同的已标定同步
边代价；同一 pipe 内的 dependency 由 FIFO pipe 顺序保证，不再添加独立代价。这样可
保留新增 reuse 边所依赖的基础同步 slack。

已有的 `sync_edges`、synchronization group、barrier record 和物理地址都不会进入 base
graph。对于 placement 中物理 overlap 的每对 buffer，scorer 会把导出的 pre-InsertSync
access provenance join 为一条有向 address-reuse hazard。每条唯一 hazard 只插入一次，并
使用经过校准的正 `sync_latency_cycles` edge weight。完整 placement penalty 为：

```text
penalty(P) = LP(G_no_reuse + E_reuse(P)) - LP(G_no_reuse)
```

该值不是 pairwise penalty 之和。一次有限 longest-path 计算会捕获重复 edge、传递顺序、
共享 slack，以及所有已选 relation 之间的交互。静态 loop 会展开为动态 operation
occurrence，因此 distance-one hazard 会把 iteration `i` 连接到 `i + 1`，并在每条 exposed
recurrence 上计入 synchronization weight。

正 edge weight 除了表示新 precedence constraint 破坏的 overlap 外，还表示
synchronization mechanism 本身的代价。零值只是 dependency lower bound，model v2 会将
其拒绝为未校准。初始版本可以使用一个 architecture-level constant，后续再细化为
pipe-pair 或 signature-specific weight；但该值必须独立于待评估 placement 冻结。

Model v5 不再拒绝所有结构化控制流。raw-PTO bridge 会为每个 `scf.if` 附加稳定的
predicate identity 与 polarity。对同一物化 predicate 的重复测试共享一个 scenario
variable；位于互斥路径上的 candidate site 不会生成 reuse edge。scorer 枚举可达的
结构化路径，而不是猜测 branch frequency。

只有 raw PTO 能证明 predicate 定义在所有 enclosing loop 之外（或它是 function
argument）时，一个 scenario bit 才会跨 loop iteration 共享。对于 loop 内 predicate，bridge
会沿 integer cast 与算术追踪 scalar SSA chain。如果它可归约为 static induction variable 与
constant 的比较，scorer 会记录每次 iteration 的精确 Boolean sequence。runtime-loaded 或
其他无法解析的 loop 内 branch 标记为 `INCOMPLETE`；model 不会用 all-then/all-else 路径代替
未知的 per-iteration mixed profile。

dynamic loop 使用参数模型，而不是展开到任意选定的有限 trip count。raw PTO 能证明
lower bound、upper bound 与 step 相同的 loop 共享一个符号参数 `N`。scorer 在
`N = 1, 2, 3, 4` 上精确计算；只有 placement extension 在这些 probe 上满足

```text
startup + (N - 1) * steady_state,  N >= 1
```

时，才可以比较所得 affine model 在 `N >= 1` 上的 dominance；但该 score 标记为
`PARAMETRIC_ASSUMPTION`，而不是 `COMPLETE`。这明确是从四个精确 probe 进行的
extrapolation，不是对任意 max-plus graph 在所有 trip count 上保持 affine 的证明。
probe 非 affine、必须混同独立 dynamic parameter，或缺失 raw-PTO identity 时会 fail
closed。静态 loop 仍使用有限展开。

lowered-access join 也会区分 exporter 漏记与 lowering 消除的 access。完整 raw-PTO
access provenance 可证明缺失的 access order 没有 materialize；但 surviving endpoint
不共享 lowered loop 并不足以证明 loop-carried recurrence 已消失。对于 peeling、unrolling
或 loop splitting，必须由 exporter 保留 original-loop identity，否则该 recurrence 会
fail closed。endpoint 同时共享多个 nested lowered loop 时也需要该 identity。对于处在
互斥 branch arm 的 distance-one handoff，recurrence scorer 还必须分别计算 source 的
iteration `i` 与 target 的 iteration `i + 1`；同一 iteration 内互斥不足以证明该 edge
不存在，因此当前也会 fail closed。

其余 fail-closed 条件包括缺失 branch node、无法解析的 access join、没有 operation-level
provenance 的 `pipeline_serialization` relation、未校准 edge weight、独立 dynamic loop
parameter 或过大的展开。旧 exporter 遗漏 branch node 的输出因此标记为 `INCOMPLETE`，
绝不会被解释为 penalty 为零的 branch-free graph。

2026 年 8 月的纯主机 re-export 最初报告 complete-placement extension 全部为零，因为它把
geometry endpoint 的 post-InsertSync graph 当作 reference。该 graph 已包含 geometry
placement 引入的 dependency。model v2 直接修复这一方法错误：它忽略全部 InsertSync
record，并从 logical SSA/allocation root 与固定 pipe order 重建 base graph。

### 全局 synchronization weight 敏感性

当前纯主机 sweep 使用一个全局 synchronization-edge weight，在
`16, 32, 64, 96, 128, 160` cycle 上重新评分 complete placement graph。它没有拟合
per-kernel 或 per-edge constant。只有归档 campaign 已建立可复现双设备 effect 时，
device ordering 才作为 label；低于阈值或无法复现的 effect 仍只作 diagnostic。

该 reanalysis 覆盖之前测量的 19 个非 Gate problem-capacity cell，以及历史上的多函数
`mtp/gate` 反例。十二个 loop-aware RMSNorm/top-k cell 在每个 weight 上仍完整。对于静态
bound 的 induction-variable predicate，可以静态导出精确的 mixed-iteration profile。对于
runtime-loaded predicate，则必须显式提供 `exact_runtime_branch_profile_v1`。该输入绑定
schedule、problem、捕获 tensor/scalar 以及 loop-trip metadata 的 hash，记录每个 active
branch occurrence，绝不会把捕获值提升为 compile-time fact。嵌套 branch 还会记录 active
flattened occurrence index。

使用归档的确定性输入后，`rmsnorm_rope_cache_write/half` 现在完整。它的两个
runtime-loaded predicate 使用实际被测 dispatch 的精确 mixed branch profile。相同 contract
也可以表达 `softmax_pool_c128/native`，但该 case 尚未使用捕获 profile 重新评分，因此这里
不把它计为新证据。

KV 缺口已经修复。fresh export 为每个 `pipeline_serialization` penalty 记录 producer 与
consumer access site。每个 arm 都实现 64 个 pair；Cypress 有 34 个、DSA-RP 有 40 个
pipeline-serialization pair，其中 16 和 20 个分别 materialize 为 lowered operation edge，
其余由唯一 raw-PTO join 证明已被消除。虽然 unit cost 相同（`64 -> 64`），complete-placement
score 在冻结 weight grid 上给 Cypress 分别增加
`408, 560, 1200, 1840, 2480, 3120` cycle，而 DSA-RP 始终为零。这在不读取 InsertSync 的
情况下正确预测了已测 DSA-RP 胜出。

Gate 已在冻结 compiler revision 上完成 product-faithful re-export。新的 source-loop marker
可以穿过 PTO codegen，使 `gate_aic` 把多个 lowered loop 重新关联到原始 source loop；
`gate_aic` 和 `x_norm_quant` 现在都完整。Pinned PTO-ISA estimate 还覆盖了 Gate 的
`tmul(fp32, 1x4096)`、`tfree` 和 `tfillpad`，没有引入本地拟合常量。但 parent aggregate
score 仍 fail closed：`ffn_norm`、`gate_aiv` 和 `route_sort` 分别需要尚未支持的
`trecip`/`tabs`、`tmaxs` 和 `trowexpanddiv` signature。因此不能从两个完整 child 推断
parent-wide Gate ordering。

全局 calibration 仍无法辨识。确认的 KV 和 RMS 胜出现在整个冻结 weight grid 上都可解释，
但 Gate 的确认 Cypress ordering 没有完整 parent score。此外，完整的 Gumbel model 在所有
符号 branch scenario 中都预测 DSA-RP 更优，而 q1 上两台设备测得 latency null。因此可用于
leave-one-workload-out 的确认且 model-complete 的方向 label 仍太少，不选择任何 weight。

指定的代表性用例直接暴露了 model coverage 与 mechanism 缺口：

| 用例 | 已有双设备证据 | Unit objective | 整函数 model 状态 |
| ---- | -------------- | -------------- | ----------------- |
| `rmsnorm_rope_cache_write/half` | DSA-RP 对 Cypress `-7.19%/-7.43%` | `8 -> 0` | 使用精确且 digest-bound 的 runtime profile 后完整。在 `16--160` weight 上，Cypress 增加 `62, 97, 161, 354, 610, 761` cycle，DSA-RP 增加零。 |
| `kv_score_proj_c128/native` | DSA-RP 对 Cypress `-2.50%/-2.30%` | `64 -> 64` | 完整。pipeline-serialization provenance 与修复后的 lowered-access join 在每个测试 weight 上都给 Cypress 正 critical-path extension、给 DSA-RP 零。 |
| `mtp/gate` | Cypress 在四种 capacity 都更快，约 `1.6--4.1%` | DSA-RP 的 count 相同或更优 | 已 re-export 但 parent 不完整：source-loop identity 已修复，两个 child 完整，三个因不支持的 duration signature 而 fail closed。 |
| `gumbel_argmax/q1` | DSA-RP 对 Cypress `+0.22%/+0.11%`，为 latency null | `14 -> 0` | 完整，但在每个 weight 上都 false-confident。将同一 predicate 的三个重复测试关联起来可删除不可能路径，却不能解释该 null。 |
| `hc_post/native` | 在不同 campaign 中 effect 很弱、符号反转或无法复现 | 原始 cell 为 `33 -> 24` | 在一个 correlated dynamic-loop parameter 下为 `PARAMETRIC_ASSUMPTION`；预测 DSA-RP 不更差，但 extrapolated score 与 device label 都不能用于 validation。 |

持久化 device 数据源是
`dsa-rp-loop-aware-model-prospective-0820ab418-final.tar.gz`
（`1a7e5d5ffe93a43b260012d47af98321cb5a10156ecc8486dbc37f00767374d2`）和
`dsa-rp-four-candidate-physical-penalty-aeba32c70-final.tar.gz`
（`a05aad5829865d196bc7d7a415b40d8c06b3e6b566d1f80fce269352c78765a0`）。
每个 `score-realized-grid` 结果都会记录 schedule、problem、solution、duration model 与可选
non-materialization evidence 的 SHA-256，并记录所选 function 与 fail-closed duration policy。
不打开 latency 数据即可对每个 frozen placement 重现 grid scoring：

```bash
python -m pypto.tools.dsa_schedule_model score-realized-grid \
  SCHEDULE.jsonl PROBLEM.dsa.json SOLUTION.dsa.solution.json \
  --function FUNCTION --model DURATION_MODEL.json \
  --sync-latency-grid 16,32,64,96,128,160 -o ARM_GRID.json
python -m pypto.tools.dsa_penalty_model_evaluation sync-weight-grid-input.tsv \
  --sync-weight-grid --split development --minimum-device-effect 0.02 \
  --required-device-count 2 -o sync-weight-grid-evaluation.json
```

该分析未通过 incremental critical-path planner 的 gate。planner 没有改动，也不应根据此
grid 启动新的 device task。static 与精确 runtime mixed-iteration branch、source-loop
identity、KV pipeline provenance 与 KV access join 已完成。但预先声明的 scientific gate
仍失败：Gate 的 duration 不完整，Gumbel 在每个测试 weight 上都是 false-confident
prediction。因此刻意没有实现 incremental greedy planner。下一项本地工作是解决这两个失败，
并增加足够多确认且 model-complete 的独立方向用例，以支持有意义的
leave-one-workload-out calibration。

对每个 arm 的实际 post-InsertSync graph 评分，才得到预期的 analysis oracle。在同一回顾
corpus 上，40 个 strict model ordering 全部与实测方向一致；device effect 至少 2% 的
24 个 strict comparison 也全部一致。RMSNorm geometry 为 12,736 cycle，Cypress 和
DSA-RP 都为 12,249 cycle：预测的 speedup 方向正确，但幅度只有约 3.8%，而实测为
21.6--22.7%。单 edge graph ablation 隔离了该差异：geometry 增加一条从第一个 loop 末尾
到第二个 loop 开始处的 V-to-MTE2 dependency。删除它会把 12,736 变为 12,249 cycle；
把它加入 Cypress 会把 12,249 变为 12,736。因此，post-InsertSync latency approach 在
该 corpus 上具有方向区分力，但幅度模型仍不完整。

同一 re-export 也为五个 cell 的 physical-penalty corpus 恢复了 product-faithful graph，
以及不使用 timing 的逻辑 candidate 与物理 placement catalog。这些 target 包含结构化
branch，因此 candidate model v1 仍会 fail closed，而不会在互斥路径之间虚构无条件
edge。KV graph 还缺少 128 条 raw candidate record 中 48 条对应的 lowered site。
因此下一项 modeling requirement 是 branch-aware candidate-to-join mapping；恢复的
catalog 已保留实现该功能所需的全部 branch 与 loop context。

### 带符号的 post-InsertSync 边际代价

合法 pair ablation 表明，reuse relation 不一定产生正 latency obligation。对于固定的
周围 placement `P` 和一条 relation `r`，分析 oracle 因此定义为：

```text
p(r | P) = L(InsertSync(P + r)) - L(InsertSync(P))
```

其中 `L` 是完整 post-InsertSync schedule 的 loop/resource-aware makespan estimate。
该值有意保留符号：负值表示加入 `r` 后删除了更昂贵的 synchronization dependency。
`evaluate` 命令将其导出为 `signed_marginal_sync_cost_cycles`，并在任一 latency
graph 不完整时 fail closed。它目前是 analysis oracle，还不是 DSA solver 使用的稀疏近似。

evaluator 还会从 baseline 加上最终 InsertSync edge 的 multiset delta 来重建 candidate
dependency graph，并要求重建后的 modeled makespan 与真实 candidate 完全一致。随后，它在
baseline context 中独立评分每条新增或删除的最终 edge：

```text
q(P' | P) = sum(e in E(P') - E(P)) delta_add(e | P)
          + sum(e in E(P) - E(P')) delta_remove(e | P)
```

报告会分别保留 `q` 与非加性 residual `p - q`。同时导出的确定性 sequential
attribution 必须 telescoping 到精确 marginal。这样既能发现重复 synchronization edge，
也能发现多条单独 exposed 的 dependency 实际覆盖同一段 critical path 的情况。

在现有 RMSNorm/top-k development slice 上，final-edge approximation 给出 40 个 strict
prediction，40 个都与 device 方向一致；device effect 至少 2% 的 24 个 comparison 也全部
一致。72 个 arm/device comparison 的 interaction residual 均为零。该结果值得继续，但它
还不是 solver penalty：approximation 仍需读取最终 InsertSync edge delta。下一步 compiler
bridge 必须从 logical reuse relation 及其周围 partial placement 预测该 delta，包括
loop-boundary lifting 与 dependency implication。

受控 Gumbel 实验在 operation order 与 address-translation control 不变的情况下隔离了
四条 relation。下表的 synchronization 变化是各 relation contrast 的相关量，本身不是
因果归因：

| Relation | 最终 synchronization 变化 | 双设备 latency 结果 |
| -------- | ------------------------- | ------------------- |
| `(2,39)` | 删除一个 V-pipe barrier | `-2.07%/-2.30%` 与 `-2.16%/-2.35%` |
| `(38,42)` | 增加一个 barrier、set 和 wait | 两台设备约 `+1.9%` |
| `(3,38)` | 最终增加一个 barrier | 约 `+0.4%` |
| `(38,79)` | 增加 set/wait site，但不增加 barrier | latency null |

从结构上看，对于 `(2,39)`，D0 包含从上一轮 `trowargmax` read 到下一轮 else-arm `tmov` write 的
loop-carried V-to-V WAR，因此 InsertSync 在 `tmov` 前插入 barrier。恢复 overlap 后，
`trowargmax` scratch 与下一轮 MTE2 load destination alias，新增 V-to-MTE2 recurrence；
已有的 MTE2-to-V load-completion handoff 随后使直接 V-to-V dependency 由传递关系蕴含。
该 barrier 在 `InsertSyncAnalysis` 内部消失；逐 phase dump 证明在 `MoveSyncState`、
`RemoveRedundantSync` 和 event-ID allocation 之前它就已不存在。因此机制是 dependency
implication，而不是 event coalescing。

精确的逻辑访问到 lowering endpoint 追踪如下：

| Relation | DSA access order | Lowered operation | Post-InsertSync 变化 | 动态位置 |
| -------- | ---------------- | ----------------- | -------------------- | -------- |
| `(2,39)` | `139/140 -> 142` 与 distance-one `142 -> 99` | `tadd`/else `tmov` -> `trowargmax`，随后 `trowargmax` -> `tload` | 删除 V-pipe barrier `52 -> 49`；把 `-> 11` 的 recurrence source 从 `51` 改为 `52` | 63-trip outer loop 内 |
| `(3,38)` | `114 -> 139` 与 distance-one `144 -> 101` | `tcolexpand` -> `tadd`，随后 scalar `tgetval` -> `texpands` | 增加 V-pipe recurrence barrier `52 -> 13` | 63-trip outer loop 内 |
| `(38,42)` | `144 -> 153` | scalar `tgetval` -> post-loop `texpands` | 增加 S-to-V handoff `59 -> 61` 与 V barrier `52 -> 61` | loop 后执行一次 |
| `(38,79)` | `144 -> 194/195` | scalar `tgetval` -> branch `tadd`/`tmov` | 增加 branch-lifted S-to-V handoff `59 -> 64` | loop 后执行一次 |

其余 relation 将实测 effect 与结构计数区分开来。`(38,42)` 增加 post-loop
synchronization，并产生可复现的正 effect；`(3,38)` 增加 V barrier，但仅慢约 0.4%；
`(38,79)` 增加 event pair，却是 latency null。因此，仅统计 logical reuse 或 barrier
都不能构成可靠的 penalty model。

branch-aware schedule graph 为每个 `(branch-or-loop marker, pipe)` 建立一个零 duration
control point。then/else arm 从同一 per-pipe frontier 开始，并通过取最大值汇合，绝不被
串行化。附着在 `IF_BEGIN`、`IF_END` 或 loop marker 上的 sync operation 会绑定到对应的
pipe-specific control point。legacy PTOAS debug import 会保留已打印的 branch skeleton，
并在存在时保留 barrier dependency node。缺失 barrier dependency 或 branch node 时，
`latency_graph_complete` 仍为 false。

该支持适用于完整 post-InsertSync arm 的评分。旧的 `candidate_v1` hypothetical-edge
scorer 在 candidate endpoint 位于 conditional 内时仍会 fail closed，因为把某个 arm 的
set 提升到 branch join 是 InsertSync transformation，不是简单加入一条 graph edge。

归档的 KV endpoint 早于 research pipeline 当前使用的 pre-DSA Simplify placement。
其中 candidate access order 98 与 103 在 lowered schedule 前已被删除。当前 export 会直接
完成 join；分析旧 endpoint 时则必须提供绑定 digest 的 non-materialization evidence，
不能为这些 order 虚构 schedule site。

纯主机回顾分析使用产品 PTOAS v0.57 InsertSync 实现重建了全部八个 Gumbel endpoint。
每个 endpoint 都有 93 个 operation node、100% exact/pinned duration coverage，以及完整的
结构化 control-flow graph。但折叠后的 operation-only 带符号 oracle 仍不能解释实测排序：

| Relation | 预测 marginal | 既有双设备结果 | 解释 |
| -------- | ------------- | -------------- | ---- |
| `(2,39)` | `0` cycle | 约 `-2.1%/-2.3%` | 正确显示 collapsed DAG 上没有 barrier exposure，但不能解释 placement effect |
| `(3,38)` | `0` cycle | 约 `+0.4%` | effect 很小，按 slack 处理是合理的 |
| `(38,42)` | `+189` cycle | 约 `+1.9%` | 符号正确，但低估幅度 |
| `(38,79)` | `+56` cycle | null | 较小的结构性 false positive |

queue/event 模型 `static_unrolled_pipe_event_branch_extremes_v2` 显式展开静态有界 loop，
保留每条 pipe 的 FIFO issue order，并把 loop-carried event 从 iteration `i` 映射到
`i + 1`。前瞻设备验证独立恢复出实际 branch profile：六个 THEN 和两个 ELSE block。
在十个 topology contrast 上，该模型消除了 unsigned reuse count 的两个符号错误，但没有
胜过 emitted barrier-site count，并且漏掉了核心 `(2,39)` 结果：

- `(2,39)`：实际 profile 的 marginal 为 `0` cycle；既有结果为约
  `-2.1%/-2.3%` 的 beneficial effect。
- `(3,38)`：active 时为 `+63` cycle；既有结果为约 `+0.4%` 的小幅 regression。
- `(38,42)`：`+192` cycle；既有结果为约 `+1.9%` 的 regression。
- `(38,79)`：`+56` cycle；既有结果为 latency null。

被删除的 node-49 barrier 位于 ELSE arm。它在六个较长的 THEN block 中不会执行，但这些
block 承担了大部分实测收益。因此，该实验**没有**证明删除 barrier 导致 `(2,39)` 提速。
要支持该结论，必须执行 placement-by-barrier 2x2 factorial，并在 device disassembly 后加入
same-footprint code-layout control。

同一验证否定了 per-pipe constant：device-0 上 `(3,38)` 与 `(38,42)` 得到的 calibration
相差 4.40x。PTO-ISA 解释了原因：barrier cost 包含 barrier instruction、排空 predecessor
尚未完成的 tail work、清空 stream，以及 successor 重新支付 startup。公共 evaluator 因此
还会报告 `queue_drain_restart_signed_marginal`，其 site cost 为：

```text
barrier instruction + predecessor pending tail + successor stream restart
```

startup/tail split 按完整 operation signature 从固定 PTO-ISA 数据或显式 calibration 中解析；
无法解析的 transfer 会 fail closed。node 49 的 `tmov` 正应如此：在 factorial 与 disassembly
确认 device mechanism 前不能拟合数值。barrier dependency provenance 直接来自公共 export
字段 `sync_groups.operations.dependency_node`，evaluator 不再需要 campaign-private 重建路径。

```bash
python -m pypto.tools.dsa_schedule_model evaluate arm-manifest.json \
    --model duration-model.json -o signed-marginals.json
```

```bash
python -m pypto.tools.ptoas_sync_summary --arm-manifest post-sync-arms.json \
    -o post-sync-summary.json
```

operation duration 由 provider snapshot 提供；
该 snapshot 从 `runtime/pto_isa.pin` 指定的精确 PTO-ISA revision 加载，并包含 A2/A3
拟合公式、传输带宽、频率、源文件哈希和完整 revision。默认情况下，不支持的 operation
会直接失败。探索性运行可以显式选择 `--unsupported-policy fallback`，但每个 fallback
node 都会被标记，结果也会报告精确覆盖率与 fallback 覆盖率。simulator instruction
中位数可以覆盖 analytical provider，同时保留 PTO-ISA provenance。

```bash
python -m pypto.tools.dsa_schedule_model snapshot-duration \
    --pto-isa-root <exact-pto-isa-checkout> -o duration-pinned.json
python -m pypto.tools.dsa_schedule_model score schedule.jsonl \
    --model duration-pinned.json -o score.json
python -m pypto.tools.dsa_schedule_model validate-perf-sim trace-*.json \
    --model duration-pinned.json -o perf-sim-validation.json
python -m pypto.tools.dsa_schedule_model calibrate instr_metrics.json \
    --base-model duration-pinned.json -o duration-calibrated.json
```

该 pinned provider 删除了 PyPTO 之前静默使用的粗粒度 pipe 常数；但它本身**不会**让
结构化 schedule graph 变成 cycle-accurate 模型。Perf-Sim 在可用时优先采用更丰富的
CCE mock 记录 cycle，仅在没有该记录时才退回 PTO-ISA lightweight formula。在 pinned
A2/A3 验证集的 102 个 formula-supported event 上，lightweight provider 相对有效
Perf-Sim event 的平均绝对误差为 82 cycle，平均绝对百分比误差为 73.6%。elementwise
`TMUL` 较接近（MAPE 11.3%），reduction 与 exponential 则不接近。

配对设备验证得出同样结论。对于当前四容量开发数据集中无 branch 的 RMSNorm 用例，
provider 精确覆盖 40% node，其余 node 都明确标记为 fallback。模型在四种容量下都预测
三个物理 endpoint 的完整 makespan 相同；但已归档的设备结果在 native、half 和 quarter
容量下显示 geometry 到 penalty-aware placement 约有 13% 改善。因此，该 provider 是
可审计且固定版本的起点，但当前 critical-path model 仍不能解释已观察到的 placement
effect。在把 cycle score 用作 DSA-RP weight 之前，必须用逐 kernel 的 Perf-Sim
instruction trace 进行校准。

当前 critical-path 校准子集使用 `structured_branch_static_loop_v2` eligibility policy。
它接受具有已导出非负静态 trip count 的 loop 与结构化 if/else branch；只排除动态边界
loop。该过滤器只使用结构信息，不读取 solver objective 或既有设备结果。在冻结该分析
子集前使用：

```bash
python -m pypto.tools.dsa_schedule_model qualify schedule-*.jsonl \
    -o schedule-eligibility.json
```

语料库发现首先从 driver 出发。只有当静态检查能够证明源码包含本地 JIT entry、tensor
specification、direct golden 和可执行的 `run_jit` contract 时，才会考虑该源码。DSA
problem 只能来自完全相同 PyPTO-Lib revision 的 fresh export。旧 inventory 只能提示应当
重新导出哪些源码，不能直接形成 current candidate。发现工具还会记录真实 measurement
unit，而不会把每个子 DSA problem 都计作独立 kernel：

- 一个 submit 中只有一个 DSA problem 时，它是 single-kernel driver；
- 一个 submit 中包含多个 DSA problem 时，整体构成一个 complete mixed group；
- 包含多个 submit 时，它是一个 parent-wide policy workload。

```bash
python .claude/skills/incore-profiling/discover_dsa_direct_golden_corpus.py \
    --pypto-lib-root <pypto-lib> \
    --invocations <fresh-export>/invocations.tsv \
    --inventory-revision <exact-pypto-lib-sha> \
    --export-status <fresh-export>/export-status.tsv \
    --output-root <discovery>
```

base problem identity 是 semantic DSA fingerprint。受控 tiling variant 必须具有单独的
显式 tiling identity，并按 base problem 分组；不能把它们当作独立 workload family。
discovery 会拒绝 current export inventory 中的性能字段。它可以附加以前的 terminal
status，但该注释绝不影响 membership 或 ordering。

随后仅按“可测量性”冻结设备语料，而不按已观察到的性能或求解器目标筛选。一个
用例必须在原生、half、q1 和 tight 四种容量下，对四个逻辑策略（几何 first-fit、
几何 canonical greedy、Cypress 和 DSA-RP canonical greedy）都可行、可运行且正确。
schedule eligibility 单独记录，并且只限制 critical-path prediction；branch 或动态 loop
不会使可在设备上测量的用例失去资格。语料冻结工具会拒绝包含时延、加速比、目标值或预测
关键路径字段的输入表：

```bash
python -m pypto.tools.dsa_measurement_cohort preflight.tsv results/ \
    --minimum 20 --maximum 40
```

在确认 launchability 后，为每个 workload 选择一个 evaluation capacity，且不读取设备
时间。该 capacity 的目标是暴露无权重 Cypress relaxation 与 DSA-RP 结构化目标之间的
差异，而不是单纯最大化内存压力。只有满足以下条件的 capacity 才是
**opportunity capacity**：

- 四个逻辑策略均可行，并且通过独立验证；
- Cypress 实际产生至少一个带 penalty 的 reuse relation；
- geometry first-fit、Cypress 与 DSA-RP 具有三个不同的完整 map；
- DSA-RP 的实际 reuse-penalty objective 严格低于 Cypress。

在所有 opportunity capacity 中，selector 依次最大化 Cypress 减 DSA-RP 的 objective
gap、实际带 penalty relation 的对称差，以及全部实际 reuse relation 的对称差；只有前述
字段全部相同时，才优先选择更紧的 capacity。该规则完全基于结构信息：solver runtime 与
device latency 都不允许作为输入。没有 opportunity capacity 的 workload 会保留为明确
标记的 null control，而不会把只有两个 map 或 objective 相同的 cell 静默提升为主用例。
强制不相交 size shortage 仍作为审计字段记录，但不参与选择。

当前已经测量的 workload 是开发语料：可以把其结构上冻结的 opportunity capacity 与既有
timing 连接以改进模型，但不能把该连接称为前瞻性证据。只有在查看任何新 workload 的
timing 之前使用同一规则冻结 capacity，它们才能构成 holdout：

```bash
python .claude/skills/incore-profiling/select_dsa_workload_capacity.py \
    --cohort inputs/results/corpus-frozen.tsv \
    --instances fresh-export/invocations.tsv \
    --feasibility inputs/results/full-policy-feasibility.tsv \
    --maps inputs/results/map-digests.tsv \
    --workload-status inputs/results/workload-status.tsv \
    --corpus-root fresh-export --replay-root inputs \
    --output-root evaluation-capacities

# 仅用于开发分析，并且必须在 evaluation-freeze.json 已写入后运行：
python .claude/skills/incore-profiling/evaluate_dsa_opportunity_freeze.py \
    --freeze evaluation-capacities/evaluation-freeze.json \
    --pairwise-effects prior-timing/results/pairwise-effects.tsv \
    --output-root development-analysis

# 前瞻性 holdout，必须在任何 timing 之前执行：
python .claude/skills/incore-profiling/freeze_dsa_opportunity_holdout.py \
    --opportunity-freeze new-candidates/evaluation-freeze.json \
    --development-freeze \
      .claude/skills/incore-profiling/dsa_driver_first_opportunity_development_v1.json \
    --minimum 8 --maximum 12 --output-root prospective-holdout
```

holdout freezer 会拒绝带 performance 字段的输入，同时排除开发集已经使用的脚本和语义
DSA problem fingerprint。这样，新的 wrapper 文件名就不能把已经观察过的问题重新标记为
prospective。该工具先按 source class 与 model family 的多样性进行确定性选择，再使用结构
opportunity 信息，并在读取 device timing 表之前封存最终 capacity 行。

该规则的第一次应用冻结在 `dsa_driver_first_opportunity_development_v1.json` 中。
排除一个被 stock golden 阻塞的 gate workload 后，共有 19 个 workload：16 个 opportunity
cell 和 3 个结构 null control，分别使用 11 个 tight、4 个 quarter、2 个 half 和 2 个
native capacity。冻结后的开发集连接得到 4 个确认的 Cypress 与 DSA-RP 排序，其中 3 个
符合结构化 objective，1 个不符合；只有一个 cell 确认完整的
geometry-first-fit > Cypress > DSA-RP 时延排序。在这个小型开发集中，objective gap 的大小
与实测 DSA-RP 优势呈负的 rank correlation，因此不能把更大的 unit-cost gap 当作时延预测。
这些观察结果说明需要改进 penalty weight，但不构成对选择规则的前瞻性验证。

若合格用例超过 40 个，则按模型族和父程序进行确定性的轮询选择。代码端点相同的
逻辑策略仍保留在矩阵中，但只需进行一次物理测量。

planner 比较使用配对 schedule graph，并采用 `candidate / baseline - 1` 约定，负数表示
预测 candidate 更快。evaluator 首先要求两个 arm 具有相同 operation stream，然后报告
placement 变化新增和移除的 synchronization dependency。held-out cohort 的 comparison
manifest 可以同时省略两边的 observed latency，但不能只提供单边 observation。
evaluator 会对 manifest、schedule 输入和 prediction 做 content address，从而在
device timing 前冻结 held-out prediction：

```bash
python -m pypto.tools.dsa_schedule_model calibrate \
    instr_metrics.json -o duration-v0.json
python -m pypto.tools.dsa_schedule_model evaluate \
    comparisons.json --model duration-v0.json -o predictions.json
```

该模型是研究基础设施，不是当前 `unit_v1` weight policy。在用于分配 DSA-RP weight
之前，它必须能够预测 fresh arm pair 的 latency 方向与排序。尤其是，单个 schedule
graph 可以验证 graph construction，但不能解释 placement 导致的 latency 差异。

原始 candidate 与 schedule 坐标通过显式标识连接。设置
`PYPTO_EMIT_DSA_ACCESS_PROVENANCE=1` 后，PTO codegen 会用
`pypto.access.N` NameLoc 包裹已标记的 lowered operation。`N` 在构造 DSA
问题时写入源 Call，并在后续 lowering 中保留；它不会根据 lowered statement
顺序重新计算。因此它与 candidate record 的 `sites=prior->next` 一致。PTOAS 将该整数
复制到 schedule graph。现在 `InitMemRef` 前会防御性地运行一次 `Simplify`，
在静态恒死的 pipeline slot 分支进入 DSA lifetime 或 candidate 之前将其删除。
若 site 缺失，或 candidate route 没有经过验证的 PTOAS pipe mapping，连接会直接
失败；不会把 SSA node number 或源码行号误当作同一坐标。

在该边界引入前生成的历史 problem，可能仍包含其配对 lowered schedule 中并不存在的
operation candidate record。只有提供显式 `--nonmaterialized-access-evidence` 文件，
并且其中 SHA-256 与该 problem 和 schedule 精确绑定时，才允许对其评分。这些 record
保留 logical unit penalty 以供审计，但对 executable relation、physical group、
synchronization execution 与 critical-path predictor 的贡献均为零。该例外必须由证据
驱动；没有该文件时连接仍然 fail closed。

solver 提升的 `pipeline_serialization` penalty 描述被放松的 pipeline stage
separation。v5 exporter 会同时记录 penalty reason 及 producer/consumer access
site，因此当前 problem 可以像其他 reuse relation 一样把它们连接到 lowered
operation。缺少这些 v5 record 的历史 export 只要实现了此类 relation 就仍为
incomplete；绝不能把缺失 provenance 解释为 operation 未 materialize 或模型成本为零。

首个 candidate-weight prototype 保留 reference schedule 中已有的全部 sync，
并从 prior site 的 terminal macro phase 向 next site 的 initial macro phase
加入一条 hypothetical completion edge。非负 weight 是加入该 edge 后 longest
path cycle 的增量。它还会把共享同一 consumer 的所有 candidate edge 联合加入，
并报告 combined cost 与 singleton cost 总和的差值，从而显式呈现 release
coalescence，而不是重复计数：

```bash
PYPTO_EMIT_DSA_ACCESS_PROVENANCE=1 python my_export.py
python -m pypto.tools.dsa_schedule_model score-candidates \
    schedule.jsonl problem.dsa.json --model duration-v0.json \
    -o candidate-weights.json
```

提供 solution 后，报告还会生成 `edge_explanations`。每一行会把一条 logical reuse
relation 依次连接到其 canonical 物理重叠范围、lowered producer/consumer operation、
已插入的 sync group、loop execution multiplier，以及 critical-path 或 recurrence slack。
duration calibration 按 pinned PTO-ISA 的完整 signature 索引；不支持的 signature 会
fail closed，而不会退化为 instruction-family median。

版本 1 保留 distance-zero candidate 的无环 longest-path 评分，并为 distance-one
candidate 增加 lower-bound 评分。它连接两个 PTOAS site，验证二者共享真实 loop，并选择
最内层的公共 loop。base initiation-interval lower bound 是单次迭代中每条 pipe 的工作量
以及所有已有 single-recurrence cycle 的最大值。对假设 edge
`source(i) -> target(i+1)`，模型寻找 intra-iteration 路径 `target -> source`；若该路径
存在，路径时长加 edge latency 会形成新的 recurrence bound。非负 weight 是该 bound 相对
base 的增量。若不存在返回路径，该 edge 只改变 phase，不提高此 throughput bound，因此
weight 为零。

这仍然只是 lower bound，而不是完整的 modulo-scheduling 模型；它尚未搜索包含多个新
recurrence edge 的 cycle。多个 candidate record 若连接到同一个
`(loop, source, target)`，会在 `loop_recurrence_edges` 中折叠，避免下游分析重复累加证据。
distance-zero record 同样会在 `distance_zero_edges` 中折叠；
`candidate_weight_summary` 的 count、sum 与 max 基于这些唯一 schedule edge，而不是原始
buffer-pair record。

对于早于 JSONL 导出器的 PTOAS 版本，旧版 level-3 调试日志导入器可以从原始
PTO 中恢复相同的稳定访问坐标。该连接要求可执行操作顺序完全一致；如果操作或
`pypto.access.N` 位置缺失，它会失败而不会猜测：

```bash
python -m pypto.tools.dsa_schedule_model import-debug insert-sync.log \
    --function kernel --pto kernel.pto -o schedule.jsonl
```

PTOAS 也会给某些函数级事件生命周期附加 loop-end identity。仅当每个活动的
set/wait endpoint 都是该 loop 内的 operation 时，bridge 才会将该 group 分类为
recurrence。prologue-to-loop-end 或 prologue-to-epilogue lifecycle 仍是 boundary
dependency，不会被误当成 initiation-interval constraint。

raw-PTO join 还会保留 operand/result type 与标量常量 operand，并从静态 tile 与
partition type 推导 `static_work_bytes`。这足以为 transfer 定价，同时不会虚构
DSA allocation size：
legacy SyncIR buffer identifier 与 raw PTO SSA name 不一致，因此 allocation size
仍明确标记为缺失。只有当 raw operation 包含 accumulator input 时，trace 侧的
`pto.tmatmul.acc` 才能与 raw `pto.tmatmul` 对应；普通双输入 matmul 仍保持不同，
并触发 mismatch 检查。

版本 0 只为 inbound/outbound DMA、L1-to-L0、L0-to-external、vector、matrix
和 scalar resource 提供经过验证的 pipe mapping。其余 transfer route 在 PTOAS
pipe mapping 确认前会被拒绝。

## 剩余验证

completion-frontier factorial 已经完成。它确认同一 consumer 的多个 active
predecessor 可能合并到一个 release frontier，但隔离出的 frontier extension 是
latency-neutral，因为已有 drain 已使该 resource 静止。因此实验支持
consumer-aware deduplication，但不支持为它赋予正权重。

下一项研究应从具有稳定 RP-versus-compact latency 差异的 kernel 反向分析：

1. 找出 endpoint placement 改变的 candidate pair；
2. 构造 exact-XOR single-pair 与小型 factorial placement，优先采用保持 capacity
   的 address exchange；
3. 在通过 PTOAS 编译前冻结 mechanism 预测；
4. 比较完整 synchronized instruction topology 与 predecessor identity，而不是只看
   group count；
5. 使用真实 kernel input 与 scalar 验证所有 written output；
6. 在两个 device 上测量 kernel-only latency，只有初始 confidence interval 有信息量
   时才增加 sample。

直接目标是已经确认 endpoint speedup、但 pair-level mechanism attribution 仍不完整
的 UB 与 L1 kernel。只有一个 mechanism 能跨 fresh kernel 和 placement background
预测效果方向，才应获得正权重。checked-in v5 unit model 可以生成 algorithm-study
instance，但在完成该 calibration 前仍不支持 production promotion。
