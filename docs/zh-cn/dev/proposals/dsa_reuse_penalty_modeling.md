# DSA 复用惩罚建模

## 状态

PyPTO 的复用惩罚 recognizer 仍是默认关闭的实验功能。必须分开理解：

1. 稳定的 DSA-RP 优化问题；
2. PyPTO 当前实现的 recognizer 与 promotion policy；
3. 用于判断哪些 candidate 应获得正权重的实验依据。

现有证据支持继续使用简单的非负优化模型，但不支持在生产环境中启用当前 promotion
policy。

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
2. 收集执行期读写，包括 tuple result、mutating operation、base allocation 和已知
   byte range；
3. 将 access 映射到抽象 source/destination route 与执行资源；
4. 为每个 allocation 构造 terminal-access 和 initial-write frontier；
5. 扫描一个 address space 内所有 lifetime-compatible buffer pair；
6. 记录 distance-zero 与 distance-one WAR/WAW handoff，并保留 route、range、loop、
   control-flow 和 ordering provenance。

raw record 位于 `metadata.recognized_reuse_candidate_records_v4`。SSA 可达记录带有
确定性的 region/statement `dag_path`，无序记录使用 `dag_path=none`。可用：

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
- 尚无实验支持负优化权重；同步减少没有产生可重复的 latency 收益。

最新 exact ordered-pair study 给出了对旧 v4 policy 的两个重要反例：

| Pair 类型 | 结果 |
| --- | --- |
| SSA-ordered `V -> MTE2` WAR | overlap 增加一个匹配的 PTOAS handoff |
| unordered `M -> MTE1` WAR | overlap 删除冗余 handoff，但没有确认 latency 收益 |
| 另外四个匹配 pair | 同步不变 |

因此 v5：

- 将 SSA `dag_path` 保留为 provenance，而不是 suppression predicate；
- 纳入 distance-one candidate，而不是抑制所有 loop-carried handoff；
- 仍把所有已构造 edge 当作尚未校准的 unit obligation。

实验也说明，给所有 cross-resource obligation 相同的正性能权重仍然过于宽泛：

- 不能给所有改变同步的 pair 正权重；
- 非负 pair model 仍然成立：neutral 或表面 beneficial 的 pair 可以不产生正 edge。
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
| --- | --- |
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
dominant pair，而不是重复求和。OR group、hyperedge、负权重和全局 event-budget
term 继续推迟。

## Critical-path model v0

下一层模型使用 schedule graph，而不是再增加 recognizer filter。PTOAS diagnostic
exporter 记录 post-InsertSync operation node、每条 pipe 的 issue-order edge、
synchronization edge、loop membership 与已知 allocation size。
`pypto.tools.dsa_schedule_model` 为 operation 赋予 duration，并计算两个有向无环图：

- 只包含每条 pipe stream edge 的 baseline；
- 加入非 loop-carried synchronization edge 后的完整图。

版本 0 把两者最长路径之差报告为 `synchronization_exposure_cycles`，并相对 baseline
critical path 评估每条 synchronization edge。这与简单统计 synchronization group
不同：如果一条 edge 的延迟已经被 critical path 上的工作覆盖，它的 exposure 为零。

在 whole-function DAG 中，静态 loop 按 trip count 聚合。由于模型没有可靠的
multiplier，动态 loop 会直接失败。loop-carried synchronization edge 会从该 DAG 中排除
并显式报告；candidate 评分会用下文的 recurrence lower bound 单独处理它们。operation
duration 由 provider snapshot 提供；
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

首个 critical-path 校准子集使用 `static_loop_v1` eligibility policy。它接受具有已导出
非负静态 trip count 的 loop，并排除 branch 与动态边界 loop。该过滤器只使用结构信息，
不读取 solver objective 或既有设备结果。在冻结该分析子集前使用：

```bash
python -m pypto.tools.dsa_schedule_model qualify schedule-*.jsonl \
    -o schedule-eligibility.json
```

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
复制到 schedule graph。若 site 缺失，或 candidate route 没有经过验证的
PTOAS pipe mapping，连接会直接失败；不会把 SSA node number 或源码行号误当作
同一坐标。

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
