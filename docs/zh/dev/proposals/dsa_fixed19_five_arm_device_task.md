# 设备任务：固定 19 个工作负载、五种 DSA 算法对比

## 状态：草案——尚不可派发

第五种算法尚未实现；当前只有完整布局 DAG 评分器，没有以延迟为目标的布局搜索。
发布包通过源代码门禁之前，不要配置或构建设备活动，不要启动设备。
缺少算法应报告 SOURCE_GATE_BLOCKED，而不是语料或时长覆盖率不足。
设备代理不得现场实现算法。英文同名文档为权威协议。

## 目标与五个实验组

| 实验组 | 布局目标 |
| ------ | -------- |
| geometry_ff | 几何首次适配 |
| geometry_cg | 几何 canonical greedy |
| cypress | 冻结的、不依赖延迟或惩罚权重的约束放松组合 |
| dsa_rp_structural | 使用结构性复用惩罚的现有 canonical greedy |
| dsa_rp_latency | 使用完整布局 DAG 增量代价的实验 canonical greedy |

延迟目标为 `p_w(P) = LP(G0 + E_reuse(P), w) - LP(G0)`。
从无地址复用的 SSA/流水线依赖图出发，同时添加所有布局引入的定向依赖，
并赋予同步权重；不得将独立缓冲对的最长路径增量简单求和。
搜索边际为 `p_w(P_candidate) - p_w(P_current)`，
部分布局的求值规则必须冻结；保留所有生命周期、别名、容量及对齐硬约束。

这是开发集比较，不是未见工作负载的留出集。新测量不会恢复分析者盲性。
不扩展语料，不改变容量，不拟合参数，不开展模拟器、指令追踪或因果消融活动。

## 0. 派发前必须在本地完成的发布包

负责人必须提交、推送并发布：

- PyPTO 分析工具、实验 planner/dsa-solver、研究 PTOAS 的完整 SHA，
  以及已验证命令。不得替换为分支尖端、未推送代码、虚构参数或私有函数绕行。
- 模型 JSON：时长来源哈希、证据分类、一个全局同步周期权重、循环/分支与
  递归语义、部分布局求值、决胜规则和父程序组合策略。不得逐内核调参。
  固定的通用近似不能标为签名校准；缺失时长和来源不能默认为零。
- 19 行结构清单、所有子问题、完整逐池容量向量、五组完整 replay map、
  独立验证结果、基线身份及模型适用性。
- 测试证明延迟目标真正参与搜索，组合边相互作用被考虑，缓存结果等于合法
  布局上的完整重算。对结构算法输出事后评分不构成第五种算法。
- 含实际问题、布局和复现源码的输入压缩包、SHA-256 sidecar 和内部清单。
- 发布 JSON，固定全部文件及重构、编译、验证、冻结、测量命令。
  派发版任务必须写出压缩包名称和完整 SHA-256。

本草案不填写猜测的发布值。远程可获取性和小型端到端主机冒烟通过后，
负责人才能标记 READY。缺少发布包或第五算法，立即 SOURCE_GATE_BLOCKED，
不要开展数日活动来重新发现这个事实。

## 1. 固定工作负载和端点

权威清单为 PyPTO 工具提交
`11c70802312700e18476ae268d6b5569f7400a0b` 的
`docs/en/dev/proposals/data/current-paper-development/primary-manifest.json`。
恰好 19 个 driver/argument 工作负载：13 个单函数驱动窗口和 6 个父程序窗口。
`paper-primary.tsv` 仅提供已有测量模式，不重新选样。
不同 target 或容量不是独立工作负载。

保持每行已选容量及完整向量：历史方案只压缩目标池，其他子问题/池为 native。
不得改成全部池同时压缩。其他容量及最新四个扩展工作负载不在本任务范围。
不得替换困难、无差异或退化的行。

| 组件 | 固定版本 |
| ---- | -------- |
| PyPTO | `3eabcfd22894151cbda6c752dcb300708a34f28d` |
| PyPTO-Lib | `83e19f6b06eb2125eb14c06232358f342941c8d5` |
| runtime | `4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0` |
| PTO-ISA | `a8040450238f162985d8b596fbebeb54bfba2bf5` |
| 原有 solver | `60ac39b02b008dc3907b876ad47a1cc342aa01fa` |
| 产品 PTOAS | 官方 v0.57 wheel，SHA-256 `4858c837e12b1b588f281207c95916dc20968a7cdc656aadd549a87658a06692` |

实验搜索可在独立工具中生成 replay map；设备代码所有组必须采用相同产品工具链。
任何产品编译器修改均需新发布包及全部组的共同版本重生成。
使用原始驱动、确定性输入和其 Torch golden。
父程序的每个子函数都由相应算法放置，不采用 target-only 替换。
第五组不得将未覆盖的子函数偷偷回退为其他算法；平凡子函数可证明零目标。
逐函数代价与父程序预测分开：求和不等于父程序关键路径；可分离父级搜索目标
必须明确冻结并标注。

## 2. 主机、源码和 stock 门禁

使用前验证压缩包；记录版本、remote、干净状态、子模块、二进制与模型哈希、
CANN/编译器、设备型号、命令和环境。复用同版本已验证环境。
按需使用 in-core profiling skill，但本协议优先于旧的三/四组示例。

95 个工作负载/算法槽位必须都有主机终态。独立验证完整 map，包括零惩罚
子问题及 native replay 合法性。既有组重建必须匹配冻结的问题语义和 map 摘要。
只规范化已证明非语义的 SSA 名称，不把 JSON 字节漂移误判为求解问题改变。
fingerprint retarget 只能修改元数据，并验证原生和收紧两种容量。

记录 Cypress 组合、选中变体、alias、放松边和尝试次数。结构与延迟 CG 使用
发布包规定的相同顺序、种子、重启及预算；必要差异明确披露。
求解时间和评估次数单独报告，不计入设备延迟。

每个工作负载先运行 stock，使用固定输入重复验证。
stock golden/执行失败或本征非确定性仅阻塞该工作负载，保留终态，
不能修改容差或筛选种子。实现存在但模型不适用或第五组无合法布局时，
报告具体状态，不声称完整五组结果。继续有效五组工作负载，
不要重跑一个缺少第五组的四组活动；历史基线数据单列。

## 3. Replay、可比性与正确性

相同完整 map 只编译一个物理端点，保留五个逻辑标签。
不同 map 共享执行需独立证明可执行文件和 ABI 等价；
非稳定源码计数器不是语义差异，也不是充分等价证明。

逐端点证明 replay 消费、逐函数布局匹配、分配实际发出或有合法 view/
晚期消除来源；仅地址包含不足以证明 replay。
函数、内核、submit、ABI、标量、block 几何以及规范化 pre-InsertSync
操作流跨组一致。递归发现产物，拒绝空清单、零分母和遗漏子函数。
产品 InsertSync 必须启用，保存真实 post-InsertSync 摘要。
研究导出器使用前证明产品一致性；官方 v0.57 无研究图 CLI。

负控制必须拒绝操作、dtype、shape、语义标量、遗漏解、错误实验组、
错误函数/图身份和非法布局；地址规范化不得隐藏非地址常量。

每个物理端点在两台设备各运行三次，每次还原相同输入和标量，
验证原始全部 golden、组内确定性及跨组全部输出 bit identity。
若原始契约允许数值波动须明确报告，不得暗改确定性面板要求。
放置正确性失败停止该工作负载并保存小型复现；先排除共享基础设施故障，
确认 harness 可信后才继续其他工作负载。

## 4. 冻结后测量设备 wall time

首次 timing 前冻结 19 行状态、五组 map/二进制、容量、适用性、模型、
结构/DAG 预测、种子、设备、测量模式和顺序；两次生成语义字节一致。
不得利用历史延迟修改选择。既有数据始终标为开发数据。

两台安静兼容设备，各一条 timing lane，独立 build/run/output 目录，
不得共享可写 dfx_outputs。构建显式指定适合远程机器的 worker 数，
禁止无界并行；复用构建，逐端点清理。

同一行各组测量相同驱动的 device window；单函数与父程序分层。
不计编译/golden 主机时间，不求和 per-core 时间，不单测 mixed group 一半，
不按 dispatch 数相除。确认设备域样本非零、无 host fallback、
span 名称正确，timing 二进制就是已验证二进制。

使用验证过的 Williams/counterbalanced 顺序，按物理端点数生成完整设计。
每设备至少八个完整 block，并上取整至完整设计周期；
五个端点用奇数组的十序列设计。机械检查位置和有向 carryover 计数，
设备二 counterbalance。每次端点出现有三次 warmup、二十次计时样本。
不得为追求显著性追加运行。

加入一个同二进制双标签控制；冻结输入中已有时，加入小型已知非零仪器控制。
完整性失败只隔离并记录原因后重跑受影响 block；不能删除有效慢样本。
不再扫容量或收集 chip trace，保存逐端点同步摘要及可执行文件哈希。

## 5. 分析与结论

逐工作负载、逐设备报告十个逻辑成对比较、中位数、ratio、
`100 * (candidate / reference - 1)` 和配对 block bootstrap CI；正值表示变慢。
同端点为 PHYSICAL_NULL，不是独立观测。设备不可混池，block 内样本不可
当独立 launch。确认效果要求两设备同方向、至少 2%、95% CI 排除零；
同时报告连续结果、小效果和方向不一致。单函数与父程序分别聚合，
所有比较用相同覆盖集合。

重点比较延迟 DSA-RP 与结构 DSA-RP、两者与 Cypress、两者与几何 FF；
几何 CG 是搜索控制。预测评估保留错向、遗漏、tie、缺失。
DAG tie 不是校准后的设备 null，严格周期排序也不是百分比预测。

- FIVE_ARM_TIMING_COMPLETE：19/19，五组、两设备全部通过。
- FIVE_ARM_TIMING_PARTIAL：至少一个有效五组行，其余全部有具体终态。
- NO_FIVE_ARM_TIMING：无完整五组测量，解释原因。
- SOURCE_GATE_BLOCKED：缺少已发布的第五组实现或发布包。
- PROVISIONING_BLOCKED / HARNESS_INTEGRITY_BLOCKED：分别表示输入不可得或测量证明不可信。

科学结果独立于完成状态，延迟算法可赢、平或退化。
事后 DAG 评分不能把结构组重新命名为延迟组；相同新 map 是合法 null。

## 6. 交付与存储

根目录 `/opt/dsa-rp-fixed19-five-arm/`，保留输入包，逐工作负载 checkpoint，
定期报告状态和存储。先用两个主机合格工作负载验证 harness，再完成固定面板，
不能按运行结果选样。

保留实际问题/maps、驱动/模型/发布源码、命令、紧凑来源和同步摘要、
预测、原始样本与终态。验证证据后删除构建和通过样本的 tensor dump。
不保留大 args.bin 或模拟器输出。

交付 REPORT.md、HANDOFF.md、PINS.md、freeze、语义哈希、复现命令、adapters，
以及 workload-status、model-coverage、planner-settings/costs、map-identities、
placement-validation、replay-provenance、comparability、negative-controls、
correctness、post-insertsync-summary、frozen-predictions、timing-raw、
timing-blocks、instrument-controls、pairwise-results、predictor-evaluation 表。

显式 allowlist 逐文件暂存后生成
`/opt/pypto/dsa-rp-fixed19-five-arm-final.tar.gz`、sidecar 和内部清单。
普通证据 <=256 MiB，普通单文件 <=25 MiB，单独声明失败复现 <=128 MiB，
总量 <=384 MiB。不得包含 symlink、构建、二进制、缓存、venv 或通过的 tensor。
超限报告 ARCHIVE_SIZE_BLOCKED，不能压缩整个根目录。
重新解压，验证精确清单覆盖和全部哈希；结束时设备空闲、无活动进程、源码干净。
