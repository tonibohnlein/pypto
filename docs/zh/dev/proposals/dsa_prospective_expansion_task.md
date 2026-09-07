# 设备任务：扩展当前开发集之外的 workload

这是 [英文任务](../../../en/dev/proposals/dsa_prospective_expansion_task.md) 的中文说明；
精确源 revision、归档文件名和 SHA-256 以英文任务中的完整表为准。
usual task 文件名为 `DEVICE_TASK_DSA_RP_DEVELOPMENT_EXPANSION.md`。

## 目标与固定输入

从 12 个 reserved candidate 构建并验证独立 driver，按结构冻结每个新 workload
的一个容量配置，在两个设备上测 geometry_ff、geometry_cg、Cypress、dsa_rp_cg。
目标为 8–12 个新 workload，不改 planner 或 duration model。
如果有界搜索后不足 8 个，仍冻结并测量有效项，报告 `EXPANSION_TIMED_PARTIAL`，
不能声称完成八项目标。模型覆盖率与 hidden-norm 调查均不阻塞设备结果。

使用已发布的 PyPTO `3eabcfd22894151cbda6c752dcb300708a34f28d`、
PyPTO-Lib `83e19f6b06eb2125eb14c06232358f342941c8d5`，以及英文 pin 表
指定的 runtime、PTO-ISA、solver 和 official PTOAS v0.57 wheel。
不要替换 branch tip；不依赖本地未提交的汇总工具。
先验证 `rebased-corpus-expansion-3eabcfd22` 和
`rebased-corpus-device-continuation-3eabcfd22` 两个归档的 sidecar 与全部 manifest。
不得通过已有 latency 选择新候选、shape、capacity 或 seed。
复用同 pin 且已证明的构建，记录 binaries/flags/imports；远程机器自行选择显式
worker 数，不重建无关 toolchain 或全部生产 parent。

## 候选与新颖性

| 原始来源 | Reserved instance | 注意事项 |
| --- | --- | --- |
| dspark `prefill_compressor_ratio4.py` | `prefill_c4_cache_write` | 可能为 structural null |
| dspark `prefill_sparse_attn.py` | `build_bias` 与三个编号副本 | 同 family，已有开发集近亲 |
| 同上 | `merge_rope_pack` 与三个编号副本 | 同 family，隔离无关 `proj_a_mm` sibling |
| pro `qkv_proj_rope.py` | `kv_proj_matmul` | 原先 map 不同但无 unit-objective gap |
| 同上 | `kv_rms_norm_rope` | 隔离无关 `qproj_matmul_aic` sibling |
| 同上 | `q_rope_prepare` | 已有开发集近亲 |

编译/计时前登记 source helper/revision/body hash、shape、dtype、tiling、launch
geometry 和 golden contract；导出后加入语义 problem 与 solver-field identity。
与已测 driver/helper/shape 相同的配置不是新 holdout；名字变化不算新 workload。
按 family 分组，分别报告熟悉 family 的新变体与新 family。
无法从结构记录证明完整历史新颖性时明确说明。

优先调用原始 inline helper；否则隔离原始 body 并记录对应关系。仅允许 wrapper、
确定性输入和独立 Torch golden 改动；保留真实模型 shape 与原运算。
不要用 parent 完成后的缓冲区作单 kernel reference。
新 source 放 campaign 内，不改共享源树；需要语义 kernel 修复的候选记 blocked。

若去重后不足八项，可在读取新 latency 前再登记**至多八个**补充 helper/shape，
优先缺失运算 family。允许真实模型 shape/tiling 变体但必须分组；仅此一轮补充，
不得看结果后换样本或无限搜索。前三个 driver 资格确定后报告进度与存储。

## Host screen 与正确性

每 driver 导出全部 child，在四种容量和四种算法下求解。
使用 profiling skill 的 `screen_dsa_capacity_corpus.py` capacity helpers，
对所有 child/pool 同时施加 profile；native 语义不变，保存完整容量向量。
沿用 frozen screener 设置，记录 seed/restarts/Cypress portfolio 与选择结果。
Cypress 选择不能读取 penalty weight 或 latency。保存 actual alias、relaxed-edge、
peak、order、seed，以便之后评估真实 objective 而非 proxy。

独立核验完整 map 对 derived/native problem 的合法性。需要 fingerprint retarget
时仅改 metadata 并记录。capacity/alignment/temporal/hard-edge 注入必须被拒绝。
先运行 stock golden；执行、golden 或内在不确定性失败不归因于 placement。

对可运行项证明 replay、全部地址/late-elimination provenance、kernel/dispatch
inventory、ABI、scalar 和 block dimensions。递归发现 artifact，拒绝空证明。
仅规范化已证明无语义的名字/地址；op、dtype、shape、常量、missing/wrong map
mutation controls 必须有效。相同完整 map 共享 physical endpoint，保留四个逻辑
算法。Parent 必须全体 child 使用同一 policy，不做 target-only 比较。

所有可行容量的不同 endpoint 先在一个设备验证；选定 primary 后，每个 endpoint
两个设备各三次，第二设备反转顺序，同输入、原 golden、完整输出比较和确定性。
placement-specific 失败停止该 workload 并保留小 reproducer；检查是否共享基础设施
问题。模型不支持不影响 device eligibility。

## 结构 freeze

仅使用正确性、feasibility、map 和 objective 记录。优先 Cypress 真实带 penalty
复用、三个主要完整 map 不同、DSA-RP objective 严格更低的容量；依次最大化
objective 差、penalized relation 差、全部 relation 差，以更紧容量作最终 tie-break。
可用时运行 skill 内 `select_dsa_workload_capacity.py`；若用 adapter，测试并保存，
不得假称运行原工具。无 opportunity 的项保留 control，不算正确性失败。

每 workload 一个 primary capacity；其他容量仅作正确性/敏感性记录。
超过十二项时按 family 多样性再按结构键选择。`cohort-frozen.json` 生成两遍必须
语义字节一致；含所有源 pins、driver hash、solver 设置、seed、容量向量、map
digest、计量模式与排除理由。计时/solver wall-clock 不得进入身份。

同时冻结 unit objective、实际 Cypress metrics、以及已经可用且 scope-matched
的 DAG bound。Official v0.57 没有 research graph CLI；不得为此新建 toolchain
或修图。缺预测在计时前标 `MODEL_UNAVAILABLE`，不删除设备 workload。
Child bound 不能当 parent prediction，也不拟合新权重或 duration。

## 计时与评价

最多两个 device lane、每设备一个 timing process。每 lane 独立 build/run/output
目录，禁止共享 `dfx_outputs`。单函数 driver 用 device effective_us；多函数必须
标 `PARENT_PROGRAM_WINDOW`，全体 child 同 policy，不除 dispatch count。
所有值非零且无 host fallback。

每 endpoint/device 八个预先平衡的 blocks，每 block 三次 warmup、二十个 sample；
第二设备反转/配平顺序。一个代表 cell 增加 same-binary label null control。
不追加大扫测、不为过阈值反复 launch；除计时完整性失败外不做 instruction profiling。

以 balanced block 为 bootstrap 单位，设备不合并，保留连续 ratio/CI/null/退化。
两个设备均达到 2% 且 CI 不含零才进入 decided 子集。预测按匹配覆盖率与全体分别
报告 missed/wrong/null/unavailable；严格 score 排序不等于校准的百分比。
分别报告 opportunity/control、熟悉/新 family；kernel/parent 不混合 geomean。

## 存储与交付

工作目录 `/opt/dsa-rp-development-expansion/`；不在 `/tmp` 放大文件。
每次 compile/run 提取证据后清理 build 和 passing tensors，保留 source、maps、
小 provenance 与 raw timing。失败 reproducer 至多一个、≤128 MiB。

交付 REPORT/HANDOFF/PINS、candidate/novelty/exclusion、feasibility/maps、correctness、
comparability/controls、结构审计、冻结 cohort/predictions 与 hash、raw timing、
pairwise 和预测评价表；包含新 driver 和必要 adapter source。
精确报告 workload/shape/family 数，不把 target 改名成独立 kernel。

显式 allowlist staging 到 `/opt/pypto/dsa-rp-development-expansion-final.tar.gz`。
压缩前限 ≤256 MiB、普通文件 ≤25 MiB，无 symlink/build/binary/cache/venv。
另带 sidecar 和内部 manifest；fresh extraction 核验完整覆盖与 hashes。
输入归档不变。只有 8–12 个真正新且已测 workload 才可报告
`PROSPECTIVE_EXPANSION_COMPLETE`；否则交付真实 partial 数据。
退出时 source clean、进程停止、设备 idle。论文和 hidden-norm 调查不互相阻塞。
