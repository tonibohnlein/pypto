# Run L · 910B2 上的 Qwen prefill 接收槽释放与合并

分别测量三项结果：提前释放 AIC 接收槽对部分注意力耗时的影响、删除入口同步对 AIV 合并耗时的影响，以及完整候选版本相对 stock 的 first256 总耗时。

## 从已发布源码构建

在独立检出目录中获取 tonibohnlein/pypto 的 `codex/pypto-autosync-clean` 分支，切换至[固定提交 5d13044ee](https://github.com/tonibohnlein/pypto/commit/5d13044eebb185507f0b674c69fd7e1aa963e74e)。使用未经修改的 hw-native-sys/pypto-lib `182990263ee87ef29c835ead20a716e3f4aa3d19`、Simpler `ec5dd4fc0fa29e977b6b71a533c191439fb96ff9`、PTO-ISA `45435c0624c1d0d056bf38aa60d4f56dc0ef3b6b` 和 PTOAS **0.78**。按设备机器的常规流程构建此检出的 Python 扩展和运行时。各组使用相同的 CANN 编译器、参数和设备。记录版本、扩展加载路径和设备配置。

在编译器检出目录中用 Bash 执行。将 LIBRARY 设为上游模型的绝对路径，OUT 设为磁盘上全新的绝对目录，CARD 设为分配的 910B2。先激活项目 Python，将 PTOAS_ROOT 指向固定工具链，为 stock 编译预留足够的队列时间。

```bash
set -eu
test "$(git rev-parse HEAD)" = 5d13044eebb185507f0b674c69fd7e1aa963e74e
export LIBRARY OUT CARD PYPTO_PTOAS_TIMEOUT_SECONDS=3600 PYPTO_REBUILD_FROM_PTO=0
export PYTHONPATH="$PWD/python:$PWD/runtime/python:$PWD/tests/ut/ir:$PWD:$LIBRARY"
python -c 'from pypto import pypto_core; print(pypto_core.__file__)'
mkdir -p "$OUT"
for size in 128 256; do
  for mode in pypto ptoas; do
    flags=()
    if [[ $mode == pypto ]]; then flags=(--require-pypto --require-domain-owners --audit-ptoas-insertion); fi
    python -m transforms.qualify_qwen_whole_model --library "$LIBRARY" --output "$OUT/$size-$mode" \
      --entry prefill --batch 1 --layers 2 --sequence "$size" --mode "$mode" \
      --policy conservative_prefix --gm-alias-policy may-alias --scalar-gm-dma "${flags[@]}"
  done
  python tests/benchmarks/prefill_sync/partial_attention/prepare.py --sequence "$size" \
    --candidate "$OUT/$size-pypto" --stock "$OUT/$size-ptoas" --output "$OUT/arms/$size"
done
```

每个输入长度保存两份编译回执。候选版本应拥有 32/32 个叶函数，29/29 项下游插入审计应一致。准备脚本核对十个完整程序的指纹；若不符，先查明构建来源再计时。

## 实验组和启动验证

| 标签 | 程序目录 | 运行器 `--arm` | 含义 |
| --- | --- | --- | --- |
| A | A | B | 两项均为旧同步方案 |
| B | B | B | 仅改变读者释放 |
| M | M | B | 仅改变合并同步 |
| D | D | B | 完整候选版本 |
| Dp | D | Bp | 复用 D 的同一份原生对象 |
| R | R | R | 本次构建的 stock PTOAS 同步 |

A/B/M 由 D 按已提交的对照差异构造并校验指纹。混合单元的两个包装源文件均包含 cube 代码。对于给定输入长度，A/B/M/D 的其他源码、编排与 ABI 保持相同。实际执行的程序由目录确定。

以下命令运行一次带数值检查、可识别内核的冒烟测试：

```bash
runner=tests/benchmarks/prefill_sync/matched/qwen_harness.py
python "$runner" --entry prefill --case first256 --arm B --label D --device "$CARD" --seed 4107 \
  --runtime-dir "$OUT/arms/256/D" --capture --out "$OUT/smoke/first256-D"
```

- 对 first128 和 first256，在 A/B/M/D/R 上分别用种子 4107、4111，每个种子运行三次。每种形状另检查 Dp 一次。要求 logits 有限、按模型的 rtol=atol=0.005 通过参考值比较、每行胜出 token 相同。核对各组输入摘要；保留错误张量，停止对失败组和形状计时。
- 确认在指定卡上以 ONBOARD 模式运行，逐形状、逐组、逐次启动记录实际加载的原生代码哈希。Dp 必须加载 D 的同一对象。检查 AIC 与 AIV 实际生效的代码；若原生编译消除了源码差异，单独记录。须解释目标单元外的意外差异。
- 每组、每种形状均采集带名称的依赖轨迹。first256 预期执行部分注意力与合并；first128 是单块路由对照。按调用 token 报告实际路由、次数和两个端点。区分合并内核已调度和其累积分支实际执行两种证据。

## 计时与交付

每组、每种形状先预热。仅在同一卡上串行采集计时数据。first256 完成 **24 个六组配对块**，first128 完成 **12 个**。首行顺序为 A B R M Dp D，按字母表 A B M D Dp R 循环替换所有标签以得到其余五行。每轮用完六行，用记录的种子洗牌，每个块内保持输入种子相同。确认冒烟数据可归因后使用 `--timing`。每次启动使用独立输出目录，复用程序目录前移走刚生成的轨迹。不完整的块整块重跑。

| first256 主比较 | 每次启动的设备指标 |
| --- | --- |
| B−A | 各部分注意力调用跨度之和，按调用取两端最早开始到最晚完成 |
| M−A | 各合并调用跨度之和 |
| D−R | 首个设备任务开始至最后一个任务完成的总时间 |

报告块内差值（µs）及比值的中位数，给出按块配对 bootstrap 的逐指标 95% 区间（至少 10,000 次抽样）。每项同时报告 Dp−D。只有变化本身和配对扣除同对象对照后的区间均沿相同方向排除零，才判为已分辨。附 D−A、D−B、D−M、交互项 D−B−M+A、first128 总耗时、两个端点的 worker 分布和无法归因的结果。Run K 在早期版本上测得 first256 慢 3.28%；本次使用新测 R 进行判定。不同调用可能重叠，不能通过相加跨度把延迟归因于某一次等待。

交付简短报告及带校验和的归档，含正确性及路由回执、原始命名轨迹和依赖、实际加载哈希、目标原生对象、启动顺序、块级数据、构建命令和分析脚本。分别给出读者释放、合并、整模型差距和正确性的结论。归档排除权重和缓存。监控采集过程中的临时输入，清理失败或跳过采集产生的本任务文件，完成后确保设备和队列空闲。
