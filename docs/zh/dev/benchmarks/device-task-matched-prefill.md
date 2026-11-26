# 910B2 设备任务：Qwen 预填充投影的配对测量

测量持久化 GEMM 的加速能否在未经修改的 Qwen 预填充投影和完整模型中复现。矩阵正确性单独验证。工具和预期指纹均位于 PyPTO 仓库。

## 版本和环境

从 <https://github.com/tonibohnlein/pypto.git> 获取 `codex/pypto-autosync-clean`。将 CURRENT 指向交付版本 `373c49ab82edda3aebf4ed55b01565cf29ad4e98`，BEFORE 指向独立的 `03f912b1ab737a83255b6cf32eecfd66d5f7d765` 检出目录。分别构建两个版本的 Python 扩展。CURRENT 的编译行为对应 7178c3bc2。各组使用 Run J 的同一套工具：上游 hw-native-sys/pypto-lib `182990263ee87ef29c835ead20a716e3f4aa3d19`、Simpler `a5349f42c71666bb8cacfb3759f34eeda1572beb`、PTO-ISA `45435c0624c1d0d056bf38aa60d4f56dc0ef3b6b` 和 PTOAS 0.77。记录 CANN、驱动、设备、仓库版本和实际加载的扩展路径。

将 LIBRARY 设为模型检出目录的绝对路径，OUT 设为新的磁盘结果目录，CARD 设为分配的设备编号。使用可用的 910B2 运行环境。PTOAS 超时设为 3600 秒，队列时限须覆盖原版预填充编译。为每组和每个版本使用独立的构建及缓存目录，命令失败即停止。

```bash
set -eu
export LIBRARY PYPTO_PTOAS_TIMEOUT_SECONDS=3600
TOOLS="$CURRENT/tests/benchmarks/prefill_sync/matched"
mkdir -p "$OUT"
# Select the Python environment built for the argument before switching checkouts
# if the two extensions were built with different Python interpreters.
use_tree() {
    cd "$1"
    export PYTHONPATH="$1/python:$1/runtime/python:$1/tests/ut/ir:$1:$LIBRARY"
    python -c 'from pypto import pypto_core; print(pypto_core.__file__)'
}
for arm in current before stock; do
    mode=pypto
    flags=(--audit-ptoas-insertion --require-pypto --require-domain-owners)
    if [[ $arm == current ]]; then use_tree "$CURRENT"; else use_tree "$BEFORE"; fi
    if [[ $arm == stock ]]; then mode=ptoas; flags=(); fi
    python -m transforms.qualify_qwen_whole_model --library "$LIBRARY" --output "$OUT/$arm" \
        --entry prefill --layers 40 --batch 1 --sequence 128 --mode "$mode" \
        --policy conservative_prefix --gm-alias-policy may-alias --scalar-gm-dma "${flags[@]}"
done
use_tree "$CURRENT"
for arm in stock pypto disjoint; do
    python "$TOOLS/compile_control.py" "$arm" "$OUT/prepared/persistent/$arm"
done
python "$TOOLS/prepare.py" "$OUT/prepared" --current "$OUT/current" --before "$OUT/before" --stock "$OUT/stock"
bash "$TOOLS/run-device.sh" "$OUT/matrix" "$CARD"
```

核对准备结果与发布的清单。两个 PyPTO 模型版本之间只有 gate_up_proj.cpp 应有变化。矩阵分别运行 48 行和 80 行，每种形状每组启动 100 次，交替输入，输出预填 NaN，核对 C++ 指纹，逐值精确比较并保存四份记录。

## 三类工作负载

1. **持久化 GEMM** 使用相同输入运行 current、stock 和 disjoint。current 使用与 Qwen 相同的 may-alias 策略，disjoint 复现历史阳性对照；分别报告加速比。
2. **投影工作核** 对 gate_up_proj 和 down_proj 的四个 band 分别运行 current、before 和 stock。gate/up 启动 24 个逻辑工作核，down 启动 20 个。gate 和 up 各覆盖 17 个输出分块。在 band j 中，逻辑块 b 满足 `(b+6j) mod 24 < 17` 时执行 gate，满足 `(b+6j+17) mod 24 < 17` 时执行 up。若可观测逻辑块编号，应区分 10 个执行两次投影的工作核和 14 个只执行一次的工作核。单凭物理核编号无法分类。down 的 before/current 是相同代码对照。
3. **完整预填充** 在相同种子和输入下比较两层模型的 first128 和 first256，运行 current、stock 及使用 current 同一二进制的对照。before 使用 BEFORE 编译器和相同测量程序。保存完整的内核名称映射和依赖图，核对记录完整性与内核归属。分别报告各 band 的 gate/up、down 耗时和完整模型总耗时，重建每种形状的实际关键路径。若工具支持，记录 Cube 活动和传输停顿。

基准准备完成后可用以下命令运行：

```bash
python "$TOOLS/run_leaf.py" persistent current --root "$OUT/prepared" --device "$CARD" --seed 4107
python "$TOOLS/run_leaf.py" gate_up_proj before --root "$OUT/prepared" --band 0 --device "$CARD" --capture
python "$TOOLS/run_leaf.py" down_proj stock --root "$OUT/prepared" --band 3 --device "$CARD" --launches 5
python "$TOOLS/qwen_harness.py" --entry prefill --case first128 --arm B --label current \
    --device "$CARD" --seed 4107 --capture --out "$OUT/first128-current-smoke"
```

完整模型测量中 B 代表 PyPTO，R 代表 stock。每次构建使用所选编译器。使用种子 4107 和 4111，独立内核逐值精确比较，两层预填充使用已公布的 5e-3 容差。失败时停止该项计时，保存输入、输出和记录。全深度数值验证为可选项，判据须在执行前记录。

## 计时与交付

先试跑一个可以归属到具体内核的测量块。核对每组和同一二进制对照实际加载的原生指令哈希。每块使用同一卡，平衡各组运行顺序，丢弃预热，每次采集使用独立目录。从泳道中提取设备任务时间，不计编译、参考计算、输入暂存和进程启动。持久化对照和 gate/up 的 band 0 目标为 24 个完整配对块；其余 band、down 和两种完整模型输入至少各 12 个平衡块。分别比较 gate/up 的 before/current 和 current/stock。报告配对中位差、比值、按块自助抽样的 95% 置信区间和同一二进制对照。与对照波动相当的变化记为无法区分。

提交一份报告及带校验和的归档，包括指纹、正确性记录、原始测量块、完整跟踪和命令。排除权重及缓存；清理失败或跳过采集时暂存的输入，列明未测指标。
