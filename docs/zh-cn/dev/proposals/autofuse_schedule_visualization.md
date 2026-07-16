# AutoFuse 调度可视化

## 目的

AutoFuse 提供两种互补的 Graphviz 视图：

1. 带融合分区的张量/算子 DAG；
2. 每个已选择的同构 vector 或 cube kernel 的算法时间线。

分区视图回答“哪些算子被融合”，算法视图回答“一个逻辑工作单元实际执行什么”。算法图
不会展开所有 SPMD block；标题记录完整网格，主体只展示一个由求解器定义的 region/tile
及其内部流式循环。

## 生成视图

编译时导出问题与最终解：

```bash
mkdir -p build/autofuse-dump
PYPTO_AUTOFUSE_DUMP=build/autofuse-dump \
PYPTO_AUTOFUSE_GENERIC_EMIT=1 \
python -m pytest tests/ut/ir/transforms/test_auto_fuse.py -k case_name -q
```

若函数名为 `kernel`，生成分区图和指定 kernel 的算法图：

```bash
python 3rdparty/pto-fusebox/scripts/visualize.py solution \
  build/autofuse-dump/kernel.dag.json \
  build/autofuse-dump/kernel.sol.json \
  build/autofuse-dump/kernel.dot

python 3rdparty/pto-fusebox/scripts/visualize.py algorithm \
  build/autofuse-dump/kernel.dag.json \
  build/autofuse-dump/kernel.sol.json 0 \
  build/autofuse-dump/kernel-kernel-0.dot

dot -Tpng build/autofuse-dump/kernel.dot \
  -o build/autofuse-dump/kernel.png
dot -Tpng build/autofuse-dump/kernel-kernel-0.dot \
  -o build/autofuse-dump/kernel-kernel-0.png
```

`algorithms` 模式会为每个同构 vector/cube step 生成一个 DOT 文件：

```bash
python 3rdparty/pto-fusebox/scripts/visualize.py algorithms \
  build/autofuse-dump/kernel.dag.json \
  build/autofuse-dump/kernel.sol.json \
  build/autofuse-dump/kernel
```

Fusebox 的 `scripts/render_fusion_dag.sh` 可一次生成分区 PNG 和全部逐 kernel 算法 PNG。

## 分区视图

算子仅通过所选 kernel 的颜色分组，不再使用会改变 DAG 布局的 cluster 外框。紧凑图例将
颜色映射到 kernel 类型、最大 region、网格、split、核心数和模型延迟。张量会区分 GM
可见值与 kernel 内部片上值。边采用正交、端口锚定和先于节点绘制的方式，避免箭头覆盖
算子标签。

## 逐 kernel 算法视图

Vector 与 cube 视图采用不同布局，因为两者描述的算法结构不同。DOT 仍是交换格式，但
cube 渲染器使用 Graphviz 绘制以 tile 为中心的流图，而不是把流水阶段错误地画成算子
依赖 DAG。

Vector 时间线仍是有序事件/存活性视图。每一行包含一个动作以及该动作完成后的片上状态。
它直接来自 `VectorStreamPlan`，展示：

- 一个逻辑 region 及其物理 UB 分配；
- strip driver 或流式 statistics/apply 阶段；
- 位于 stage-2 循环之外的串行 init、ragged tail 和 finalize；
- P4 在线统计生成的工作；
- 按求解器 pebbling/拓扑顺序执行的源算子；
- 边界加载、中间值最后一次使用后的释放以及 GM 存储。

Cube 调度采用以 tile 为中心的流图，来自 `CubeSchedulePlan` 及其共享的 `L0MatmulPlan`
子计划。每个 matmul request 展示 output-tile 循环的一个代表性迭代：

- 外层 output-tile 循环以及所有 full/tail tile 变体；
- 上方 K-slice 操作数 tile 依次经过 fill、首次重叠、重复稳态和流水 drain；
- 下方表示**同一个 output tile** 驻留在 L0C 中，并由连续 K slice 的 Matrix 操作逐步累加；
- 只有真正更新 output tile 的阶段才绘制虚线 Matrix 更新箭头；纯 feed 的 fill 阶段不会
  错误地显示为修改了 C；
- request 标题概括子 L0 调度，每次 Matrix 更新执行该调度；
- 存在 ragged K 时，位于 stage-2 ring 之外的串行 tail；
- output tile 完成后的唯一最终 FIXPIPE drain；
- 递归 matmul request 之间的 L1 结果保留与最后使用后释放。

可选的 split-K 零值 seed 会显示为独立的 AIV prologue。其 work-unit 数量表示 UB 安全 seed
store 的数量，而不是 cube spatial region 的数量。

解序列化器只为最终选择的 step 重建这些描述符，它们不会进入局部搜索的 `CostResult`
缓存。缺少 `vector_stream` 或 `cube_schedule` 的旧 `.sol.json` 必须重新生成后才能绘制算法图。
