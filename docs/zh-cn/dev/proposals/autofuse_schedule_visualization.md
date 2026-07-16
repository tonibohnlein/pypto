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

每一行表示一个有序算法事件：左列是动作，右列是该动作完成后的片上存活状态。颜色固定：
蓝色表示加载，绿色表示计算，紫色表示循环/流水，黄色表示 carry，橙色表示 drain/store，
灰色表示释放。

Vector 时间线直接来自 `VectorStreamPlan`，展示：

- 一个逻辑 region 及其物理 UB 分配；
- strip driver 或流式 statistics/apply 阶段；
- 位于 stage-2 循环之外的串行 init、ragged tail 和 finalize；
- P4 在线统计生成的工作；
- 按求解器 pebbling/拓扑顺序执行的源算子；
- 边界加载、中间值最后一次使用后的释放以及 GM 存储。

Cube 时间线来自 `CubeSchedulePlan` 及其共享的 `L0MatmulPlan` 子计划，展示：

- 一个 spatial/split-K 工作单元和可选的零值 seed；
- 按执行顺序排列的递归 matmul request；
- output/L0C tile 变体；
- GM→L1 K-window 的 init、rolled overlap 和 ragged tail；
- 嵌套的 L1→L0A/L0B 加载与 `TMATMUL`/`TMATMUL_ACC`；
- 唯一的最终 FIXPIPE drain（到 L1 或 GM）；
- L1 中间值保留到模型定价的最后一个消费者后再释放。

解序列化器只为最终选择的 step 重建这些描述符，它们不会进入局部搜索的 `CostResult`
缓存。缺少 `vector_stream` 或 `cube_schedule` 的旧 `.sol.json` 必须重新生成后才能绘制算法图。
