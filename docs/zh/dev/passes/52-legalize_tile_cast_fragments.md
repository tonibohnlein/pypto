# LegalizeTileCastFragments Pass

在物理 tile 布局与存储规划完成后，物化目标相关的原生 cast 宽度限制。

## 为什么是后置 Pass

`LegalizeTileCast` 判断数据类型转换是否原生。后端还可通过
`BackendHandler::GetTcvtSafeFragmentWidth(src, dst)` 限制该原生指令的物理列宽。
在源和目标父缓冲区的行跨度尚未确定时，无法安全实现此限制。

因此默认流水线在 `InsertCommFence` 之后、`MaterializeValidShapeSymbols` 之前运行
`LegalizeTileCastFragments`。此时相关 tile 的布局、内存空间和分配标识均已确定。

## 改写

对不安全的 `tile.cast`，本 Pass：

1. 保留原目标分配及其 `valid_shape`；
2. 在行循环外一次性计算各分片的有效宽度；
3. 只遍历有效行，并保护运行时为空的分片；
4. 为源和目标父缓冲区建立零拷贝 `tile.slice` 视图；
5. 用内部的目标传入式 `tile.cast_fragment` 写入每对视图。

`tile.slice` 直接降级为 `pto.subview`，因此源和目标的父缓冲区行跨度均被保留。
分片视图不拥有 MemRef，也不分配紧密临时缓冲区。`tile.cast_fragment` 机械地
一对一降级为 `pto.tcvt`；PTO 代码生成不选择分片宽度、不插入循环，也不重组结果。

Ascend910B 对 `INT32→FP16` 和 `FP16→INT8` 使用 128 元素的修复粒度。例如，
物理宽度 224 会按每个有效行发出 `128 + 96`；物理宽度 256、有效宽度 129 的
填充 frame 会保留父行跨度，并发出有效宽度 `128 + 1`。不超过 128 的完整无填充
frame，以及完全由 128 元素对齐分片组成的完整 frame，保留原单条 `tile.cast`。

`valid_shape` 之外的存储内容未定义。运行时测试只比较逻辑矩形，并覆盖 128 元素
边界两侧的剩余宽度。

## API

| C++ | Python |
| --- | ------ |
| `pass::LegalizeTileCastFragments()` | `passes.legalize_tile_cast_fragments()` |

`tile.cast_fragment` 仅供编译器内部使用。用户程序仍使用 `pl.cast` / `pl.tile.cast`。

## 另请参阅

- [17-legalize_tile_cast.md](17-legalize_tile_cast.md) — 数据类型对合法化
- [53-materialize_valid_shape_symbols.md](53-materialize_valid_shape_symbols.md) — 默认流水线中的下一个 Pass
- [00-pass_manager.md](00-pass_manager.md) — 默认顺序与 Pass 属性
