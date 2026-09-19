# LegalizeWideGmToMatLoads Pass

当 GM 到 Mat 的加载源行跨度超过目标可直接编码的 leading dimension 时，对其进行改写。

## 为什么是后置 Pass

该限制属于目标后端，而不是 DSL。后端通过
`BackendHandler::GetMaxGmToMatRowStrideElements(dtype)` 报告此能力。本 Pass 紧随
[`InitMemRef`](34-init_memref.md) 运行；此时 tensor stride、tile 内存空间、布局和
分配标识均已确定。

在 Ascend910B 上，GM 到 L1 的 ND-to-NZ 路径最多能在源行跨度字段中编码 65,535
个元素。到 65,536 时该字段会静默回绕，Mat 操作数的每一行会别名到首行。普通 GM
加载不受影响。

## 改写

对不安全的 rank-2 ND `tile.load` 到 Mat，本 Pass：

1. 保留原目标分配和逻辑 `valid_shape`；
2. 遍历逻辑源行；
3. 使用无界 IR `index` 算术计算每一行的扁平源偏移；
4. 将原始 GM 指针重定位到该行；
5. 向目标提供紧密的 `[1, C]` ND 视图，并直接加载到对应的目标子视图。

编译器内部操作 `tile.load_rebased_row` 记录该决定。PTO 代码生成机械地将其降级为
`pto.addptr`、紧密的 `pto.make_tensor_view`、`pto.partition_view` 和 `pto.tload`。
因此原父行跨度仅用于地址算术，不会进入有界的目标字段。该改写不会增加 GM 中间值，
也不会增加密集 Mat 重组。

若目标能力查询返回 `std::nullopt`，则保留原加载；不超过目标上限的加载同样保持不变。

## API

| C++ | Python |
| --- | ------ |
| `pass::LegalizeWideGmToMatLoads()` | `passes.legalize_wide_gm_to_mat_loads()` |

`tile.load_rebased_row` 仅供编译器内部使用。用户程序仍使用 `pl.load` / `pl.tile.load`。

## 另请参阅

- [34-init_memref.md](34-init_memref.md) — 建立本 Pass 保留的分配标识
- [36-materialize_semantic_aliases.md](36-materialize_semantic_aliases.md) — 默认流水线中的下一个 Pass
- [00-pass_manager.md](00-pass_manager.md) — 默认顺序与 Pass 属性
