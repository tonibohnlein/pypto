# 紧凑同步控制

本检查点减少现有物理发射器中的重复控制代码，同时保持有序命令轨迹和物理 ID。
它不重新计算需求，不删除 SSA token 的实际使用，也不改变分配算法。
直接诊断路径和生产 token 降低路径共用该实现。

## 其他框架提供的参考

[AIR 的并发流水线](https://github.com/Xilinx/mlir-air/blob/main/docs/AIRAsyncConcurrency.md)
将依赖提取、传递约简与操作规范化分开。依赖边约简不能直接证明可以删除已经分配的
消费型 SET/WAIT 对。
[MLIR SCF 的 `CombineIfs`](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SCF/IR/SCF.cpp)
合并条件相同的连续分支，并保持分支体的顺序，是本次前缀共享的直接参考。
[MLIR CSE](https://mlir.llvm.org/docs/Passes/#-cse)依赖内存效果证据；
[规范化](https://mlir.llvm.org/docs/Canonicalization/)属于尽力优化，正确性不能依赖它。
这里复用 PyPTO 的结构比较和结构化 IR，没有引入 AIR/MLIR 依赖或新后端。

一般表达式 CSE、循环不变量外提和 token 依赖约简仍是独立工作。本次不推测执行条件、
不外提计算，也不移动同步命令跨越载荷。

## 诊断、构造与有效性

Qwen gate/up 的 H=5120、band 2 用例原有 362 个物理分支，但只有 29 种不同的条件
表达式，每个最多七个节点。主要问题是已经完成资格验证的 case 之间重复构造激活条件
和端点测试。有限 matmul、Vec 和 softmax 原本没有同步条件；旋转缓冲区还会独立测试
每个物理 ID 的余数。

`physical_sync_realization.cpp` 实现三个局部变化：

1. 在原始载荷之前、之后的各个切点保留完整命令序列及闭合 block 区间。只有数值相邻、
   完整序列结构相等的区间才合并；自由 SSA 变量按身份比较。遗漏的 case 保持为空洞。
   合并之后再生成激活条件。
2. 使用保持顺序的可变树，共享发射器生成的命令森林中连续相等的条件前缀。叶节点顺序
   和重数不变，带 else 的 ID 决策树作为不透明节点，不跨越原始控制或载荷边界。
3. 字面 ID 选择使用 `if`/`else if`。最后一个相等测试仍保留，因此不需要新的穷尽范围
   前提。所有余数运算仍处于原有端点存在条件之内。

结构比较不可用时保持原始表示，不形成新的编译拒绝。没有内核名称特判、提高限制、
循环展开或替代着色。测试入口 `share_sync_command_guards` 仅用于符合发射器契约的
纯坐标条件和物理命令，不是任意 IR 优化 pass。

有效性理由：每个允许的调用原本只有一个 case 生效。相邻域上的相同完整序列在其并集
上产生同一有序轨迹。按有序树归纳，共享前缀不会改变活动叶节点；条件读取已就绪且
不可变的原始坐标，命令不重定义这些值。同名变量不一定相同，共同活动的重复命令也
不能去重。余数相等测试互斥，命中后跳过后续测试不会遗漏活动命令。因此原有模型中
的载荷顺序、匹配和因果 ID 复用保证保持不变。原生指令和返回边界仍需独立资格证明。

构造扫描输入控制节点并一次性冻结输出。只比较相邻完整 case，不做全配对或不断复制
增长的命令体。结构比较遍历单独计费，不宣称仅按不同 DAG 节点计费。
`control_statement_visits` 只计输入森林检查，不包含比较和冻结。
`constructed_guard_sites`、`shared_guard_sites` 是逐 case 构造计数；`case_sites`
位于跨 case 合并之前。顶层计数来自最终 IR；`if_sites` 包含原始载荷控制，
`guard_sites` 不包含它。

## 结果

基线为 `3f437376e4188a31e5ddf5dc5e284dc99a9d742b`。
[机器可读记录](../../../en/dev/ir/compact-sync-control.json)固定了脏源码、扩展、输入、
命令和有界结果。28 个输入哈希全部与基线一致。

| 用例 | 物理 IR 分支：之前 → 之后 | PTO 算术：之前 → 之后 |
| ---- | ------------------------- | --------------------- |
| Matmul / accumulation / Vec / softmax | 0 → 0 | 不变 |
| DDR split-K | 19 → 18 | 50 → 48 |
| 旋转缓冲区，7 次访问 | 20 → 20 | 76 → 76 |
| 两个旋转区域 | 36 → 36 | 111 → 111 |
| 嵌套 / collective / 库 GEMM | 不变 | 不变 |
| Qwen output H=5120 | 27 → 27 | 93 → 93 |
| Qwen gate/up band 0 | 237 → 187 | 547 → 419 |
| Qwen gate/up band 1 | 292 → 242 | 691 → 563 |
| Qwen gate/up band 2 | 362 → 312 | 742 → 614 |
| Qwen gate/up band 3 | 292 → 242 | 693 → 565 |

245 次有界执行的载荷/命令有序轨迹完全一致，包括方向、ID 和载荷位置。其中九次另外
运行独立的原始访问、闭包、覆盖边、匹配和复用检查器。更广的测试门还检查小型 Qwen
调用边界及两个 layer。大型 Qwen 控制轨迹比较本身不是独立访问证明。

band 2 的双 tile 执行由 7,364 次条件测试降至 6,562，假条件由 4,568 降至 3,766；
两个版本仍各执行 719 次 SET 和 WAIT。24 个 block 的条件总数由 123,368 降至
109,734（11.1%）。旋转 7 次访问由 122 降至 80 次条件；70 次由 1,382 降至 962。
`condition_nodes` 是到达条件中的语法节点出现数，包含可能被短路的操作数，不是指令数。

28 对新旧输出均成功生成 C++，下游 InsertSync 保持启用。band 2 的最终 C++ 分支仍由
362 降至 312，源码由 57,869 降至 53,460 字节。静态 SET/WAIT 从 112/112 变为
99/94，因为发布和消费切点共享 case 的方式不同；动态匹配不变，静态数量不必相等。

这是源码和控制改善，不是设备加速结果。一次生产计时中 band 2 从约 12 ms 变为
16 ms，规范化增加了主机构造工作，不宣称编译加速。所有比较保持相同协议和 ID 池。
旋转计数增加到 70、7,000、1,000,000，嵌套计数增加到 (1,000,000, 1,000,000) 时，
静态表示大小不增长。零次旋转用例保持原有事务式委托，不宣称新增空所有权支持。

完整测试门通过 **1,284** 项测试（36 个弃用警告）。格式化后重建的条件和结构比较测试
通过 **168/168**。架构、正确性和算法审阅均接受实现。修复测试类型收窄及翻译表格对齐
后 hooks 通过；现有 Python 3.11 hook 环境支持固定 torch 依赖，运行测试使用 Python 3.14 扩展。

## 复现与限制

使用 Linux 测试环境、单工作线程构建/测试、固定 PTOAS v0.72 和在 PATH 中供其启动器
使用的 Python 3.11。本次扩展使用 Python 3.14。不要加载旧 macOS `env.sh`。

```bash
source .claude/skills/testing/load-env.sh
cmake --build build-py314 --parallel 1
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYPTO_CODEGEN_MAX_WORKERS=1 /usr/bin/python3 -m pytest -q \
  tests/ut/ir/transforms/test_sync_command_control.py
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYPTO_CODEGEN_MAX_WORKERS=1 /usr/bin/python3 -m \
  tests.ut.ir.transforms.qualify_sync_control \
  --baseline build-py314/guard-audit --output build-py314/guard-control/final --emit-cpp
```

基线目录包含修改前固定的 `.msgpack`。重建时用相同驱动、基线扩展、不传 `--baseline`，
写入另一个目录。驱动和指标助手可配合旧 Python/扩展树使用。记录将普通生产计时与
额外调用的只读数学/物理诊断分开。

完整序列不同时，仍有重复 case 条件和算术。跨 case 共享部分序列或外提不变谓词需要
单独的顺序、可用性证明和成本检查。本次建立通用局部表示优化，不改变论文的需求、
实现或分配保证。设备计时和原生入口/返回边界前提仍不在本次主机验证范围内。
