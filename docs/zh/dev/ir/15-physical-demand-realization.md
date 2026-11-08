# 保留需求的物理实现

这是显式启用的研究原型，消费 `CompactSyncAnalysis`，不是面向任意前端 token
程序的通用 `LowerSync`。默认 PyPTO/PTOAS 同步选择不变。英文版本是权威说明。

## 已接受的计划及上游关系

2026-11-09 的独立审查核对了草稿、仓库、GitHub 和经过认证的
[任务 3](https://taskboard.pypto.cloud/?task=5276dfa6-98d8-45c9-ad11-5df3af2f78c8)。
3.1 仍为 TODO，由 Feng Siyuan 负责；3.2 和 3.2.1 已提供标志及屏障 API。
[PR 3291](https://github.com/hw-native-sys/pypto/pull/3291) 提供逻辑 token，
物理降低仍是后续工作。[PR 3124](https://github.com/hw-native-sys/pypto/pull/3124)
和 [PR 3147](https://github.com/hw-native-sys/pypto/pull/3147) 的资源目录与宏资源
声明是输入，不能代替分配证明。没有修改上游任务或 PR。

1. 每个跨管线需求族产生独立配对的交接（handoff）族。扇出使用独立交接。
   原始坐标和谓词限定实际消费者；最终源只在最后一次访问发布，不合成终端清理。
2. 同管线 M/V 屏障必须证明紧邻的前一载荷完成本来就是目标的前置条件。
   无法证明则拒绝，不改用 ALL 屏障。
3. 有限交接通过首次复用阈值和区间着色分配 ID；复用要求消费到再次发布的因果路径。
4. 重复族使用已证明的循环槽，每槽保留首次发布到最后消费的完整区间。
   这是充分构造，不声称重复程序的最少 ID 数。
5. 使用目标目录、方向、保留 ID 和已限定宏资源。未知所有权仍不可用。
   调用者列表只能缩小合法池。余数分支发出字面量 ID，不插入虚假标志。
6. 独立检查实际命令、原始访问、逻辑实现、匹配、因果复用和全部载荷顺序。

## 接口和正确性论证

```python
from pypto import testing

retained = testing.analyze_placed_sync(function, True)  # Explicit disjoint-GM contract.
physical = retained.realize_physical("a2a3", [0, 1, 2, 3, 4, 5])
function_with_flags = physical["function"]
```

`None` 使用目录中合法且未保留的 ID，并排除宏声明的资源。六个 ID 是测试配置，
不是虚构的硬件统一容量。结果包含原始族坐标、循环 ID、静态命令和谓词数量、
槽数量、查询工作量及 `finite_minimum`。C++ 入口为
`sync::RealizePhysicalSync(const CompactSyncResult&, target, ids)`。

`physical_sync_realization.cpp` 读取不可变数学结果。逻辑与物理发射器共用
`sync_placement.h`，定位原始插入点、保留初始化分支、跳过零次循环体。
使用既有 `system.set_flag/wait_flag/bar_m/bar_v` 和后端发射器。
Buffer IR 预检查现允许这些已有操作和显式 `bar_all`，不重建 Tile IR。

在规范直接放置及不同源/目标管线的前提下，交接 h 的消费到 g 的发布由
`C_consumer(h) -> I_source(g)` 证明。WAIT 约束其消费者，源完成约束 SET。
同管线屏障只在不增加载荷顺序时接受。分配不通过新增边来制造复用证明。

同方向发布与消费分别位于有序本地链上。可复用的目标构成发布序列的后缀，
其首索引随消费顺序单调变化。排序后扫描只调用线性数量的保留查询；这些查询自身
的成本另计。区间着色给出该有限固定协议的最优分配。`Fixed-plan scarcity`
仅指这个已限定协议及可用池，不能推论所有协议都不可行。

重复族使用保留的周期完成到开始距离确定槽周期。原始谓词限制有效端点。
跨族复用整槽区间是充分条件，可能拒绝别的可行分配，因此报告
`Allocation unproved`。片段结束或计数器复位不自动重置物理通知；跨片段
复用同样需要保留顺序查询。失败不改变原 IR 或数学结果，也不通过新增命令修复容量。

例如 `a[i] -> b[i+1]` 仅在 `i+1 < trips` 时发布，`i > 0` 时等待。
两端使用同一源序号选择 ID；两槽时分别为 `i % 2` 和 `(i-1) % 2`。

## 本地和下游限定

主机检查使用 PTOAS **v0.72**，审查运行时固定的 PTO-ISA 修订
`f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7` 的手动 a2a3 模式。
允许的 AIC 变体为 TLOAD(Mat)、TSTORE(Acc)、TMOV/TEXTRACT(Mat 到 Left/Right)、
TMATMUL 和 TMATMUL_ACC。这些实现没有隐藏的 AIC 标志。
公共包装器传递空 `WaitAllEvents` 参数包，返回平凡 `RecordEvent`，生成代码没有
Event 转换。TEXTRACT 的 V/S 标志属于排除的 Vec 重载。这是逐变体审查的限定，
不表示目录中缺少条目的任意宏都安全。未知所有者或非固定资源绑定保持不可用。

生成 PTO 使用 `--pto-level=level3 --pto-arch=a2 --enable-insert-sync=false`。
三个实测内核在开启和关闭自动同步时输出相同；测试比较内核体的方向、ID 序列和命令数。
PTOAS 还生成未被调用的 `ptoas_auto_sync_tail` 辅助函数，其定义不计为执行的命令，
也不用于证明返回时完成。此审查不覆盖启用 `__PTO_AUTO__` 或 `PTO_FLAG_TEST`
的后续构建。没有进行设备编译或执行。

通知入口空闲、出口闭合不代表所有载荷已完成；调用者/运行时仍须提供原有完成及可见性
契约。这与逻辑闭包、物理复用及原生正确性分别报告。

## 复现和证据

使用当前工作树构建的 Python，`PTOAS_ROOT` 指向 v0.72，PATH 为其启动器提供
兼容的 Python 3.11。此次 PyPTO 使用 Python 3.14 构建；若 PATH 不同，显式调用它。

```bash
source .claude/skills/testing/load-env.sh
cmake --build build-py314 --parallel 2
PYTHONPATH=$PWD/python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -m tests.ut.ir.transforms.run_physical_sync_prototype \
  --native --output build-py314/physical-sync-prototype/campaign
PYTHONPATH=$PWD/python python3 -m pytest \
  tests/ut/ir/transforms/test_physical_sync_realization.py \
  --require-ptoas -n 1
```

收据固定未提交源码、输入/输出序列化、示例、扩展库、Python 和 PTOAS 的身份。
分析与物理构造分别计时，单次计时不证明加速。
独立解释器从原始 IR 提取物理访问，构造冲突和本地事件图，检查实际覆盖、载荷顺序、
匹配、无环及因果复用，不添加复用边使结果通过。独立二分匹配计算固定图方向宽度，
小图还枚举反链。检查器共享声明的访问和原生语义模型，不共享生产生成器或分配器。

测试覆盖三个原始矩阵乘、零/短循环、1–3 次顺序调用、初始化/累加、尾部存储、
非零起点/非单位步长、1–4 个静态阶段、稀疏池、扇出、序列化和重复请求。
缺失 WAIT、错误 ID、文本配对但无因果复用的变异必须失败。
有限宽度为二的例子对单 ID 报容量不足；重复族对单 ID 保留“未证明”结果。
K 到一百万时只保留静态族和有界槽，独立解释器仅展开小例子。

| 已建立结果 | 仍需前提 | 可作出的论断 |
| ---------- | -------- | ------------ |
| 有界检查中的物理命令与逻辑需求和顺序一致 | 已限定原始访问及事件语义 | 支持子类的直接物理实现 |
| 独立检查有限宽度和充分循环槽 | 空闲入口、方向池及调用接口 | 固定协议复用；仅限定有限情况声称最优 |
| PTOAS 保留显式标志计划 | 调用者完成及原生契约 | 主机降低兼容，不是设备验证 |
| K 增大时静态表示保持不变 | 静态族及可执行原始端点 | 紧凑重复实现，不是通用 token 降低 |

独立源码审查在补充复用与容量负例后接受了实现及通用性。
通用 token 降低、分支历史、任意嵌套区域、容量修复和调用完成合成不属于此里程碑。

## 本次主机结果

新的集中测试通过 **639 个 Python 测试和六个 C++ 测试**。三个原始例子均通过
PTOAS v0.72 生成 C++。已检查执行满足模型内全部载荷顺序相等及匹配闭合，
在给定六 ID 池内每个方向最多使用 ID 0–1。

| 原始例子 | 静态 SET / WAIT / 屏障 | 执行 SET / WAIT / 屏障 |
| -------- | ---------------------- | ---------------------- |
| `matmul_64` | 4 / 4 / 0 | 4 / 4 / 0 |
| `matmul_acc_64` | 10 / 10 / 0 | 10 / 10 / 0 |
| `ddr_split_k` | 15 / 15 / 4 | 19 / 19 / 6 |

受控的两次调用衍生例在 K 为 4、8、16、64、1024、1,000,000 时始终保留
37 个族、33 个槽区间、60 次复用查询、33 个静态 SET 和 WAIT、四个屏障及
41 个谓词。十五个独立有界执行覆盖零到五次迭代及一到三次调用。
这些是固定静态阶段的衍生例，不是不同 AutoTile 配置。

提交前的重跑还把同步控制与原始初始化分支分开计数。两个有限矩阵乘例子新增
标量操作和实际谓词测试均为零。Split-K 在下游优化前新增 38 个 PTO `arith.*`
操作，在固定输入中执行 38 次新增谓词测试（静态谓词为 19 个）。这些不是设备
指令数或周期估计。独立命令图没有额外载荷前置条件；每方向最多两个 ID，与
有界图宽度一致。这支持模型内的顺序质量和小资源用量，不能证明静态命令数或
控制代码全局最优。扇出合并及控制简化可作为后续工作。三个例子的单次保留分析
及物理构造各小于一毫秒，前端捕获和下游编译分别记录。

新的提交前独立审查接受了实现及这些限定结论。639 个 Python 测试和六个 C++
测试重新通过。不据此推断设备延迟、数值执行或函数返回时的载荷完成保证。
