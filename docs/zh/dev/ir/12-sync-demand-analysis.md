# 有限同步需求分析

`ir::sync::AnalyzeFiniteSync` 实现具体有限情形的构造：物理访问类 → 存储生成边 → 非原生覆盖边。
它是 `transforms/utils/finite_sync_analysis.h` 中的只读 C++ 分析服务。显式启用的 autosync
通过合格区域提供者消费它；服务本身不插入同步命令。英文文档为规范来源。

## 输入契约

输入按参考顺序列出 occurrence、具体 pipe、已确定的存在性和读写 cell。
索引保留原 occurrence 身份，缺席 occurrence 不导致重新编号；导入器负责保留原语句和调用坐标。
cell 表示物理重叠，包括不同 SSA 根复用的字节。此服务不从算子效应恢复 footprint，
也不解释隐式 scratch、标量可见性或多管线宏。

`covered_cells` 声明完整覆盖的存储域。域外访问（包括缺席描述中的访问）被拒绝；
域外查询返回 `nullopt`，域内无访问则返回可用但 selector 为空的边界。
局部存储覆盖不能把缺失的 GM 信息变为空答案。

`exact_accesses=true` 是提供者给出的精确冲突证据，false 表示可靠的保守包围。
两者均返回所选模型的最小需求，只有前者才能继承精确物理需求保证。
该标记本身不证明访问提取或原生语义。

原生图包含每个 occurrence 的 start→completion 及每条 pipe 上独立的 start/start、
completion/completion 链，不自动串行化同 pipe payload。
额外原生 C→I 边与必要前置条件分开提供，前者在约简前参与。
标量或累加器的硬件豁免必须由提供者证明，不能仅凭 pipe 名称推断。
边仅在两端均存在时生效；未解析谓词和其他原生事件边形式不在本契约内。

## 构造与输出

每个 cell 保留前一 writer 及其后的只读访问。新访问接收前一 writer 的边，
新 writer 还接收读者边，再替换 frontier。RMW 先查询旧状态再成为 writer。
合并重复端点后，按 source 逆序检查各 consumer 的入边，仅对必要需求合并 rank 行。

例如同一 cell 的 MTE2 写、V 读、MTE2 覆盖，生成边为 (0,1)、(1,2)、(0,2)，
约简需求为 (0,1)、(1,2)。MTE2 completion 顺序不能替代读完成后才覆盖的要求。

数值行支持全部四种严格 start/completion 可达查询；缺席端点为 false，非法索引报错。
存储边界返回首末 writer、每条 pipe 上首 writer 前的首 reader 和末 writer 后的末 reader。
无 writer 时 reader 边界覆盖整个区域。边界不证明 payload 完成或物理 flag 可重置。

设 h 为覆盖 cell 数、n 为 occurrence 数、c 为输入关联数、e 为必要前置边数、
a 为额外原生边数、k 为 pipe 数、f 为输出需求数，期望时间为
O(hk+n+c+e+k(n+a+f))，空间为 O(hk+nk+c+e+a)，硬件 pipe 集合固定。
不存在 read/read 二次扫描或全对闭包。计数器记录访问输入、活动 cell、reader 遍历、
witness、候选测试和需求合并。Python 内部测试接口复制数值行以检查结果，不是生产区域导出器。

独立测试直接枚举冲突和事件图，验证闭包、实际覆盖边、四种查询与边界，涵盖 RMW、
原生前置条件、已解析存在性、物理字节重叠和区域划分。读密集扩展测试至 4096 readers。
枚举测试中的 guard 取值不等于实现符号 guarded 分析。

后续导入器须从最终 IR 建立上述输入契约，不能丢弃未解析效应。
符号 guard 和紧凑循环需要独立构造，不能通过此接口展开整个循环。
逻辑 token 可表达结果；可执行 guard、合法插入点、matching、调用闭合、物理 ID 复用
及设备验证仍是独立义务。

[已分配矩阵乘原型](https://github.com/tonibohnlein/pypto/blob/365b803a45072f1c00e8eca5da9942bd9828685f/docs/zh/dev/ir/13-matmul-sync-prototype.md) 已提供具有限定契约的 IR 导入器和紧凑重复区域分析，其限制与独立验证见单独文档。
