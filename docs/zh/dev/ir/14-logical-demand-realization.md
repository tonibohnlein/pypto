# 保留需求的逻辑实现

`InsertSync` 把合格的保留需求转成逻辑完成 token IR，保持原 occurrence 坐标、守卫和合法插入
位置，不分配物理事件 ID。`sync.record` 是不可变、非消耗的完成事实，可被多个 wait 使用；
join 合并要求，completed token 表示已建立的空前缀。它不等于消耗型硬件通知。
详见[逻辑同步](11-logical-sync.md)。

`OptimizeSync` 与生产 `RealizeSync` 共用逻辑验证服务。`LowerSync` 消费实际 token 定义和
使用，而不重新生成最小需求来替换输入依赖。合法 token IR 仍可能缺少可执行降低所需的坐标、
就绪性或资源证据；此时事务性拒绝。显式启用的生产路径将不支持的完整函数原样交给 PTOAS；
编译器生成的非法 IR 仍是编译器错误。

[生产 pass](../passes/59-realize_sync.md) 说明选择与所有权，
[架构契约](33-sync-architecture-contracts.md) 说明阶段边界。
早期有限诊断构造及历史验证保存在[归档检查点](https://github.com/tonibohnlein/pypto/blob/365b803a45072f1c00e8eca5da9942bd9828685f/docs/en/dev/ir/14-logical-demand-realization.md)，
不代表当前生产覆盖范围的完整契约。
