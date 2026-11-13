# Logical realization of retained demands

`InsertSync` turns qualified retained demands into logical completion-token IR.
It preserves original occurrence coordinates, guards and legal insertion cuts.
The logical service does not assign physical event IDs.

A `sync.record` represents an immutable, non-consuming completion fact. Multiple
waits may use that fact. Joins combine completion requirements; completed tokens
represent established empty prefixes. This is different from a consuming
hardware notification. See [logical synchronization](11-logical-sync.md) for the
IR contract and verifier.

`OptimizeSync` and production `RealizeSync` share the logical verification
service. `LowerSync` consumes the actual token definitions and uses; it does not
rerun minimum-demand generation to replace supplied dependencies. Valid logical
IR can still lack an admitted executable lowering. Unsupported coordinates,
readiness or resource evidence cause a transactional refusal.

The opt-in production route delegates an unsupported whole function unchanged
to PTOAS; compiler-generated invalid IR remains a compiler error. See
[RealizeSync](../passes/59-realize_sync.md) for selection and ownership, and
[architecture contracts](33-sync-architecture-contracts.md) for stage boundaries.

The earlier finite diagnostic construction, its restrictions and historical
qualification results remain in the [archived checkpoint](https://github.com/tonibohnlein/pypto/blob/365b803a45072f1c00e8eca5da9942bd9828685f/docs/en/dev/ir/14-logical-demand-realization.md).
They are not the complete current production coverage contract.
