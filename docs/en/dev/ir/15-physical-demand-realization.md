# Physical synchronization realization

The shared physical services lower qualified actual-token handoffs and diagnostic
retained-demand plans. They do not support arbitrary token control flow.
Allocation, command scheduling and materialization are separate stages; successful
allocation moves the assigned families into an immutable `AssignedSyncPlan`.

## Resource contract

Each handoff has matched publication and consumption occurrences, original
coordinates, activity guards and a direction. Fanout may require several physical
handoffs even when logical uses share a completion token. Reuse requires a causal
consumption-before-republication path, not merely source order or equal counter
values. A child boundary or counter reset does not rearm an ID.

Finite qualified handoffs use ordered reuse thresholds and interval coloring.
Repeated families use sufficient cyclic-lane reservations and certified boundary
paths. An unproved repeated assignment is not proof of fixed-plan scarcity.
Target eligibility, reserved IDs and native macro ownership restrict the pool;
an optional caller pool can restrict it further, never enlarge it. No extra
synchronization is inserted to force an assignment.

Strict direct placement admits same-pipe barriers only under its existing
order-preservation premises and native exemptions. Endpoint guards must be ready
at their original cuts. Schedule selection preserves ordered commands and IDs;
shared guard construction is not permission to change the logical protocol.

## Production and qualification

`RealizeSync` commits only a complete qualified whole-function plan. Downstream
PTOAS autosync stays enabled: it skips functions with explicit qualifying flags
and handles delegated functions. Empty or barrier-only candidates delegate rather
than manufacturing a flag. See [the production pass](../passes/59-realize_sync.md).

Fresh notification entry, closed matching, completed payloads at return, physical
reuse and native correctness are separate obligations. Closing notifications does
not prove that the final asynchronous payload completed before return. No general
native entry/return guarantee or device performance claim follows from host tests.

See [architecture contracts](33-sync-architecture-contracts.md) for ownership and
[the historical physical checkpoint](https://github.com/tonibohnlein/pypto/blob/365b803a45072f1c00e8eca5da9942bd9828685f/docs/en/dev/ir/15-physical-demand-realization.md)
for the original derivation and pinned experiments.
