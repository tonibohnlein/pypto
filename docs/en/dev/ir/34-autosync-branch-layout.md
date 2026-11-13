# Autosync branch layout and qualification

The development branch is organized around four review units: Buffer access and
layout prerequisites; logical completion-token IR; integrated demand analysis and
qualified physical lowering; and maintained documentation. The public pipeline is
`InsertSync` → `OptimizeSync` → `LowerSync`, composed transactionally by opt-in
`RealizeSync`. See [logical synchronization](11-logical-sync.md),
[finite analysis](12-sync-demand-analysis.md), [logical realization](14-logical-demand-realization.md),
[physical realization](15-physical-demand-realization.md) and
[architecture contracts](33-sync-architecture-contracts.md).

## Reconstruction boundary

The published `codex/pypto-buffered-intervals` branch at
`365b803a45072f1c00e8eca5da9942bd9828685f` is preserved. The new
`codex/pypto-autosync-clean` branch starts from upstream
`d0127ff79` and reconstructs the merged net change rather than replaying the
14 development commits. Four newer upstream commits were merged without
conflicts; their CI, A5 VF, non-finite constant and descriptor-ABI changes remain.

No production algorithm, token contract, allocation policy, physical-ID assignment,
fixture or independent checker was removed. Source review found that the retained
diagnostic APIs have test consumers; being diagnostic is not evidence of dead code.
The first two packages establish prerequisites; the third keeps coupled entry
points together rather than introducing temporary declarations without definitions.

## Evidence archive

Chronological chapters 13 and 16–32 and documentation JSON receipts are archived
at the [immutable published source](https://github.com/tonibohnlein/pypto/tree/365b803a45072f1c00e8eca5da9942bd9828685f/docs/en/dev/ir).
This removes 79 files and 73,014 historical lines from the new branch's net change.
Existing links to that evidence now target the immutable revision. Current
contracts and all executable fixtures, fixture manifests, qualification drivers
and independent oracles remain in the branch. Chapters 14 and 15 now state the
current logical and physical boundaries; their original derivations remain in
the archive.

The [reconstruction receipt](autosync-branch-layout.json) records the exact
archived paths and hashes, package boundaries and validation. Retaining an old
receipt is evidence of its original run, not a new execution on this branch.
Native invocation-return semantics and device performance remain separately
qualified; this reconstruction makes no new device claim.

## Validation

The initial reconstructed tree built successfully and passed 1,814 focused tests. All
16 generated C++ outputs are byte-identical to the published source snapshot;
235 ordered command traces agree and nine fixtures pass independent access,
ordering, matching and physical-reuse checks. Documentation parity/navigation,
headers and English-only checks pass. The reviewer accepted the complete result.

Package boundaries received static dependency review. Intermediate package
builds were not separately run; the build and tests cover the complete tree.
The subsequent source cleanup is folded into these same commits: duplicate emission
and obsolete production wrappers are absent from the implementation commit. Its
focused gate passes 363 tests plus two counter regressions, with the same 16
byte-identical outputs and 235 traces. See the ownership contract in chapter 33.
