# Qualified structured synchronization schedules

The baseline is `868fdd471`. Exact demand analysis, token matching, physical
allocation and hardware exemptions remain unchanged. `LowerSync` now extracts
qualified boundary commands from its existing assigned-command descriptors,
before creating scalar statements. It does not run another synchronization
analysis or optimize already materialized scalar code.

## Supported construction

The initial construction admits a known-nonempty, literal-bound sequential
INDEX loop with positive step, a representable final increment, and at least two
visits. There are no loop-carried results, special loop attributes, definitions,
unknown effects or nested control. A previously qualified initialization/
accumulation alternative sharing one unconditional command cut is supported.
Zero/singleton and unsupported loops retain their existing lowering.

Only the first payload's incoming cut and the last payload's outgoing cut are
eligible. For entry extraction, a descriptor must execute no commands on any
nonfirst visit. For exit extraction, it must execute none on any nonlast visit.
The existing arithmetic analyzer proves this over intervals enclosing all
remaining IV values; failure is an unavailable proof, not inactivity. This is
sound for nonunit steps without enumerating the IV lattice.

Within an eligible cut, scan from the corresponding boundary. A preceding
(entry) or following (exit) command is a movement fence unless it is itself
extracted or proved absent on that boundary visit. Preserve the order inside
multi-command descriptors. Replace the original IV with its literal boundary
value in the extracted descriptor and retain unknown enclosing predicates.

For example, this loop-entry sequence:

```text
if i > first: WAIT(release)
if i == first: WAIT(readiness)
payload
```

can place `WAIT(readiness)` before the loop because the preceding release WAIT
is absent on its first visit. A readiness WAIT before a second payload cannot
cross the first payload, even if they use different pipes.

## Validity and ownership

1. The retained descriptor is inactive on every visit from which it is removed.
2. On its boundary visit, it crosses no executed command or payload. Every
   crossed descriptor is proved empty there.
3. The restricted body introduces no SSA definitions or hidden scalar effects.
   Captured values dominate the loop; literal IV substitution removes the only
   value whose scope changes. Original branch control remains in place.
4. The nonempty loop and its enclosing guards stay in their original positions.
   Each outer invocation repeats the same transformation, preserving resets.
5. Literal actions, directions, IDs and ordered command multiplicities remain
   unchanged. Entry and exit use distinct inventories even for a single-payload
   body. Notification closure and physical reuse follow from the unchanged
   complete modeled payload/command trace, not from a new boundary assumption.

The descriptor owner remains immutable. Each accepted loop gets a small local
cut override; unsuccessful qualification retains the original descriptors.
No payload body or loop is cloned. Allocation, token identity and storage
coordinates are not reconstructed.

## Cost and rejected alternative

A candidate must retain the static command count, reduce static guard count and
not increase expression occurrences in its affected descriptors. This is a
control-description gate, not a universal bound on serialized IR or C++ bytes.
The whole output is measured separately. Descriptor-level guard evaluations do
not increase: extraction removes boundary tests and evaluates surviving enclosing
predicates at most once instead of on every visit. Moving predicates between
scopes can affect later scalar-value sharing, so final scalar-operation work is
a measured outcome, not a universal consequence. Loop-header work is unchanged.

Body/cut inspection, descriptor visits, rejected candidates and before/after
control size are counted. The traversal does not depend on trip count; supplied
expression proof work is additional and uses the existing arithmetic analyzer.

An initial first/interior/last body-splitting experiment was rejected on all 28
fixtures. For large gate/up, its estimated innermost representation grew from
1,092 to 2,576 nodes. That experiment is not in production. Explicit phase
splitting or symbolic family unification would need their own cost and validity
contracts.

## Qualification

The receipt `structured-sync-control.json` records pinned source/tool/input
identities, commands, actual output sizes and ordered-trace comparisons.
Synthetic tests exercise nonunit bounds, negative starts, zero/singleton loops,
outer skips, resets, guarded command groups, fences, holes and unavailable scope
proofs. They test lowering only; independent access/command checks on production
fixtures establish modeled coverage, matching, acyclicity and causal reuse.

Exact demand analysis is not disproved by excessive guard code. These fixtures
retain compact mathematical results; this repair addresses how their assigned
commands use the original loop structure. It is deliberately narrower than a
uniform structured-schedule theorem. Native invocation entry/return and new
device performance remain unqualified by this host-only checkpoint.

## Measured result

Ordinary token lowering, compared with the immediately preceding baseline:

| Fixture | IR branches | C++ conditional sites | C++ bytes | Evaluated conditions over the same executions |
| ------- | ----------: | --------------------: | --------: | --------------------------------------------: |
| Qwen output, H=512 | 14 → 11 | 14 → 11 | 15,142 → 14,992 | 224 → 176 |
| Qwen gate/up, band 2, H=512 | 83 → 73 | 81 → 70 | 30,453 → 29,691 | 7,938 → 6,774 |
| Qwen gate/up, band 2, H=5120 | 147 → 143 | 141 → 137 | 39,004 → 38,729 | 117,858 → 112,826 |
| Qwen output, H=5120 | 27 → 27 | 27 → 27 | 17,307 → 17,307 | 36,320 → 36,320 |

All 477 ordered traces across 28 fixtures are identical. All 56 baseline/
candidate C++ outputs are generated successfully. The nine small independent access/command checks
pass. Fifteen emitted-C++ graph checks on ten supported fixtures retain exactly
the baseline's comparison against PTOAS: neither added nor removed
completion-to-start prerequisites change. Six existing comparison limitations
remain explicit (vector preprocessor handling or delegated slot selection).

For the checked small band-2 gate/up executions, PTOAS has 202 excess
completion-to-start pairs and PyPTO has zero, with strict relation containment.
That is finite modeled evidence, not a uniform proof or a device speedup. The
boundary repair itself changes neither relation nor executed command count.

Synthetic boundary inspection/materialization counts remain constant from four
to four million trips. The existing rotating and nested million-trip fixtures
retain their compact representation. Construction counters in the campaign
come from a separate explicit mathematical-owner diagnostic; they are not a
production trace. In particular, diagnostic acceptance on large output
projection does not imply an ordinary token-path output improvement there.
Single-run production times are descriptive; no compiler speedup is claimed.

The remaining controls include first-use waits at interior payload cuts,
nonfinal reuse publications, reset/outer-invocation decisions and case selection.
This strict transformation cannot move those across executed payloads. It does
not establish that all remaining control is necessary or optimally represented.

The broad host gate passed 1,593 tests (37 warnings). A final rebuilt extension
passed 144 focused tests, including all 18 boundary tests; two negative boundary
tests were added after the broad gate had collected its original 16. Only C++
formatting changed between the broad gate's build and the final rebuild.
Compiler-expert review accepted the design, code, validity argument and measured
performance scope. No commit or device run is part of this checkpoint.
