# Runtime valid axes, reductions and direct casts

Status: design/native-source qualification accepted; all 60 focused runtime
family tests pass. Final integrated hooks/commit gates are recorded in the
companion receipt. This extends the M2 operation/metadata contract
and feeds conservative M3/M4 composition. It adds no recognition backend,
allocator, private branch-history state, invocation bound or kernel-name rule.

## Shared physical and active-domain owners

`PhysicalLogicalVecDescriptor` recovers only placed-candidate physical shape,
layout, origin and stride from existing Tile/Buffer descriptions. Its capacity
projection is never an exact active-access or selector answer. The existing
static descriptor helper consumes this geometry and separately checks original
static valid extents. Dynamic families retain actual valid expressions through
per-use metadata restoration and `BufferRuntimeValidAnalysis`.

The new family supports FP32 row sum/max and column-carrier row expansion with
runtime rows and fixed positive full source columns. Source/output active rows
must agree at the use site; the column carrier must have those same active rows
and valid column one. A full carrier is not silently narrowed to a partial
matrix. Native calls require positive rows. A zero-row visit issues no such
operation, while metadata and unrelated payload controls retain their meaning.
The usual `0 <= rows <= capacity` operation-validity premise remains explicit.

Whole-capacity contiguous FP32 column `[M,1]` and row `[1,M]` interpretations
can retain the original active-prefix axis map `[r,1]` to/from `[1,r]`. Both
native descriptor objects have independently owned metadata and point to the
same proved placed byte interval. Lowering restores each original logical axis
map at every data use. It does not propagate metadata by pointer aliasing or
reinterpret an addressless runtime descriptor. Requested reshape dimensions
must also agree with the declared output; equal byte counts are insufficient.

BF16/FP32 casts reuse the same physical geometry and actual metadata service.
They retain matching `[M,C]` physical shapes, fixed positive full columns and
matching actual runtime rows, existing rounding/saturation restrictions and
existing source/destination disjointness. Runtime casts use their owned native
handles; a borrowed subview cannot stand in for missing metadata evidence.

## Native implementation and bounds

The selected implementation is pinned pto-isa `5487a6b0`, A2/A3 only. Other
targets and unsupported native families remain unavailable.

* FP32 row reduction uses the generic vector helper when valid rows are runtime.
  Integer scalar-reduction branches with numbered V/S events are not selected.
  Sum emits one scalar per repeat; max selects ONLY_VALUE. With destination
  stride one, the output is the active contiguous prefix. Inactive output rows
  are untouched and are not implicitly initialized to zero. These units follow
  the [repeat-sum instruction contract](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/C-API/vector_compute/asc_repeat_reduce_sum.md)
  and [repeat-max contract](https://asc.gitcode.com/api/SIMD-API/c_api/vector_compute/asc_repeat_reduce_max.html).
* The explicit reduction workspace encloses each helper's per-row partial
  segments and masked tail. Runtime rows fit uint16 UB-copy burst counts,
  source/workspace row strides in FP32 blocks fit uint16 reduction fields,
  and source/output/workspace element products fit uint32 pointer arithmetic.
  The existing resource catalog's conservative integer-variant event reservation
  is retained; this task does not change resource assignment.
* Row expansion stages `ceil(r/8)` carrier groups into its existing fixed 8 KiB
  workspace. Thus carrier reads can include inactive rounded padding, contained
  in the physical column because `M` is divisible by eight. The small-row Count
  and Norm paths and large-row 240-row chunks stay inside that workspace. Large
  physical-row Norm paths require uint8 repeat strides. Element products fit
  uint32. Effects conservatively cover the whole allocation and scratch; the
  carrier footprint is not advertised as exact active-row reads.
* The selected direct BF16/FP32 cast head/tail loops use actual destination rows
  and columns with source/destination physical row strides. They do not select
  the integer scratch/event branches. Full columns are multiples of 16, physical
  row strides fit uint16 before native block division, and row element offsets
  fit uint32. Existing modes and disjoint-window contracts remain authoritative.

These are instruction encoding premises, not larger admission limits or
benchmark-specific bounds. Independent target/device execution remains separate.

## Validation contract and remaining boundary

The new ordinary-frontend fixtures combine runtime-row reduction, both contiguous
axis maps, matched-valid scalar arithmetic, row expansion and output transfer.
Small executions check rows zero/one/tails/full capacity, selected/skipped visits,
outer resets and changing actual metadata across visits. Independent interpreters
expand original active bytes, account separately for rounded carrier reads and
scratch, and compare full token-induced versus physical payload event order.
Matching, acyclicity, directional six-ID eligibility and causal reuse remain
separate checks. Prefix results use sound conservative effects; they make no
minimum-demand or exact dynamic-selector claim.

Negative tests distinguish actual metadata mismatch, absent positive guards,
partial columns, incompatible targets, aliasing, native field overflow, forged
reshape shapes and unavailable addressless metadata ownership. Large literal
trip-count derivatives check retained size without unfolding those visits.

The original Qwen post-RMSNorm witness still has a full-valid accumulator plus a
partial-valid reduction value yielding a full-valid add result, followed by a
full column carrier used with a partial matrix. This construction supplies the
axis map and matched-domain native recipes. It does not supply inactive zeros,
a caller-specific `r=M` premise, or a proof for those mixed domains. Such uses
remain explicitly unavailable. Conservative address effects cannot repair an
unqualified operation-value or metadata contract.

| Established evidence | Separate limitation | Paper consequence |
| -------------------- | ------------------- | ----------------- |
| Shared geometry and actual active-domain metadata | Capacity is not an exact active-access export | Preserve input qualification separately from demand reduction |
| Original coordinate/valid-axis restoration at placed aliases | No arbitrary dynamic/addressless reinterpretation | Regional interfaces preserve their declared domains |
| Conservative effects plus actual logical/physical order checks | Dynamic exact selectors unavailable | Qualified conservative synchronization, not exact minimum demands |
| Existing six-ID physical protocol and reuse certificates | Native execution and device timing untested | Host-model feasibility only |

## Fresh production capture

The unchanged two-layer Qwen prefill capture (batch one, sequence 128; library
`b72c66d9`) generates all 32 C++ leaves under the explicit conservative-prefix
policy. PyPTO owns 16 leaves, up from 14 at the preceding checkpoint: `x_gamma`
and `x_gamma_0` now use the shared runtime BF16/FP32 cast and broadcast path.
Both report whole-memory-space enclosures and preservation of the selected
actual-token order. The complete original model is not PyPTO-owned or device
validated. Full compiler, source, binary, tool and command hashes are in the
capture and companion receipt. The one inclusive 6.487-second capture measurement
is descriptive compilation work, not minimum-demand time or a speedup claim.

The focused checks include 40 axis/reduction executions, 32 direct-cast
executions and the 24 preserved broadcast executions. The new encoding tests
use disjoint synthetic windows and require the native field diagnostic, so a
capacity/alias refusal cannot accidentally count as a successful encoding test.
