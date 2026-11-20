# GM scalar staging: operation-contract checkpoint

Status: the bounded implementation passes its first coherent 53-test host gate
and its source review. The post-format build, corrected operator gate and pinned
whole-model capture also pass. Final hooks and accepted commit publication are
pending. Native instruction correctness remains
the existing selected-TLOAD premise, not a newly proved hardware result. This extends M2/M3 of the
[architecture plan](41-general-autosync-architecture-plan.md). It follows the
qualified [local scalar bridge](43-autosync-value-and-scalar-contracts.md).
The whole-model host capture improves local ownership from 17 to 19 leaves;
complete Qwen ownership and device results are not claimed.

## Packed rank-two point reads

`PassContext(enable_scalar_gm_dma=True)` selects point DMA for packed, unpadded
ND INT32 and FP32 tensors of rank one or two on A2/A3 AIV. It preserves the
original tensor, coordinates, and execution cut. A rank-two read partitions
the original tensor at `[row, col]` with extent `[1, 1]`. The column extent
must be static and the parent stride must fit the native burst-gap field.
The row extent may be dynamic. Explicit strides must agree with the packed
layout. The recipe requires full-valid GM views.

The unit partition gives one four-byte GM burst for either dtype. The
original parent row pitch still contributes to the starting address even
when the selected window is one row tall. TLOAD may touch the complete
32-byte UB carrier. Conservative synchronization imports that enclosure and
leaves exact physical recovery unavailable.

The A2/A3 public DMA contract accepts both four-byte dtypes and permits GM
start addresses that are unaligned to 32 bytes. The native implementation
requires a 32-byte-aligned local carrier and preserves the original
in-bounds tensor access precondition. [CANN DataCopyPad contract](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0265.html)

## Supported construction and preserved semantics

Start with a direct SSA definition `value = tensor.read(source, [index])` in an
A2/A3 AIV leaf. The source is an ordinary packed, unpadded ND rank-one INT32 GM
tensor; `index` is a ready integer expression at that original cut. Original
operation definedness requires an in-bounds, non-wrapping address. Do not infer
that condition from a benchmark, a different SSA pointer or a numerical hull.
Input task/publication and local memory visibility contracts remain necessary.

```text
original: value = tensor.read(original_source, [original_index])

candidate at the same executed cut:
  TLOAD original_source[original_index : original_index + 1]
        -> private Vec INT32 capacity [1,8], valid [1,1]
  value = buffer.read(private_carrier, [0,0])
```

The original tensor parameter, ABI rank, allocation identity, index expression,
scalar result identity and surrounding control remain intact. Do not substitute
an invented `[1,N]` caller tensor. The fresh carrier reserves 32 bytes through
ordinary storage services. It supplies one defined scalar; no operation may
observe its inactive lanes. Account conservatively for native writes within the
whole carrier, including a possible alignment tail. The source read is exactly
one INT32 element; never read eight GM elements to obtain an aligned transfer.

`TLOAD` is an asynchronous MTE2 payload and the qualified local read is an S
payload. Existing prefix services must establish MTE2 completion before the
scalar read; the blocking S result then establishes availability of the original
SSA for values, addresses or control. Carrier overwrite on a later visit also
requires the existing causal reuse proof. Neither scalar native order nor closed
notification matching establishes an unrelated payload's completion.

The implementation accepts only normalized direct assignments. A discarded or nested GM
read, an effectful while predicate without an original statement cut, a write,
a different dtype/core/target or unavailable source layout refuses. Do not hoist
a read out of a branch or omit the final false while evaluation. Reads of
cross-core publication/polling state need that existing protocol's visibility
contract; DMA staging does not manufacture one.

## Native evidence and remaining qualification

The selected native pin is PTO-ISA
`5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f`:

- [TLOAD contract](https://github.com/hw-native-sys/pto-isa/blob/5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f/docs/isa/TLOAD.md)
  admits the A2/A3 matching-layout INT32 Vec path with positive source and valid
  extents. This is distinct from the block-aligned Mat paths.
- `include/pto/common/arch/memory/tload_common.hpp`, `TLoadGm2ubNd2nd`, computes one burst
  of four bytes for source shape `[1,1,1,1,1]` and valid `[1,1]`.
- `include/pto/npu/a2a3/TLoad.hpp`, `TLoadInstrGm2ubNative`, selects
  `copy_gm_to_ubuf_align_b32` for four-byte element types.
- The [official CANN nonaligned-copy reference](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0265.html)
  explicitly supports Atlas A2/A3 INT32 GM-to-UB copies. GlobalTensor source
  addresses have no address-alignment restriction; LocalTensor destinations
  require 32-byte alignment. Burst lengths are in bytes, including four bytes.
  Thus an element-aligned INT32 source plus an integral element index need not
  be 32-byte aligned. This public contract must be bound to the pinned selected
  intrinsic; it does not prove arbitrary caller address validity.

The source and public nonaligned-copy model permit a UB alignment tail. Reserve
and summarize its complete 32-byte enclosing domain, regardless of padding
values. The implementation does not assert that `PadValue::Null` proves only four
native UB bytes are touched. Exact physical-footprint qualification and any
future exact-selector extension are separate from the sound enclosure.

The architecture review accepts construction under the existing normative
selected-TLOAD contract, which already supplies native instruction correctness
to ordinary loads. The pinned source chooses the four-byte transfer, and the
public contract permits that transfer. Their exact CCE intrinsic/public-API
implementation correspondence has not been independently reproduced: the CANN
headers are unavailable locally and the authoritative source service is rate
limited. Keep that qualification explicit rather than presenting the new bridge
as a hardware proof. The existing local event contract must order DMA completion
before GetValue. No device validation is available.

## Implemented owners and interfaces

| Owner | Bounded change |
| ----- | -------------- |
| Shared transfer/native recipe | Recognize a packed rank-one INT32 single-point GM transfer to the dedicated rank-two carrier; preserve source and native-write precision separately |
| `SelectBufferIRFunctions` | Stage a complete immutable candidate only for supported direct reads; ordinary whole-function admission remains authoritative |
| `memref_utils.h` | Reuse `CreateAllocStatement` and `InsertAllocsIntoBody` for top-level carrier allocation before existing placement; no second allocator |
| Buffer transfer deduction and runtime-valid analysis | Map source offset/extent `[i]`/`[1]` to destination valid `[1,1]`; reject other rank-one transfer forms until separately qualified |
| `LowerTileToBuffer` | Consume the qualified staged recipe at the original cut and preserve the original scalar result; reuse the static local read bridge |
| PTO Buffer codegen | Admit the qualified rank-one ABI and reuse ordinary tensor/partition-view emission, native TLOAD and GetValue |
| Effect import and actual-token lowering | Conservative DDR/Vec facts cover the source and full carrier; retain source root/alias assumptions and existing S readiness/protocol checks |

These are shared operation/lowering changes, not a kernel-name recognizer. There
is no new backend, public buffer representation, flag allocator, pool, admission
limit or watchdog. `StageBufferScalarReads` runs only when PyPTO autosync and Buffer IR are enabled
under the explicit `conservative_prefix` policy in A2/A3 AIV. With autosync off
or Buffer IR disabled, the original Functional read remains. When the PTOAS
planner is selected, staged Buffer input still delegates physical ownership; the
real downstream PTOAS C++ path is tested. No partial candidate is published
when ordinary whole-function admission refuses.

Exact providers may report unavailable rank-one extraction or GM selectors.
Those are not empty answers and must not discard useful local mathematics.
A conservative request may use sound whole-memory-space effects; that establishes
coverage rather than minimum physical demands.

## Original acceptance obligations

1. Independent access/value checks for element indices 0, 1, 7, 8 and the final
   valid element, including sources of length one. Poison adjacent GM elements
   and carrier padding; require one-element source reads and correct scalar SSA.
2. Zero/one/multiple visits, conditional skips, tail/control use and counter
   resets. Preserve original effectful evaluation cuts and scalar dependencies.
3. Actual logical/physical order, matching, acyclicity, eligible directions and
   consumption before republication, including surrounding parent commands.
4. Both alias policies under their stated contracts, physically overlapping
   views, prior local GM writes and missing external publication evidence.
5. All planners and Buffer-disabled delegation. Inspect real generated PTO/C++
   at non-32-byte GM offsets and run the existing local scalar/pitch regressions.
6. Refresh the existing original Qwen capture only after the reusable family
   passes. Retain the old denominator and report each newly reached obligation;
   rank-one admission does not qualify mixed-core hand protocols automatically.

Construction and retained state must depend on static read sites, not runtime
indices or trip counts. Count recovery, staging, logical protocol construction,
physical realization and source generation separately. This design does not
promise a device speedup: point staging adds a DMA and its required handshake.

## Implemented host gates and limitations

The first coherent gate passes **53 tests in 0.86 s** under the unchanged
120-second watchdog. The exact command is:

```bash
timeout 120s python -m pytest -q --require-ptoas \
  tests/ut/ir/transforms/test_gm_scalar_staging.py
```

The [receipt](44-autosync-gm-scalar-staging.json) retains the design/native evidence
and records implementation files, logs and publication status. The gate covers
all three existing planners, source lengths one and nine, source indices
0/1/7/8, both alias policies, and a symbolic rank-one extent. Small executions
vary empty/one/multiple loops, skipped branches and counter resets while keeping
one retained function. Loaded positive/negative values drive original scalar
control and a local write. The independent byte/command interpreter distinguishes
the four-byte GM source from the entire 32-byte UB write enclosure, checks output
values and verifies coverage, matching, acyclicity and causal physical reuse.
May-alias executions include permitted overlapping GM roots; no disjointness is
inferred from pointer names.

Real PTOAS-generated C++ contains TLOAD and GetValue for all three planners.
Autosync-disabled and Buffer-disabled paths preserve original `tensor.read`;
discarded reads and INT64 preserve whole-function delegation. Runtime trip
values do not clone carriers or commands: the fixture retains one scalar-read
site, two load sites including its ordinary input, and the original two loops.
The source's rank, allocation root, actual index and scalar SSA remain unchanged.

The selected transfer remains an asynchronous DMA with a qualified blocking
local scalar consumer, not a fictional one-pipe scalar operation. Static sites
require private carriers and can add handshakes or conservative ordering. No
minimum-demand, minimum-ID or device-speedup result follows. This gate does not
independently prove peer publication, native cache behavior, arbitrary effectful
while predicates, INT64 staging or exact GM boundary exports. Those premises
remain explicit. The original M1–M6 goals and complete Qwen ownership remain incomplete.

### Broader gate and unchanged whole-model capture

The post-format coherent build succeeds. A broader deduplicated regression run
records **3,132 passed, two failed, four warnings in 430.90 s**. Both failures
were stale `rank-2` diagnostic expectations: rank-one FP32 load/store inputs
still refuse under the new, more precise INT32 point-transfer diagnostic. Only
the expected diagnostic and its explanation were corrected. A fresh operator
module plus GM staging run passes **202 tests in 1.21 s**. These are separate
receipts; there is no claimed single green 3,134-test run. Final hooks run after
all documentation is frozen.

The original two-layer, sequence-128, batch-1 Qwen capture generates all 32 leaf
C++ files in 6.692 s inclusive. It reports 47 observed task-control owners,
**19 PyPTO leaves and 13 delegated leaves**, adding `token_embed_single` and
`token_embed_group` through this shared recipe. The previous 17/15 result remains
the baseline. This is one descriptive host measurement, not device or
analysis-only timing.

The remaining first obligations are four mixed-valid ADD leaves
(`out_proj_aiv`, `out_proj_aiv_0`, `rms_recip`, `rms_recip_0`), `qkpv_plan`'s
`tensor.write`, `rope_kv_cache`'s `tile.reshape`, `lm_head`'s Acc descriptor,
and six mixed-peer leaves whose existing GM pipe-buffer type is not admitted.
The latter are `qk_pv_partial_plan_aic`, `qk_pv_partial_plan_aiv`,
`qk_pv_single_block_0_aic`, `qk_pv_single_block_0_aiv`,
`qk_pv_single_block_aic` and `qk_pv_single_block_aiv`. These first diagnostics
are not all algorithm failures or proof of infeasibility. Existing peer
protocols remain their own owners; this package constructs no cross-core
synchronization. Exact exports and native qualification remain separate.

| Established result | Remaining premise or limitation | Appropriate claim |
| ------------------ | ------------------------------- | ----------------- |
| Direct INT32 GM read lowers through ordinary DMA plus local S payloads | Selected native TLOAD contract and original access-validity/publication premises | Reusable shared lowering recipe, not a new hardware theorem |
| Original ABI, SSA and control plus sound carrier enclosure pass bounded checks | Unsupported reads retain delegation; exact GM selectors unavailable | Conservative local synchronization under explicit policy |
| All three planners generate C++; actual PyPTO command checks use the existing pool | No device execution or uniform optimality result | Host feasibility for the qualified family |
