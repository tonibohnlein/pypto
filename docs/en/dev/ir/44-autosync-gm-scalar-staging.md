# GM scalar staging: proposed operation-contract subgate

Status: architecture accepted for implementation; implementation and semantic
validation are pending. Native instruction correctness remains the existing
selected-TLOAD premise, not a newly proved hardware result. This extends M2/M3 of the
[architecture plan](41-general-autosync-architecture-plan.md). It follows the
qualified [local scalar bridge](43-autosync-value-and-scalar-contracts.md).
No whole-Qwen ownership improvement or device result is claimed.

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

Initially accept only normalized direct assignments. A discarded or nested GM
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
- `include/pto/npu/a2a3/tload_common.hpp`, `TLoadGm2ubNd2nd`, computes one burst
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
values. The proposal does not assert that `PadValue::Null` proves only four
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

## Existing owners and proposed changes

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
limit or watchdog. The producer must not assume a newly introduced transfer is
synchronized when autosync is disabled: qualify its existing downstream owner
and test that path, or limit construction to the explicitly supported request.

Exact providers may report unavailable rank-one extraction or GM selectors.
Those are not empty answers and must not discard useful local mathematics.
A conservative request may use sound whole-memory-space effects; that establishes
coverage rather than minimum physical demands.

## Acceptance subgate

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
