# Autosync review repairs

## R1: bounded discovery and standalone physical output

Closed-prefix discovery now visits shared expression nodes once while retaining
statement-occurrence service accounting. Token-use summaries are memoized by
expression identity and charged at each actual use. Only proved empty completion
facts and their pure SSA aliases are erased. Unsupported token frames refuse
before mutation.

Every successful logical-to-physical lowering checks that logical operations,
SyncTokenType values, tuple members and function signatures are absent. Runtime
TaskIds remain separate. Invalid compiler output raises an internal error before
invocation ownership publication.

Fresh host validation includes the 28-shared-join regression, eight completed
alias cases (payload/command-only, optimization on/off and serialization),
existing token carry/capture and retained-child tests. The first five-module
repair run passed 166 tests. The interface run passed 154 tests and exposed six
parser failures in the previously paused, unqualified active-domain test package;
those failures remain recorded separately and are not a green milestone claim.

These checks establish admitted host-model behavior, not native/device correctness.
The logical protocol, eligible ID pool and allocation algorithms are unchanged.

## R2: policy-independent native workspace legality

Selected native operation qualification now exposes one operand/workspace
separation predicate. Exact access recovery supplies its retained physical bounds;
conservative analysis and actual-token resource qualification use an indexed local
placement inventory. An unavailable bound remains unavailable. Adding ordering
between calls cannot repair overlap within one native instruction.

All relevant explicit operands must be disjoint from each selected same-space
implicit workspace. Half-open edge equality is legal; arithmetic is checked before
publishing bounds. Aliases preserve allocation identity. Runtime subviews inherit
the Buffer IR contract that their window stays inside the source allocation
([types](02-types.md)); their parent enclosure is conservative, not an exact
selector. Slot indices require separate evidence; this adapter currently accepts
static in-range selectors and refuses unsupported dynamic selectors for workspace
qualification. Borrowed operands lacking allocation provenance can already fail
Buffer IR verification, independently of the synchronization-policy check.

Public regression routes cover exact/conservative analysis, insertion, lowering
of actual closed tokens and enabled transactional realization, with read/write
operands, view aliases, workspace edges, unavailable placement and slots. Implicit
workspace effects remain in inter-operation hazard analysis as well.

## R3: static row descriptors after dynamic alias promotion

Static dense row-major row-recipe operands and outputs now use the existing
static descriptor-window bridge. The ordinary per-use metadata restoration runs
first. Each use keeps its original valid extent and physical pitch even if a
different alias promoted the shared allocation descriptor to runtime metadata.
Column-major carriers retain their existing independent typed-root descriptors;
dynamic operands retain their guarded full-column contract.

Sixteen focused tests cover PYPTO, DSA_RP and PTOAS placement, partial/full
columns, sum/subtraction and max/multiplication, alias reuse after the row call,
zero/short metadata values, Buffer-disabled controls and unproved-metadata
refusal. The actual row-call descriptors and independent Torch values are
checked. The final combined descriptor/workspace/metadata run passed 134 tests.
No payload arithmetic, guards, physical layout or synchronization policy changes.

## R4: nested publication schedules and fresh gates

The depth sweep isolated a separate representation expansion in publication
ranking. The general grammar partitions every frame against all raw family
boundaries; canonical carry families can then produce one piece per publication.
An optional producer now recognizes a complete mixed-radix successor partition
for one source/consumer site pair in one original leaf. Every nonsingleton carry
class must occur exactly once, with matching original frames, owner, domains,
unit lattice steps, bases and checked products. Its disjoint classes partition
all source ordinals except the final one, so each raw label receives the affine
rank equal to its original source ordinal. No raw label or matching map changes.

Before accepting this representation, the existing actual-token order service
proves consumption at each target occurrence precedes the next publication's
source payload in that same visit, uniformly over the whole class. That is a
stronger sufficient query than the required WAIT-to-SET path. Consecutive reuse
and matching SET-to-WAIT paths also justify each positive budget. Failed shape
qualification or proof retains the existing grammar. The allocator, eligibility,
protocol and logical tokens are unchanged. This is a bounded producer improvement;
arbitrary family unions and other large interfaces remain separate limitations.
Structural qualification/rank construction costs O(FD) for F raw classes and D
frames; existing transition-query costs are additional.

Rebuilt, binary-pinned single runs on identical retained token inputs measured:

| Depth | Baseline pieces | Candidate pieces | Baseline SET sites | Candidate SET sites | Baseline guards | Candidate guards | Baseline physical seconds | Candidate physical seconds |
| ----- | --------------- | ---------------- | ------------------ | ------------------- | --------------- | ---------------- | ------------------------- | -------------------------- |
| 8 | 255 | 8 | 260 | 13 | 2007 | 88 | 0.10430 | 0.00518 |
| 12 | 4095 | 12 | 4100 | 17 | 32711 | 180 | 2.14734 | 0.00976 |
| 16 | not attempted | 16 | not attempted | 21 | not attempted | 304 | not attempted | 0.01616 |

The same 12-second per-stage watchdog applies. These are host construction
observations, not device speedups. Independent bounded checks compare actual
ranks/labels, token and flag payload order, coverage, matching, acyclicity and
reuse. Duplicate uses, genuine activity holes, foreign owners and missing reuse
paths retain conservative behavior. Operator regression updates preserve empty
static-domain refusals and move runtime metadata evidence to function/use sites.

The final isolated repair gate passed **2,910 test executions, zero failures**
with four existing deprecation warnings in 397.39 seconds. Two explicit module
paths repeat entries in the broad manifest; this count is executions, not unique
tests. The [receipt](../../../en/dev/ir/42-autosync-review-repairs.json) retains
commands, binary/source pins and raw-log hashes. R1–R4 repair qualification does
not complete the original M1–M6 coverage goal.
