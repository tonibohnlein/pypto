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
