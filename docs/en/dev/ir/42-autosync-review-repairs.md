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
