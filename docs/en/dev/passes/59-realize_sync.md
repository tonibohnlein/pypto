# RealizeSync (experimental)

This optional pass tries PyPTO synchronization per whole kernel after storage
lowering. **PTOAS autosync remains enabled in both modes.** Default compilation
continues to delegate synchronization to PTOAS.

```python
from pypto import passes

# Explicit caller promise: distinct GM arguments denote disjoint allocations.
with passes.PassContext([], enable_pypto_autosync=True,
                        autosync_disjoint_gm=True):
    compiled = kernel.compile(a, b, c, config=config)

# Turn PyPTO's pass off; PTOAS still handles synchronization.
with passes.PassContext([], enable_pypto_autosync=False):
    baseline = kernel.compile(a, b, c, config=config)
```

PyPTO attempts the existing placed-access importer, compact minimum-demand
analysis and physical realization only for physically placed Buffer IR on the
qualified AIC matmul fragment and manual a2a3 macro variants. Missing placement,
GM disjointness, target, access, control, endpoint or resource qualifications
leave the **original kernel body** for PTOAS. This delegation does not promise
that PTOAS supports every input; its failures still propagate. Internal compiler
errors are not converted into fallback. The retained-analysis research APIs
remain strict and report qualification failures directly.

A successful attempt publishes its complete body and the boolean
`pypto_autosync=True` attribute. PTOAS skips insertion independently for each
function containing explicit SET/WAIT or record/wait events. Thus a module can
contain both PyPTO-synchronized and delegated functions. **Barriers alone do not
trigger this skip.** Empty or barrier-only PyPTO candidates are discarded and the
original body is delegated; the pass never manufactures a dummy flag. The pass
never publishes partial synchronization on a failed attempt. Already present
user synchronization remains governed by PTOAS's existing skip rule, rather
than being completed or independently validated by this switch.

For an opted-in compilation, `report/autosync.json` lists each attempted kernel's
selected route (`pypto` or `ptoas`) and the fallback reason. This is selection
metadata, not a certificate that PTOAS inserted commands or that compilation
succeeded. External-source, builtin-template and runtime-bound functions retain
their external ownership and are not included in this report. Do not edit owned
bodies or forge the attribute: like physical addresses, it is trusted compiler
output, not a proof checker for arbitrary user edits. Reapplying the pass to an
owned plan is refused.

The toggle and disjointness contract are part of JIT cache identity. Dumping and
profiling preserve them. A `PassManager` cannot change either after construction.
Allocation algorithms are unchanged. This interface does not expose an unsafe
synchronization-free execution mode.

The qualified [physical construction](../ir/15-physical-demand-realization.md)
establishes modeled payload order, matching and legal directional reuse. It
assumes fresh invocation entry and closes notifications; it does **not** prove
that returning from the function has completed its last asynchronous payload.
Native macro semantics and runtime invocation completion require separate
device qualification.
