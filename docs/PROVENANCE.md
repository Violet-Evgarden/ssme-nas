# Source provenance

The released implementation is derived from the NAS-Bench-101 Q200 and
NAS-Bench-201 Q100 main-experiment sources. It includes the structural edit
memory, native mutations, evolutionary controller, frozen Flash concepts and
initial populations used by the four supplied configurations.

## Modules and portability

| Module | Source and role |
|---|---|
| `core/memory.py` | Main-experiment relation completion, conditional partition compiler, rank readout and mutation selection |
| `core/nb101.py`, `core/nb201.py` | Native graph adapters, structural descriptors and candidate enumeration |
| `parent_selection.py`, `population.py` | Tournament draws, initial population rebuilding and elite retention |
| `controller.py`, `selection.py` | Search ordering, validation-based winner selection and winner sealing |
| `evaluators.py`, `runner.py` | Portable benchmark paths, stage-restricted measurements and persistent run outputs |

The port preserves numerical constants, edit ordering, random seed namespaces,
comparison tolerances and tie rules. Explicit search-space objects replace
process-global bindings. Local benchmark readers replace machine-specific
storage access; the NAS-Bench-101 reader follows the official TFRecord and
protobuf layouts, and the NAS-Bench-201 readers use the official benchmark APIs.

## Release verification

Checks against the experiment sources and recorded executions covered:

- 2,800 controller steps, including parent selection and survivor decisions.
- 14 complete memory compilations and 12 mutation proposals compared with the
  original implementation in the same Python runtime.
- 20 final memories and 60 mutation proposals compared with archived outputs.
  The largest floating-point discrepancy was approximately `1.78e-15`.

The repository tests also cover graph validity, concept validation, selection
rules, benchmark metric isolation, TFRecord integrity and end-to-end execution
with synthetic observations.

## Source fingerprints

SHA-256 fingerprints identify the original source files used for the port:

| Source | SHA-256 |
|---|---|
| NB101 search loop | `968f4549c2bcd4a3b350e795c8e36e2aac2bd3914e0ff377c755e30ff8276b41` |
| NB201 CIFAR-10 search loop | `0b1cd8931fe5d392548df2379f87cc3d29490b55585ae42b23fcde0130b2f114` |
| Parent selection | `1f90cf1741e3313d6f823d1132ee8ff8d6a5017bd0f3410174838baed1b6fc17` |
| Population retention | `c7ec212850a355938dba9cc13c0ff29e3d2310451c0f827d84c21bc44027aba5` |
