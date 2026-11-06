# Source provenance

The released implementation is derived from the NAS-Bench-101 Q200 and
NAS-Bench-201 Q100 main-experiment sources. It includes the structural edit
memory, native mutations, evolutionary controller, frozen Flash concepts and
initial populations used by the four supplied configurations.

## Twenty-run public inputs

On 2026-11-06, the four defaults were extended from ten to twenty search runs.
Pools 0–9 and their seeds were preserved exactly from commit `aa26fae`.
Pools 10–19 were generated once without reading any validation or test scores;
they are public reproduction inputs, not a reconstruction of runs performed
on another computer or evidence of twenty measured results. No benchmark
search or new LLM call was performed for this configuration update.

The generation namespace is `seme-nas-public-starts-extension-20261106-v1`.
For each artifact stem and pool index, its seed is
`int(digest([namespace, artifact_stem, pool])[:16], 16) & ((1 << 63) - 1)`,
using `ssme_nas._utils.digest`. New seeds are distinct from all retained seeds
and from one another. With `random.Random(seed).sample(domain, 10)`, each pool
samples ten distinct architectures uniformly without replacement; overlap
between pools is allowed. No outcome-driven redrawing or structural filtering
is used.

- NAS-Bench-101 samples from all 423,624 official identities sorted by hash.
  Only structural metadata is read. The existing `core.nb101.padded` lift uses
  `int(digest([seed, 'inactive-node-labels', architecture_hash])[:16], 16)`
  as its separate random seed. The resulting official identity is checked.
- NAS-Bench-201 samples from `itertools.product(core.nb201.OPS, repeat=6)`,
  containing all 15,625 structures in the original operation order.

The [extension manifest](../artifacts/starts/EXTENSION_20_RUNS.json) records
the new seeds and identities, domain fingerprint and preserved-input hashes.
All twenty pools' identity draws were replayed against the original sampling
rules. The frozen concepts, scoring, controller, repeat aggregation and
per-run query budgets are unchanged; the initial population remains ten.
Changing the run count does not change any historical result table.

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
