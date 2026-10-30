# Selected source provenance

The preview was prepared on 2026-10-30 from local, archived main-experiment
sources. It does not include the private project's Git history or storage paths.
Repository name `ssme-nas` is the requested public name; some historical private
artifacts used earlier working names. No experiment was rerun or renamed in its
original archive for this release.

## Archive identities

| Source run | Selected scope |
|---|---|
| `nb101-new-flash-paired-recheck-q200-20261028` | Current full-coverage Flash, native NAS-Bench-101, Q200 |
| `nb201-flash-fullcoverage-c10-q100-20261029` | Current Flash, NAS-Bench-201/CIFAR-10, Q100 |
| `nb201-flash-fullcoverage-c100-q100-20261029` | Corresponding CIFAR-100 main run |
| `nb201-flash-fullcoverage-in16-q100-20261029` | Corresponding ImageNet16-120 main run |

## Selection and adaptations

- `parent_selection.py`: selected from the archived tournament component;
  reformatted and imports moved to a public utility module. The legacy random
  namespace is retained to avoid silently changing deterministic draws.
- `population.py`: selected bootstrap/elite-retention component, with explicit
  input exceptions in place of assertions. The comparison tolerance and tie
  rules are retained.
- `controller.py`: a readable integration scaffold adapted from the two archived
  search loops. The query budget is parameterized for the documented examples;
  private readers, manifests, checkpointing, detailed instrumentation, and core
  implementations are not copied. It is **not a byte-identical runnable release**.
- `selection.py`: extracted selection rule plus a new portable seal helper.
  It preserves raw-score argmax and identity ties; it is not the archived
  benchmark access-control implementation.
- `contracts.py`, `private_core.py`, CLI, tests, and documentation are release
  scaffolding, not additional scientific algorithms.

## Original file fingerprints

These SHA-256 fingerprints identify selected private source files, not a claim
that the adapted public files are byte-identical.

| Source | SHA-256 |
|---|---|
| NB101 `run.py` | `968f4549c2bcd4a3b350e795c8e36e2aac2bd3914e0ff377c755e30ff8276b41` |
| NB201 C10 `run.py` | `0b1cd8931fe5d392548df2379f87cc3d29490b55585ae42b23fcde0130b2f114` |
| `parent_selection.py` | `1f90cf1741e3313d6f823d1132ee8ff8d6a5017bd0f3410174838baed1b6fc17` |
| `population.py` | `c7ec212850a355938dba9cc13c0ff29e3d2310451c0f827d84c21bc44027aba5` |

Original private source, failed runs, credentials, and numerical records remain
outside this repository. No unverified user-supplied summary is converted into
a measured result by publishing this code preview.
