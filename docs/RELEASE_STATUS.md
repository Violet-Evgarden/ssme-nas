# Execution guide

## Benchmark installation

Install the repository package:

```bash
python -m pip install -e .
```

NAS-Bench-101's native TFRecord reader needs no TensorFlow installation.
The optional CRC package accelerates file-integrity checks:

```bash
python -m pip install google-crc32c
```

For the NATS-Bench topology table:

```bash
python -m pip install nats-bench torch
```

For the NAS-Bench-201 `.pth` table instead:

```bash
python -m pip install nas-bench-201 torch numpy
```

The benchmark adapters load official tables locally; they do not train
networks. Follow the benchmark providers' download instructions:

- [NAS-Bench-101](https://github.com/google-research/nasbench): use
  `nasbench_only108.tfrecord` or the full TFRecord.
- [NATS-Bench](https://github.com/D-X-Y/NATS-Bench): use the topology-space
  `NATS-tss-*.pickle.pbz2` table.
- [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201): use
  `NAS-Bench-201-v1_1-096897.pth`.

Only load benchmark files obtained from trusted sources; the official
PyTorch and pickle-based formats deserialize Python objects.

## Select and run a configuration

```bash
python run_search.py --config configs/nb101_cifar10_q200.json --benchmark /path/to/nasbench_only108.tfrecord --output runs/nb101
python run_search.py --config configs/nb201_cifar10_q100.json --benchmark /path/to/NATS-tss-v1_0-3ffb9.pickle.pbz2 --output runs/nb201_cifar10
```

The JSON configurations specify the space, dataset, query budget, twenty search
runs, and benchmark measurement policies. Validation selects one of the three
official training repeats uniformly, using an independent seed derived from the
search seed and architecture identity. This single observation is cached within
the run and its repeat ID is recorded. The initial ten architectures
are included in each run's query budget. The CIFAR-100 and ImageNet16-120
configurations use the same topology benchmark with their respective datasets.

The default concepts are `artifacts/concepts/nb101_flash.json` and
`artifacts/concepts/nb201_flash.json`. Initial populations and seeds are stored
under `artifacts/starts/`. The supplied artifacts make an LLM call unnecessary
when running the released search.

Each configuration repeats the same method independently twenty times. Each
search begins with ten distinct, uniformly sampled random architectures and
uses its own seed. Seeds and populations are stored in the starts file for
reproducibility. Use `--starts FILE` to supply a different random initialization.

Optional arguments:

| Argument | Use |
|---|---|
| `--concepts FILE` | Override the frozen concept JSON |
| `--starts FILE` | Override initial populations and seeds |
| `--validation-only` | Stop after validation-based winner selection |
| `--describe` | Inspect supported execution without loading a benchmark |

The concepts remain fixed during a search. The observed relation memory,
partitions and numerical scores update after validation queries.

## Outputs

Each output directory contains:

| Path | Contents |
|---|---|
| `run.json` | Run metadata |
| `inputs/config.json` | Configuration used |
| `inputs/concepts.json` | Frozen concepts used |
| `inputs/starts.json` | Initial populations and seeds used |
| `pools/<pool>/events.jsonl` | Query intents, validation measurements and decisions |
| `pools/<pool>/result.json` | Search archive, final population and selected winner |
| `WINNERS.json`, `WINNER_SEAL.json` | Winners fixed using validation before test access |
| `RESULTS.json` | Per-run and aggregate results |

With `--validation-only`, no winner test measurements are requested.
Use a new output directory rather than overwriting an earlier run.

## Python interfaces

```python
from ssme_nas.spaces import get_space
from ssme_nas.private_core import compile_memory, propose_mutation

space = get_space("nb201")
memory = compile_memory(records, concepts, space=space)
proposal = propose_mutation(parent, visited, memory, seed, step)
```

`records` contains evaluated architectures and their validation measurements.
`concepts` is the validated concept list, including stable identifiers and
structural-field bundles. `memory.data` exposes compiled partitions, retained
relations, rank information and relation provenance. `proposal` contains
`architecture_id`, `ops`, the selected edit and its readout diagnostics.

The complete search loop is
[`search_pool`](../ssme_nas/controller.py). Its evaluator supplies validation
measurements, and its audit sink persists the query and decision sequence.
Direct evaluator calls must include `search_seed`, `pool`, `query_index`, and
`architecture_id` in the validation intent. Benchmark files must contain all
three validation repeats for every queried architecture. The final winner's
test accuracy retains its all-available-repeat mean; test values do not enter
the search.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic data and regression fixtures. They do not call an LLM,
train a network or start a benchmark search.
