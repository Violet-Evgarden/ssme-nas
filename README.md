# SEME-NAS

**Evolutionary Neural Architecture Search Driven by Structural Edit Memory with LLM-Refined Concepts**

SEME-NAS refines structural concepts once before search and uses observed edit
outcomes to guide evolutionary mutation. This repository contains the complete
search implementation, native search-space adapters, benchmark evaluators,
frozen concepts and initial populations. The Python package name is `ssme_nas`.

## Requirements

- Python 3.10 or newer.
- The memory, mutation, controller and offline tests use the Python standard
  library.
- NAS-Bench-101 uses a native TFRecord reader; NAS-Bench-201 uses its benchmark
  API. Both read local official benchmark tables. No network training, GPU or LLM API key is needed
  for tabular search with the supplied frozen concepts.

Install this package from the repository root:

```bash
python -m pip install -e .
```

Benchmark installation instructions are in the
[execution guide](docs/RELEASE_STATUS.md#benchmark-installation). Obtain the
benchmark files from [NAS-Bench-101](https://github.com/google-research/nasbench),
[NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201), or
[NATS-Bench](https://github.com/D-X-Y/NATS-Bench).

## Run the search

NAS-Bench-101:

```bash
python run_search.py --config configs/nb101_cifar10_q200.json --benchmark /path/to/nasbench_only108.tfrecord --output runs/nb101
```

NAS-Bench-201, CIFAR-10:

```bash
python run_search.py --config configs/nb201_cifar10_q100.json --benchmark /path/to/NATS-tss-v1_0-3ffb9.pickle.pbz2 --output runs/nb201_cifar10
```

Each default configuration runs the same method independently twenty times,
with ten randomly sampled initial architectures per search. Seeds and initial
populations are saved for reproducibility. NAS-Bench-201 also accepts its
official `.pth` benchmark.
Use `--concepts FILE` and `--starts FILE` to supply alternative JSON artifacts;
use `--validation-only` to finish after validation-based winner selection.
Choose a fresh output directory for each run.

Read-only inspection and offline tests:

```bash
python run_search.py --describe
python -m unittest discover -s tests -v
```

## Configurations

| Configuration | Dataset | Queries per run | Initial population | Search runs |
|---|---|---:|---:|---:|
| `nb101_cifar10_q200.json` | CIFAR-10 | 200 | 10 | 20 |
| `nb201_cifar10_q100.json` | CIFAR-10 | 100 | 10 | 20 |
| `nb201_cifar100_q100.json` | CIFAR-100 | 100 | 10 | 20 |
| `nb201_imagenet16_120_q100.json` | ImageNet16-120 | 100 | 10 | 20 |

Initial-population evaluations count toward the query budget. Search uses
validation measurements; test measurements are read only after all
validation-selected winners have been sealed.

A query reveals the recorded validation performance of one distinct architecture.
NAS-Bench-101 uses the 108-epoch records; NAS-Bench-201 uses the 200-epoch
records in the NATS-Bench topology release
`NATS-tss-v1_0-3ffb9.pickle.pbz2`. Thus a NAS-Bench-201 run evaluates at most
100 distinct architectures, including its ten initial architectures, and selects
its final architecture from their validation feedback. The supplied evaluators
read official training records rather than retraining networks during search.

### Contextual bandwidth

The contextual evidence kernel is `exp(-distance / H)`. The implementation uses
`H = 0.34090909090909094`, defined in
[`ssme_nas/core/memory.py`](ssme_nas/core/memory.py).
`H` is a positive bandwidth hyperparameter set before search: smaller values
concentrate weight on nearby edit relations, while larger values retain more
distant evidence. The supplied experiments use this same value throughout;
it is not updated by the LLM or by search feedback.

### Released prompts

The complete concept-refinement requests are available as
[`prompts/nb101_flash_request.json`](prompts/nb101_flash_request.json) and
[`prompts/nb201_flash_request.json`](prompts/nb201_flash_request.json).
They include the system and user prompts, structural definitions, unlabeled
examples, model identifier and generation settings. Response schemas and usage
instructions are in the [prompt guide](prompts/README.md).

## Programs and files

| Path | Purpose |
|---|---|
| `run_search.py` | Command-line configuration, execution and result export |
| `ssme_nas/core/memory.py` | Observed-neighbor completion, conditional partitions, rank readout and mutation |
| `ssme_nas/core/nb101.py`, `nb201.py` | Native graph operations, structural fields and candidate edits |
| `ssme_nas/spaces.py` | Search-space adapter selection |
| `ssme_nas/controller.py` | Initialization, mutation, evaluation and memory updates |
| `ssme_nas/parent_selection.py`, `population.py` | Parent tournaments and elite survivor retention |
| `ssme_nas/evaluators.py` | Benchmark measurements and validation/test stage separation |
| `ssme_nas/selection.py` | Validation-based winner selection and sealing |
| `ssme_nas/private_core.py` | Compatibility import path for the released core |
| `configs/` | Four main-experiment configurations |
| `artifacts/concepts/` | Frozen, validated Flash concepts for both search spaces |
| `artifacts/starts/` | Initial populations and seeds for each configuration |
| `ssme_nas/transfer.py`, `artifacts/transfer/` | Deterministic cross-space field adapters and instantiated source bundles |
| `prompts/` | Actual concept-generation request JSON and response schemas |
| `tests/` | Offline unit, evaluator and archived-parity regression checks |

Outputs include `run.json`, copied inputs, per-pool records,
`WINNERS.json`, `WINNER_SEAL.json` and `RESULTS.json`.
See the [execution guide](docs/RELEASE_STATUS.md) for their formats and the
[prompt guide](prompts/README.md) for concept generation.
The [cross-space adapter guide](docs/TRANSFER.md) specifies field projections
and the exact bundles used for exploratory concept reuse.

## License

Source availability does not change the repository's licensing terms.
See [NOTICE.md](NOTICE.md) for project rights and third-party attribution.
