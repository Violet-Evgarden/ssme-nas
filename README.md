# SEME-NAS

**Evolutionary Neural Architecture Search Driven by Structural Edit Memory with LLM-Refined Concepts**

SEME-NAS uses an offline LLM to refine structural concepts, then guides
evolutionary mutation with edit memory updated from observed validation feedback.
The Python package name remains `ssme_nas`.

## Requirements

- Python 3.10 or newer.
- No third-party Python packages are required for the included controller
  components and tests.
- Inspecting the JSON prompts and running the tests require neither a GPU nor
  an API key.

## Usage

From the repository root:

```bash
python run_search.py --describe
python -m unittest discover -s tests -v
```

These commands inspect the interface and test the included components. Full
search execution requires the separate core implementation.

## Programs and files

| Path | Purpose |
|---|---|
| `run_search.py` | Command-line interface and implementation-availability check |
| `ssme_nas/controller.py` | Search-loop integration: initialization, mutation, evaluation and feedback |
| `ssme_nas/parent_selection.py` | Bootstrap scheduling and validation-based parent tournaments |
| `ssme_nas/population.py` | Population rebuilding and elite survivor retention |
| `ssme_nas/selection.py` | Final validation-based selection and winner sealing |
| `ssme_nas/contracts.py` | Search-space, evaluator, memory and record interfaces |
| `ssme_nas/private_core.py` | Interface boundary for the separate memory and mutation implementation |
| `configs/` | NAS-Bench-101 and NAS-Bench-201 experiment configurations |
| `prompts/` | Actual offline concept-generation requests and response-format schemas |
| `tests/` | Controller, configuration and prompt-artifact checks |

## Configurations

| Configuration | Dataset | Validation queries | Initial population |
|---|---|---:|---:|
| `nb101_cifar10_q200.json` | CIFAR-10 | 200 | 10 |
| `nb201_cifar10_q100.json` | CIFAR-10 | 100 | 10 |
| `nb201_cifar100_q100.json` | CIFAR-100 | 100 | 10 |
| `nb201_imagenet16_120_q100.json` | ImageNet16-120 | 100 | 10 |

Initial-population evaluations count toward the query budget. Search uses
validation measurements; test measurements are read for the selected winners.

Benchmark resources:
[NAS-Bench-101](https://github.com/google-research/nasbench) and
[NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201).

## LLM prompts

The two request files contain the actual system message, user message and
generation settings used for offline concept construction:

- [NAS-Bench-101 request](prompts/nb101_flash_request.json)
- [NAS-Bench-201 request](prompts/nb201_flash_request.json)

Each user message supplies the native grammar, 22 structural-field definitions
and 16 unlabelled edit examples. Each request uses `deepseek-flash`,
`temperature=0`, `max_tokens=4096`, disabled thinking and JSON output.
The NAS-Bench-201 request is shared across its three datasets.

The expected output is a `concepts` array; each concept contains a `features`
array of existing field names and a short structural `meaning`.
See [prompt format](prompts/README.md) and the accompanying JSON schemas.
The request files contain no credentials and are not executed by the test suite.

## License

See [NOTICE.md](NOTICE.md).
