# Configuration and integration guide

## Inspect the interface

Run these commands from the repository root:

```bash
python run_search.py --describe
python -m unittest discover -s tests -v
```

For requirements and supported execution, see the [README](../README.md).

## Select a configuration

The JSON files in [`configs/`](../configs/) specify the search space, dataset,
query budget, initial population and number of search runs. The `validation`
and `test` objects identify the benchmark measurements and repeat aggregation.
Initial-population evaluations are included in `queries_per_run`.

## Prepare the concepts

Use the request JSON for the selected search space, parse the model's JSON
response, and validate its structure, field coverage and bundle uniqueness.
The [prompt guide](../prompts/README.md) gives the steps and response schemas.
Save the validated object as `concepts.json` and retain its request and response
alongside it. Load the object's `concepts` array for the controller's `concepts`
argument. Keep that array fixed during search; observed edit evidence is updated
after each validation query.

## Connect the interfaces

[`search_pool`](../ssme_nas/controller.py) describes the search-loop integration:

| Argument | Role |
|---|---|
| `starts`, `seed`, `pool` | Ten distinct initial genotypes, the search seed and the run identifier |
| `concepts` | Validated concept list from `concepts.json` |
| `query_budget` | The configuration's `queries_per_run` |
| `space` | Canonical architecture identity and native genotype/edit validation |
| `evaluator` | Validation measurements for queried architectures |
| `audit` | Persistence of query, measurement and decision records |

The backend contracts are defined in [`contracts.py`](../ssme_nas/contracts.py).
The controller selects parents, evaluates proposed mutations, updates the
archive and memory, and returns the validation-selected winner. Final selection
and winner sealing are implemented in [`selection.py`](../ssme_nas/selection.py).
