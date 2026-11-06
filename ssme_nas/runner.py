"""Portable inputs, persistent query records, and winner-only Test evaluation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics

from . import __version__
from ._utils import digest
from .controller import search_pool
from .core.memory import validate_concepts
from .evaluators import create_evaluator, _protocol
from .selection import seal_winners
from .spaces import get_space


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


class JsonlAudit:
    """Append and flush each event before the controller takes its next action."""
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("x", encoding="utf-8", newline="\n")

    def append(self, kind, record):
        self._handle.write(json.dumps({"kind": kind, "record": record},
                                     ensure_ascii=False, allow_nan=False) + "\n")
        self._handle.flush()

    def close(self):
        self._handle.close()


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def load_inputs(config_path, concepts_path=None, starts_path=None):
    config_path = Path(config_path).expanduser().resolve()
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    for key in ("space", "dataset", "queries_per_run", "search_runs", "initial_population",
                "online_llm", "imported_performance_labels", "validation", "test"):
        if key not in config:
            raise ValueError(f"Missing configuration field: {key}")
    _positive_integer(config["queries_per_run"], "queries_per_run", 10)
    _positive_integer(config["search_runs"], "search_runs")
    if _positive_integer(config["initial_population"], "initial_population") != 10:
        raise ValueError("The controller uses an initial population of ten.")
    if config["online_llm"] is not False or config["imported_performance_labels"] != 0:
        raise ValueError("Search uses frozen concepts and current-run validation feedback only.")
    if not isinstance(config["test"], dict) or config["test"].get("winner_only_after_all_winners_sealed") is not True:
        raise ValueError("Test requires sealing all validation-selected winners first.")
    if not isinstance(config["validation"], dict):
        raise ValueError("Validation settings must be a JSON object.")
    _protocol(config)
    space = get_space(config["space"])

    def artifact(override, field):
        if override is not None:
            return Path(override).expanduser().resolve()
        if not isinstance(config.get(field), str):
            raise ValueError(f"Missing artifact path: {field}")
        return (config_path.parent / config[field]).resolve()

    raw_concepts = read_json(artifact(concepts_path, "concepts_file"))
    starts = read_json(artifact(starts_path, "starts_file"))
    if not isinstance(raw_concepts, dict) or raw_concepts.get("space", space.name) != space.name:
        raise ValueError("Concept artifact search space mismatch.")
    raw = raw_concepts.get("concepts", raw_concepts.get("accepted"))
    if not isinstance(raw, list) or not (8 <= len(raw) <= 12 if space.name == "nb101" else len(raw) == 12):
        raise ValueError("Expected 8-12 NB101 concepts or 12 NB201 concepts.")
    # Explicit IDs determine split tie breaks and must survive loading.
    try:
        concepts, rejected = validate_concepts({"concepts": raw}, space)
    except (AssertionError, KeyError, TypeError) as exc:
        raise ValueError("Invalid structural concept schema.") from exc
    if rejected:
        raise ValueError("Invalid or duplicated structural concept bundles.")
    for concept, source in zip(concepts, raw):
        if not source["meaning"].strip():
            raise ValueError("Each concept requires a nonempty structural meaning.")
        if "id" in source:
            concept["id"] = source["id"]
        concept["features"] = list(source["features"])
    identities = [concept["id"] for concept in concepts]
    if any(not isinstance(identity, str) or not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("Concept IDs must be unique, nonempty strings.")
    if {field for concept in concepts for field in concept["features"]} != set(space.CATALOG):
        raise ValueError("Concepts must cover the complete native field catalog.")
    if not isinstance(starts, dict) or starts.get("space", space.name) != space.name:
        raise ValueError("Initial-population search space mismatch.")
    seeds, pools = starts.get("seeds"), starts.get("pools")
    if not isinstance(seeds, list) or len(seeds) < config["search_runs"] or not isinstance(pools, dict):
        raise ValueError("Initial populations and seeds must cover every configured run.")
    for pool in range(config["search_runs"]):
        _positive_integer(seeds[pool], "search seed", 0)
        population = pools.get(str(pool))
        if not isinstance(population, list) or len(population) != 10:
            raise ValueError(f"Pool {pool} must contain ten native genotypes.")
        for ops in population:
            if not isinstance(ops, list):
                raise ValueError(f"Pool {pool} contains a malformed native genotype.")
            space.validate(tuple(ops))
        if len({space.canonical(tuple(ops)) for ops in population}) != 10:
            raise ValueError(f"Pool {pool} contains duplicate canonical architectures.")
    return config, concepts, starts, space


def _summary(values):
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def run(config_path, benchmark_path, output_path, *, concepts_path=None,
        starts_path=None, validation_only=False, progress=None):
    """Execute configured pools, seal all winners, then read final Test."""
    config, concepts, starts, space = load_inputs(config_path, concepts_path, starts_path)
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Output directory already exists; choose a new path to preserve its records.")
    benchmark = Path(benchmark_path).expanduser().resolve()
    if progress:
        progress("Loading and checking the local benchmark; the initial scan may take several minutes.")
    evaluator = create_evaluator(config, benchmark, space)
    started = datetime.now(timezone.utc).isoformat()
    output_created = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        output_created = True
        write_json(output / "inputs/config.json", config)
        write_json(output / "inputs/concepts.json", {"space": space.name, "concepts": concepts})
        write_json(output / "inputs/starts.json", starts)
        write_json(output / "run.json", {
            "implementation_version": __version__, "python": platform.python_version(),
            "started_utc": started, "benchmark_file": benchmark.name,
            "benchmark_bytes": benchmark.stat().st_size,
            "config_sha256": digest(config), "concepts_sha256": digest(concepts),
            "starts_sha256": digest(starts), "validation_only": validation_only,
        })
        winners = []
        for pool in range(config["search_runs"]):
            if progress:
                progress(f"Run {pool + 1}/{config['search_runs']}: {config['queries_per_run']} validation queries")
            pool_dir = output / "pools" / str(pool)
            audit = JsonlAudit(pool_dir / "events.jsonl")
            try:
                result = search_pool(starts["pools"][str(pool)], starts["seeds"][pool],
                                     concepts, config["queries_per_run"], pool, space, evaluator, audit)
            finally:
                audit.close()
            write_json(pool_dir / "result.json", result)
            winners.append(result["winner"])
        write_json(output / "WINNERS.json", winners)
        seal = seal_winners(output / "WINNER_SEAL.json", winners)
        evaluator.seal_winners(winners)
        final = []
        for winner in winners:
            row = {"pool": winner["pool"], "architecture_id": winner["architecture_id"],
                   "ops": winner["ops"], "validation": winner["validation"]}
            if not validation_only:
                row["test"] = evaluator.test(tuple(winner["ops"]), winner)
                write_json(output / "pools" / str(winner["pool"]) / "test.json", row["test"])
            final.append(row)
        results = {"method": "SEME-NAS", "space": config["space"], "dataset": config["dataset"],
                   "search_runs": config["search_runs"], "queries_per_run": config["queries_per_run"],
                   "winners_sha256": seal["winners_sha256"], "winners": final,
                   "validation": _summary([r["validation"]["value"] for r in final]),
                   "completed_utc": datetime.now(timezone.utc).isoformat()}
        if not validation_only:
            results["test"] = _summary([r["test"]["value"] for r in final])
        write_json(output / "RESULTS.json", results)
        return results
    except BaseException as exc:
        if output_created and output.is_dir() and not (output / "FAILURE.json").exists():
            write_json(output / "FAILURE.json", {"error_type": type(exc).__name__, "message": str(exc),
                                                "utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        evaluator.close()
