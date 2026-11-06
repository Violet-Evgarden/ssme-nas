"""End-to-end tests with generated labels, never benchmark measurements."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_search import main
from ssme_nas._utils import digest
from ssme_nas.evaluators import BenchmarkEvaluator
from ssme_nas.runner import load_inputs, read_json, run


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/nb201_cifar10_q100.json"


def save(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def fixture_config(folder, budget=25, runs=2):
    config = read_json(BASE_CONFIG)
    for field in ("concepts_file", "starts_file"):
        config[field] = str((BASE_CONFIG.parent / config[field]).resolve())
    config["queries_per_run"] = budget
    config["search_runs"] = runs
    return save(folder / "config.json", config)


class SyntheticBackend:
    """A pure identity-to-number function, not an architecture quality model."""
    def __init__(self):
        self.closed = False

    def read(self, identity, stage, protocol):
        value = 20 + int(digest([identity, stage])[:8], 16) % 6000 / 100
        return [{"repeat_id": 0, "value": value - 0.1},
                {"repeat_id": 1, "value": value + 0.1}]

    def close(self):
        self.closed = True


class TraceEvaluator(BenchmarkEvaluator):
    def __init__(self, config, space, output):
        self.backend = SyntheticBackend()
        super().__init__(config, space, self.backend)
        self.trace = []
        self.output = output

    def validation(self, ops, intent):
        result = super().validation(ops, intent)
        self.trace.append(("validation", intent["pool"], intent["query_index"]))
        return result

    def seal_winners(self, winners):
        super().seal_winners(winners)
        # The persistent seal and every result must precede enabling Test.
        if not (self.output / "WINNER_SEAL.json").is_file():
            raise AssertionError("Missing persistent seal before Test authorization")
        for pool in range(self.config["search_runs"]):
            if not (self.output / f"pools/{pool}/result.json").is_file():
                raise AssertionError("Missing completed pool before global seal")
        self.trace.append(("seal",))

    def test(self, ops, intent):
        result = super().test(ops, intent)
        self.trace.append(("test", intent["pool"]))
        return result


class RunnerTest(unittest.TestCase):
    def test_complete_chain_query_budget_seal_and_test_order(self):
        for budget in (10, 24, 25):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                config_path = fixture_config(folder, budget=budget)
                benchmark = save(folder / "synthetic.json", {})
                output = folder / "output"
                constructed = []

                def factory(config, path, space):
                    evaluator = TraceEvaluator(config, space, output)
                    constructed.append(evaluator)
                    return evaluator

                with patch("ssme_nas.runner.create_evaluator", side_effect=factory):
                    results = run(config_path, benchmark, output)
                evaluator = constructed[0]
                expected = [("validation", pool, index)
                            for pool in range(2) for index in range(budget)]
                expected += [("seal",), ("test", 0), ("test", 1)]
                self.assertEqual(evaluator.trace, expected)
                self.assertTrue(evaluator.backend.closed)
                self.assertEqual(results, read_json(output / "RESULTS.json"))
                self.assertEqual(results["validation"]["n"], 2)
                self.assertEqual(results["test"]["n"], 2)
                for pool in range(2):
                    result = read_json(output / f"pools/{pool}/result.json")
                    records = result["queried"]
                    self.assertEqual(len(records), budget)
                    self.assertEqual(len({q["architecture_id"] for q in records}), budget)
                    self.assertEqual(len(result["events"]), budget - 10)
                    self.assertEqual(result["final_memory"]["online"]["executed_edits"], budget - 10)
                    self.assertEqual(set(result["final_memory"]["source_ids"]),
                                     {q["architecture_id"] for q in records})
                    events = [json.loads(line) for line in
                              (output / f"pools/{pool}/events.jsonl").read_text().splitlines()]
                    self.assertEqual(sum(e["kind"] == "QUERY_INTENT" for e in events), budget)
                    self.assertEqual(sum(e["kind"] == "VALIDATION_RESULT" for e in events), budget)
                    self.assertFalse(any("test" in q["validation"] for q in records))
                    if budget > 20:
                        self.assertEqual(records[20]["proposal"]["parent_selection"]["reason"],
                                         "VALIDATION_TWO_WAY_TOURNAMENT")
                if budget == 25:
                    reasons = {q["proposal"]["reason"] for pool in range(2)
                               for q in read_json(output / f"pools/{pool}/result.json")["queried"][10:]}
                    self.assertIn("MEMORY_GREEDY", reasons)

    def test_validation_only_has_no_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config_path = fixture_config(folder, budget=25, runs=1)
            benchmark = save(folder / "synthetic.json", {})
            output = folder / "output"
            config, _, _, space = load_inputs(config_path)
            evaluator = TraceEvaluator(config, space, output)
            with patch("ssme_nas.runner.create_evaluator", return_value=evaluator):
                results = run(config_path, benchmark, output, validation_only=True)
            self.assertEqual(evaluator.trace[-1], ("seal",))
            self.assertFalse(any(event[0] == "test" for event in evaluator.trace))
            self.assertNotIn("test", results)
            self.assertFalse((output / "pools/0/test.json").exists())
            self.assertTrue(evaluator.backend.closed)

    def test_cli_with_exported_json_and_no_mocked_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config_path = fixture_config(folder, budget=10, runs=2)
            config, _, starts, space = load_inputs(config_path)
            table = {"format": "seme-nas-benchmark-v1", "space": "nb201", "dataset": "cifar10",
                     "validation": config["validation"], "test": config["test"], "architectures": {}}
            backend = SyntheticBackend()
            for pool in range(2):
                for ops in starts["pools"][str(pool)]:
                    identity = space.canonical(tuple(ops))
                    table["architectures"][identity] = {"ops": ops, **{
                        stage: backend.read(identity, stage, config[stage])
                        for stage in ("validation", "test")}}
            benchmark = save(folder / "synthetic.json", table)
            output = folder / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["--config", str(config_path), "--benchmark", str(benchmark),
                               "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertEqual(read_json(output / "RESULTS.json")["queries_per_run"], 10)

    def test_existing_output_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config_path = fixture_config(folder)
            output = folder / "output"
            output.mkdir()
            marker = save(output / "keep.json", {"preserve": True})
            before = marker.read_bytes()
            with patch("ssme_nas.runner.create_evaluator") as factory:
                with self.assertRaises(FileExistsError):
                    run(config_path, folder / "nonexistent.json", output)
                factory.assert_not_called()
            self.assertEqual(marker.read_bytes(), before)
            self.assertEqual([p.name for p in output.iterdir()], ["keep.json"])

    def test_concurrent_output_creation_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config_path = fixture_config(folder)
            benchmark = save(folder / "synthetic.json", {})
            output = folder / "output"
            config, _, _, space = load_inputs(config_path)
            evaluator = TraceEvaluator(config, space, output)

            def factory(*args):
                output.mkdir()
                save(output / "other-process.json", {"preserve": True})
                return evaluator

            with patch("ssme_nas.runner.create_evaluator", side_effect=factory):
                with self.assertRaises(FileExistsError):
                    run(config_path, benchmark, output)
            self.assertEqual([p.name for p in output.iterdir()], ["other-process.json"])
            self.assertTrue(evaluator.backend.closed)

    def test_invalid_configuration_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config_path = fixture_config(folder)
            original = read_json(config_path)
            changes = [("queries_per_run", 9), ("queries_per_run", True),
                       ("queries_per_run", 25.0), ("search_runs", 0),
                       ("search_runs", 11), ("initial_population", 10.0),
                       ("initial_population", 11), ("online_llm", True),
                       ("imported_performance_labels", 1), ("validation", []),
                       ("test", {"winner_only_after_all_winners_sealed": False})]
            for field, value in changes:
                with self.subTest(field=field, value=value):
                    config = deepcopy(original)
                    config[field] = value
                    save(config_path, config)
                    with patch("ssme_nas.runner.create_evaluator") as factory:
                        with self.assertRaises(ValueError):
                            run(config_path, folder / "missing.json", folder / "output")
                        factory.assert_not_called()
                    self.assertFalse((folder / "output").exists())

    def test_malformed_concepts_and_genotypes_raise_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config_path = fixture_config(folder)
            malformed = save(folder / "concepts.json", {"concepts": [{}] * 12})
            with self.assertRaises(ValueError):
                load_inputs(config_path, concepts_path=malformed)
            _, _, starts, _ = load_inputs(config_path)
            starts["pools"]["0"][0] = None
            malformed = save(folder / "starts.json", starts)
            with self.assertRaises(ValueError):
                load_inputs(config_path, starts_path=malformed)

    def test_all_shipped_inputs_and_large_integer_seeds_load_losslessly(self):
        configs = sorted((ROOT / "configs").glob("*.json"))
        self.assertEqual(len(configs), 4)
        for path in configs:
            config, concepts, starts, _ = load_inputs(path)
            raw_starts = read_json(path.parent / config["starts_file"])
            raw_concepts = read_json(path.parent / config["concepts_file"])
            self.assertEqual(starts, raw_starts)
            self.assertEqual(concepts, raw_concepts["concepts"])
            self.assertTrue(all(type(seed) is int for seed in starts["seeds"]))
            if config["dataset"] == "ImageNet16-120":
                self.assertGreater(max(starts["seeds"]), 2 ** 53)
                self.assertEqual(starts["seeds"][0], 7798545461926963494)


if __name__ == "__main__":
    unittest.main()
