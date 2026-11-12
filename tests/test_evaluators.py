"""Synthetic API-contract tests; no official data, training, or search scores."""
import base64
from copy import deepcopy
import json
import random
from pathlib import Path
import tempfile
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ssme_nas.evaluators import BenchmarkDataError, create_evaluator, _crc32c, _masked_crc32c


ROOT = Path(__file__).resolve().parents[1]


def config(name="nb201_cifar10_q100", runs=1, queries=2):
    value = json.loads((ROOT / "configs" / (name + ".json")).read_text(encoding="utf-8"))
    value.update(search_runs=runs, queries_per_run=queries)
    return value


class Space:
    def canonical(self, ops):
        return ops[0]


def intent(identity, index=0, pool=0, seed=123):
    return dict(pool=pool, search_seed=seed, query_index=index, architecture_id=identity)


def fixture(cfg):
    return {
        "format": "seme-nas-benchmark-v1", "space": cfg["space"], "dataset": cfg["dataset"],
        "validation": deepcopy(cfg["validation"]), "test": deepcopy(cfg["test"]),
        "architectures": {
            "a": {"ops": ["a"], "validation": [{"repeat_id": 999, "value": 80},
                                                  {"repeat_id": 777, "value": 82},
                                                  {"repeat_id": 888, "value": 85}],
                  "test": [{"repeat_id": 777, "value": 95}]},
            "b": {"ops": ["b"], "validation": [{"repeat_id": n, "value": 91}
                                                  for n in (777, 888, 999)],
                  "test": [{"repeat_id": 888, "value": 90}, {"repeat_id": 777, "value": 94}]},
        },
    }


def varint(value):
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def tfrecord_row(identity="a", validation=0.91, test=0.94, epochs=108, include_test=True):
    evaluation = bytes([4 << 3 | 1]) + struct.pack("<d", validation)
    if include_test:
        evaluation += bytes([5 << 3 | 1]) + struct.pack("<d", test)
    model = (b"\x0a" + varint(len(evaluation)) + evaluation) * 3
    row = [identity, epochs, "0100", "input,output", base64.b64encode(model).decode()]
    data = json.dumps(row).encode()
    length = struct.pack("<Q", len(data))
    return length + struct.pack("<I", _masked_crc32c(length)) + data + struct.pack("<I", _masked_crc32c(data))


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def make(self, cfg=None, data=None):
        cfg = cfg or config()
        path = self.root / "benchmark.json"
        path.write_text(json.dumps(data if data is not None else fixture(cfg)), encoding="utf-8")
        return create_evaluator(cfg, path, Space())

    def test_one_random_repeat_and_cached_percentages(self):
        evaluator = self.make()
        record = evaluator.validation(("a",), intent("a"))
        original = deepcopy(record)
        self.assertIn(record["value"], (80, 82, 85))
        self.assertEqual(record["repeat_ids"], [record["selected_repeat_id"]])
        self.assertEqual(record["available_repeat_ids"], [777, 888, 999])
        self.assertEqual(record["repeat_count"], 1)
        self.assertEqual(record["key"]["aggregation"], "random_one_of_three")
        self.assertEqual(record["stage"], "validation")
        self.assertNotIn("test", record)
        record["repeats"][0]["value"] = -1
        self.assertEqual(evaluator.validation(("a",), intent("a")), original)

    def test_reproducible_independent_draws_and_run_local_cache(self):
        first, second = self.make(config(runs=30, queries=1)), self.make(config(runs=30, queries=1))
        state = random.getstate()
        drawn = []
        for pool in range(30):
            request = intent("a", pool=pool, seed=1000 + pool)
            one = first.validation(("a",), request)
            # Other code's global RNG use cannot perturb evaluator sampling.
            random.random()
            two = second.validation(("a",), request)
            self.assertEqual(one, two)
            drawn.append(one["selected_repeat_id"])
        random.setstate(state)
        self.assertEqual(set(drawn), {777, 888, 999})
        self.assertEqual(len(first._cache), 30)
        with self.assertRaises(ValueError):
            first.validation(("a",), intent("a", seed=99999))

    def test_validation_draw_does_not_advance_global_random_state(self):
        evaluator = self.make()
        before = random.getstate()
        evaluator.validation(("a",), intent("a"))
        self.assertEqual(random.getstate(), before)

    def test_repeat_choice_is_independent_of_query_order(self):
        first, second = self.make(), self.make()
        first_a = first.validation(("a",), intent("a", 0))
        first_b = first.validation(("b",), intent("b", 1))
        second_b = second.validation(("b",), intent("b", 0))
        second_a = second.validation(("a",), intent("a", 1))
        self.assertEqual(first_a, second_a)
        self.assertEqual(first_b, second_b)

    def test_validation_requires_three_repeats_and_explicit_seed(self):
        for count in (0, 1, 2, 4):
            data = fixture(config())
            data["architectures"]["a"]["validation"] = [
                {"repeat_id": n, "value": 90} for n in range(count)]
            with self.assertRaises(BenchmarkDataError):
                self.make(data=data).validation(("a",), intent("a"))
        evaluator = self.make()
        request = intent("a")
        del request["search_seed"]
        with self.assertRaises(ValueError):
            evaluator.validation(("a",), request)

    def test_winners_use_each_runs_sampled_value(self):
        cfg = config(runs=2, queries=2)
        data = fixture(cfg)
        data["architectures"]["b"]["validation"] = [
            {"repeat_id": n, "value": 83} for n in (777, 888, 999)]
        evaluator = self.make(cfg, data)
        winners = []
        for pool, seed in enumerate((0, 9)):
            scored = []
            for index, identity in enumerate(("a", "b")):
                request = intent(identity, index, pool, seed)
                scored.append((evaluator.validation((identity,), request)["value"], identity))
            best = min(scored, key=lambda item: (-item[0], item[1]))[1]
            winners.append(intent(best, pool=pool, seed=seed))
        self.assertEqual([winner["architecture_id"] for winner in winners], ["b", "a"])
        evaluator.seal_winners(winners)
        for winner in winners:
            evaluator.test((winner["architecture_id"],), winner)

    def test_test_guard_requires_every_run_and_best_winners(self):
        evaluator = self.make(config(runs=2))
        with self.assertRaises(PermissionError):
            evaluator.test(("a",), intent("a"))
        for index, identity in enumerate(("a", "b")):
            evaluator.validation((identity,), intent(identity, index, pool=0))
        winners = [intent("b", pool=0), intent("b", pool=1)]
        with self.assertRaises(ValueError):
            evaluator.seal_winners(winners)
        for index, identity in enumerate(("a", "b")):
            evaluator.validation((identity,), intent(identity, index, pool=1))
        with self.assertRaises(ValueError):
            evaluator.seal_winners([intent("a", pool=0), intent("b", pool=1)])
        evaluator.seal_winners(winners)
        self.assertEqual(evaluator.test(("b",), intent("b", pool=0))["value"], 92)
        with self.assertRaises(PermissionError):
            evaluator.test(("a",), intent("a"))
        with self.assertRaises(RuntimeError):
            evaluator.validation(("a",), intent("a"))
        evaluator.seal_winners(winners)  # Identical seal is idempotent.
        with self.assertRaises(ValueError):
            evaluator.seal_winners([intent("a", pool=0), intent("b", pool=1)])

    def test_sequential_unique_budgeted_queries(self):
        evaluator = self.make()
        with self.assertRaises(ValueError):
            evaluator.validation(("a",), intent("a", 1))
        evaluator.validation(("a",), intent("a", 0))
        with self.assertRaises(ValueError):
            evaluator.validation(("b",), intent("b", 0))
        with self.assertRaises(ValueError):
            evaluator.validation(("a",), intent("a", 1))
        with self.assertRaises(ValueError):
            evaluator.validation(("b",), intent("b", 2))
        with self.assertRaises(ValueError):
            evaluator.validation(("b",), intent("a", 1))

    def test_missing_nonfinite_and_duplicate_repeats_fail_closed(self):
        for rows in ([], [{"repeat_id": 1, "value": float("nan")}],
                     [{"repeat_id": 1, "value": 90}, {"repeat_id": 1, "value": 92}],
                     [{"repeat_id": 1, "value": 101}], [{"repeat_id": 1}]):
            with self.subTest(rows=rows):
                data = fixture(config())
                data["architectures"]["a"]["validation"] = rows
                with self.assertRaises(BenchmarkDataError):
                    self.make(data=data).validation(("a",), intent("a"))
        for invalid in (float("nan"), float("inf"), -1, 101, True):
            data = fixture(config())
            for row in data["architectures"]["a"]["validation"]:
                row["value"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(BenchmarkDataError):
                self.make(data=data).validation(("a",), intent("a"))

    def test_tied_winner_uses_canonical_identity(self):
        data = fixture(config())
        for entry in data["architectures"].values():
            for row in entry["validation"]:
                row["value"] = 81
        evaluator = self.make(data=data)
        evaluator.validation(("b",), intent("b", 0))
        evaluator.validation(("a",), intent("a", 1))
        with self.assertRaises(ValueError):
            evaluator.seal_winners([intent("b")])
        evaluator.seal_winners([intent("a")])

    def test_protocol_mismatch_fails_before_metric_access(self):
        data = fixture(config())
        data["validation"]["epoch_index"] = 11
        with self.assertRaises(BenchmarkDataError):
            self.make(data=data)
        cfg = config()
        cfg["validation"]["metric"] = "ori-test"
        with self.assertRaises(ValueError):
            self.make(cfg)

    def test_metadata_contains_no_metric_and_close_blocks_reads(self):
        evaluator = self.make()
        self.assertEqual(evaluator.architecture_ids(), ("a", "b"))
        self.assertEqual(list(evaluator.structures()), [
            {"architecture_id": "a", "ops": ["a"]},
            {"architecture_id": "b", "ops": ["b"]}])
        evaluator.close()
        with self.assertRaises(RuntimeError):
            evaluator.validation(("a",), intent("a"))

    def test_nb101_official_metric_selection_and_conversion(self):
        accesses = []

        class Row(dict):
            def __getitem__(self, key):
                accesses.append(key)
                return super().__getitem__(key)

        rows = [Row(final_validation_accuracy=0.91, final_test_accuracy=0.95),
                Row(final_validation_accuracy=0.93, final_test_accuracy=0.97),
                Row(final_validation_accuracy=0.94, final_test_accuracy=0.96)]
        api = SimpleNamespace(NASBench=lambda path: SimpleNamespace(
            computed_statistics={"a": {108: rows}},
            fixed_statistics={"a": {"module_adjacency": [[0, 1], [0, 0]],
                                      "module_operations": ["input", "output"]}}))
        path = self.root / "official.tfrecord"
        path.touch()
        cfg = config("nb101_cifar10_q200", queries=1)
        cfg["nb101_reader"] = "nasbench_api"
        with patch.dict("sys.modules", {"nasbench": SimpleNamespace(api=api)}):
            evaluator = create_evaluator(cfg, path, Space())
        result = evaluator.validation(("a",), intent("a"))
        self.assertIn(result["value"], (91, 93, 94))
        self.assertEqual(accesses, ["final_validation_accuracy"])
        evaluator.seal_winners([intent("a")])
        self.assertEqual(evaluator.test(("a",), intent("a"))["value"], 96)
        self.assertEqual(accesses[-3:], ["final_test_accuracy"] * 3)

    def test_native_nb101_crc_known_vector_and_official_schema(self):
        self.assertEqual(_crc32c(b"123456789"), 0xE3069283)
        path = self.root / "native.tfrecord"
        path.write_bytes(tfrecord_row(validation=0.91) + tfrecord_row(validation=0.93)
                         + tfrecord_row(validation=0.95)
                         + tfrecord_row(epochs=12, validation=0.60))
        cfg = config("nb101_cifar10_q200", queries=1)
        with patch("ssme_nas.evaluators._final_nb101_metric", side_effect=AssertionError("No labels in metadata")):
            evaluator = create_evaluator(cfg, path, Space())
            self.assertEqual(evaluator.architecture_ids(), ("a",))
            self.assertEqual(list(evaluator.structures()), [{"architecture_id": "a",
                "matrix": [[0, 1], [0, 0]], "ops": ["input", "output"]}])
        measurement = evaluator.validation(("a",), intent("a"))
        self.assertIn(measurement["value"], (91, 93, 95))
        self.assertEqual(len(measurement["repeat_ids"]), 1)
        self.assertEqual(measurement["available_repeat_ids"], [0, 1, 2])
        self.assertEqual(measurement["key"]["metric"], "final_validation_accuracy")
        evaluator.seal_winners([intent("a")])
        self.assertAlmostEqual(evaluator.test(("a",), intent("a"))["value"], 94)
        evaluator.close()

    def test_native_nb101_corrupt_and_truncated_tfrecord_are_rejected(self):
        valid = tfrecord_row()
        length_corrupt, data_corrupt = bytearray(valid), bytearray(valid)
        length_corrupt[8] ^= 1
        data_corrupt[-5] ^= 1
        for raw in (valid[:6], valid[:10], valid[:-3], length_corrupt, data_corrupt):
            with self.subTest(length=len(raw)):
                path = self.root / "corrupt.tfrecord"
                path.write_bytes(raw)
                with self.assertRaises(BenchmarkDataError):
                    create_evaluator(config("nb101_cifar10_q200", queries=1), path, Space())

    def test_native_nb101_validation_never_requires_or_decodes_test(self):
        for kwargs in ({"include_test": False}, {"test": float("nan")}):
            path = self.root / "partial.tfrecord"
            path.write_bytes(tfrecord_row(**kwargs) * 3)
            evaluator = create_evaluator(config("nb101_cifar10_q200", queries=1), path, Space())
            self.assertEqual(evaluator.validation(("a",), intent("a"))["value"], 91)
            evaluator.seal_winners([intent("a")])
            with self.assertRaises(BenchmarkDataError):
                evaluator.test(("a",), intent("a"))
            evaluator.close()

    def _nb201_api(self, accesses):
        class Repeat:
            def __init__(self, dataset, seed):
                self.dataset, self.seed = dataset, seed

            def get_eval(self, metric, epoch):
                accesses.append((self.dataset, self.seed, metric, epoch))
                return {"accuracy": {777: 91, 888: 94, 999: 93}[self.seed]}

        datasets = ("cifar10-valid", "cifar10", "cifar100", "ImageNet16-120")
        meta = SimpleNamespace(
            get_dataset_seeds=lambda dataset: [999, 888, 777],
            all_results={(dataset, seed): Repeat(dataset, seed)
                         for dataset in datasets for seed in (777, 888, 999)})
        return SimpleNamespace(query_index_by_arch=lambda arch: 0, meta_archs=["a"],
                               arch2infos_dict={0: {"200": meta}})

    def test_nb201_pth_one_validation_repeat_and_cifar10_split_isolation(self):
        accesses, loads = [], []
        api = self._nb201_api(accesses)
        torch = SimpleNamespace(load=lambda *args, **kwargs: loads.append((args, kwargs)) or {})
        official = SimpleNamespace(NASBench201API=lambda data, verbose: api)
        path = self.root / "official.pth"
        path.touch()
        with patch.dict("sys.modules", {"torch": torch, "nas_201_api": official}):
            evaluator = create_evaluator(config(queries=1), path, Space())
        self.assertFalse(loads[0][1]["weights_only"])
        self.assertIn(evaluator.validation(("a",), intent("a"))["value"], (91, 93, 94))
        self.assertEqual(len(accesses), 1)
        self.assertTrue(all(row[0] == "cifar10-valid" and row[2:] == ("x-valid", 199)
                            for row in accesses))
        evaluator.seal_winners([intent("a")])
        evaluator.test(("a",), intent("a"))
        self.assertEqual(accesses[-1], ("cifar10", 777, "ori-test", 199))

    def test_nats_supports_c100_and_imagenet16_with_same_final_endpoint(self):
        for name, dataset in (("nb201_cifar100_q100", "cifar100"),
                              ("nb201_imagenet16_120_q100", "ImageNet16-120")):
            accesses, creates = [], []
            api = self._nb201_api(accesses)
            nats = SimpleNamespace(create=lambda *args, **kwargs:
                                   creates.append((args, kwargs)) or api)
            path = self.root / "official.pickle.pbz2"
            path.touch()
            with patch.dict("sys.modules", {"nats_bench": nats}):
                evaluator = create_evaluator(config(name, queries=1), path, Space())
            self.assertEqual(creates[0][0][1], "tss")
            self.assertFalse(creates[0][1]["fast_mode"])
            result = evaluator.validation(("a",), intent("a"))
            self.assertEqual(accesses, [(dataset, result["selected_repeat_id"], "x-valid", 199)])
            evaluator.seal_winners([intent("a")])
            evaluator.test(("a",), intent("a"))
            self.assertEqual(accesses[-1], (dataset, 777, "x-test", 199))


if __name__ == "__main__":
    unittest.main()
