"""Synthetic mechanics tests only: these are not NAS performance experiments."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from run_search import main
from ssme_nas.controller import search_pool
from ssme_nas.parent_selection import select_parent
from ssme_nas.population import bootstrap_archive, retain
from ssme_nas.private_core import CoreNotReleasedError, compile_memory, propose_mutation
from ssme_nas.selection import seal_winners, validation_winner


def record(identity, score):
    return dict(architecture_id=identity, validation=dict(value=float(score)))


class PublicComponentsTest(unittest.TestCase):
    def test_bootstrap_round_robin(self):
        parents = [record(str(i), i) for i in range(10)]
        for step in range(10):
            self.assertEqual(select_parent(parents, 123, step)[0], step)

    def test_tournament_deterministic_and_validation_based(self):
        parents = [record(str(i), i) for i in range(10)]
        for step in range(10, 35):
            selected, info = select_parent(parents, 123, step)
            self.assertEqual((selected, info), select_parent(parents, 123, step))
            self.assertEqual(selected, max(info["drawn_slots"]))
            self.assertEqual(len(set(info["drawn_slots"])), 2)

    def test_tournament_tie_identity(self):
        parents = [record(str(i), 1) for i in range(10)]
        selected, info = select_parent(parents, 123, 15)
        self.assertEqual(str(selected), min(str(i) for i in info["drawn_slots"]))

    def test_bootstrap_versus_elite_survival(self):
        parents = [record(str(i), i) for i in range(10)]
        child = record("new", 5.5)
        self.assertFalse(retain(parents, child, 9, 9)[1]["accepted"])
        updated, decision = retain(parents, child, 9, 10)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["replacement_slot"], 0)
        self.assertEqual(updated[9], parents[9])
        self.assertEqual(parents[0]["architecture_id"], "0")

    def test_tied_or_negative_child_not_retained(self):
        parents = [record(str(i), i) for i in range(10)]
        for score in (-1, 0, 1e-8):
            self.assertFalse(retain(parents, record("new", score), 9, 10)[1]["accepted"])

    def test_population_rebuild(self):
        initial = [record(str(i), i) for i in range(20)]
        self.assertEqual([x["validation"]["value"] for x in bootstrap_archive(initial)],
                         list(range(19, 9, -1)))

    def test_duplicate_identity_rejected(self):
        parents = [record(str(i), i) for i in range(10)]
        with self.assertRaises(ValueError):
            retain(parents, record("0", 50), 0, 10)

    def test_raw_not_rounded_winner(self):
        self.assertEqual(validation_winner([record("a", 1.000000001), record("b", 1.000000002)])
                         ["architecture_id"], "b")
        self.assertEqual(validation_winner([record("z", 1), record("a", 1)])
                         ["architecture_id"], "a")

    def test_nonfinite_and_duplicate_archive_rejected(self):
        for score in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validation_winner([record("bad", score)])
        with self.assertRaises(ValueError):
            validation_winner([record("same", 1), record("same", 2)])

    def test_winner_seal_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "winner.json"
            winner = dict(pool=0, **record("a", 1))
            first = seal_winners(path, [winner])
            self.assertEqual(first, seal_winners(path, [winner]))
            with self.assertRaises(ValueError):
                seal_winners(path, [dict(pool=0, **record("b", 2))])
            self.assertEqual(json.loads(path.read_text()), first)

    def test_core_stubs_never_supply_scores(self):
        for function in (compile_memory, propose_mutation):
            with self.assertRaises(CoreNotReleasedError):
                function()

    def test_controller_stops_before_using_any_backend(self):
        with self.assertRaises(CoreNotReleasedError):
            search_pool(None, None, None, None, None, None, None, None)

    def test_cli_is_explicitly_disabled(self):
        message = io.StringIO()
        with contextlib.redirect_stderr(message):
            code = main(["--config", "nonexistent-config-must-not-be-opened.json"])
        self.assertEqual(code, 2)
        self.assertIn("CoreNotReleasedError", message.getvalue())

    def test_describe_is_read_only(self):
        message = io.StringIO()
        with contextlib.redirect_stdout(message):
            code = main(["--describe"])
        self.assertEqual(code, 0)
        self.assertIn("preview-core-withheld", message.getvalue())

    def test_config_examples(self):
        root = Path(__file__).resolve().parents[1]
        configs = list((root / "configs").glob("*.json"))
        self.assertEqual(len(configs), 4)
        for path in configs:
            config = json.loads(path.read_text())
            self.assertEqual(config["release_status"], "preview_core_withheld")
            self.assertFalse(config["online_llm"])
            self.assertEqual(config["imported_performance_labels"], 0)
            self.assertTrue(config["test"]["winner_only_after_all_winners_sealed"])


if __name__ == "__main__":
    unittest.main()
