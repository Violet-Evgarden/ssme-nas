"""Deterministic controller regression vectors using synthetic observations only.

No benchmark, historical accuracy value, network, or model API is used here.
The vectors protect ordering and tie rules when refactoring the search runner.
"""
import random
import unittest

from ssme_nas.parent_selection import select_parent
from ssme_nas.population import EPS, bootstrap_archive, retain
from ssme_nas.selection import validation_winner


def record(index, score):
    return {
        "architecture_id": f"arch-{index:02d}",
        "validation": {"value": float(score)},
    }


class ControllerRegressionTest(unittest.TestCase):
    def test_seeded_tournament_draw_vectors(self):
        parents = [record(i, i) for i in range(10)]
        vectors = (
            (123, 10, [5, 2], 5),
            (123, 11, [4, 0], 4),
            (123, 19, [5, 3], 5),
            (123, 20, [9, 7], 9),
            (987654321, 70, [4, 8], 8),
            (987654321, 189, [5, 7], 7),
        )
        for seed, step, drawn, winner in vectors:
            with self.subTest(seed=seed, step=step):
                slot, info = select_parent(parents, seed, step)
                self.assertEqual(info["drawn_slots"], drawn)
                self.assertEqual(slot, winner)

    def test_controller_does_not_use_global_random_state(self):
        parents = [record(i, i) for i in range(10)]
        original = random.getstate()
        try:
            random.seed(123)
            before = random.getstate()
            expected = select_parent(parents, 17, 73)
            self.assertEqual(random.getstate(), before)
            random.seed(999)
            for _ in range(27):
                random.random()
            self.assertEqual(select_parent(parents, 17, 73), expected)
        finally:
            random.setstate(original)

    def test_bootstrap_uses_all_twenty_observations_and_identity_ties(self):
        queried = [record(i, 1) for i in range(20)]
        expected = [f"arch-{i:02d}" for i in range(10)]
        for observations in (queried, list(reversed(queried))):
            self.assertEqual(
                [r["architecture_id"] for r in bootstrap_archive(observations)],
                expected,
            )

    def test_worst_survivor_tie_is_identity_order_not_slot_order(self):
        parents = list(reversed([record(i, 1) for i in range(10)]))
        survivors, detail = retain(parents, record(20, 2), 0, 10)
        self.assertEqual(detail["compared_architecture"], "arch-00")
        self.assertEqual(detail["replacement_slot"], 9)
        self.assertEqual(survivors[0], parents[0])

    def test_tolerance_rejected_child_can_still_win_archive(self):
        parents = [record(i, 1) for i in range(10)]
        child = record(20, 1 + EPS / 2)
        survivors, detail = retain(parents, child, 0, 10)
        self.assertFalse(detail["accepted"])
        self.assertEqual(survivors, parents)
        self.assertEqual(validation_winner(parents + [child]), child)

    def test_retention_is_strictly_above_tolerance(self):
        parents = [record(i, 0) for i in range(10)]
        self.assertFalse(retain(parents, record(20, EPS), 0, 10)[1]["accepted"])
        self.assertTrue(retain(parents, record(20, 2 * EPS), 0, 10)[1]["accepted"])


if __name__ == "__main__":
    unittest.main()
