"""Public random-input provenance checks; no benchmark metrics or searches."""
from itertools import product
import json
from pathlib import Path
import random
import unittest

from ssme_nas._utils import digest
from ssme_nas.core import nb201
from ssme_nas.runner import load_inputs


ROOT = Path(__file__).resolve().parents[1]


class StartsExtensionTest(unittest.TestCase):
    def test_preserved_inputs_and_new_seed_provenance(self):
        manifest = json.loads((ROOT / 'artifacts/starts/EXTENSION_20_RUNS.json').read_text())
        self.assertEqual(manifest['new_benchmark_metric_reads'], 0)
        self.assertEqual(manifest['new_search_runs_executed'], 0)
        old_seeds, new_seeds = set(), []
        for path in sorted((ROOT / 'configs').glob('*.json')):
            with self.subTest(config=path.name):
                config, _, starts, space = load_inputs(path)
                stem = Path(config['starts_file']).stem
                proof = manifest['artifacts'][stem]
                prefix = dict(seeds=starts['seeds'][:10],
                              pools={str(i): starts['pools'][str(i)] for i in range(10)})
                self.assertEqual(digest(prefix), proof['preserved_first_ten_digest'])
                self.assertEqual(digest(starts), proof['complete_starts_digest'])
                self.assertEqual(starts['seeds'][10:], proof['new_seeds'])
                self.assertEqual(len(set(starts['seeds'])), 20)
                old_seeds.update(starts['seeds'][:10])
                for pool in range(10, 20):
                    seed = starts['seeds'][pool]
                    expected_seed = int(digest([manifest['namespace'], stem, pool])[:16], 16) & ((1 << 63) - 1)
                    self.assertEqual(seed, expected_seed)
                    new_seeds.append(seed)
                    rows = starts['pools'][str(pool)]
                    identities = [space.canonical(tuple(row)) for row in rows]
                    self.assertEqual(len(set(identities)), 10)
                    self.assertEqual(identities, proof['new_pool_architecture_ids'][str(pool)])
                    if config['space'] == 'nb201':
                        domain = list(product(nb201.OPS, repeat=6))
                        expected = random.Random(seed).sample(domain, 10)
                        self.assertEqual(rows, [list(row) for row in expected])
        self.assertEqual(len(new_seeds), 40)
        self.assertEqual(len(set(new_seeds)), 40)
        self.assertTrue(old_seeds.isdisjoint(new_seeds))


if __name__ == '__main__':
    unittest.main()
