"""Offline synthetic checks; no benchmark labels, API calls or model training."""
import copy
import math
import unittest

from ssme_nas.core import nb101
from ssme_nas.core.memory import H, evidence_mean, fit, tolerant_ranks
from ssme_nas.private_core import compile_memory, propose_mutation, require_private_core
from ssme_nas.spaces import get_space


def synthetic_record(space, ops, index, score, parent=None):
    identity = space.canonical(ops)
    key = {'architecture_id': identity}
    if space.name == 'nb101':
        key['metric'] = 'final_validation_accuracy'
    else:
        key.update(split='x-valid', hp=200, epoch_index=199)
    return dict(architecture_id=identity, ops=list(ops), pool=0, query_index=index,
                validation={'value': score, 'key': key}, parent=parent,
                origin='RANDOM_START' if parent is None else 'EVOLVED_CHILD')


def concept(space):
    name = 'parent_depth' if space.name == 'nb101' else 'parent_connected'
    return [{'id': 'C00', 'features': [name], 'meaning': 'Synthetic test context'}]


class PublicCoreTests(unittest.TestCase):
    def test_release_gate_is_removed(self):
        self.assertIsNone(require_private_core())

    def test_tolerant_ranks_and_absolute_shrinkage(self):
        ranks, _ = tolerant_ranks({'a': 9.711999998474122, 'b': 9.712000000712077, 'c': 80.0})
        self.assertEqual(ranks['a'], ranks['b'])
        near, mass, weights = evidence_mean([0.8], [0.0], H)
        far, _, _ = evidence_mean([0.8], [1.5], H)
        self.assertEqual(near, 0.4)
        self.assertEqual((mass, weights), (1.0, [1.0]))
        self.assertTrue(0 < far < near)
        self.assertEqual(evidence_mean([0.8, -0.8], [0, 0], H)[0], 0)
        self.assertLess(evidence_mean([-0.8], [0], H)[0], 0)

    def test_nb201_legal_candidates_and_edit_order(self):
        space = get_space('nb201')
        parent = ('none',) * 6
        options = space.candidates(parent, set())
        self.assertEqual(len(options), 264)
        self.assertEqual(sum(len(space.action(parent, ops)) == 1 for _, ops in options), 24)
        self.assertEqual(len({identity for identity, _ in options}), 264)
        for identity, ops in options:
            space.validate_edit(parent, ops)
            self.assertEqual(space.canonical(ops), identity)

    def test_observed_neighbour_completion_and_derived_reverse(self):
        space = get_space('nb201')
        p = ('none',) * 6
        q = ('skip_connect',) + p[1:]
        r = ('skip_connect', 'skip_connect') + p[2:]
        records = [synthetic_record(space, p, 0, 50), synthetic_record(space, q, 1, 51),
                   synthetic_record(space, r, 2, 49, space.canonical(q))]
        memory = compile_memory(records, concept(space), space)
        self.assertEqual(memory.data['online']['executed_edits'], 1)
        self.assertEqual(memory.data['online']['additional_observed_neighbour_pairs'], 2)
        self.assertEqual(len(memory.data['relations']), 6)
        by_pair = {(row['parent'], row['child']): row for row in memory.data['relations']}
        for row in memory.data['relations']:
            reverse = by_pair[row['child'], row['parent']]
            self.assertEqual(row['gain'], -reverse['gain'])
            self.assertEqual(row['rank_gain'], -reverse['rank_gain'])
        self.assertEqual(memory.data['relations'][0]['parent'], space.canonical(q))
        self.assertEqual(memory.data['relations'][0]['child'], space.canonical(r))

    def test_deterministic_exploration_and_exclusion(self):
        space = get_space('nb201')
        p = ('none',) * 6
        records = [synthetic_record(space, p, 0, 10)]
        memory = compile_memory(records, concept(space), space)
        visited = {space.canonical(p)}
        first = propose_mutation(p, visited, memory, 71, 0)
        self.assertEqual(first, propose_mutation(p, visited, memory, 71, 0))
        self.assertEqual(first['reason'], 'BOOTSTRAP_RANDOM_MUTATION')
        self.assertNotIn(first['architecture_id'], visited)
        visited.add(first['architecture_id'])
        self.assertNotEqual(first['architecture_id'], propose_mutation(p, visited, memory, 71, 0)['architecture_id'])
        unsupported = propose_mutation(p, visited, memory, 71, 11)
        self.assertEqual(unsupported['reason'], 'NO_POSITIVE_SUPPORT_EXPLORATION')

    def test_conditional_partition_and_tie_break(self):
        rows, labels = [], {}
        for i in range(16):
            parent, child = 'p'+str(i), 'q'+str(i)
            gain = 1 if i < 8 else -1
            labels.update({parent: 50, child: 50+gain})
            rows.append(dict(id='E'+str(i), parent=parent, child=child, p=(), q=(),
                             gain=gain, sign=gain, key='edit', f={'x': i < 8}))
        concepts = [{'id': 'B', 'features': ['x']}, {'id': 'A', 'features': ['x']}]
        data = fit(rows, concepts, labels)
        self.assertEqual(data['roots']['edit']['concept'], 'A')
        self.assertEqual(len(data['roots']['edit']['children']), 2)

    def test_nb101_structure_hash_depth_and_isomorphic_deduplication(self):
        space = get_space('nb101')
        ops = ['0'] * 21 + [nb101.OPS101[0]] * 5
        for edge in ((0, 1), (1, 6)):
            ops[nb101.LINK_INDEX[edge]] = '1'
        parent = tuple(ops)
        child = list(parent)
        child[21] = nb101.OPS101[1]
        child = tuple(child)
        space.validate_edit(parent, child)
        self.assertEqual(space.features(parent, child)['parent_depth'], 1)
        options = space.candidates(parent, set())
        self.assertEqual(len(options), len({identity for identity, _ in options}))
        self.assertNotIn(space.canonical(parent), {identity for identity, _ in options})
        for identity, candidate in options:
            space.validate_edit(parent, candidate)
            self.assertEqual(identity, space.canonical(candidate))

    def test_memories_from_both_spaces_remain_independent(self):
        s201, s101 = get_space('nb201'), get_space('nb101')
        p201 = ('none',) * 6
        q201 = ('skip_connect',) + p201[1:]
        m201 = compile_memory([synthetic_record(s201, p201, 0, 50),
                               synthetic_record(s201, q201, 1, 51)], concept(s201), s201)
        before = m201.score(p201, q201)
        p101 = ['0'] * 21 + [nb101.OPS101[0]] * 5
        p101[nb101.LINK_INDEX[(0, 6)]] = '1'
        m101 = compile_memory([synthetic_record(s101, p101, 0, 80)], concept(s101), s101)
        self.assertEqual(m101.space.name, 'nb101')
        self.assertEqual(m201.score(p201, q201), before)
        self.assertTrue(math.isfinite(before['score']))

    def test_labels_and_invalid_concepts_are_not_admitted(self):
        space = get_space('nb201')
        records = [synthetic_record(space, ('none',) * 6, 0, 50)]
        with self.assertRaises(ValueError):
            compile_memory(records, [{'id': 'C0', 'features': ['test_accuracy']}], space)
        bad = copy.deepcopy(records)
        bad[0]['validation']['key']['split'] = 'ori-test'
        with self.assertRaises(ValueError):
            compile_memory(bad, concept(space), space)


if __name__ == '__main__':
    unittest.main()
