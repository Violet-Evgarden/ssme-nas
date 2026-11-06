"""Label-free cross-space projection and released-artifact checks."""
import json
from pathlib import Path
import unittest

from ssme_nas.core import nb101
from ssme_nas.spaces import get_space
from ssme_nas.transfer import SOURCE_FIELDS, projected_features, source_view


class TransferTests(unittest.TestCase):
    def test_nb201_graph_projection_regression(self):
        empty = ('none',) * 6
        skip = ('none', 'none', 'none', 'skip_connect', 'none', 'none')
        conv = ('none', 'none', 'none', 'nor_conv_3x3', 'none', 'none')
        conv1 = ('none', 'none', 'none', 'nor_conv_1x1', 'none', 'none')
        expected = dict(edge_delta=1, edit_kind='LINK_ONLY', node_delta=0,
                        depth_delta=0, parent_depth=0, shortest_depth_delta=0,
                        parent_paths=0, path_delta=1, touched_path_coverage='NONE',
                        parent_branches=0, parent_merges=0,
                        same_path_before=False, same_path_after=False)
        self.assertEqual(source_view(empty, skip, 'nb201'), expected)
        expected.update(edit_kind='MIXED', node_delta=1, depth_delta=1,
                        shortest_depth_delta=1, parent_paths=1, path_delta=0,
                        touched_path_coverage='ALL')
        self.assertEqual(source_view(skip, conv, 'nb201'), expected)
        expected.update(edge_delta=0, edit_kind='NODE_ONLY', node_delta=0,
                        depth_delta=0, parent_depth=1, shortest_depth_delta=0)
        self.assertEqual(source_view(conv, conv1, 'nb201'), expected)

    def test_nb101_node_split_regression(self):
        parent = ['0'] * 21 + [nb101.OPS101[0]] * 5
        parent[nb101.LINK_INDEX[(0, 1)]] = '1'
        parent[nb101.LINK_INDEX[(1, 6)]] = '1'
        child = list(parent)
        child[21] = nb101.OPS101[1]
        expected = dict(parent_connected=True, child_connected=True, path_delta=0,
                        parent_paths=1, child_paths=1, parent_live_conv=True,
                        child_live_conv=True, touched_live_before=1, touched_live_after=1,
                        touched_all_paths=True, untouched_path_exists=False,
                        same_path_before=False, same_path_after=False, share_tail=False,
                        share_head=False, serial_slots=False, live_conv_delta=0,
                        live_skip_delta=0, unmodified_live_conv=False,
                        unmodified_live_pool=False, parent_branch=False, parent_merge=False)
        self.assertEqual(source_view(parent, child, 'nb101'), expected)

    def test_native_fields_and_catalog_remain_unchanged(self):
        space = get_space('nb201')
        parent = ('nor_conv_3x3',) * 6
        child = ('skip_connect',) + parent[1:]
        catalog = dict(space.CATALOG)
        native = space.features(parent, child)
        merged = projected_features(parent, child, space)
        self.assertEqual({key: merged[key] for key in catalog}, native)
        self.assertEqual(space.CATALOG, catalog)
        self.assertEqual(len(merged), len(native) + 13)
        self.assertEqual({k.removeprefix('transfer__') for k in merged if k.startswith('transfer__')},
                         set(SOURCE_FIELDS['nb201']))

    def test_missing_mapping_is_rejected(self):
        parent = ('none',) * 6
        child = ('skip_connect',) + parent[1:]
        with self.assertRaisesRegex(ValueError, 'No graph projection'):
            projected_features(parent, child, 'nb201', ['parent_conv3'])
        with self.assertRaises(ValueError):
            projected_features(parent, child, 'nb201', [0])

    def test_artifacts_preserve_bundle_ids_and_field_sets(self):
        root = Path(__file__).resolve().parents[1] / 'artifacts' / 'transfer'
        for source, target, count, ids in (
                ('nb201', 'nb101', 8, ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07']),
                ('nb101', 'nb201', 5, ['C00', 'C01', 'C02', 'C03', 'C05'])):
            data = json.loads((root / f'{source}_to_{target}.json').read_text(encoding='utf8'))
            self.assertEqual(set(data), {'source_space', 'target_space', 'numerical_catalog', 'accepted'})
            self.assertEqual((data['source_space'], data['target_space']), (source, target))
            self.assertEqual(data['numerical_catalog'], 'UNCHANGED_TARGET_NATIVE')
            self.assertEqual(len(data['accepted']), count)
            self.assertEqual([c['id'] for c in data['accepted']], ids)
            fields = set()
            for item in data['accepted']:
                self.assertEqual(set(item), {'id', 'features', 'meaning'})
                self.assertTrue(1 <= len(item['features']) <= 3)
                self.assertTrue(item['meaning'])
                self.assertTrue(all(f.startswith('transfer__') for f in item['features']))
                fields.update(f.removeprefix('transfer__') for f in item['features'])
            self.assertEqual(fields, set(SOURCE_FIELDS[target]))


if __name__ == '__main__':
    unittest.main()
