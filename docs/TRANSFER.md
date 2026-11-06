# Cross-space structural concept projections

`ssme_nas/transfer.py` implements the deterministic field projections used in
the exploratory concept-transfer experiment. It is a structural utility: it
does not change the main controller, request benchmark measurements or call an
LLM. All source fields are recomputed from the target parent and child graphs;
field positions are never copied between vocabularies.

## Interface

```python
from ssme_nas.transfer import source_view, projected_features

source_fields = source_view(parent, child, target_space="nb201")
features = projected_features(parent, child, target_space="nb201")
```

`source_view` returns the projected source-field values. `projected_features`
keeps every target-native descriptor and adds the required source fields under
the `transfer__` prefix. Its optional `required_fields` argument accepts
unprefixed source-field names. An unsupported field raises `ValueError`; no
field is filled by a default or silently dropped. The target's numerical
`CATALOG` and distance definition remain unchanged. The utility performs no
file I/O and no global module replacement.

## NAS-Bench-201 concepts on NAS-Bench-101

The node-split view represents each operation node by an operation edge and
each original connection by a routing edge. Effective paths determine
connectivity, path counts, convolution presence, edit coverage and surviving
operations. Edited slot edges determine shared endpoints and serial relations.
Here `live_skip_delta` means the change in effective **routing connections**;
it does not introduce a skip operation into NAS-Bench-101. Source path counts
are unbucketed and are kept separate from target-native bucketed counts.

The artifact is
[`nb201_to_nb101.json`](../artifacts/transfer/nb201_to_nb101.json): eight validated
source bundles using twenty distinct source fields. Prefixes are omitted below.

| ID | Source fields |
|---|---|
| C00 | child_connected, parent_connected, path_delta |
| C01 | child_paths, parent_paths, path_delta |
| C02 | touched_all_paths, untouched_path_exists |
| C03 | same_path_after, same_path_before, serial_slots |
| C04 | live_conv_delta, live_skip_delta, unmodified_live_conv |
| C05 | child_live_conv, parent_live_conv, unmodified_live_pool |
| C06 | parent_branch, parent_merge, parent_paths |
| C07 | share_head, share_tail |

## NAS-Bench-101 concepts on NAS-Bench-201

Each effective nonidentity operation edge becomes a computational node with
two routing connectors. A skip edge contributes one routing connector and no
computational node; `none` is absent. Original junctions are routing-only nodes.

| Field group | Computation on the target graph |
|---|---|
| node_delta | Sign of the change in effective nonidentity-operation count |
| edge_delta | Sign of the change in effective edges plus computational nodes |
| parent_depth, depth_delta, shortest_depth_delta | Computational-node counts along effective paths; empty-path depth is zero |
| parent_paths, parent_branches, parent_merges | Source zero/one/multiple buckets |
| path_delta, touched_path_coverage, same_path_before, same_path_after | Effective-path counts and edited-slot coverage |
| edit_kind | Nonidentity-to-nonidentity is NODE_ONLY; none/skip changes are LINK_ONLY; transitions between these classes are MIXED |

The artifact is
[`nb101_to_nb201.json`](../artifacts/transfer/nb101_to_nb201.json): five validated
source bundles using thirteen distinct source fields. Their identifiers and
field combinations are preserved. Prefixes are omitted below.

| ID | Source fields |
|---|---|
| C00 | edge_delta, edit_kind, node_delta |
| C01 | depth_delta, parent_depth, shortest_depth_delta |
| C02 | parent_paths, path_delta, touched_path_coverage |
| C03 | edge_delta, parent_branches, parent_merges |
| C05 | edit_kind, same_path_after, same_path_before |

These are the bundles accepted for the transfer experiment, not the twelve
bundles in each main-experiment vocabulary. A transferred bundle supplies a
candidate partition condition; it need not be selected as an actual split.
Source numerical outcomes are not part of either artifact.
