"""Deterministic graph-field projections for exploratory cross-space concepts.

This utility computes structural fields only. It performs no file I/O, changes
no module bindings, and does not alter the main search or numerical distance.
"""
from .core import nb101, nb201
from .spaces import get_space


SOURCE_FIELDS = {
    "nb101": ["child_connected","child_live_conv","child_paths","live_conv_delta","live_skip_delta","parent_branch","parent_connected","parent_live_conv","parent_merge","parent_paths","path_delta","same_path_after","same_path_before","serial_slots","share_head","share_tail","touched_all_paths","unmodified_live_conv","unmodified_live_pool","untouched_path_exists"],
    "nb201": ["depth_delta","edge_delta","edit_kind","node_delta","parent_branches","parent_depth","parent_merges","parent_paths","path_delta","same_path_after","same_path_before","shortest_depth_delta","touched_path_coverage"],
}


def sg(x):
    return (x > 0) - (x < 0)


def _nb201_fields_on_nb101(p,q):
    graph, action, SLOT_EDGES = nb101.graph, nb101.action, nb101.SLOT_EDGES
    pp,pl=graph(p);qp,ql=graph(q); touched={i for i,_,_ in action(p,q)}
    ee=[SLOT_EDGES[i] for i in sorted(touched)]
    def cnt(o,live,kind):
        if kind=='routing':return sum(i<21 for i in live)
        return sum(i>=21 and (o[i].startswith('conv') if kind=='conv' else o[i]=='maxpool3x3') for i in live)
    sg=lambda x:(x>0)-(x<0)
    return dict(parent_connected=bool(pp),child_connected=bool(qp),path_delta=sg(len(qp)-len(pp)),
        parent_paths=len(pp),child_paths=len(qp),parent_live_conv=bool(cnt(p,pl,'conv')),child_live_conv=bool(cnt(q,ql,'conv')),
        touched_live_before=len(touched&pl),touched_live_after=len(touched&ql),
        touched_all_paths=bool(pp) and all(touched.intersection(x) for x in pp),
        untouched_path_exists=any(not touched.intersection(x) for x in pp),
        same_path_before=len(touched)==2 and any(touched<=set(x) for x in pp),
        same_path_after=len(touched)==2 and any(touched<=set(x) for x in qp),
        share_tail=len(ee)==2 and ee[0][0]==ee[1][0],share_head=len(ee)==2 and ee[0][1]==ee[1][1],
        serial_slots=len(ee)==2 and (ee[0][1]==ee[1][0] or ee[1][1]==ee[0][0]),
        live_conv_delta=sg(cnt(q,ql,'conv')-cnt(p,pl,'conv')),
        live_skip_delta=sg(cnt(q,ql,'routing')-cnt(p,pl,'routing')),
        unmodified_live_conv=any(i>=21 and p[i].startswith('conv') for i in pl-touched),
        unmodified_live_pool=any(i>=21 and p[i]=='maxpool3x3' for i in pl-touched),
        parent_branch=any(sum(SLOT_EDGES[i][0]==v for i in pl)>1 for v in range(14)),
        parent_merge=any(sum(SLOT_EDGES[i][1]==v for i in pl)>1 for v in range(14)))




def _nb101_fields_on_nb201(p, q):
    graph, action, EDGES = nb201.graph, nb201.action, nb201.EDGES
    pp, pl = graph(p)
    qp, ql = graph(q)
    touched = {i for i, _, _ in action(p, q)}
    # NB201 operation edges become computational nodes. Skip is identity
    # routing, none is absent. Original junctions are routing-only vertices.
    pn = {i for i in pl if p[i] != 'skip_connect'}
    qn = {i for i in ql if q[i] != 'skip_connect'}
    pd = [sum(i in pn for i in path) for path in pp]
    qd = [sum(i in qn for i in path) for path in qp]
    parent_routing = len(pl) + len(pn)  # one connector for skip, two for an op
    child_routing = len(ql) + len(qn)
    covered = sum(bool(touched.intersection(path)) for path in pp)
    kinds = set()
    for _, old, new in action(p, q):
        old_node = old not in ('none', 'skip_connect')
        new_node = new not in ('none', 'skip_connect')
        if old_node and new_node:
            kinds.add('NODE')
        elif not old_node and not new_node:
            kinds.add('LINK')
        else:
            kinds.update(('LINK', 'NODE'))
    return dict(
        edge_delta=sg(child_routing-parent_routing),
        edit_kind='MIXED' if len(kinds) == 2 else 'NODE_ONLY' if 'NODE' in kinds else 'LINK_ONLY',
        node_delta=sg(len(qn)-len(pn)),
        depth_delta=sg(max(qd, default=0)-max(pd, default=0)),
        parent_depth=max(pd, default=0),
        shortest_depth_delta=sg(min(qd, default=0)-min(pd, default=0)),
        parent_paths=min(2, len(pp)), path_delta=sg(len(qp)-len(pp)),
        touched_path_coverage='NONE' if not covered else 'ALL' if covered == len(pp) else 'SOME',
        parent_branches=min(2, sum(sum(EDGES[i][0] == v for i in pl) > 1 for v in range(4))),
        parent_merges=min(2, sum(sum(EDGES[i][1] == v for i in pl) > 1 for v in range(4))),
        same_path_before=len(touched) == 2 and any(touched <= set(path) for path in pp),
        same_path_after=len(touched) == 2 and any(touched <= set(path) for path in qp),
    )


def source_view(parent, child, target_space):
    """Recompute source-vocabulary fields from a target-space native edit.

    For target nb101, source fields describe the node-split edge view.
    For target nb201, source fields describe the computational-node view.
    """
    space = get_space(target_space)
    p, q = tuple(parent), tuple(child)
    space.validate_edit(p, q)
    if space.name == "nb101":
        return _nb201_fields_on_nb101(p, q)
    return _nb101_fields_on_nb201(p, q)


def projected_features(parent, child, target_space, required_fields=None):
    """Keep native descriptors and add namespaced source concept fields.

    The default fields are exactly those used by the released eight/five-bundle
    transfer artifacts. A requested field with no implemented projection raises
    ValueError; fields are never matched by position, filled or silently dropped.
    The target native CATALOG is not modified.
    """
    space = get_space(target_space)
    p, q = tuple(parent), tuple(child)
    projected = source_view(p, q, space)
    required = SOURCE_FIELDS[space.name] if required_fields is None else list(required_fields)
    if not all(isinstance(key, str) for key in required):
        raise ValueError("Source field names must be strings")
    missing = set(required) - set(projected)
    if missing:
        raise ValueError("No graph projection for source fields: " + ", ".join(sorted(missing)))
    native = space.features(p, q)
    return dict(native, **{"transfer__" + key: projected[key] for key in required})

