# The graph-hashing portion of official() is adapted from Google NASBench.
# Copyright 2019 The Google Research Authors. Licensed under Apache-2.0.
# See LICENSES/Apache-2.0.txt and NOTICE.md. Changes: standard-library tuple
# operations and a native effective-graph input replace the NumPy interface.
"""Label-free NB101 adapter; no benchmark accuracy or predictor access.

Seven aligned vertices are retained in the mutation genotype. Official identity
prunes inactive vertices and uses Google's isomorphism-invariant graph hash.
Node splitting maps node operations onto operation edges; original connections
are identity-routing edges, NOT learned skip operations in NAS-Bench-101.
"""
from functools import lru_cache
from itertools import combinations, product
import hashlib
import json

OPS101 = ('conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3')
LINKS = tuple(combinations(range(7), 2))
LINK_INDEX = {edge: i for i, edge in enumerate(LINKS)}
SLOT_EDGES = tuple((2*u+1, 2*v) for u,v in LINKS) + tuple((2*v,2*v+1) for v in range(1,6))
def action(p,q):
    assert len(p) == len(q) == 26
    return tuple((i,p[i],q[i]) for i in range(26) if p[i] != q[i])


def action_type(p,q):
    return json.dumps(sorted(('link' if i<21 else 'node',old,new) for i,old,new in action(p,q)),separators=(',',':'))


@lru_cache(maxsize=60000)
def graph(o):
    assert len(o)==26 and all(x in ('0','1') for x in o[:21]) and all(x in OPS101 for x in o[21:])
    paths=[]
    def walk(v,slots):
        if v==6:
            paths.append(tuple(slots));return
        for w in range(v+1,7):
            i=LINK_INDEX[v,w]
            if o[i]=='1':
                walk(w,slots+[i]+([20+w] if w<6 else []))
    walk(0,[])
    return tuple(paths),frozenset(i for p in paths for i in p)


@lru_cache(maxsize=120000)
def official(o):
    """Official effective graph and hash; equivalent to ModelSpec + hash_module.

    Hash algorithm: google-research/nasbench/lib/graph_util.py (Apache-2.0).
    Verified independently against the unmodified vendored official implementation.
    """
    pp,live=graph(o)
    if not pp or sum(i<21 for i in live)>9:return None
    nodes=sorted({v for i in live if i<21 for v in LINKS[i]})
    matrix=tuple(tuple(int(o[LINK_INDEX[u,v]]) if u<v else 0 for v in nodes) for u in nodes)
    ops=tuple('input' if v==0 else 'output' if v==6 else o[20+v] for v in nodes)
    n=len(nodes); labels=[-1]+[OPS101.index(x) for x in ops[1:-1]]+[-2]
    hashes=[hashlib.md5(str((sum(matrix[v]),sum(matrix[u][v] for u in range(n)),labels[v])).encode()).hexdigest() for v in range(n)]
    for _ in range(n):
        hashes=[hashlib.md5((''.join(sorted(hashes[u] for u in range(n) if matrix[u][v]))+'|'+
            ''.join(sorted(hashes[w] for w in range(n) if matrix[v][w]))+'|'+hashes[v]).encode()).hexdigest() for v in range(n)]
    key=hashlib.md5(str(sorted(hashes)).encode()).hexdigest()
    return key,matrix,ops


def canonical(o):
    value=official(tuple(o));assert value is not None,'Invalid NB101 effective graph'
    return value[0]


def architecture(o):
    a,m,ops=official(tuple(o))
    return dict(space='nb101',hash=a,official_id=a,matrix=[list(r) for r in m],ops=list(ops))


def padded(matrix,ops,rng):
    """Metadata-only lift; inactive node labels drawn from a separate seed stream."""
    n=len(ops); idx=list(range(n-1))+[6]
    values=['0']*21+[rng.choice(OPS101) for _ in range(5)]
    for u in range(n):
        for v in range(u+1,n):values[LINK_INDEX[idx[u],idx[v]]]=str(int(matrix[u][v]))
    for v in range(1,n-1):values[20+v]=ops[v]
    return tuple(values)


@lru_cache(maxsize=1000)
def neighbours(p):
    results={}; parent=canonical(p)
    alternatives=[tuple(x for x in (('0','1') if i<21 else OPS101) if x!=p[i]) for i in range(26)]
    for count in (1,2):
        for slots in combinations(range(26),count):
            for new in product(*(alternatives[i] for i in slots)):
                q=list(p)
                for i,x in zip(slots,new):q[i]=x
                q=tuple(q); value=official(q)
                if value is None:continue
                a=value[0]
                if a==parent:continue
                # Isomorphic actions are not extra query opportunities. Prefer
                # minimal edit count, then complete raw genotype lexicographically.
                old=results.get(a)
                if old is None or (count,q)<(len(action(p,old)),old):results[a]=q
    return tuple(sorted(results.items()))


def candidates(p,excluded):
    return [(a,q) for a,q in neighbours(tuple(p)) if a not in excluded]


CATALOG = {
    'parent_nodes': 'effective internal parent node count: 0, 1, or multiple (2)',
    'child_nodes': 'effective internal child node count: 0, 1, or multiple (2)',
    'node_delta': 'sign of change in effective internal node count',
    'parent_depth': 'maximum number of internal nodes on a parent input-output path (0 to 5)',
    'child_depth': 'maximum number of internal nodes on a child input-output path (0 to 5)',
    'depth_delta': 'sign of change in maximum internal-node path depth',
    'parent_paths': 'parent effective path multiplicity: 0, 1, or multiple (2)',
    'path_delta': 'sign of change in effective input-output path count',
    'parent_branches': 'effective parent vertices with outdegree>1: 0, 1, or multiple (2)',
    'parent_merges': 'effective parent vertices with indegree>1: 0, 1, or multiple (2)',
    'parent_conv3': 'effective parent conv3x3-bn-relu nodes: 0, 1, or multiple (2)',
    'parent_conv1': 'effective parent conv1x1-bn-relu nodes: 0, 1, or multiple (2)',
    'parent_pool': 'effective parent maxpool3x3 nodes: 0, 1, or multiple (2)',
    'remaining_conv3': 'parent conv3x3 nodes retained unchanged and effective in BOTH graphs: 0/1/multiple',
    'remaining_conv1': 'parent conv1x1 nodes retained unchanged and effective in BOTH graphs: 0/1/multiple',
    'remaining_pool': 'parent maxpool nodes retained unchanged and effective in BOTH graphs: 0/1/multiple',
    'touched_path_coverage': 'NONE, SOME or ALL parent effective paths contain an edited slot',
    'same_path_before': 'two edited slots co-occur on a parent path; false for a single edit',
    'same_path_after': 'two edited slots co-occur on a child path; false for a single edit',
    'edit_kind': 'LINK_ONLY, NODE_ONLY or MIXED link/node slots (one or two edited slots)',
    'edge_delta': 'sign of change in effective routing connections, not skip operations',
    'shortest_depth_delta': 'sign of change in minimum internal-node count across input-output paths',
}


def bucket(n):
    return min(2, n)


def sign(n):
    return (n > 0) - (n < 0)


@lru_cache(maxsize=160000)
def features(p, q):
    pp, pl = graph(p)
    qp, ql = graph(q)
    assert pp and qp, 'Only legal connected NB101 candidates'
    touched = {i for i, _, _ in action(p, q)}
    pd = [sum(i >= 21 for i in path) for path in pp]
    qd = [sum(i >= 21 for i in path) for path in qp]
    pn, qn = {i for i in pl if i >= 21}, {i for i in ql if i >= 21}
    retained = (pn & qn) - touched
    count = lambda ids, op: bucket(sum(p[i] == op for i in ids))
    covered = sum(bool(touched.intersection(path)) for path in pp)
    kinds = {i < 21 for i in touched}
    edges = [LINKS[i] for i in pl if i < 21]
    result = dict(
        parent_nodes=bucket(len(pn)), child_nodes=bucket(len(qn)),
        node_delta=sign(len(qn)-len(pn)), parent_depth=max(pd), child_depth=max(qd),
        depth_delta=sign(max(qd)-max(pd)), parent_paths=bucket(len(pp)),
        path_delta=sign(len(qp)-len(pp)),
        parent_branches=bucket(sum(sum(u == v for u, _ in edges) > 1 for v in range(7))),
        parent_merges=bucket(sum(sum(w == v for _, w in edges) > 1 for v in range(7))),
        parent_conv3=count(pn, 'conv3x3-bn-relu'), parent_conv1=count(pn, 'conv1x1-bn-relu'),
        parent_pool=count(pn, 'maxpool3x3'),
        remaining_conv3=count(retained, 'conv3x3-bn-relu'),
        remaining_conv1=count(retained, 'conv1x1-bn-relu'),
        remaining_pool=count(retained, 'maxpool3x3'),
        touched_path_coverage='NONE' if not covered else 'ALL' if covered == len(pp) else 'SOME',
        same_path_before=len(touched) == 2 and any(touched <= set(path) for path in pp),
        same_path_after=len(touched) == 2 and any(touched <= set(path) for path in qp),
        edit_kind='MIXED' if len(kinds) == 2 else 'LINK_ONLY' if True in kinds else 'NODE_ONLY',
        edge_delta=sign(sum(i < 21 for i in ql)-sum(i < 21 for i in pl)),
        shortest_depth_delta=sign(min(qd)-min(pd)),
    )
    assert set(result) == set(CATALOG)
    return result

