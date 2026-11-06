"""Native NAS-Bench-201 genotype, legal edits and structural descriptors."""
from functools import lru_cache
from itertools import combinations, product
import hashlib
import json

OPS = ('none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3')
EDGES = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))
PATHS = ((3,), (0, 4), (1, 5), (0, 2, 5))
EPS = 1e-7
CATALOG = {
 'parent_connected': 'parent has at least one effective input-output path',
 'child_connected': 'child has at least one effective input-output path',
 'path_delta': 'sign of change in number of effective paths',
 'parent_paths': 'number of effective parent paths (0 to 4)',
 'child_paths': 'number of effective child paths (0 to 4)',
 'parent_live_conv': 'any convolution on an effective parent path',
 'child_live_conv': 'any convolution on an effective child path',
 'touched_live_before': 'number of edited slots on effective parent paths',
 'touched_live_after': 'number of edited slots on effective child paths',
 'touched_all_paths': 'all effective parent paths contain an edited slot; false if disconnected',
 'untouched_path_exists': 'at least one effective path avoids every edited slot',
 'same_path_before': 'two edited slots share a parent effective path; false for single edits',
 'same_path_after': 'two edited slots share a child effective path; false for single edits',
 'share_tail': 'two edited edges have the same source node',
 'share_head': 'two edited edges have the same destination node',
 'serial_slots': 'an edited edge feeds another edited edge',
 'live_conv_delta': 'sign of change in number of effective convolution edges',
 'live_skip_delta': 'sign of change in number of effective skip edges',
 'unmodified_live_conv': 'unmodified convolution exists on an effective parent path',
 'unmodified_live_pool': 'unmodified pooling exists on an effective parent path',
 'parent_branch': 'an effective parent path graph branches',
 'parent_merge': 'an effective parent path graph merges',
}

def canonical(o):
    assert len(o) == 6 and all(x in OPS for x in o)
    return f'|{o[0]}~0|+|{o[1]}~0|{o[2]}~1|+|{o[3]}~0|{o[4]}~1|{o[5]}~2|'

def action(p, q):
    return tuple((i, p[i], q[i]) for i in range(6) if p[i] != q[i])

def action_type(p, q):
    return json.dumps(sorted((old, new) for _, old, new in action(p, q)), separators=(',', ':'))

def sign(x):
    return 0 if abs(x) <= EPS else 1 if x > 0 else -1

@lru_cache(maxsize=40000)
def graph(o):
    paths = tuple(p for p in PATHS if all(o[i] != 'none' for i in p))
    live = frozenset(i for p in paths for i in p)
    return paths, live

@lru_cache(maxsize=160000)
def features(p, q):
    pp, pl = graph(p); qp, ql = graph(q)
    touched = {i for i, _, _ in action(p, q)}
    edges = [EDGES[i] for i in sorted(touched)]
    def count(o, live, prefix): return sum(o[i].startswith(prefix) for i in live)
    out = dict(parent_connected=bool(pp), child_connected=bool(qp),
        path_delta=sign(len(qp)-len(pp)), parent_paths=len(pp), child_paths=len(qp),
        parent_live_conv=bool(count(p,pl,'nor_conv')), child_live_conv=bool(count(q,ql,'nor_conv')),
        touched_live_before=len(touched&pl), touched_live_after=len(touched&ql),
        touched_all_paths=bool(pp) and all(touched.intersection(x) for x in pp),
        untouched_path_exists=any(not touched.intersection(x) for x in pp),
        same_path_before=len(touched)==2 and any(touched<=set(x) for x in pp),
        same_path_after=len(touched)==2 and any(touched<=set(x) for x in qp),
        share_tail=len(edges)==2 and edges[0][0]==edges[1][0],
        share_head=len(edges)==2 and edges[0][1]==edges[1][1],
        serial_slots=len(edges)==2 and (edges[0][1]==edges[1][0] or edges[1][1]==edges[0][0]),
        live_conv_delta=sign(count(q,ql,'nor_conv')-count(p,pl,'nor_conv')),
        live_skip_delta=sign(count(q,ql,'skip')-count(p,pl,'skip')),
        unmodified_live_conv=any(p[i].startswith('nor_conv') for i in pl-touched),
        unmodified_live_pool=any(p[i]=='avg_pool_3x3' for i in pl-touched),
        parent_branch=any(sum(EDGES[i][0]==n for i in pl)>1 for n in range(4)),
        parent_merge=any(sum(EDGES[i][1]==n for i in pl)>1 for n in range(4)))
    assert set(out) == set(CATALOG)
    return out


def candidates(p,excluded):
    out=[]
    for n in (1,2):
        for slots in combinations(range(6),n):
            for new in product(*(tuple(o for o in OPS if o!=p[i]) for i in slots)):
                q=list(p)
                for i,o in zip(slots,new):q[i]=o
                a=canonical(q)
                if a not in excluded:out.append((a,tuple(q)))
    assert len(out)==len({a for a,q in out})
    return out

