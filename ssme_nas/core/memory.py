"""Evidence completion, conditional partitions and contextual rank readout.

Ported from the archived NAS-Bench-101 and NAS-Bench-201 main runs. All
architecture operations are supplied by the memory's search-space adapter;
memories for different spaces can coexist without mutable module bindings.
"""
import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations

EPS = 1e-7
# Preserve the exact archived numerical setting (reported rounded in the paper).
H = 0.34090909090909094


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sign(value):
    return 0 if abs(value) <= EPS else 1 if value > 0 else -1


def rank_labels(labels):
    vals=sorted(labels.values()); n=len(vals)
    return {a:(sum(v<x for v in vals)+0.5*sum(v==x for v in vals))/n for a,x in labels.items()}

def validate_concepts(raw, space):
    CATALOG = space.CATALOG
    accepted=[]; rejected=[]; seen=set()
    assert isinstance(raw.get('concepts'),list)
    for i,c in enumerate(raw['concepts']):
        ok=isinstance(c,dict) and isinstance(c.get('features'),list) and 1<=len(c['features'])<=3
        ok=ok and all(isinstance(x,str) and x in CATALOG for x in c['features'])
        ok=ok and len(set(c['features']))==len(c['features']) and isinstance(c.get('meaning'),str)
        if not ok: rejected.append(dict(index=i,reason='INVALID_CONCEPT_SCHEMA',original=c)); continue
        key=tuple(sorted(c['features']))
        if key in seen: rejected.append(dict(index=i,reason='DUPLICATE_BUNDLE',original=c));continue
        seen.add(key); accepted.append(dict(id=f'C{i:02d}',features=list(key),meaning=c['meaning']))
    assert accepted, 'BUILD_FAILED: no valid model-authored concept bundle'
    return accepted,rejected

def impurity(rows):
    counts=Counter(r['sign'] for r in rows); n=len(rows)
    return 1-sum((v/n)**2 for v in counts.values()) if n else 0.0

def branch_key(f, fields): return json.dumps([f[x] for x in fields],separators=(',',':'))

def fit(rows, concepts, labels):
    roots={}; splits=[]
    def node(rr,depth,key):
        out=dict(members=[r['id'] for r in rr],n=len(rr),parent_groups=len({r['parent'] for r in rr}),
            sign_counts=dict(Counter(str(r['sign']) for r in rr)),impurity=impurity(rr))
        if depth>=3 or len(rr)<8 or out['impurity']==0:return out
        options=[]
        for c in concepts:
            groups=defaultdict(list)
            for r in rr:groups[branch_key(r['f'],c['features'])].append(r)
            if len(groups)<2 or any(len({r['parent'] for r in g})<4 for g in groups.values()):continue
            reduction=out['impurity']-sum(len(g)*impurity(g) for g in groups.values())/len(rr)
            # Fixed complexity charge; no post-outcome threshold tuning.
            objective=reduction-0.02*(len(groups)-1)-0.01*(len(c['features'])-1)
            if objective>0: options.append((-objective,c['id'],c,groups,reduction))
        if not options:return out
        _,_,c,groups,reduction=min(options,key=lambda x:(x[0],x[1]))
        out.update(concept=c['id'],fields=c['features'],reduction=reduction,
            children={k:node(g,depth+1,key) for k,g in sorted(groups.items())})
        splits.append(dict(action_type=key,depth=depth,concept=c['id'],n=len(rr),reduction=reduction,
            witness_positive=[r['id'] for r in rr if r['sign']>0],
            witness_negative=[r['id'] for r in rr if r['sign']<0],
            witness_zero=[r['id'] for r in rr if r['sign']==0]))
        return out
    grouped=defaultdict(list)
    for r in rows:grouped[r['key']].append(r)
    for k,rr in sorted(grouped.items()):roots[k]=node(rr,0,k)
    ranks=rank_labels(labels)
    compact=[dict(id=r['id'],parent=r['parent'],child=r['child'],p=r['p'],q=r['q'],
        gain=r['gain'],sign=r['sign'],key=r['key'],f=r['f'],rank_gain=ranks[r['child']]-ranks[r['parent']]) for r in rows]
    return dict(roots=roots,relations=compact,concepts=concepts,splits=splits,
        source_ids=sorted(labels),rank_readout='average-tie source percentile child-minus-parent')


def tolerant_ranks(labels):
    """Deterministic max-span-EPS tie groups, then average percentile rank."""
    ordered=sorted(labels, key=lambda a:(labels[a],a))
    ranks={};clusters=[];start=0
    while start<len(ordered):
        stop=start+1;anchor=labels[ordered[start]]
        while stop<len(ordered) and labels[ordered[stop]]-anchor<=EPS:
            stop+=1
        value=(start+0.5*(stop-start))/len(ordered)
        for a in ordered[start:stop]:ranks[a]=value
        clusters.append(dict(ids=ordered[start:stop],low=anchor,
                             high=labels[ordered[stop-1]],rank=value))
        start=stop
    return ranks,clusters


def distance(p,q,r,space):
    features, action, CATALOG = space.features, space.action, space.CATALOG
    f=features(tuple(p),tuple(q));touched={i for i,_,_ in action(p,q)}
    structural=sum(f[k]!=r['f'][k] for k in CATALOG)/len(CATALOG)
    background=sum(p[i]!=r['p'][i] for i in range(len(p)) if i not in touched)/(len(p)-len(touched))
    return structural+background



def repair_compiled(original,labels,bandwidth):
    result=copy.deepcopy(original)
    ranks,clusters=tolerant_ranks(labels)
    changed=0
    for r in result['relations']:
        old=r['rank_gain']
        gain=labels[r['child']]-labels[r['parent']]
        assert gain==r['gain']
        new=0.0 if abs(gain)<=EPS else ranks[r['child']]-ranks[r['parent']]
        assert new==0 or sign(new)==sign(gain)
        r['previous_rank_gain']=old;r['rank_gain']=new
        changed+=new!=old
    by_pair={(r['parent'],r['child']):r for r in result['relations']}
    for r in result['relations']:
        assert r['rank_gain']==-by_pair[(r['child'],r['parent'])]['rank_gain']
        if r['sign']==0:assert r['rank_gain']==0
    assert result['roots']==original['roots'] and result['concepts']==original['concepts']
    result['readout_revision']=dict(version='tolerant-rank-absolute-evidence-v2',EPS=EPS,
        bandwidth=bandwidth,zero_prior_mass=1.0,kernel='exp(-distance / bandwidth)',
        formula='sum(w_i * rank_gain_i)/(1 + sum(w_i))',nearest_limit=10,
        changed_rank_relations=changed,tie_clusters=clusters)
    return result


def evidence_mean(gains,distances,bandwidth):
    weights=[math.exp(-d/bandwidth) for d in distances]
    mass=sum(weights)
    return sum(w*g for w,g in zip(weights,gains))/(1.0+mass),mass,weights



class Memory:
    def __init__(self,data,space):
        self.data=data
        self.space=space
        self.rows={r['id']:r for r in data['relations']}
        self.bandwidth=data['readout_revision']['bandwidth']

    def score(self,p,q,refined=True):
        action_type, features = self.space.action_type, self.space.features
        key=action_type(p,q);f=features(tuple(p),tuple(q));root=self.data['roots'].get(key)
        if root is None:return None
        leaf=root;depth=0;backoff=False
        while refined and 'children' in leaf:
            k=branch_key(f,leaf['fields'])
            if k not in leaf['children']:backoff=True;break
            leaf=leaf['children'][k];depth+=1
        ordered=sorted((self.rows[i] for i in leaf['members']),key=lambda r:(distance(p,q,r,self.space),r['id']))[:10]
        distances=[distance(p,q,r,self.space) for r in ordered]
        score,mass,weights=evidence_mean([r['rank_gain'] for r in ordered],distances,self.bandwidth)
        return dict(score=score,depth=depth,ancestor_backoff=backoff,support=len(leaf['members']),
            evidence=[r['id'] for r in ordered],root_support=root['n'],
            absolute_evidence_mass=mass,nearest_distance=min(distances),
            distances=distances,weights=weights,bandwidth=self.bandwidth,zero_prior_mass=1.0)



def observed_relations(queried, space):
    canonical, action = space.canonical, space.action
    action_type, features = space.action_type, space.features
    labels = {q['architecture_id']: q['validation']['value'] for q in queried}
    assert len(labels) == len(queried)
    assert [q['query_index'] for q in queried] == list(range(len(queried)))
    assert len({q['pool'] for q in queried}) <= 1
    by_id = {q['architecture_id']: q for q in queried}
    for q in queried:
        assert canonical(tuple(q['ops'])) == q['architecture_id']
        assert math.isfinite(labels[q['architecture_id']]) and 0 <= labels[q['architecture_id']] <= 100
        space.validate_measurement(q)

    pairs, seen = [], set()
    # Preserve the original executed-row order, then append the extra relations.
    for q in queried:
        if q['origin'] == 'RANDOM_START':
            continue
        parent = by_id[q['parent']]
        assert parent['query_index'] < q['query_index']
        pair = frozenset((parent['architecture_id'], q['architecture_id']))
        assert pair not in seen and len(pair) == 2
        assert 1 <= len(action(parent['ops'], q['ops'])) <= 2
        seen.add(pair)
        pairs.append((parent, q, 'EXECUTED'))
    executed = len(pairs)
    for left, right in combinations(queried, 2):
        pair = frozenset((left['architecture_id'], right['architecture_id']))
        if pair in seen:
            continue
        difference = 0
        for a, b in zip(left['ops'], right['ops']):
            difference += a != b
            if difference > 2:
                break
        if difference not in (1, 2):
            continue
        seen.add(pair)
        pairs.append((left, right, 'OBSERVED_NEIGHBOUR'))
    rows, provenance = [], {}
    for first, second, origin in pairs:
        availability = max(first['query_index'], second['query_index'])
        for left, right, reverse in ((first, second, False), (second, first, True)):
            p, c = tuple(left['ops']), tuple(right['ops'])
            gain = labels[right['architecture_id']] - labels[left['architecture_id']]
            rid = 'E' + digest([left['architecture_id'], right['architecture_id']])[:20]
            assert rid not in provenance
            rows.append(dict(id=rid, parent=left['architecture_id'], child=right['architecture_id'],
                p=p, q=c, gain=gain, sign=sign(gain), key=action_type(p,c), f=features(p,c)))
            provenance[rid] = dict(origin=origin, available_after_query=availability,
                derived_reverse=reverse, independent_measurement=False)
    assert len(rows) == 2*len(seen)
    return rows, labels, provenance, executed


def model_from_records(queried, concepts, space):
    rows, labels, provenance, executed = observed_relations(queried, space)
    compiled = repair_compiled(fit(rows, concepts, labels), labels, H)
    compiled['relation_provenance'] = provenance
    compiled['online'] = dict(observed_architectures=len(labels),
        executed_edits=executed, observed_edits=len(rows)//2,
        additional_observed_neighbour_pairs=len(rows)//2-executed,
        directed_records=len(rows), reverse_is_derived_not_independent=True,
        shared_architectures_not_independent=True, historical_labels_loaded=0,
        revision='raw-aligned-observed-neighbour-completion-v1')
    return Memory(compiled, space)


def random_mutation(options, parent, seed, step, space):
    """Fixed seeded 1/2 single versus double, then uniform within that order."""
    action = space.action
    rng = random.Random(int(digest([seed, 'edit-exploration-v1', step])[:16], 16))
    groups = {n: [x for x in options if len(action(parent, x[1])) == n] for n in (1, 2)}
    available = [n for n in (1, 2) if groups[n]]
    assert available
    order = rng.choice(available)
    return rng.choice(groups[order])


def best(scored):
    return min(scored, key=lambda x: (-x[2]['score'], x[0])) if scored else None


def propose(parent, visited, memory, seed, step):
    """Only past observations may enter memory; no candidate label is available."""
    space = memory.space
    candidates, action = space.candidates, space.action
    options = candidates(tuple(parent), visited)
    assert options
    round_index, slot = divmod(step, 10)
    reason = ('BOOTSTRAP_RANDOM_MUTATION' if round_index == 0 else
              'SCHEDULED_EXPLORATION' if (round_index-1+slot) % 5 == 0 else 'MEMORY_GREEDY')
    scored, root_scored = [], []
    if reason == 'MEMORY_GREEDY':
        for a, q in options:
            for refined, output in ((True, scored), (False, root_scored)):
                s = memory.score(tuple(parent), q, refined=refined)
                if s is not None:
                    assert math.isfinite(s['score'])
                    if s['score'] > 0:
                        output.append((a, q, s))
        selected = best(scored)
        if selected is None:
            reason = 'NO_POSITIVE_SUPPORT_EXPLORATION'
    else:
        selected = None
    if reason != 'MEMORY_GREEDY':
        a, q = random_mutation(options, parent, seed, step, space)
        selected = (a, q, memory.score(tuple(parent), q))
    a, q, s = selected
    root = best(root_scored)
    return dict(architecture_id=a, ops=list(q), reason=reason,
        action=[list(x) for x in action(parent, q)], memory_score=s,
        allowed_actions=len(options), supported_positive_actions=len(scored),
        formula_calls=(2*len(options) if reason in ('MEMORY_GREEDY', 'NO_POSITIVE_SUPPORT_EXPLORATION') else 0)
                      + (reason != 'MEMORY_GREEDY'),
        root_only_argmax=root[0] if root else None,
        semantic_argmax_changed=(root is None or root[0] != a) if reason == 'MEMORY_GREEDY' else None)


def accept(gain):
    return gain > EPS

