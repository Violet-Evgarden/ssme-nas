"""Validation-only search with online structural edit memory and elite retention."""
from ._utils import digest
from .contracts import AuditSink, SearchSpace, ValidationEvaluator
from .parent_selection import select_parent
from .population import bootstrap_archive, retain, value
from .private_core import compile_memory, propose_mutation
from .selection import validation_winner


def search_pool(starts, seed, concepts, query_budget, pool,
                space: SearchSpace, evaluator: ValidationEvaluator, audit: AuditSink):
    if isinstance(query_budget, bool) or not isinstance(query_budget, int) or query_budget < 10:
        raise ValueError("The query budget must include at least ten initial architectures.")
    if len(starts) != 10 or len({space.canonical(tuple(s)) for s in starts}) != 10:
        raise ValueError("Expected ten distinct random initial architectures.")
    archive, parents, events, visited = [], [], [], set()

    def evaluate(ops, slot, parent=None, proposal=None, memory_version=None):
        ops = tuple(ops)
        space.validate(ops)
        identity = space.canonical(ops)
        if identity in visited or len(archive) >= query_budget:
            raise ValueError("Revisited identity or exhausted query budget.")
        intent = dict(
            pool=pool, search_seed=seed, query_index=len(archive), architecture_id=identity,
            ops=list(ops), slot=slot, parent=parent, proposal=proposal,
            memory_version_before=memory_version,
            origin="RANDOM_START" if parent is None else "EVOLVED_CHILD",
        )
        audit.append("QUERY_INTENT", intent)
        measurement = evaluator.validation(ops, intent)
        record = dict(**intent, validation=measurement)
        value(record)
        audit.append("VALIDATION_RESULT", record)
        archive.append(record)
        visited.add(identity)
        return record

    for slot, ops in enumerate(starts):
        parents.append(evaluate(ops, slot))
    memory = compile_memory(archive, concepts, space=space)
    for step in range(query_budget - 10):
        if step == 10:
            parents = bootstrap_archive(archive)
        slot, parent_event = select_parent(parents, seed, step)
        parent = parents[slot]
        if set(memory.data["source_ids"]) != visited:
            raise ValueError("Memory contains missing or out-of-run identities.")
        version = digest(memory.data)
        proposal = propose_mutation(tuple(parent["ops"]), visited, memory, seed, step)
        proposal["parent_selection"] = parent_event
        space.validate_edit(tuple(parent["ops"]), tuple(proposal["ops"]))
        if space.canonical(tuple(proposal["ops"])) != proposal["architecture_id"]:
            raise ValueError("Proposed genotype and identity disagree.")
        child = evaluate(proposal["ops"], slot, parent["architecture_id"], proposal, version)
        # Gain is against the actual parent, not the replaced worst survivor.
        gain = value(child) - value(parent)
        parents, retention = retain(parents, child, slot, step)
        # Rejected, zero-gain, and negative-gain observations remain in archive.
        memory = compile_memory(archive, concepts, space=space)
        event = dict(
            step=step, parent=parent["architecture_id"], child=child["architecture_id"],
            gain_pp=gain, parent_selection=parent_event, retention=retention,
            memory_before=version, memory_after=digest(memory.data),
        )
        audit.append("DECISION", event)
        events.append(event)
    if len(archive) != query_budget or len(visited) != query_budget:
        raise ValueError("Incomplete query ledger.")
    winner = validation_winner(archive)
    # The runner seals all pools' validation winners before accessing Test.
    return dict(pool=pool, queried=archive, events=events, winner=winner,
                parents=parents, final_memory=memory.data)
