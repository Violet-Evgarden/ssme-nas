"""Compatibility facade for the fully released structural edit memory core.

The historical import path is retained so existing integrations keep working.
"""
from .core.memory import Memory, model_from_records, propose, validate_concepts
from .spaces import get_space


class CoreNotReleasedError(RuntimeError):
    """Legacy exception name; the public core no longer raises this exception."""


def require_private_core():
    """Compatibility no-op: the scientific core is distributed in this package."""
    return None


def compile_memory(records, concepts, space=None):
    """Compile queried validation outcomes with the frozen concept list.

    Concepts may be the accepted list or a released CONCEPTS.json object.
    Space is inferred from the native genotype width when omitted.
    """
    if space is None:
        if not records:
            raise ValueError("Provide a search space when compiling an empty archive")
        width = len(records[0]['ops'])
        if width not in (6, 26):
            raise ValueError("Unknown genotype width; expected 6 or 26 slots")
        space = get_space('nb101' if width == 26 else 'nb201')
    else:
        space = get_space(space)
    if isinstance(concepts, dict):
        if 'accepted' in concepts:
            concepts = concepts['accepted']
        else:
            concepts, rejected = validate_concepts(concepts, space)
            if rejected:
                raise ValueError("Concept response contains invalid or duplicated bundles")
    if not isinstance(concepts, (list, tuple)) or not concepts:
        raise ValueError("At least one frozen structural concept is required")
    ids, bundles = set(), set()
    for concept in concepts:
        fields = concept.get('features')
        if (not isinstance(fields, list) or not 1 <= len(fields) <= 3 or
                len(set(fields)) != len(fields) or any(f not in space.CATALOG for f in fields)):
            raise ValueError("Concepts may contain only one to three native structural fields")
        identity = concept.get('id')
        bundle = tuple(sorted(fields))
        if not isinstance(identity, str) or identity in ids or bundle in bundles:
            raise ValueError("Concept identities and feature bundles must be distinct")
        ids.add(identity)
        bundles.add(bundle)
    return model_from_records(records, list(concepts), space)


def propose_mutation(parent, visited, memory: Memory, seed, step):
    """Select the archived exploration or evidence-guided mutation, unchanged."""
    return propose(tuple(parent), visited, memory, seed, step)
