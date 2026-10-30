"""Selected archived bootstrap and elite retention for ten-parent examples."""
import math

EPS = 1e-7


def value(record):
    result = record["validation"]["value"]
    if not math.isfinite(result):
        raise ValueError("Validation must be finite.")
    return result


def bootstrap_archive(queried):
    if len(queried) != 20 or len({q["architecture_id"] for q in queried}) != 20:
        raise ValueError("Bootstrap requires twenty distinct observed architectures.")
    return sorted(queried, key=lambda q: (-value(q), q["architecture_id"]))[:10]


def retain(parents, child, parent_slot, step):
    if len(parents) != 10 or len({p["architecture_id"] for p in parents}) != 10:
        raise ValueError("Expected ten distinct active parents.")
    if child["architecture_id"] in {p["architecture_id"] for p in parents}:
        raise ValueError("Child identity is already active.")
    if not 0 <= parent_slot < 10 or step < 0:
        raise ValueError("Invalid slot or mutation step.")
    victim = parent_slot if step < 10 else min(
        range(10), key=lambda i: (value(parents[i]), parents[i]["architecture_id"])
    )
    gain = value(child) - value(parents[victim])
    kept = gain > EPS
    updated = list(parents)
    if kept:
        updated[victim] = child
    return updated, dict(
        accepted=kept, replacement_slot=victim if kept else None,
        compared_slot=victim, compared_architecture=parents[victim]["architecture_id"],
        comparison_gain_pp=gain,
        scope="OWN_PARENT_BOOTSTRAP" if step < 10 else "LIVE_POPULATION_WORST",
    )
