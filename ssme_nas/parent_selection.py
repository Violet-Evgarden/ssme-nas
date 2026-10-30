"""Selected archived parent controller for the explicit ten-parent examples."""
import random

from ._utils import digest


def select_parent(parents, seed, step):
    if len(parents) != 10 or len({p["architecture_id"] for p in parents}) != 10:
        raise ValueError("The archived configuration requires ten distinct parents.")
    if step < 0:
        raise ValueError("Mutation step must be nonnegative.")
    if step < 10:
        slots = [step]
        reason = "ROUND_ROBIN_BOOTSTRAP"
    else:
        # Historical namespace retained exactly for deterministic archived draws.
        rng = random.Random(int(digest([seed, "nb101-parent-tournament-v1", step])[:16], 16))
        slots = rng.sample(range(10), 2)
        reason = "VALIDATION_TWO_WAY_TOURNAMENT"
    winner = min(slots, key=lambda s: (
        -parents[s]["validation"]["value"], parents[s]["architecture_id"], s
    ))
    return winner, dict(
        reason=reason, drawn_slots=slots, selected_slot=winner,
        participants=[dict(
            slot=s, architecture_id=parents[s]["architecture_id"],
            Validation=parents[s]["validation"]["value"],
        ) for s in slots],
    )
