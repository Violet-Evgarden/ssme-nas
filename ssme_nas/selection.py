"""Raw Validation selection and portable immutable winner seals; no Test reader."""
import json
from pathlib import Path

from ._utils import digest
from .population import value


def validation_winner(records):
    if not records:
        raise ValueError("Cannot select a winner from an empty archive.")
    if len({r["architecture_id"] for r in records}) != len(records):
        raise ValueError("Archive identities must be unique.")
    return min(records, key=lambda r: (-value(r), r["architecture_id"]))


def seal_winners(path, winners):
    """Create once, or verify the identical existing seal without overwriting it."""
    if not winners or len({w["pool"] for w in winners}) != len(winners):
        raise ValueError("Exactly one winner per completed pool is required.")
    selected = []
    for winner in winners:
        value(winner)
        selected.append({
            "pool": winner["pool"], "architecture_id": winner["architecture_id"],
            "validation": winner["validation"],
        })
    selected.sort(key=lambda x: x["pool"])
    payload = {
        "rule": "Unrounded Validation argmax, then ascending canonical identity",
        "winners": selected, "winners_sha256": digest(selected),
        "status": "WINNERS_FIXED_BEFORE_TEST",
    }
    target = Path(path)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("Existing winner seal differs; refusing to overwrite.")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return payload
