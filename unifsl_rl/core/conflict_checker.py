from dataclasses import dataclass
from typing import List


@dataclass
class ParamOwnership:
    slot_id: str
    param_keys: List[str]


def check_slot_conflicts(slots):
    owner = {}
    for slot in slots:
        for k in slot.owns:
            if k in owner:
                raise RuntimeError(f"Slot conflict on '{k}': {owner[k]} vs {slot.id}")
            owner[k] = slot.id
    return owner
