from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Protocol:
    name: str
    allowed_signals: List[str]
    selection_split: str
    can_use_labels_for_reward: bool
    can_use_labels_for_state: bool


TRAIN_PROTOCOL = Protocol(
    name="train",
    allowed_signals=["support", "val", "meta"],
    selection_split="val",
    can_use_labels_for_reward=True,
    can_use_labels_for_state=False,
)

EVAL_PROTOCOL = Protocol(
    name="eval",
    allowed_signals=["support", "test", "meta"],
    selection_split="test",
    can_use_labels_for_reward=False,
    can_use_labels_for_state=False,
)

PROBE_PROTOCOL = Protocol(
    name="probe",
    allowed_signals=["support", "meta"],
    selection_split="support",
    can_use_labels_for_reward=True,
    can_use_labels_for_state=False,
)
