from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Protocol:
    name: str
    allowed_signals: Tuple[str, ...]
    selection_split: str
    can_use_labels_for_reward: bool
    can_use_labels_for_state: bool


TRAIN_PROTOCOL = Protocol(
    name="offline_train",
    allowed_signals=("support", "val", "meta", "prompt", "branch"),
    selection_split="val",
    can_use_labels_for_reward=True,
    can_use_labels_for_state=False,
)
EVAL_PROTOCOL = Protocol(
    name="offline_eval",
    allowed_signals=("support", "val", "test", "meta", "prompt", "branch"),
    selection_split="val",
    can_use_labels_for_reward=True,
    can_use_labels_for_state=False,
)
PROBE_PROTOCOL = Protocol(
    name="test_time_probe",
    allowed_signals=("support", "meta", "prompt", "branch"),
    selection_split="support",
    can_use_labels_for_reward=True,
    can_use_labels_for_state=False,
)

# legacy aliases
OFFLINE_TRAIN_PROTOCOL = TRAIN_PROTOCOL
OFFLINE_EVAL_PROTOCOL = EVAL_PROTOCOL
TEST_TIME_PROBE_PROTOCOL = PROBE_PROTOCOL
STRICT_RL_PROTOCOL = Protocol(
    name="strict_rl",
    allowed_signals=("support", "val", "meta", "prompt", "branch"),
    selection_split="val",
    can_use_labels_for_reward=True,
    can_use_labels_for_state=False,
)
