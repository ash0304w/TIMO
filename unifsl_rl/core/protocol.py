from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Protocol:
    name: str
    allowed_signals: List[str]
    selection_split: str
    can_use_labels_for_reward: bool
    can_use_labels_for_state: bool


OFFLINE_TRAIN_PROTOCOL = Protocol("offline_train", ["support", "val", "meta"], "val", True, False)
OFFLINE_EVAL_PROTOCOL = Protocol("offline_eval", ["support", "val", "test", "meta"], "val", True, False)
TEST_TIME_PROBE_PROTOCOL = Protocol("test_time_probe", ["support", "meta"], "support", True, False)
STRICT_RL_PROTOCOL = Protocol("strict_rl", ["support", "val", "meta"], "val", True, False)
