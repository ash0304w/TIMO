from dataclasses import dataclass


@dataclass(frozen=True)
class TIMOProtocol:
    name: str
    use_val_for_selection: bool
    use_support_loo: bool


OFFLINE_TRAIN = TIMOProtocol("offline_train", True, False)
OFFLINE_EVAL = TIMOProtocol("offline_eval", True, False)
TEST_TIME_PROBE = TIMOProtocol("test_time_probe", False, True)
STRICT_RL = TIMOProtocol("strict_rl", True, False)
