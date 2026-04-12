import pytest

from unifsl_rl.core.access_guard import GuardedMapping, ensure_probe_reward_safe
from unifsl_rl.core.protocol import PROBE_PROTOCOL


def test_probe_forbid_val_test_access():
    ctx = GuardedMapping({"support_vecs": 1, "val_features": 2}, protocol=PROBE_PROTOCOL, stage="selection")
    _ = ctx["support_vecs"]
    with pytest.raises(RuntimeError):
        _ = ctx["val_features"]


def test_probe_reward_split_forbidden():
    with pytest.raises(RuntimeError):
        ensure_probe_reward_safe(PROBE_PROTOCOL, "val")
