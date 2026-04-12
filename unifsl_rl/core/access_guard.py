from collections.abc import Mapping

FORBIDDEN_STATE_LABEL_KEYS = {"query_labels", "test_labels", "val_labels", "support_labels"}
FORBIDDEN_PROBE_KEYS = {
    "val_features",
    "val_labels",
    "test_features",
    "test_labels",
    "query_features",
    "query_labels",
}


class GuardedMapping(Mapping):
    def __init__(self, data: dict, protocol, stage: str = "state"):
        self._data = data
        self._protocol = protocol
        self._stage = stage

    def __getitem__(self, key):
        self._ensure_key_allowed(key)
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def get(self, key, default=None):
        if key in self._data:
            self._ensure_key_allowed(key)
        return self._data.get(key, default)

    def _ensure_key_allowed(self, key):
        if self._protocol.name == "test_time_probe" and key in FORBIDDEN_PROBE_KEYS:
            raise RuntimeError(f"[AccessGuard] Probe stage={self._stage} cannot access '{key}'")


def ensure_state_safe(protocol, payload: dict, location: str):
    if protocol.can_use_labels_for_state:
        return
    for k in payload.keys():
        if k in FORBIDDEN_STATE_LABEL_KEYS or k.endswith("_labels"):
            raise RuntimeError(f"[AccessGuard] Illegal state label access at {location}: {k}")


def ensure_probe_reward_safe(protocol, split: str):
    if protocol.name != "test_time_probe":
        return
    if split in {"val", "test", "query"}:
        raise RuntimeError("[AccessGuard] Probe reward cannot use val/test/query labels")
