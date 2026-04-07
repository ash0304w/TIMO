FORBIDDEN_LABEL_KEYS = {"query_labels", "test_labels", "val_labels"}


def ensure_state_safe(protocol, payload: dict, location: str):
    if protocol.can_use_labels_for_state:
        return
    for k in payload.keys():
        if k in FORBIDDEN_LABEL_KEYS or k.endswith("_labels"):
            raise RuntimeError(f"[AccessGuard] Illegal state label access at {location}: {k}")


def ensure_probe_reward_safe(protocol, split: str):
    if protocol.name != "probe":
        return
    if split in {"val", "test", "query"}:
        raise RuntimeError("[AccessGuard] Probe reward cannot use val/test/query labels")
