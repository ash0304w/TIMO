from unifsl_rl.methods import build_adapter


def test_registry_has_second_method():
    cfg = {"dataset": "x", "cache_dir": "/tmp", "shots": 1}
    adapter_cls = build_adapter("gda_clip", cfg, device="cpu").__class__.__name__
    assert adapter_cls == "GDAClipAdapter"
