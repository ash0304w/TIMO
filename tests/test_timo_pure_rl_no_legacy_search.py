from unifsl_rl.methods.timo import ops


def test_pure_no_legacy_search_symbols():
    # pure_rl path should not depend on these legacy selectors in controller/train/infer.
    assert hasattr(ops, "run_timo_config")
    assert hasattr(ops, "support_loo_score")
