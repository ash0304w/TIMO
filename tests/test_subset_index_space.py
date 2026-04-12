from unifsl_rl.core.constraints import ConstraintEngine


def test_subset_index_space_is_original_prompt_id():
    engine = ConstraintEngine()
    actions = {"beta": 3, "subset_scores": __import__('torch').tensor([0.1, 0.9, 0.2, 0.8]), "alpha": 1.0, "gamma": 1.0}
    ctx = {"prompt_num": 4}
    fixed, _ = engine.apply(actions, ctx)
    assert fixed["subset_indices"] == sorted(fixed["subset_indices"])
    assert all(0 <= i < 4 for i in fixed["subset_indices"])
