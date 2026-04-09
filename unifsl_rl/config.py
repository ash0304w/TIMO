from dataclasses import dataclass


@dataclass
class RLConfig:
    timo_mode: str = "safe_rl"
    rl_mode: str = "eval"
    ckpt: str = ""
    save_rl_outputs: int = 1

    safe_floor_mode: str = "safe"
    require_significant_gain: int = 1
    significance_margin: float = 0.0
    verify_repeats: int = 3
    verify_use_ci: int = 1

    subset_mode: str = "residual_prefix_swap"
    swap_window: int = 8
    swap_budget: int = 2
    beta_domain_mode: str = "repo_compat"

    rl_train_tasks: str = "auto"
    rl_train_epochs: int = 20
    rl_batch_size: int = 8
    rl_lr: float = 1e-3
    rl_entropy_coef: float = 1e-3
    rl_value_coef: float = 0.5
    rl_cost_lambda: float = 1e-4
    rl_violation_lambda: float = 1e-4
    rl_seed: int = 1
    rl_device: str = "cuda"
    warm_start_ckpt: str = ""

    rl_trials: int = 3
    rl_budget: int = 2
    probe_metric: str = "support_loo"
    probe_refine_neighbors: int = 1

    dump_jsonl: int = 1
    dump_csv: int = 1
    dump_candidate_pool: int = 1
