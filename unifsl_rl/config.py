from dataclasses import dataclass


@dataclass
class RLConfig:
    mode: str = "eval"
    train_episodes: int = 50
    lr: float = 1e-3
    budget: int = 2
    trials: int = 1
    cost_lambda: float = 0.01
    violation_lambda: float = 0.1
    align_lambda: float = 0.0
    anom_lambda: float = 0.0
    seed: int = 1
    device: str = "cuda"
    ckpt: str = ""
