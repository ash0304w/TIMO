from dataclasses import dataclass

from .residual_actions import ALPHA_DELTAS, GAMMA_DELTAS, BETA_DELTAS


@dataclass
class AlphaFusionSlot:
    alpha_grid: list
    residual_deltas: list = None

    def __post_init__(self):
        self.residual_deltas = ALPHA_DELTAS


@dataclass
class GammaSharpnessSlot:
    gamma_grid: list
    residual_deltas: list = None

    def __post_init__(self):
        self.residual_deltas = GAMMA_DELTAS


@dataclass
class BetaPromptCountSlot:
    residual_deltas: list = None

    def __post_init__(self):
        self.residual_deltas = BETA_DELTAS


@dataclass
class PromptSubsetSlot:
    subset_mode: str = "residual_prefix_swap"
