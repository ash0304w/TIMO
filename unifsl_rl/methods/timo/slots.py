import torch

from unifsl_rl.core.action_spec import Discrete, SubsetK
from unifsl_rl.core.slot import Slot
from .ops import ALPHA_GRID, GAMMA_GRID


class GammaSharpnessSlot(Slot):
    id = "timo.gamma"
    stage = 0

    def __init__(self):
        self._spec = Discrete(values=GAMMA_GRID)

    @property
    def owns(self):
        return ["gamma", "gamma_idx"]

    @property
    def action_spec(self):
        return self._spec

    def observe(self, ctx):
        stats = ctx["state_stats"]["gamma_stats"]
        return {"stats": torch.tensor(stats, dtype=torch.float32)}

    def apply(self, ctx, action):
        return {"gamma_idx": int(action["idx"]), "gamma": float(action["value"])}


class BetaPromptCountSlot(Slot):
    id = "timo.beta"
    stage = 0

    def __init__(self, prompt_num: int):
        self._spec = Discrete(values=list(range(1, int(prompt_num) + 1)))

    @property
    def owns(self):
        return ["beta", "beta_idx"]

    @property
    def action_spec(self):
        return self._spec

    def observe(self, ctx):
        stats = ctx["state_stats"]["beta_stats"]
        return {"stats": torch.tensor(stats, dtype=torch.float32)}

    def apply(self, ctx, action):
        return {"beta_idx": int(action["idx"]), "beta": int(action["value"])}


class PromptSubsetSlot(Slot):
    id = "timo.subset"
    stage = 0

    def __init__(self, prompt_num: int):
        self.prompt_num = int(prompt_num)
        self._spec = SubsetK(n_items=self.prompt_num)

    @property
    def owns(self):
        return ["subset_scores", "subset_indices"]

    @property
    def action_spec(self):
        return self._spec

    def observe(self, ctx):
        stats = ctx["state_stats"]["subset_stats"]
        return {"stats": torch.tensor(stats, dtype=torch.float32)}

    def apply(self, ctx, action):
        return {"subset_scores": action}


class AlphaFusionSlot(Slot):
    id = "timo.alpha"
    stage = 1

    def __init__(self):
        self._spec = Discrete(values=ALPHA_GRID)

    @property
    def owns(self):
        return ["alpha", "alpha_idx"]

    @property
    def action_spec(self):
        return self._spec

    def observe(self, ctx):
        stats = ctx["state_stats"]["alpha_stats"]
        return {"stats": torch.tensor(stats, dtype=torch.float32)}

    def apply(self, ctx, action):
        return {"alpha_idx": int(action["idx"]), "alpha": float(action["value"])}
