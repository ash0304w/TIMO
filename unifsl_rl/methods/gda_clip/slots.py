import torch

from unifsl_rl.core.action_spec import Discrete
from unifsl_rl.core.slot import Slot
from unifsl_rl.methods.timo.ops import ALPHA_GRID


class GDAAlphaSlot(Slot):
    id = "gda_clip.alpha"
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
        return {"stats": torch.tensor(ctx["state_stats"]["alpha_stats"], dtype=torch.float32)}

    def apply(self, ctx, action):
        return {"alpha_idx": int(action["idx"]), "alpha": float(action["value"])}


class GDADummyStage0Slot(Slot):
    id = "gda_clip.stage0"
    stage = 0

    def __init__(self):
        self._spec = Discrete(values=[1.0])

    @property
    def owns(self):
        return ["gamma", "gamma_idx", "beta", "beta_idx"]

    @property
    def action_spec(self):
        return self._spec

    def observe(self, ctx):
        return {"stats": torch.tensor([0.0, 1.0], dtype=torch.float32)}

    def apply(self, ctx, action):
        return {"gamma": 1.0, "gamma_idx": 0, "beta": 1, "beta_idx": 0}
