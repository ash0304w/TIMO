import torch

from unifsl_rl.core.action_spec import Discrete, SubsetK
from unifsl_rl.core.slot import Slot
from .ops import ALPHA_CANDIDATES, GAMMA_CANDIDATES, build_state_stats


class AlphaFusionSlot(Slot):
    def __init__(self):
        super().__init__(id="alpha", owns=["alpha"], action_spec=Discrete(ALPHA_CANDIDATES))

    def observe(self, ctx):
        stats = build_state_stats(ctx, diagnostics=ctx.get("branch_diagnostics"), protocol_name=ctx["protocol"].name, remaining_budget=ctx.get("remaining_budget", 0))
        return {"stats": stats}

    def apply(self, ctx, action):
        ctx["alpha"] = float(action)


class BetaPromptCountSlot(Slot):
    def __init__(self, prompt_num):
        super().__init__(id="beta", owns=["beta"], action_spec=Discrete(list(range(1, prompt_num + 1))))

    def observe(self, ctx):
        stats = build_state_stats(ctx, diagnostics=ctx.get("branch_diagnostics"), protocol_name=ctx["protocol"].name, remaining_budget=ctx.get("remaining_budget", 0))
        return {"stats": stats}

    def apply(self, ctx, action):
        ctx["beta"] = int(action)


class PromptSubsetSlot(Slot):
    def __init__(self, prompt_num, per_class=False):
        super().__init__(id="subset_scores", owns=["subset"], action_spec=SubsetK(n_items=prompt_num, per_class=per_class))

    def observe(self, ctx):
        stats = build_state_stats(ctx, diagnostics=ctx.get("branch_diagnostics"), protocol_name=ctx["protocol"].name, remaining_budget=ctx.get("remaining_budget", 0))
        return {"stats": stats}

    def apply(self, ctx, action):
        if isinstance(action, torch.Tensor):
            ctx["subset_scores"] = action
        else:
            ctx["subset_scores"] = torch.tensor(action, device=ctx["device"]).float()


class GammaSharpnessSlot(Slot):
    def __init__(self):
        super().__init__(id="gamma", owns=["gamma"], action_spec=Discrete(GAMMA_CANDIDATES))

    def observe(self, ctx):
        stats = build_state_stats(ctx, diagnostics=ctx.get("branch_diagnostics"), protocol_name=ctx["protocol"].name, remaining_budget=ctx.get("remaining_budget", 0))
        return {"stats": stats}

    def apply(self, ctx, action):
        ctx["gamma"] = float(action)
