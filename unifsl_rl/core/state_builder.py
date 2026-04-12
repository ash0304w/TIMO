from typing import Dict, List

import torch

from .access_guard import GuardedMapping, ensure_state_safe


class JointStateBuilder:
    def __init__(self, slots: List, fixed_dim: int = 128):
        self.slots = slots
        self.fixed_dim = fixed_dim

    def build(self, ctx: Dict, protocol, stage: int = 0):
        states = []
        guarded_ctx = GuardedMapping(ctx, protocol=protocol, stage=f"state_stage{stage}")
        for slot in self.slots:
            if getattr(slot, "stage", 0) != stage:
                continue
            obs = slot.observe(guarded_ctx)
            ensure_state_safe(protocol, obs, f"slot.observe:{slot.id}")
            stats = obs["stats"].flatten().float()
            states.append(stats)

        x = torch.cat(states, dim=0) if states else torch.zeros(0)
        if x.numel() >= self.fixed_dim:
            return x[: self.fixed_dim]
        return torch.cat([x, torch.zeros(self.fixed_dim - x.numel(), device=x.device)], dim=0)
