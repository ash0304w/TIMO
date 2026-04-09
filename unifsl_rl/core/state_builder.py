from typing import Dict, List

import torch

from .access_guard import ensure_state_safe


class JointStateBuilder:
    def __init__(self, slots: List, fixed_dim: int = 64):
        self.slots = slots
        self.fixed_dim = fixed_dim

    def build(self, ctx: Dict, protocol):
        states = []
        for slot in self.slots:
            obs = slot.observe(ctx)
            ensure_state_safe(protocol, obs, f"slot.observe:{slot.id}")
            states.append(obs["stats"].flatten())

        x = torch.cat(states, dim=0).float()
        if x.numel() >= self.fixed_dim:
            x = x[: self.fixed_dim]
        else:
            x = torch.cat([x, torch.zeros(self.fixed_dim - x.numel(), device=x.device)], dim=0)
        return x
