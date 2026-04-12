from typing import Dict

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .action_spec import CompositeActionSpec, ContinuousBox, Discrete, SubsetK


class MultiHeadPolicy(nn.Module):
    def __init__(self, state_dim: int, composite_spec: CompositeActionSpec, hidden_dim=128):
        super().__init__()
        self.spec = composite_spec
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.heads = nn.ModuleDict()
        for k, s in self.spec.specs.items():
            if isinstance(s, Discrete):
                self.heads[k] = nn.Linear(hidden_dim, s.n)
            elif isinstance(s, SubsetK):
                self.heads[k] = nn.Linear(hidden_dim, s.n_items)
            elif isinstance(s, ContinuousBox):
                dim = int(torch.tensor(s.shape).prod().item())
                self.heads[k] = nn.Linear(hidden_dim, dim)
            else:
                self.heads[k] = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.encoder(state)
        return h, self.value_head(h)

    def sample_actions(self, state, deterministic=False):
        h, value = self.forward(state)
        actions: Dict = {}
        log_prob = torch.zeros(1, device=h.device)
        entropy = torch.zeros(1, device=h.device)

        for k, s in self.spec.specs.items():
            logits = self.heads[k](h)
            if isinstance(s, Discrete):
                dist = Categorical(logits=logits)
                idx = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
                idx = idx.squeeze(0)
                actions[k] = {
                    "idx": int(idx.item()),
                    "value": float(s.values[int(idx.item())]),
                }
                log_prob = log_prob + dist.log_prob(idx)
                entropy = entropy + dist.entropy()
            elif isinstance(s, SubsetK):
                actions[k] = logits.squeeze(0)
            elif isinstance(s, ContinuousBox):
                mean = logits
                std = self.log_std.exp()
                dist = Normal(mean, std)
                a = mean if deterministic else dist.sample()
                actions[k] = a.squeeze(0)
                log_prob = log_prob + dist.log_prob(a).sum(-1)
                entropy = entropy + dist.entropy().sum(-1)
        return actions, log_prob.squeeze(0), entropy.squeeze(0), value.squeeze(-1).squeeze(0)


class PolicyFactory:
    @staticmethod
    def build(state_dim: int, composite_spec: CompositeActionSpec, hidden_dim: int = 128):
        return MultiHeadPolicy(state_dim, composite_spec, hidden_dim=hidden_dim)


def build_policy(state_dim, composite_spec, hidden_dim=128):
    return PolicyFactory.build(state_dim, composite_spec, hidden_dim)
