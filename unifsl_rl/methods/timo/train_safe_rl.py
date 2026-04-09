import os

import torch
import torch.nn as nn
from torch.distributions import Categorical

from unifsl_rl.core.metrics import delta_acc
from .residual_actions import apply_residual


class ResidualPolicy(nn.Module):
    def __init__(self, state_dim=96, subset_candidates=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.alpha_head = nn.Linear(128, 5)
        self.gamma_head = nn.Linear(128, 5)
        self.beta_head = nn.Linear(128, 9)
        self.subset_head = nn.Linear(128, subset_candidates)
        self.value = nn.Linear(128, 1)

    def forward_logits(self, x):
        h = self.enc(x)
        return {
            "alpha_delta": self.alpha_head(h),
            "gamma_delta": self.gamma_head(h),
            "beta_delta": self.beta_head(h),
            "subset_id": self.subset_head(h),
            "value": self.value(h),
        }

    def sample(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.forward_logits(x)
        d_alpha = Categorical(logits=out["alpha_delta"])
        d_gamma = Categorical(logits=out["gamma_delta"])
        d_beta = Categorical(logits=out["beta_delta"])
        d_subset = Categorical(logits=out["subset_id"])

        a = {
            "alpha_delta_idx": int(d_alpha.sample().item()),
            "gamma_delta_idx": int(d_gamma.sample().item()),
            "beta_delta_idx": int(d_beta.sample().item()),
            "subset_id": int(d_subset.sample().item()),
        }
        log_prob = d_alpha.log_prob(torch.tensor(a["alpha_delta_idx"], device=x.device)) + \
            d_gamma.log_prob(torch.tensor(a["gamma_delta_idx"], device=x.device)) + \
            d_beta.log_prob(torch.tensor(a["beta_delta_idx"], device=x.device)) + \
            d_subset.log_prob(torch.tensor(a["subset_id"], device=x.device))
        entropy = d_alpha.entropy() + d_gamma.entropy() + d_beta.entropy() + d_subset.entropy()
        return a, log_prob, entropy.mean(), out["value"].mean()


def train_safe_rl_step(policy, optimizer, state, incumbent, subset_bank, prompt_num, beta_domain_mode, chosen, incumbent_result, cost_lambda=1e-4, violation_lambda=1e-4):
    action, log_prob, entropy, value = policy.sample(state)
    ai, beta, gi = apply_residual(incumbent, action, prompt_num, beta_domain_mode)
    subset = subset_bank[min(action["subset_id"], len(subset_bank) - 1)]
    reward = (chosen.selection_score - incumbent_result.selection_score) - cost_lambda * chosen.cost - violation_lambda * chosen.violation

    adv = reward - float(value.detach().item())
    loss = -(log_prob * adv) - 0.001 * entropy + 0.5 * (value - reward) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return reward, action


def save_policy(policy, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(policy.state_dict(), path)


def load_policy(policy, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(state)
    return policy
