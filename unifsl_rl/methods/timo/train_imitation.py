"""Warm-start imitation: supervised residual heads towards oracle/safe incumbents."""

import os
import torch


def run_imitation(policy, optimizer, states, targets, ckpt_path):
    policy.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    total = 0.0
    for x, y in zip(states, targets):
        out = policy.forward_logits(x.unsqueeze(0))
        loss = (
            loss_fn(out["alpha_delta"], y["alpha_delta"].unsqueeze(0))
            + loss_fn(out["gamma_delta"], y["gamma_delta"].unsqueeze(0))
            + loss_fn(out["beta_delta"], y["beta_delta"].unsqueeze(0))
            + loss_fn(out["subset_id"], y["subset_id"].unsqueeze(0))
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(policy.state_dict(), ckpt_path)
    return total / max(1, len(states))
