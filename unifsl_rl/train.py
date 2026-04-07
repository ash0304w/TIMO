import os

import torch

from unifsl_rl.core.action_spec import CompositeActionSpec
from unifsl_rl.core.conflict_checker import check_slot_conflicts
from unifsl_rl.core.controller import RLCoordinator
from unifsl_rl.core.policy_factory import build_policy
from unifsl_rl.core.protocol import TRAIN_PROTOCOL
from unifsl_rl.core.reward import RewardWeights, compute_reward
from unifsl_rl.core.state_builder import JointStateBuilder


def train_rl(adapter, rl_cfg):
    cache = adapter.build_cache()
    slots = [s for s in adapter.get_slots(cache)]
    check_slot_conflicts(slots)

    spec0 = CompositeActionSpec({
        "gamma": slots[0].action_spec,
        "beta": slots[1].action_spec,
        "subset_scores": slots[2].action_spec,
    })
    spec1 = CompositeActionSpec({"alpha": slots[3].action_spec})

    state_builder = JointStateBuilder(slots=slots, fixed_dim=64)
    policy0 = build_policy(64, spec0).to(adapter.device)
    policy1 = build_policy(64, spec1).to(adapter.device)

    params = list(policy0.parameters()) + list(policy1.parameters())
    optim = torch.optim.Adam(params, lr=rl_cfg.lr)
    weights = RewardWeights(
        align_lambda=rl_cfg.align_lambda,
        anom_lambda=rl_cfg.anom_lambda,
        cost_lambda=rl_cfg.cost_lambda,
        violation_lambda=rl_cfg.violation_lambda,
    )
    coordinator = RLCoordinator(adapter, slots, policy0, policy1, state_builder, TRAIN_PROTOCOL, compute_reward, weights, budget=rl_cfg.budget)

    logs = []
    for ep in range(rl_cfg.train_episodes):
        ctx = adapter.build_context(cache, TRAIN_PROTOCOL, split="val")
        step = coordinator.decide_and_run(ctx)
        advantage = step.reward - float(step.value.detach().cpu().item())
        loss = -(step.log_prob * advantage) - 0.001 * step.entropy + 0.5 * (step.value - step.reward) ** 2

        optim.zero_grad()
        loss.mean().backward()
        optim.step()

        logs.append((ep, step.reward, step.actions))
        print(f"[RL-TRAIN] ep={ep} reward={step.reward:.4f} alpha={step.actions['alpha']} beta={step.actions['beta']} gamma={step.actions['gamma']} subset={step.actions.get('subset').tolist() if step.actions.get('subset') is not None else None}")

    ckpt_dir = os.path.join(adapter.cfg["cache_dir"], "rl_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"unifsl_rl_{adapter.cfg['dataset']}_{adapter.cfg['shots']}shot_seed{adapter.cfg['seed']}.pt")
    torch.save({"policy0": policy0.state_dict(), "policy1": policy1.state_dict(), "cfg": adapter.cfg, "logs": logs}, ckpt_path)
    return ckpt_path, logs
