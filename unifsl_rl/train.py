import json
import os

import torch

from unifsl_rl.core.action_spec import CompositeActionSpec
from unifsl_rl.core.conflict_checker import check_slot_conflicts
from unifsl_rl.core.controller import RLCoordinator
from unifsl_rl.core.policy_factory import PolicyFactory
from unifsl_rl.core.protocol import TRAIN_PROTOCOL
from unifsl_rl.core.reward import RewardWeights, compute_reward
from unifsl_rl.core.state_builder import JointStateBuilder


def _build_stage_specs(slots):
    stage0 = {}
    stage1 = {}
    for slot in slots:
        if slot.stage == 0:
            stage0_key = slot.owns[0].split("_")[0] if slot.owns else slot.id
            if "gamma" in slot.owns:
                stage0_key = "gamma"
            elif "beta" in slot.owns:
                stage0_key = "beta"
            elif "subset_scores" in slot.owns:
                stage0_key = "subset_scores"
            stage0[stage0_key] = slot.action_spec
        else:
            stage1["alpha"] = slot.action_spec
    return CompositeActionSpec(stage0), CompositeActionSpec(stage1)


def _save_outputs(out_dir, train_rows, best_step, policy0, policy1):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "best_pure_rl.pt")
    torch.save({"policy_stage0": policy0.state_dict(), "policy_stage1": policy1.state_dict()}, ckpt)
    with open(os.path.join(out_dir, "train_log.csv"), "w", encoding="utf-8") as f:
        f.write("epoch,reward,cost,violation,base_acc\n")
        for r in train_rows:
            f.write(f"{r['epoch']},{r['reward']},{r['cost']},{r['violation']},{r['base_acc']}\n")
    with open(os.path.join(out_dir, "final_decision.json"), "w", encoding="utf-8") as f:
        json.dump(best_step, f, indent=2)
    return ckpt


def train_pure_rl(adapter, rl_cfg):
    cache = adapter.build_cache()
    ctx = dict(adapter.build_protocol_view(TRAIN_PROTOCOL, cache))
    slots = adapter.get_slots()
    check_slot_conflicts(slots)
    state_builder = JointStateBuilder(slots, fixed_dim=128)
    spec0, spec1 = _build_stage_specs(slots)

    policy0 = PolicyFactory.build(128, spec0).to(adapter.device)
    policy1 = PolicyFactory.build(128, spec1).to(adapter.device)
    params = list(policy0.parameters()) + list(policy1.parameters())
    opt = torch.optim.Adam(params, lr=rl_cfg.rl_lr)

    weights = RewardWeights(cost_lambda=rl_cfg.rl_cost_lambda, violation_lambda=rl_cfg.rl_violation_lambda)
    coord = RLCoordinator(adapter, slots, policy0, policy1, state_builder, TRAIN_PROTOCOL, compute_reward, weights, budget=rl_cfg.rl_budget)

    train_rows = []
    best = {"reward": -1e9}
    for ep in range(rl_cfg.rl_train_epochs):
        step = coord.decide_and_run(ctx, deterministic=False)
        advantage = step.reward - float(step.value.detach().cpu().item())
        policy_loss = -(step.log_prob * advantage)
        value_loss = (step.value - step.reward) ** 2
        entropy_bonus = -rl_cfg.rl_entropy_coef * step.entropy
        loss = policy_loss + rl_cfg.rl_value_coef * value_loss + entropy_bonus
        opt.zero_grad()
        loss.backward()
        opt.step()

        row = {
            "epoch": ep,
            "reward": float(step.reward),
            "cost": float(step.cost),
            "violation": float(step.violation),
            "base_acc": float(step.run_output.get("acc", 0.0)),
        }
        train_rows.append(row)
        if row["reward"] > best["reward"]:
            best = {"reward": row["reward"], "actions": {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in step.actions.items()}}

    out_dir = os.path.join("outputs", "unifsl_rl", adapter.cfg["dataset"], f"{adapter.cfg['shots']}shot_seed{adapter.cfg['seed']}")
    ckpt = _save_outputs(out_dir, train_rows, best, policy0, policy1)
    return ckpt, best


# backward-compat name used by old main paths
train_rl = train_pure_rl
