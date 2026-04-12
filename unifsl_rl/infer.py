import os

import torch

from unifsl_rl.core.action_spec import CompositeActionSpec
from unifsl_rl.core.conflict_checker import check_slot_conflicts
from unifsl_rl.core.controller import RLCoordinator
from unifsl_rl.core.policy_factory import PolicyFactory
from unifsl_rl.core.protocol import EVAL_PROTOCOL, PROBE_PROTOCOL, OFFLINE_EVAL_PROTOCOL, STRICT_RL_PROTOCOL, TEST_TIME_PROBE_PROTOCOL
from unifsl_rl.core.reward import RewardWeights, compute_reward
from unifsl_rl.core.state_builder import JointStateBuilder
from unifsl_rl.core.incumbent import IncumbentProvider
from unifsl_rl.methods.timo.infer_safe import run_safe_inference
from unifsl_rl.methods.timo.logs import append_csv, ensure_dir
from unifsl_rl.methods.timo.train_safe_rl import ResidualPolicy, load_policy


def _build_stage_specs(slots):
    stage0 = {}
    stage1 = {}
    for slot in slots:
        if slot.stage == 0:
            if "gamma" in slot.owns:
                stage0["gamma"] = slot.action_spec
            elif "beta" in slot.owns:
                stage0["beta"] = slot.action_spec
            elif "subset_scores" in slot.owns:
                stage0["subset_scores"] = slot.action_spec
        else:
            stage1["alpha"] = slot.action_spec
    return CompositeActionSpec(stage0), CompositeActionSpec(stage1)


def _load_policies(adapter, ckpt, slots):
    state_builder = JointStateBuilder(slots, fixed_dim=128)
    spec0, spec1 = _build_stage_specs(slots)
    p0 = PolicyFactory.build(128, spec0).to(adapter.device)
    p1 = PolicyFactory.build(128, spec1).to(adapter.device)
    data = torch.load(ckpt, map_location=adapter.device)
    p0.load_state_dict(data["policy_stage0"])
    p1.load_state_dict(data["policy_stage1"])
    return p0, p1, state_builder


def eval_pure_rl(adapter, rl_cfg):
    cache = adapter.build_cache()
    ctx = dict(adapter.build_protocol_view(EVAL_PROTOCOL, cache))
    slots = adapter.get_slots()
    check_slot_conflicts(slots)
    p0, p1, sb = _load_policies(adapter, rl_cfg.ckpt, slots)
    weights = RewardWeights(cost_lambda=rl_cfg.rl_cost_lambda, violation_lambda=rl_cfg.rl_violation_lambda)
    coord = RLCoordinator(adapter, slots, p0, p1, sb, EVAL_PROTOCOL, compute_reward, weights, budget=rl_cfg.rl_budget)
    return coord.decide_and_run(ctx, deterministic=True)


def probe_pure_rl(adapter, rl_cfg):
    cache = adapter.build_cache()
    ctx = dict(adapter.build_protocol_view(PROBE_PROTOCOL, cache))
    slots = adapter.get_slots()
    check_slot_conflicts(slots)
    p0, p1, sb = _load_policies(adapter, rl_cfg.ckpt, slots)
    weights = RewardWeights(cost_lambda=rl_cfg.rl_cost_lambda, violation_lambda=rl_cfg.rl_violation_lambda)
    coord = RLCoordinator(adapter, slots, p0, p1, sb, PROBE_PROTOCOL, compute_reward, weights, budget=rl_cfg.rl_budget)

    best = None
    for _ in range(max(1, rl_cfg.rl_budget)):
        step = coord.decide_and_run(ctx, deterministic=False)
        if best is None or step.reward > best.reward:
            best = step
    return best


# legacy functions kept as ablations
def eval_joint_exact(adapter, cfg):
    cache = adapter.build_cache()
    ctx = adapter.build_context(cache, OFFLINE_EVAL_PROTOCOL)
    inc = IncumbentProvider(adapter)
    return inc.get_joint_exact_incumbent(ctx)


def eval_safe_or_strict(adapter, rl_cfg, strict=False):
    cache = adapter.build_cache()
    protocol = STRICT_RL_PROTOCOL if strict else OFFLINE_EVAL_PROTOCOL
    ctx = adapter.build_context(cache, protocol)
    policy = ResidualPolicy(state_dim=96, subset_candidates=max(16, rl_cfg.swap_window * 4)).to(adapter.device)
    load_policy(policy, rl_cfg.ckpt, adapter.device)
    return run_safe_inference(adapter, ctx, protocol, policy, rl_cfg, strict_rl=strict)


def probe_safe(adapter, rl_cfg, strict=False):
    cache = adapter.build_cache()
    ctx = adapter.build_context(cache, TEST_TIME_PROBE_PROTOCOL)
    policy = ResidualPolicy(state_dim=96, subset_candidates=max(16, rl_cfg.swap_window * 4)).to(adapter.device)
    load_policy(policy, rl_cfg.ckpt, adapter.device)
    best = None
    for _ in range(max(1, rl_cfg.rl_budget)):
        res = run_safe_inference(adapter, ctx, TEST_TIME_PROBE_PROTOCOL, policy, rl_cfg, strict_rl=strict)
        if best is None or res["chosen"].selection_score > best["chosen"].selection_score:
            best = res
    out_dir = os.path.join("outputs", "unifsl_rl", adapter.cfg["dataset"], f"{adapter.cfg['shots']}shot_seed{adapter.cfg['seed']}")
    ensure_dir(out_dir)
    append_csv(os.path.join(out_dir, "probe_log.csv"), {"raw_selection_acc": best["chosen"].selection_score})
    return best


def eval_rl(adapter, rl_cfg):
    step = eval_pure_rl(adapter, rl_cfg)
    return step.run_output.get("acc", 0.0) * 100.0, step


def probe_rl(adapter, rl_cfg):
    step = probe_pure_rl(adapter, rl_cfg)
    return step.run_output.get("acc", 0.0) * 100.0, step
