import torch

from unifsl_rl.core.action_spec import CompositeActionSpec
from unifsl_rl.core.conflict_checker import check_slot_conflicts
from unifsl_rl.core.controller import RLCoordinator
from unifsl_rl.core.policy_factory import build_policy
from unifsl_rl.core.protocol import EVAL_PROTOCOL, PROBE_PROTOCOL
from unifsl_rl.core.reward import RewardWeights, compute_reward
from unifsl_rl.core.state_builder import JointStateBuilder
from unifsl_rl.methods.timo.probe import run_probe


def _build_runtime(adapter, protocol, rl_cfg, ckpt_path):
    cache = adapter.build_cache()
    slots = [s for s in adapter.get_slots(cache)]
    check_slot_conflicts(slots)
    spec0 = CompositeActionSpec({"gamma": slots[0].action_spec, "beta": slots[1].action_spec, "subset_scores": slots[2].action_spec})
    spec1 = CompositeActionSpec({"alpha": slots[3].action_spec})

    policy0 = build_policy(64, spec0).to(adapter.device)
    policy1 = build_policy(64, spec1).to(adapter.device)
    ckpt = torch.load(ckpt_path, map_location=adapter.device, weights_only=False)
    policy0.load_state_dict(ckpt["policy0"])
    policy1.load_state_dict(ckpt["policy1"])

    weights = RewardWeights(
        align_lambda=rl_cfg.align_lambda,
        anom_lambda=rl_cfg.anom_lambda,
        cost_lambda=rl_cfg.cost_lambda,
        violation_lambda=rl_cfg.violation_lambda,
    )
    coordinator = RLCoordinator(adapter, slots, policy0, policy1, JointStateBuilder(slots, 64), protocol, compute_reward, weights, budget=rl_cfg.budget)
    return cache, coordinator


def eval_rl(adapter, rl_cfg):
    cache, coordinator = _build_runtime(adapter, EVAL_PROTOCOL, rl_cfg, rl_cfg.ckpt)
    ctx = adapter.build_context(cache, EVAL_PROTOCOL, split="test")
    step = coordinator.decide_and_run(ctx)
    test_acc = (step.run_output["logits_final"].argmax(-1) == ctx["eval_labels"]).float().mean().item() * 100.0
    return test_acc, step


def probe_rl(adapter, rl_cfg):
    cache, coordinator = _build_runtime(adapter, PROBE_PROTOCOL, rl_cfg, rl_cfg.ckpt)
    ctx = adapter.build_context(cache, PROBE_PROTOCOL, split="support")
    step = run_probe(adapter, coordinator, ctx, budget=rl_cfg.budget)

    final_actions = step.actions
    test_acc, _ = adapter.run_final_inference(cache, final_actions)
    return test_acc, step
