from dataclasses import dataclass

import torch

from .constraints import ConstraintEngine


@dataclass
class StepResult:
    actions: dict
    report: dict
    run_output: dict
    reward: float
    cost: float
    violation: float
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class RLCoordinator:
    def __init__(self, wrapper, slots, policy_stage0, policy_stage1, state_builder, protocol, reward_fn, reward_weights, budget=2):
        self.wrapper = wrapper
        self.slots = slots
        self.policy_stage0 = policy_stage0
        self.policy_stage1 = policy_stage1
        self.state_builder = state_builder
        self.protocol = protocol
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.constraint = ConstraintEngine(budget=budget)

    def decide_stage0(self, ctx, deterministic=False):
        s0 = self.state_builder.build(ctx, self.protocol, stage=0)
        a0, lp0, ent0, v0 = self.policy_stage0.sample_actions(s0, deterministic=deterministic)
        actions = {
            "gamma_idx": a0["gamma"]["idx"],
            "gamma": a0["gamma"]["value"],
            "beta_idx": a0["beta"]["idx"],
            "beta": int(a0["beta"]["value"]),
            "subset_scores": a0["subset_scores"],
            "alpha": 1.0,
        }
        return actions, lp0, ent0, v0

    def materialize(self, ctx, stage0_actions):
        return self.wrapper.materialize(stage0_actions, ctx)

    def decide_stage1(self, ctx, deterministic=False):
        s1 = self.state_builder.build(ctx, self.protocol, stage=1)
        a1, lp1, ent1, v1 = self.policy_stage1.sample_actions(s1, deterministic=deterministic)
        return {
            "alpha_idx": a1["alpha"]["idx"],
            "alpha": a1["alpha"]["value"],
        }, lp1, ent1, v1

    def apply_constraints(self, actions, ctx):
        return self.constraint.apply(actions, ctx)

    def evaluate(self, ctx, actions):
        run_output = self.wrapper.run_with_action(ctx, actions)
        eval_stats = self.wrapper.evaluate(ctx, run_output, self.protocol)
        return run_output, eval_stats

    def decide_and_run(self, ctx, deterministic=False):
        stage0_actions, lp0, ent0, v0 = self.decide_stage0(ctx, deterministic=deterministic)
        stage0_actions, report0 = self.apply_constraints(stage0_actions, ctx)

        ctx2 = self.materialize(ctx, stage0_actions)
        alpha_actions, lp1, ent1, v1 = self.decide_stage1(ctx2, deterministic=deterministic)
        actions = dict(stage0_actions)
        actions.update(alpha_actions)
        actions, report1 = self.apply_constraints(actions, ctx)

        run_output, eval_stats = self.evaluate(ctx2, actions)
        reward = self.reward_fn(
            eval_stats["base_acc"],
            eval_stats.get("align_gain", 0.0),
            eval_stats.get("anom_risk", 0.0),
            self.constraint.cost(actions),
            report0.violation + report1.violation,
            self.reward_weights,
        )
        return StepResult(
            actions=actions,
            report={"stage0": report0, "stage1": report1},
            run_output=run_output,
            reward=float(reward),
            cost=self.constraint.cost(actions),
            violation=float(report0.violation + report1.violation),
            log_prob=lp0 + lp1,
            entropy=ent0 + ent1,
            value=(v0 + v1) / 2,
        )
