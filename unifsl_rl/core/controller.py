from dataclasses import dataclass

import torch

from .constraints import ConstraintEngine


@dataclass
class StepResult:
    actions: dict
    report: object
    run_output: dict
    reward: float
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

    def decide_and_run(self, ctx, deterministic=False):
        # Step 0: gamma/beta/subset
        s0 = self.state_builder.build(ctx, self.protocol)
        a0, lp0, ent0, v0 = self.policy_stage0.sample_actions(s0)
        actions = {
            "gamma": a0["gamma"],
            "beta": int(a0["beta"]),
            "subset_scores": a0["subset_scores"],
            "alpha": 1.0,
        }
        actions, report0 = self.constraint.apply(actions, ctx)

        draft_out = self.wrapper.run_with_config(ctx, actions)
        ctx2 = dict(ctx)
        ctx2["branch_diagnostics"] = draft_out["diagnostics"]

        # Step 1: alpha
        s1 = self.state_builder.build(ctx2, self.protocol)
        a1, lp1, ent1, v1 = self.policy_stage1.sample_actions(s1)
        actions["alpha"] = float(a1["alpha"])
        actions, report1 = self.constraint.apply(actions, ctx)

        run_output = self.wrapper.run_with_config(ctx, actions)
        eval_stats = self.wrapper.evaluate(ctx, run_output, self.protocol)
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
            reward=reward,
            log_prob=lp0 + lp1,
            entropy=ent0 + ent1,
            value=(v0 + v1) / 2,
        )
