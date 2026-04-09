from dataclasses import dataclass


@dataclass
class RewardWeights:
    align_lambda: float = 0.0
    anom_lambda: float = 0.0
    cost_lambda: float = 0.0
    violation_lambda: float = 0.0


def compute_reward(base_acc, align_gain, anom_risk, cost, violation, weights: RewardWeights):
    return (
        base_acc
        + weights.align_lambda * align_gain
        - weights.anom_lambda * anom_risk
        - weights.cost_lambda * cost
        - weights.violation_lambda * violation
    )
