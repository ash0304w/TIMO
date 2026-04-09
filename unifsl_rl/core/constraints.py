from dataclasses import dataclass

import torch


@dataclass
class ConstraintReport:
    violation: float
    repaired: bool


class ConstraintEngine:
    def __init__(self, budget=2):
        self.budget = budget

    def _repair_subset(self, scores, beta):
        k = max(1, int(beta))
        k = min(k, scores.numel())
        idx = torch.topk(scores, k=k).indices
        return idx

    def apply(self, actions: dict, ctx: dict):
        violation = 0.0
        repaired = False

        beta = int(actions["beta"])
        p = int(ctx["prompt_num"])
        if beta < 1 or beta > p:
            repaired = True
            violation += 1.0
            beta = max(1, min(beta, p))
            actions["beta"] = beta

        subset_scores = actions.get("subset_scores")
        if subset_scores is not None:
            subset_idx = self._repair_subset(subset_scores, beta)
            actions["subset"] = subset_idx
            if subset_idx.numel() != beta:
                repaired = True
                violation += 1.0

        actions["gamma"] = float(max(1e-4, actions["gamma"]))
        actions["alpha"] = float(max(1e-4, actions["alpha"]))

        return actions, ConstraintReport(violation=violation, repaired=repaired)

    def cost(self, actions):
        beta = float(actions.get("beta", 1))
        return beta
