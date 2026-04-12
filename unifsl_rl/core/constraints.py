from dataclasses import dataclass

import torch


@dataclass
class ConstraintReport:
    violation: float
    repaired: bool


class ConstraintEngine:
    def __init__(self, budget=2):
        self.budget = budget

    def _repair_subset(self, scores, beta, prompt_num):
        k = max(1, min(int(beta), int(prompt_num)))
        idx = torch.topk(scores, k=k).indices
        idx = idx.clamp(0, prompt_num - 1).long()
        idx = torch.unique(idx, sorted=True)
        if idx.numel() < k:
            fill = [i for i in range(prompt_num) if i not in set(idx.tolist())][: k - idx.numel()]
            idx = torch.cat([idx, torch.tensor(fill, device=idx.device, dtype=torch.long)], dim=0)
        return idx[:k]

    def apply(self, actions: dict, ctx: dict):
        violation = 0.0
        repaired = False
        prompt_num = int(ctx["prompt_num"])

        beta = int(actions["beta"])
        if beta < 1 or beta > prompt_num:
            repaired = True
            violation += 1.0
            beta = max(1, min(beta, prompt_num))
            actions["beta"] = beta

        if "subset_scores" in actions:
            subset_idx = self._repair_subset(actions["subset_scores"], beta, prompt_num)
            actions["subset_indices"] = [int(i) for i in subset_idx.tolist()]
            if len(actions["subset_indices"]) != beta:
                violation += 1.0
                repaired = True

        gamma = float(actions.get("gamma", 1.0))
        alpha = float(actions.get("alpha", 1.0))
        gamma = float(max(1e-4, gamma))
        alpha = float(max(1e-4, alpha))
        actions["gamma"] = gamma
        actions["alpha"] = alpha

        return actions, ConstraintReport(violation=violation, repaired=repaired)

    def cost(self, actions):
        return float(actions.get("beta", 1))

    def violation_penalty(self, report: ConstraintReport, lam: float):
        return lam * float(report.violation)
