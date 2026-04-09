from dataclasses import asdict

from .types import CandidateConfig, CandidateResult


class IncumbentProvider:
    def __init__(self, adapter):
        self.adapter = adapter

    def get_paper_incumbent(self, ctx) -> CandidateResult:
        cand = self.adapter.get_paper_incumbent_candidate(ctx)
        return self.adapter.evaluate_candidate(ctx, cand)

    def get_joint_exact_incumbent(self, ctx) -> CandidateResult:
        cand = self.adapter.get_joint_exact_candidate(ctx)
        res = self.adapter.evaluate_candidate(ctx, cand)
        return res

    def get_safe_incumbent(self, ctx) -> CandidateResult:
        paper = self.get_paper_incumbent(ctx)
        joint = self.get_joint_exact_incumbent(ctx)
        return joint if joint.selection_score >= paper.selection_score else paper
