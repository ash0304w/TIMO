import math
from typing import List

from .types import CandidateConfig, VerificationResult


class ExactVerifier:
    def __init__(self, adapter, verify_repeats=1, use_ci=False):
        self.adapter = adapter
        self.verify_repeats = max(1, int(verify_repeats))
        self.use_ci = bool(use_ci)

    def verify(self, ctx, candidates: List[CandidateConfig], protocol) -> VerificationResult:
        out = []
        for cand in candidates:
            scores = []
            last = None
            for _ in range(self.verify_repeats):
                last = self.adapter.evaluate_candidate(ctx, cand, protocol=protocol)
                scores.append(last.selection_score)
            mean_s = sum(scores) / len(scores)
            if self.use_ci and len(scores) > 1:
                var = sum((s - mean_s) ** 2 for s in scores) / (len(scores) - 1)
                ci = 1.96 * math.sqrt(var / len(scores))
            else:
                ci = 0.0
            last.metadata = dict(last.metadata)
            last.metadata["verify_mean"] = mean_s
            last.metadata["verify_ci"] = ci
            last.selection_score = mean_s
            out.append(last)
        out.sort(key=lambda x: x.selection_score, reverse=True)
        return VerificationResult(ranked=out, protocol=protocol.name)
