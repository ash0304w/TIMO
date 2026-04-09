
class SafetyGuard:
    def __init__(self, margin=0.0, require_significant_gain=True, strict_rl=False):
        self.margin = float(margin)
        self.require_significant_gain = bool(require_significant_gain)
        self.strict_rl = bool(strict_rl)

    def select(self, incumbent, verified):
        best = verified.ranked[0]
        if self.strict_rl:
            return best
        delta = best.selection_score - incumbent.selection_score
        if not self.require_significant_gain:
            return best if delta >= 0 else incumbent

        ci = best.metadata.get("verify_ci", 0.0)
        lower = delta - ci
        if lower > self.margin:
            return best
        return incumbent
