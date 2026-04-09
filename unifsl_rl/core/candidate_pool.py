class CandidatePool:
    def __init__(self):
        self.items = []

    def add(self, cand):
        self.items.append(cand)

    def extend(self, cands):
        self.items.extend(cands)

    def unique(self):
        seen = set()
        uniq = []
        for c in self.items:
            key = (c.alpha_idx, c.beta, c.gamma_idx, tuple(c.subset_indices))
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        self.items = uniq
        return self.items
