from unifsl_rl.core.candidate_pool import CandidatePool
from unifsl_rl.core.types import CandidateConfig
from .subset_candidates import make_subset_candidates
from .ops import ALPHA_GRID, GAMMA_GRID


def build_candidate_pool(ctx, paper_inc, joint_inc, safe_inc, proposal, protocol, budget, cfg):
    pool = CandidatePool()
    pool.add(paper_inc)
    pool.add(joint_inc)
    pool.add(safe_inc)
    if proposal is not None:
        pool.add(proposal)

    def add_neighbors(base, prefix):
        for da in [-1, 0, 1]:
            ai = max(0, min(len(ALPHA_GRID) - 1, base.alpha_idx + da))
            pool.add(CandidateConfig(ai, ALPHA_GRID[ai], base.beta, base.gamma_idx, base.gamma_value, base.subset_indices, f"{prefix}_alpha", base.mode_name, {}))
        for dg in [-1, 0, 1]:
            gi = max(0, min(len(GAMMA_GRID) - 1, base.gamma_idx + dg))
            pool.add(CandidateConfig(base.alpha_idx, base.alpha_value, base.beta, gi, GAMMA_GRID[gi], base.subset_indices, f"{prefix}_gamma", base.mode_name, {}))
        for db in [-2, -1, 1, 2]:
            b = max(1, min(ctx["prompt_num"], base.beta + db))
            subset = base.subset_indices[:b] if len(base.subset_indices) >= b else list(range(b))
            pool.add(CandidateConfig(base.alpha_idx, base.alpha_value, b, base.gamma_idx, base.gamma_value, subset, f"{prefix}_beta", base.mode_name, {}))

    add_neighbors(safe_inc, "local_inc")
    if proposal is not None:
        add_neighbors(proposal, "local_prop")

    swap_bank = make_subset_candidates(ctx["prompt_num"], safe_inc.beta, swap_window=cfg.swap_window, swap_budget=cfg.swap_budget)
    for s in swap_bank[: max(3, budget * 3)]:
        pool.add(CandidateConfig(safe_inc.alpha_idx, safe_inc.alpha_value, safe_inc.beta, safe_inc.gamma_idx, safe_inc.gamma_value, s, "subset_swap", safe_inc.mode_name, {}))

    pool.unique()
    return pool.items
