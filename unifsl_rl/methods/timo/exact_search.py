from unifsl_rl.core.types import CandidateConfig
from .ops import ALPHA_GRID, GAMMA_GRID, build_prefix_subset, evaluate_timo_config


def run_joint_exact(cfg, ctx, beta_domain_mode="repo_compat"):
    p = ctx["prompt_num"]
    bmax = p if beta_domain_mode == "paper_strict" else p * 2
    best = None
    for ai, _ in enumerate(ALPHA_GRID):
        for gi, _ in enumerate(GAMMA_GRID):
            for beta in range(1, bmax + 1):
                subset = build_prefix_subset(p, min(beta, p))
                acc, _, info = evaluate_timo_config(cfg, ctx, ai, min(beta, p), gi, subset, ctx["val_features"], ctx["val_labels"])
                if best is None or acc > best[0]:
                    best = (acc, ai, min(beta, p), gi, subset)
    _, ai, beta, gi, subset = best
    return CandidateConfig(
        alpha_idx=ai,
        alpha_value=ALPHA_GRID[ai],
        beta=beta,
        gamma_idx=gi,
        gamma_value=GAMMA_GRID[gi],
        subset_indices=subset,
        source_tag="joint_exact",
        mode_name="joint_exact",
        metadata={},
    )
