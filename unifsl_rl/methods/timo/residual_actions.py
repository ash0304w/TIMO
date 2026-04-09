from .ops import ALPHA_GRID, GAMMA_GRID

ALPHA_DELTAS = [-2, -1, 0, 1, 2]
GAMMA_DELTAS = [-2, -1, 0, 1, 2]
BETA_DELTAS = [-8, -4, -2, -1, 0, 1, 2, 4, 8]


def apply_residual(incumbent, action, prompt_num, beta_domain_mode="repo_compat"):
    ai = max(0, min(len(ALPHA_GRID) - 1, incumbent.alpha_idx + ALPHA_DELTAS[action["alpha_delta_idx"]]))
    gi = max(0, min(len(GAMMA_GRID) - 1, incumbent.gamma_idx + GAMMA_DELTAS[action["gamma_delta_idx"]]))
    bmax = prompt_num if beta_domain_mode == "paper_strict" else prompt_num * 2
    beta = max(1, min(bmax, incumbent.beta + BETA_DELTAS[action["beta_delta_idx"]]))
    return ai, beta, gi
