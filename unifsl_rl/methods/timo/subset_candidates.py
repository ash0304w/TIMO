from .ops import build_subset_candidate_bank


def make_subset_candidates(prompt_num, beta, swap_window=8, swap_budget=2):
    prefix = list(range(max(1, min(beta, prompt_num))))
    win = list(range(min(prompt_num, beta + swap_window)))
    bank = build_subset_candidate_bank(prefix, win, swap_budget=swap_budget)
    return bank
