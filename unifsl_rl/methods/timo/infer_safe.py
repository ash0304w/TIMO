from unifsl_rl.core.incumbent import IncumbentProvider
from unifsl_rl.core.safety_guard import SafetyGuard
from unifsl_rl.core.verifier import ExactVerifier
from unifsl_rl.methods.timo.candidate_builder import build_candidate_pool
from unifsl_rl.methods.timo.residual_actions import apply_residual
from unifsl_rl.methods.timo.subset_candidates import make_subset_candidates
from unifsl_rl.core.types import CandidateConfig
from .ops import ALPHA_GRID, GAMMA_GRID


def make_proposal_from_policy(policy, state, incumbent, prompt_num, beta_domain_mode, subset_bank):
    action, _, _, _ = policy.sample(state)
    ai, beta, gi = apply_residual(incumbent, action, prompt_num, beta_domain_mode)
    subset = subset_bank[min(action["subset_id"], len(subset_bank) - 1)]
    return CandidateConfig(
        alpha_idx=ai,
        alpha_value=ALPHA_GRID[ai],
        beta=min(beta, prompt_num),
        gamma_idx=gi,
        gamma_value=GAMMA_GRID[gi],
        subset_indices=subset,
        source_tag="rl_proposal",
        mode_name="safe_rl",
        metadata={"action": action},
    )


def run_safe_inference(adapter, ctx, protocol, policy, cfg, strict_rl=False):
    inc_provider = IncumbentProvider(adapter)
    paper_inc = inc_provider.get_paper_incumbent(ctx)
    joint_inc = inc_provider.get_joint_exact_incumbent(ctx)
    safe_inc = inc_provider.get_safe_incumbent(ctx)
    ctx["paper_joint_gap"] = joint_inc.selection_score - paper_inc.selection_score

    subset_bank = make_subset_candidates(ctx["prompt_num"], safe_inc.beta, cfg.swap_window, cfg.swap_budget)
    state = adapter.build_state(ctx, safe_inc, protocol)
    proposal = make_proposal_from_policy(policy, state, safe_inc, ctx["prompt_num"], cfg.beta_domain_mode, subset_bank)

    cands = build_candidate_pool(ctx, paper_inc, joint_inc, safe_inc, proposal, protocol, cfg.rl_budget, cfg)
    verifier = ExactVerifier(adapter, verify_repeats=cfg.verify_repeats, use_ci=cfg.verify_use_ci)
    verified = verifier.verify(ctx, cands, protocol)
    guard = SafetyGuard(margin=cfg.significance_margin, require_significant_gain=cfg.require_significant_gain, strict_rl=strict_rl)
    chosen = guard.select(safe_inc, verified)
    return {
        "paper_inc": paper_inc,
        "joint_inc": joint_inc,
        "safe_inc": safe_inc,
        "proposal": proposal,
        "verified": verified,
        "chosen": chosen,
    }
