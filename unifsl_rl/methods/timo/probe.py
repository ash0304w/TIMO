from unifsl_rl.core.access_guard import ensure_probe_reward_safe
from .ops import support_loo_score


def run_probe(wrapper, coordinator, ctx, budget=2):
    best = None
    for t in range(max(1, budget)):
        ctx["remaining_budget"] = budget - t
        step = coordinator.decide_and_run(ctx)
        ensure_probe_reward_safe(ctx["protocol"], "support")
        loo = support_loo_score(wrapper.cfg, ctx, step.actions["alpha"], step.actions["beta"], step.actions.get("subset_scores"), step.actions["gamma"])
        score = loo
        if best is None or score > best[0]:
            best = (score, step)
    return best[1]
