import os

from unifsl_rl.core.incumbent import IncumbentProvider
from unifsl_rl.core.protocol import OFFLINE_EVAL_PROTOCOL, STRICT_RL_PROTOCOL, TEST_TIME_PROBE_PROTOCOL
from unifsl_rl.methods.timo.infer_safe import run_safe_inference
from unifsl_rl.methods.timo.logs import append_csv, dump_candidate_pool, dump_final_json, ensure_dir, write_run_readme
from unifsl_rl.methods.timo.train_safe_rl import ResidualPolicy, load_policy


def eval_joint_exact(adapter, cfg):
    cache = adapter.build_cache()
    ctx = adapter.build_context(cache, OFFLINE_EVAL_PROTOCOL)
    inc = IncumbentProvider(adapter)
    joint = inc.get_joint_exact_incumbent(ctx)
    return joint


def eval_safe_or_strict(adapter, rl_cfg, strict=False):
    cache = adapter.build_cache()
    protocol = STRICT_RL_PROTOCOL if strict else OFFLINE_EVAL_PROTOCOL
    ctx = adapter.build_context(cache, protocol)

    policy = ResidualPolicy(state_dim=96, subset_candidates=max(16, rl_cfg.swap_window * 4)).to(adapter.device)
    load_policy(policy, rl_cfg.ckpt, adapter.device)

    res = run_safe_inference(adapter, ctx, protocol, policy, rl_cfg, strict_rl=strict)
    out_dir = os.path.join("outputs", "unifsl_rl", adapter.cfg["dataset"], f"{adapter.cfg['shots']}shot_seed{adapter.cfg['seed']}")
    ensure_dir(out_dir)
    if rl_cfg.dump_candidate_pool:
        dump_candidate_pool(os.path.join(out_dir, "candidate_pool.jsonl"), res["verified"].ranked)
    chosen = res["chosen"]
    inc = res["safe_inc"]
    row = {
        "raw_selection_acc": chosen.selection_score,
        "incumbent_acc": inc.selection_score,
        "delta_acc": chosen.selection_score - inc.selection_score,
        "reward": (chosen.selection_score - inc.selection_score) - rl_cfg.rl_cost_lambda * chosen.cost - rl_cfg.rl_violation_lambda * chosen.violation,
        "cost": chosen.cost,
        "violation": chosen.violation,
        "source_tag": chosen.source_tag,
    }
    append_csv(os.path.join(out_dir, "eval_log.csv"), row)
    dump_final_json(os.path.join(out_dir, "final_decision.json"), row)
    write_run_readme(os.path.join(out_dir, "README_run.md"), f"mode={'strict_rl' if strict else 'safe_rl'}")
    return res


def probe_safe(adapter, rl_cfg, strict=False):
    cache = adapter.build_cache()
    ctx = adapter.build_context(cache, TEST_TIME_PROBE_PROTOCOL)
    policy = ResidualPolicy(state_dim=96, subset_candidates=max(16, rl_cfg.swap_window * 4)).to(adapter.device)
    load_policy(policy, rl_cfg.ckpt, adapter.device)

    best = None
    for _ in range(max(1, rl_cfg.rl_budget)):
        res = run_safe_inference(adapter, ctx, TEST_TIME_PROBE_PROTOCOL, policy, rl_cfg, strict_rl=strict)
        chosen = res["chosen"]
        if best is None or chosen.selection_score > best["chosen"].selection_score:
            best = res
    out_dir = os.path.join("outputs", "unifsl_rl", adapter.cfg["dataset"], f"{adapter.cfg['shots']}shot_seed{adapter.cfg['seed']}")
    ensure_dir(out_dir)
    append_csv(os.path.join(out_dir, "probe_log.csv"), {
        "raw_selection_acc": best["chosen"].selection_score,
        "incumbent_acc": best["safe_inc"].selection_score,
        "delta_acc": best["chosen"].selection_score - best["safe_inc"].selection_score,
        "reward": best["chosen"].selection_score - best["safe_inc"].selection_score,
        "cost": best["chosen"].cost,
        "violation": best["chosen"].violation,
    })
    return best
