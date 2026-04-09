import os

import torch

from unifsl_rl.core.incumbent import IncumbentProvider
from unifsl_rl.core.protocol import OFFLINE_TRAIN_PROTOCOL
from unifsl_rl.methods.timo.infer_safe import run_safe_inference
from unifsl_rl.methods.timo.logs import append_csv, dump_final_json, ensure_dir, write_run_readme
from unifsl_rl.methods.timo.subset_candidates import make_subset_candidates
from unifsl_rl.methods.timo.train_imitation import run_imitation
from unifsl_rl.methods.timo.train_safe_rl import ResidualPolicy, save_policy


def train_rl(adapter, rl_cfg):
    cache = adapter.build_cache()
    ctx = adapter.build_context(cache, OFFLINE_TRAIN_PROTOCOL)
    out_dir = os.path.join("outputs", "unifsl_rl", adapter.cfg["dataset"], f"{adapter.cfg['shots']}shot_seed{adapter.cfg['seed']}")
    ensure_dir(out_dir)

    policy = ResidualPolicy(state_dim=96, subset_candidates=max(16, rl_cfg.swap_window * 4)).to(adapter.device)
    opt = torch.optim.Adam(policy.parameters(), lr=rl_cfg.rl_lr)

    # Stage-1 imitation (single task fallback)
    inc_provider = IncumbentProvider(adapter)
    safe_inc = inc_provider.get_safe_incumbent(ctx)
    state = adapter.build_state(ctx, safe_inc, OFFLINE_TRAIN_PROTOCOL)
    target = {
        "alpha_delta": torch.tensor(2, device=adapter.device),
        "gamma_delta": torch.tensor(2, device=adapter.device),
        "beta_delta": torch.tensor(4, device=adapter.device),
        "subset_id": torch.tensor(0, device=adapter.device),
    }
    warm_ckpt = rl_cfg.warm_start_ckpt or os.path.join(out_dir, "warm_start.pt")
    imitation_loss = run_imitation(policy, opt, [state], [target], warm_ckpt)

    # Stage-2 conservative safe RL (bandit improvement over safe incumbent)
    train_csv = os.path.join(out_dir, "train_log.csv")
    best_ckpt = os.path.join(out_dir, "best_safe_rl.pt")
    best_score = -1e9
    for ep in range(rl_cfg.rl_train_epochs):
        res = run_safe_inference(adapter, ctx, OFFLINE_TRAIN_PROTOCOL, policy, rl_cfg, strict_rl=False)
        chosen = res["chosen"]
        incumbent = res["safe_inc"]
        reward = (chosen.selection_score - incumbent.selection_score) - rl_cfg.rl_cost_lambda * chosen.cost - rl_cfg.rl_violation_lambda * chosen.violation
        row = {
            "epoch": ep,
            "raw_selection_acc": chosen.selection_score,
            "incumbent_acc": incumbent.selection_score,
            "delta_acc": chosen.selection_score - incumbent.selection_score,
            "reward": reward,
            "cost": chosen.cost,
            "violation": chosen.violation,
        }
        append_csv(train_csv, row)
        if chosen.selection_score > best_score:
            best_score = chosen.selection_score
            save_policy(policy, best_ckpt)

    dump_final_json(os.path.join(out_dir, "final_decision.json"), {"warm_start_loss": imitation_loss, "best_selection_score": best_score, "ckpt": best_ckpt})
    write_run_readme(os.path.join(out_dir, "README_run.md"), "Safe-RL training completed with incumbent-conditioned residual proposals.")
    return best_ckpt, {"warm_start_loss": imitation_loss, "best_selection_score": best_score}
