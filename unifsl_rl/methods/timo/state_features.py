import torch
import torch.nn.functional as F


def build_state_features(ctx, diagnostics, incumbent, protocol_name, subset_mode="residual_prefix_swap", beta_domain_mode="repo_compat"):
    ms = diagnostics["matching_score"]
    ms_mean = ms.mean(0)
    soft = F.softmax(ms_mean, dim=0)
    topk = torch.topk(ms_mean, k=min(max(2, incumbent.beta), ms_mean.numel())).values

    prompt_block = [
        float(ms.mean().item()), float(ms.std().item()), float(ms.max().item()), float(ms.min().item()),
        float((-soft * (soft + 1e-12).log()).sum().item()),
        float((topk[0] - topk[1]).item()) if topk.numel() > 1 else 0.0,
        float((topk[0] - topk[-1]).item()),
    ]

    sim = ctx["image_prototypes"] @ ctx["image_prototypes"].t()
    off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    anom = [
        float(off.mean().item()),
        float(off.max().item()),
        float(torch.topk(off, k=min(5, off.numel())).values.mean().item()),
        float((off > 0.3).float().mean().item()),
    ]

    cross = [
        float(diagnostics.get("agreement", 0.0)),
        float(diagnostics.get("tgi_entropy", 0.0)),
        float(diagnostics.get("igt_entropy", 0.0)),
        float(diagnostics.get("tgi_margin", 0.0)),
        float(diagnostics.get("igt_margin", 0.0)),
    ]

    inc = [
        float(incumbent.alpha_idx), float(incumbent.alpha_value), float(incumbent.beta),
        float(incumbent.gamma_idx), float(incumbent.gamma_value), float(incumbent.selection_score),
        float(ctx.get("paper_joint_gap", 0.0)),
    ]
    prot = {"offline_train": [1, 0, 0], "offline_eval": [0, 1, 0], "test_time_probe": [0, 0, 1], "strict_rl": [0, 1, 1]}[protocol_name]
    subset_oh = [1, 0] if subset_mode == "prefix" else [0, 1]
    beta_oh = [1, 0] if beta_domain_mode == "paper_strict" else [0, 1]
    meta = [float(ctx["shots"]), float(ctx["cate_num"]), float(ctx["prompt_num"]), *prot, *subset_oh, *beta_oh]

    feat = torch.tensor(anom + prompt_block + cross + inc + meta, device=ctx["support_vecs"].device).float()
    fixed_dim = 96
    if feat.numel() < fixed_dim:
        feat = torch.cat([feat, torch.zeros(fixed_dim - feat.numel(), device=feat.device)], dim=0)
    return feat[:fixed_dim]
