from typing import Dict, Optional

import torch
import torch.nn.functional as F

from utils import cls_acc, image_guide_text, vec_sort

ALPHA_CANDIDATES = [10 ** i for i in range(-4, 5)]
GAMMA_CANDIDATES = list(range(5, 101, 5))


def build_image_weights_from_cache(cache_keys, cache_values):
    cate_num = cache_values.shape[1]
    image_weights_all = torch.stack([cache_keys.t()[torch.argmax(cache_values, dim=1) == i] for i in range(cate_num)])
    image_weights = image_weights_all.mean(dim=1)
    image_weights = image_weights / image_weights.norm(dim=1, keepdim=True)
    return image_weights


def run_igt(cfg, clip_weights_all, image_weights, gamma, return_matching=True):
    out = image_guide_text(cfg, clip_weights_all, image_weights, gamma=gamma, return_matching=return_matching)
    if return_matching:
        clip_weights_igt, matching_score = out
        return clip_weights_igt.t(), matching_score
    return out.t(), None


def sort_prompt_bank(vecs_t, matching_score):
    return vec_sort(vecs_t, matching_score)


def select_tgi_prompts(vecs_t, matching_score, beta, subset_scores=None, mode="paper"):
    sorted_vecs, sorted_weights = sort_prompt_bank(vecs_t, matching_score)
    c, p, d = sorted_vecs.shape
    beta = max(1, min(int(beta), p))
    if mode == "paper" or subset_scores is None:
        selected_vecs = sorted_vecs[:, :beta, :]
        selected_weights = sorted_weights[:, :beta]
        subset_idx = None
    else:
        idx = torch.topk(subset_scores, k=beta).indices
        idx = idx.sort().values
        selected_vecs = vecs_t[:, idx, :]
        selected_weights = matching_score[:, idx]
        subset_idx = idx
    return {
        "selected_vecs": selected_vecs,
        "selected_weights": selected_weights,
        "subset_idx": subset_idx,
    }


def build_tgi_transfer_set(selected_prompts, selected_weights, support_vecs, support_labels):
    c, beta, _ = selected_prompts.shape
    weighted_prompts = selected_prompts * selected_weights.unsqueeze(-1)
    text_vecs = weighted_prompts.reshape(c * beta, -1)
    text_labels = torch.arange(c, device=text_vecs.device).unsqueeze(1).repeat(1, beta).flatten()
    vecs = torch.cat([text_vecs, support_vecs.float()])
    labels = torch.cat([text_labels.float(), support_labels.float()])
    return vecs, labels


def fit_gda_classifier(vecs, labels, class_num):
    mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(class_num)])
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(class_num)])
    cov = center_vecs.T.cov()
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * cov + cov.trace() * torch.eye(center_vecs.shape[1], device=vecs.device))
    ps = torch.ones(class_num, device=vecs.device) / class_num
    W = torch.einsum("nd,dc->cn", mus, cov_inv)
    b = ps.log() - torch.einsum("nd,dc,nc->n", mus, cov_inv, mus) / 2
    return W, b


def _diag_from_logits(logits):
    prob = logits.softmax(-1)
    ent = (-prob * (prob + 1e-12).log()).sum(-1).mean()
    top2 = torch.topk(prob, k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).mean()
    return ent.item(), margin.item()


def run_timo_config(cfg, ctx: Dict, alpha, beta, subset=None, gamma=50, mode="rl"):
    clip_weights_igt, matching_score = run_igt(cfg, ctx["clip_weights_all"], ctx["image_weights"], gamma=gamma, return_matching=True)
    sel = select_tgi_prompts(ctx["clip_weights_all"], matching_score, beta=beta, subset_scores=subset, mode=mode)
    vecs_c, labels_c = build_tgi_transfer_set(sel["selected_vecs"], sel["selected_weights"], ctx["support_vecs"], ctx["support_labels"])

    W, b = fit_gda_classifier(vecs_c, labels_c, ctx["cate_num"])
    logits_tgi = ctx["eval_features"].float() @ W + b
    logits_igt = alpha * ctx["eval_features"].float() @ clip_weights_igt.float()
    logits_final = logits_tgi + logits_igt

    sup_tgi = ctx["support_vecs"].float() @ W + b
    sup_igt = alpha * ctx["support_vecs"].float() @ clip_weights_igt.float()
    agreement = (sup_tgi.argmax(-1) == sup_igt.argmax(-1)).float().mean().item()
    ent_tgi, margin_tgi = _diag_from_logits(sup_tgi)
    ent_igt, margin_igt = _diag_from_logits(sup_igt)

    diagnostics = {
        "agreement": agreement,
        "tgi_entropy": ent_tgi,
        "igt_entropy": ent_igt,
        "tgi_margin": margin_tgi,
        "igt_margin": margin_igt,
        "matching_score": matching_score,
        "subset_idx": sel["subset_idx"],
        "beta": int(beta),
        "gamma": float(gamma),
        "alpha": float(alpha),
    }
    return {
        "logits_tgi": logits_tgi,
        "logits_igt": logits_igt,
        "logits_final": logits_final,
        "diagnostics": diagnostics,
    }


def support_loo_score(cfg, ctx, alpha, beta, subset, gamma):
    x = ctx["support_vecs"]
    y = ctx["support_labels"]
    total = len(y)
    correct = 0
    for i in range(total):
        mask = torch.ones(total, dtype=torch.bool, device=x.device)
        mask[i] = False
        sub_ctx = dict(ctx)
        sub_ctx["support_vecs"] = x[mask]
        sub_ctx["support_labels"] = y[mask]
        sub_ctx["eval_features"] = x[i : i + 1]
        out = run_timo_config(cfg, sub_ctx, alpha=alpha, beta=beta, subset=subset, gamma=gamma, mode="rl")
        pred = out["logits_final"].argmax(-1)
        correct += int(pred.item() == int(y[i].item()))
    return correct / max(total, 1)


def build_state_stats(ctx, diagnostics: Optional[Dict] = None, protocol_name="eval", remaining_budget=0):
    ms = ctx["matching_score"] if "matching_score" in ctx else diagnostics["matching_score"]
    ms_soft = F.softmax(ms.mean(0), dim=0)
    ms_top = torch.topk(ms.mean(0), k=min(5, ms.shape[1])).values
    top1_top2 = (ms_top[0] - ms_top[1]).item() if ms_top.numel() > 1 else 0.0
    top1_topk = (ms_top[0] - ms_top[-1]).item()
    prompt_block = torch.tensor([
        ms.mean().item(), ms.std().item(), ms.max().item(), ms.min().item(),
        float((-ms_soft * (ms_soft + 1e-12).log()).sum().item()),
        top1_top2, top1_topk,
    ], device=ms.device)

    sim = ctx["image_weights"] @ ctx["image_weights"].t()
    off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    anom_block = torch.tensor([
        off.mean().item(), off.max().item(), torch.topk(off, k=min(5, off.numel())).values.mean().item(),
        (off > 0.3).float().mean().item(),
    ], device=ms.device)

    if diagnostics is None:
        cross = torch.zeros(5, device=ms.device)
    else:
        cross = torch.tensor([
            diagnostics.get("agreement", 0.0),
            diagnostics.get("tgi_entropy", 0.0), diagnostics.get("igt_entropy", 0.0),
            diagnostics.get("tgi_margin", 0.0), diagnostics.get("igt_margin", 0.0),
        ], device=ms.device)

    protocol_one_hot = {
        "train": [1, 0, 0], "eval": [0, 1, 0], "probe": [0, 0, 1]
    }[protocol_name]
    meta = torch.tensor([
        float(ctx["shots"]), float(ctx["cate_num"]), float(ctx["prompt_num"]),
        *protocol_one_hot, float(remaining_budget),
    ], device=ms.device)

    return torch.cat([anom_block, prompt_block, cross, meta], dim=0)
