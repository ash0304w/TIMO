"""TIMO reusable primitives shared by paper/joint_exact/safe_rl/strict_rl."""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from utils import cls_acc, image_guide_text, vec_sort

ALPHA_GRID = [10 ** i for i in range(-4, 5)]
GAMMA_GRID = list(range(5, 101, 5))


def build_image_prototypes(cache_keys, cache_values):
    cate_num = cache_values.shape[1]
    image_weights_all = torch.stack([cache_keys.t()[torch.argmax(cache_values, dim=1) == i] for i in range(cate_num)])
    image_weights = image_weights_all.mean(dim=1)
    return image_weights / image_weights.norm(dim=1, keepdim=True)


def build_prompt_matching_scores(cfg, clip_weights_all, image_prototypes, gamma_value):
    _, matching = image_guide_text(cfg, clip_weights_all, image_prototypes, gamma=gamma_value, return_matching=True)
    return matching


def build_igt_text_weights(cfg, clip_weights_all, image_prototypes, gamma_value, return_matching=True):
    out = image_guide_text(cfg, clip_weights_all, image_prototypes, gamma=gamma_value, return_matching=return_matching)
    if return_matching:
        clip_weights_igt, matching_score = out
        return clip_weights_igt.t(), matching_score
    return out.t(), None


def sort_prompt_bank(vecs_t, matching_score):
    return vec_sort(vecs_t, matching_score)


def build_prefix_subset(prompt_num, beta):
    b = max(1, min(int(beta), int(prompt_num)))
    return list(range(b))


def build_subset_candidate_bank(prefix_subset: List[int], window_indices: List[int], swap_budget=2):
    base = list(prefix_subset)
    cands = [base]
    outside = [i for i in window_indices if i not in base]
    inside = list(base)

    for out_idx in outside:
        for in_idx in inside:
            cand = [out_idx if x == in_idx else x for x in base]
            cand = sorted(list(dict.fromkeys(cand)))
            if len(cand) == len(base):
                cands.append(cand)
    if swap_budget >= 2:
        for i, o1 in enumerate(outside):
            for o2 in outside[i + 1 :]:
                if len(inside) < 2:
                    continue
                cand = base.copy()
                cand[0] = o1
                cand[1] = o2
                cand = sorted(list(dict.fromkeys(cand)))
                if len(cand) == len(base):
                    cands.append(cand)
    uniq = []
    seen = set()
    for c in cands:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            uniq.append(c)
    return uniq


def build_tgi_augmented_vectors(clip_weights_all, matching_score, support_vecs, support_labels, beta, subset_indices):
    cate_num, prompt_num, _ = clip_weights_all.shape
    subset_indices = sorted([int(i) for i in subset_indices])
    subset_indices = [i for i in subset_indices if 0 <= i < prompt_num]
    if len(subset_indices) == 0:
        subset_indices = [0]
    if len(subset_indices) != beta:
        subset_indices = (subset_indices + list(range(prompt_num)))[:beta]

    selected_vecs = clip_weights_all[:, subset_indices, :]
    selected_weights = matching_score[:, subset_indices]
    selected_vecs = selected_vecs * selected_weights.unsqueeze(-1)

    text_vecs = selected_vecs.reshape(cate_num * beta, -1)
    text_labels = torch.arange(cate_num, device=text_vecs.device).unsqueeze(1).repeat(1, beta).flatten().float()
    vecs = torch.cat([text_vecs, support_vecs.float()])
    labels = torch.cat([text_labels, support_labels.float()])
    return vecs, labels


def fit_gda_from_vectors(vecs, labels, class_num):
    mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(class_num)])
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(class_num)])
    cov = center_vecs.T.cov()
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * cov + cov.trace() * torch.eye(center_vecs.shape[1], device=vecs.device))
    ps = torch.ones(class_num, device=vecs.device) / class_num
    W = torch.einsum("nd,dc->cn", mus, cov_inv)
    b = ps.log() - torch.einsum("nd,dc,nc->n", mus, cov_inv, mus) / 2
    return W, b


def compute_timo_logits(eval_features, clip_weights_igt, W, b, alpha_value):
    logits_tgi = eval_features.float() @ W + b
    logits_igt = alpha_value * eval_features.float() @ clip_weights_igt.float()
    return logits_tgi, logits_igt, logits_tgi + logits_igt


def materialize_branch_diagnostics(support_vecs, clip_weights_igt, W, b, alpha_value, matching_score):
    sup_tgi = support_vecs.float() @ W + b
    sup_igt = alpha_value * support_vecs.float() @ clip_weights_igt.float()
    prob_tgi = F.softmax(sup_tgi, dim=-1)
    prob_igt = F.softmax(sup_igt, dim=-1)
    agree = (sup_tgi.argmax(-1) == sup_igt.argmax(-1)).float().mean().item()
    ent_tgi = float((-prob_tgi * (prob_tgi + 1e-12).log()).sum(-1).mean().item())
    ent_igt = float((-prob_igt * (prob_igt + 1e-12).log()).sum(-1).mean().item())
    top2_tgi = torch.topk(prob_tgi, k=2, dim=-1).values
    top2_igt = torch.topk(prob_igt, k=2, dim=-1).values
    margin_tgi = float((top2_tgi[:, 0] - top2_tgi[:, 1]).mean().item())
    margin_igt = float((top2_igt[:, 0] - top2_igt[:, 1]).mean().item())
    return {
        "agreement": agree,
        "tgi_entropy": ent_tgi,
        "igt_entropy": ent_igt,
        "tgi_margin": margin_tgi,
        "igt_margin": margin_igt,
        "matching_score": matching_score,
    }


def evaluate_timo_config(cfg, ctx: Dict, alpha_idx, beta, gamma_idx, subset_indices, split_features, split_labels):
    alpha_value = ALPHA_GRID[alpha_idx]
    gamma_value = GAMMA_GRID[gamma_idx]
    clip_weights_igt, matching_score = build_igt_text_weights(cfg, ctx["clip_weights_all"], ctx["image_prototypes"], gamma_value, return_matching=True)
    vecs, labels = build_tgi_augmented_vectors(ctx["clip_weights_all"], matching_score, ctx["support_vecs"], ctx["support_labels"], beta, subset_indices)
    W, b = fit_gda_from_vectors(vecs, labels, ctx["cate_num"])
    logits_tgi, logits_igt, logits_final = compute_timo_logits(split_features, clip_weights_igt, W, b, alpha_value)
    acc = cls_acc(logits_final, split_labels) / 100.0
    diag = materialize_branch_diagnostics(ctx["support_vecs"], clip_weights_igt, W, b, alpha_value, matching_score)
    return acc, logits_final, {
        "alpha_idx": alpha_idx,
        "alpha_value": alpha_value,
        "gamma_idx": gamma_idx,
        "gamma_value": gamma_value,
        "beta": beta,
        "subset_indices": subset_indices,
        "diagnostics": diag,
    }


def evaluate_timo_candidate(cfg, ctx, candidate, protocol):
    if protocol.name in {"offline_train", "offline_eval"}:
        feats, labels = ctx["val_features"], ctx["val_labels"]
    elif protocol.name == "test_time_probe":
        return evaluate_support_loo(cfg, ctx, candidate)
    else:
        feats, labels = ctx["test_features"], ctx["test_labels"]

    acc, logits, info = evaluate_timo_config(
        cfg,
        ctx,
        candidate.alpha_idx,
        candidate.beta,
        candidate.gamma_idx,
        candidate.subset_indices,
        feats,
        labels,
    )
    return acc, info


def evaluate_support_loo(cfg, ctx, candidate):
    x, y = ctx["support_vecs"], ctx["support_labels"]
    total = len(y)
    corr = 0
    for i in range(total):
        mask = torch.ones(total, dtype=torch.bool, device=x.device)
        mask[i] = False
        sub_ctx = dict(ctx)
        sub_ctx["support_vecs"] = x[mask]
        sub_ctx["support_labels"] = y[mask]
        acc, _, _ = evaluate_timo_config(
            cfg,
            sub_ctx,
            candidate.alpha_idx,
            candidate.beta,
            candidate.gamma_idx,
            candidate.subset_indices,
            x[i : i + 1],
            y[i : i + 1],
        )
        corr += int(acc > 0.5)
    return corr / max(1, total), {"probe_metric": "support_loo"}
