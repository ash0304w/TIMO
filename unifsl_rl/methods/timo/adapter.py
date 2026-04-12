import torch

from unifsl_rl.core.method_wrapper import MethodWrapper
from unifsl_rl.core.types import CandidateConfig, CandidateResult
from unifsl_rl.core.access_guard import GuardedMapping, ensure_probe_reward_safe
from utils import loda_val_test_feature, load_few_shot_feature, image_guide_text_search
from .exact_search import run_joint_exact
from .ops import (
    ALPHA_GRID,
    GAMMA_GRID,
    build_igt_text_weights,
    build_image_prototypes,
    build_prefix_subset,
    build_state_stats,
    evaluate_timo_candidate,
    run_timo_config,
    support_loo_score,
)
from .slots import AlphaFusionSlot, BetaPromptCountSlot, GammaSharpnessSlot, PromptSubsetSlot


class TIMOAdapter(MethodWrapper):
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self._cache = None

    def build_cache(self, cfg=None):
        if self._cache is not None:
            return self._cache
        cfg = cfg or self.cfg
        clip_weights_all = torch.load(cfg["cache_dir"] + "/text_weights_cupl_t_all.pt", weights_only=False).float().to(self.device)
        cache_keys, cache_values = load_few_shot_feature(cfg)
        cache_keys = cache_keys.to(self.device).float()
        cache_values = cache_values.to(self.device).float()
        support_vecs = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float().to(self.device)
        support_labels = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float().to(self.device)
        val_features, val_labels = loda_val_test_feature(cfg, "val")
        val_features, val_labels = val_features.to(self.device).float(), val_labels.to(self.device)
        if cfg["dataset"] == "imagenet":
            test_features, test_labels = val_features, val_labels
        else:
            test_features, test_labels = loda_val_test_feature(cfg, "test")
            test_features, test_labels = test_features.to(self.device).float(), test_labels.to(self.device)

        image_prototypes = build_image_prototypes(cache_keys, cache_values)
        cate_num, prompt_num, _ = clip_weights_all.shape
        self._cache = {
            "clip_weights_all": clip_weights_all,
            "cache_keys": cache_keys,
            "cache_values": cache_values,
            "support_vecs": support_vecs,
            "support_labels": support_labels,
            "val_features": val_features,
            "val_labels": val_labels,
            "test_features": test_features,
            "test_labels": test_labels,
            "image_prototypes": image_prototypes,
            "cate_num": cate_num,
            "prompt_num": prompt_num,
            "shots": cfg["shots"],
            "meta": {"dataset": cfg["dataset"]},
        }
        return self._cache

    def build_protocol_view(self, protocol, cache):
        view = dict(cache)
        if protocol.name == "test_time_probe":
            for k in ["val_features", "val_labels", "test_features", "test_labels"]:
                view.pop(k, None)
        view["protocol"] = protocol
        view["state_stats"] = build_state_stats(view)
        return GuardedMapping(view, protocol=protocol, stage="ctx")

    # backward compat
    def build_context(self, cache, protocol, split="val"):
        return dict(self.build_protocol_view(protocol, cache))

    def get_slots(self):
        cache = self.build_cache()
        return [
            GammaSharpnessSlot(),
            BetaPromptCountSlot(cache["prompt_num"]),
            PromptSubsetSlot(cache["prompt_num"]),
            AlphaFusionSlot(),
        ]

    def materialize(self, prefix_action, ctx):
        base = dict(ctx)
        gamma_idx = int(prefix_action["gamma_idx"])
        gamma_value = GAMMA_GRID[gamma_idx]
        clip_weights_igt, matching = build_igt_text_weights(self.cfg, base["clip_weights_all"], base["image_prototypes"], gamma_value, True)
        base["clip_weights_igt"] = clip_weights_igt
        base["matching_score"] = matching
        base["state_stats"] = build_state_stats(base, branch_diagnostics={"agreement": 0.0})
        return base

    def run_with_action(self, ctx, action):
        protocol = ctx["protocol"]
        if protocol.name == "test_time_probe":
            score, info = support_loo_score(self.cfg, ctx, action)
            return {"acc": score, "diagnostics": info}

        split = "val" if protocol.selection_split == "val" else "test"
        feats = ctx[f"{split}_features"]
        labels = ctx[f"{split}_labels"]
        acc, logits, info = run_timo_config(self.cfg, ctx, action, feats, labels)
        return {"acc": acc, "logits": logits, "diagnostics": info.get("diagnostics", {})}

    def evaluate(self, ctx, run_output, protocol):
        split = protocol.selection_split
        ensure_probe_reward_safe(protocol, split)
        return {
            "base_acc": float(run_output["acc"]),
            "align_gain": 0.0,
            "anom_risk": 0.0,
        }

    def estimate_cost(self, action):
        return float(action.get("beta", 1.0))

    # -------- legacy methods for safe_rl path --------
    def get_paper_incumbent_candidate(self, ctx):
        clip_weights_igt, matching = image_guide_text_search(self.cfg, ctx["clip_weights_all"], ctx["val_features"], ctx["val_labels"], ctx["image_prototypes"])
        alpha_idx = ALPHA_GRID.index(10.0) if 10.0 in ALPHA_GRID else 5
        gamma_idx = GAMMA_GRID.index(50) if 50 in GAMMA_GRID else 9
        beta = ctx["prompt_num"]
        subset = build_prefix_subset(ctx["prompt_num"], beta)
        return CandidateConfig(alpha_idx, ALPHA_GRID[alpha_idx], beta, gamma_idx, GAMMA_GRID[gamma_idx], subset, "paper", "paper", {})

    def get_joint_exact_candidate(self, ctx):
        return run_joint_exact(self.cfg, ctx, beta_domain_mode="repo_compat")

    def evaluate_candidate(self, ctx, candidate, protocol):
        score, info = evaluate_timo_candidate(self.cfg, ctx, candidate, protocol)
        return CandidateResult(
            **candidate.__dict__,
            selection_score=float(score),
            diagnostics=info,
            cost=float(candidate.beta),
            violation=0.0,
            repair_flag=False,
            raw_accuracy=float(score),
        )
