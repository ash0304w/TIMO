import torch

from unifsl_rl.core.method_wrapper import MethodWrapper
from unifsl_rl.core.types import CandidateConfig, CandidateResult
from utils import cls_acc, loda_val_test_feature, load_few_shot_feature, image_guide_text_search
from .exact_search import run_joint_exact
from .ops import (
    ALPHA_GRID,
    GAMMA_GRID,
    build_igt_text_weights,
    build_image_prototypes,
    build_prefix_subset,
    evaluate_timo_candidate,
)
from .state_features import build_state_features
from .protocols import OFFLINE_EVAL


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
        }
        return self._cache

    def build_context(self, cache, protocol, split="val"):
        ctx = dict(cache)
        ctx["protocol"] = protocol
        return ctx

    def get_paper_incumbent_candidate(self, ctx):
        clip_weights_igt, matching = image_guide_text_search(self.cfg, ctx["clip_weights_all"], ctx["val_features"], ctx["val_labels"], ctx["image_prototypes"])
        # Use original staged behavior by evaluating TIMO-compatible defaults in wrapped form.
        # beta uses prompt_num to mimic non-grid + grid best anchor baseline.
        alpha_idx = ALPHA_GRID.index(10.0) if 10.0 in ALPHA_GRID else 5
        gamma_idx = GAMMA_GRID.index(50) if 50 in GAMMA_GRID else 9
        beta = ctx["prompt_num"]
        subset = build_prefix_subset(ctx["prompt_num"], beta)
        return CandidateConfig(alpha_idx, ALPHA_GRID[alpha_idx], beta, gamma_idx, GAMMA_GRID[gamma_idx], subset, "paper", "paper", {})

    def get_joint_exact_candidate(self, ctx):
        return run_joint_exact(self.cfg, ctx, beta_domain_mode="repo_compat")

    def evaluate_candidate(self, ctx, candidate, protocol=OFFLINE_EVAL):
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

    def run_with_config(self, ctx, config):
        cand = CandidateConfig(
            alpha_idx=config["alpha_idx"],
            alpha_value=ALPHA_GRID[config["alpha_idx"]],
            beta=config["beta"],
            gamma_idx=config["gamma_idx"],
            gamma_value=GAMMA_GRID[config["gamma_idx"]],
            subset_indices=config["subset_indices"],
            source_tag=config.get("source_tag", "direct"),
            mode_name=config.get("mode_name", "direct"),
            metadata=config.get("metadata", {}),
        )
        return self.evaluate_candidate(ctx, cand, protocol=ctx["protocol"])

    def build_state(self, ctx, incumbent, protocol):
        # diagnostics from incumbent config
        clip_weights_igt, matching = build_igt_text_weights(self.cfg, ctx["clip_weights_all"], ctx["image_prototypes"], incumbent.gamma_value, True)
        diagnostics = {
            "matching_score": matching,
            "agreement": 0.0,
            "tgi_entropy": 0.0,
            "igt_entropy": 0.0,
            "tgi_margin": 0.0,
            "igt_margin": 0.0,
        }
        return build_state_features(ctx, diagnostics, incumbent, protocol.name)
