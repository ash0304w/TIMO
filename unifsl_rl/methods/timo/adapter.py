import torch

from unifsl_rl.core.method_wrapper import MethodWrapper
from utils import cls_acc, loda_val_test_feature, load_few_shot_feature
from .episode_bank import EpisodeBank
from .ops import build_image_weights_from_cache, build_state_stats, run_timo_config, support_loo_score
from .slots import AlphaFusionSlot, BetaPromptCountSlot, GammaSharpnessSlot, PromptSubsetSlot


class TIMOAdapter(MethodWrapper):
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device

    def build_cache(self, cfg=None):
        cfg = cfg or self.cfg
        clip_weights_all = torch.load(cfg["cache_dir"] + "/text_weights_cupl_t_all.pt", weights_only=False).float().to(self.device)
        cache_keys, cache_values = load_few_shot_feature(cfg)
        cache_keys = cache_keys.to(self.device)
        cache_values = cache_values.to(self.device)
        support_vecs = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float().to(self.device)
        support_labels = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float().to(self.device)
        val_features, val_labels = loda_val_test_feature(cfg, "val")
        val_features, val_labels = val_features.to(self.device), val_labels.to(self.device)
        if cfg["dataset"] == "imagenet":
            test_features, test_labels = val_features, val_labels
        else:
            test_features, test_labels = loda_val_test_feature(cfg, "test")
            test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)

        image_weights = build_image_weights_from_cache(cache_keys, cache_values)
        cate_num, prompt_num, _ = clip_weights_all.shape
        bank = EpisodeBank(support_vecs, support_labels, val_features, val_labels, test_features, test_labels)
        return {
            "clip_weights_all": clip_weights_all,
            "cache_keys": cache_keys,
            "cache_values": cache_values,
            "support_vecs": support_vecs,
            "support_labels": support_labels,
            "val_features": val_features,
            "val_labels": val_labels,
            "test_features": test_features,
            "test_labels": test_labels,
            "image_weights": image_weights,
            "cate_num": cate_num,
            "prompt_num": prompt_num,
            "episode_bank": bank,
        }

    def build_context(self, cache, protocol, split="val"):
        eval_features, eval_labels = cache["episode_bank"].sample(split=split)
        ctx = dict(cache)
        ctx.update(
            {
                "eval_features": eval_features,
                "eval_labels": eval_labels,
                "protocol": protocol,
                "shots": self.cfg["shots"],
                "device": self.device,
                "remaining_budget": 0,
            }
        )
        # bootstrap matching for state stats
        from .ops import run_igt

        _, matching_score = run_igt(self.cfg, cache["clip_weights_all"], cache["image_weights"], gamma=50, return_matching=True)
        ctx["matching_score"] = matching_score
        return ctx

    def run_with_config(self, ctx, config):
        return run_timo_config(
            self.cfg,
            ctx,
            alpha=config["alpha"],
            beta=config["beta"],
            subset=config.get("subset_scores"),
            gamma=config["gamma"],
            mode="rl",
        )

    def evaluate(self, ctx, run_output, protocol):
        logits = run_output["logits_final"]
        labels = ctx["eval_labels"]
        base_acc = cls_acc(logits, labels) / 100.0 if protocol.can_use_labels_for_reward else 0.0
        d = run_output["diagnostics"]
        align = d.get("agreement", 0.0)
        anom = max(0.0, d.get("tgi_entropy", 0.0) - d.get("igt_entropy", 0.0))
        return {"base_acc": base_acc, "align_gain": align, "anom_risk": anom}

    def evaluate_action(self, ctx, actions):
        return self.run_with_config(ctx, actions)

    def evaluate_support_loo(self, ctx, actions):
        return support_loo_score(self.cfg, ctx, actions["alpha"], actions["beta"], actions.get("subset_scores"), actions["gamma"])

    def run_final_inference(self, cache, actions):
        from unifsl_rl.core.protocol import EVAL_PROTOCOL

        ctx = self.build_context(cache, EVAL_PROTOCOL, split="test")
        out = self.run_with_config(ctx, actions)
        acc = cls_acc(out["logits_final"], ctx["eval_labels"])
        return acc, out

    def get_slots(self, cache=None):
        if cache is None:
            cache = self.build_cache()
        p = int(cache["prompt_num"])
        return [GammaSharpnessSlot(), BetaPromptCountSlot(p), PromptSubsetSlot(p, per_class=False), AlphaFusionSlot()]

    def estimate_cost(self, config):
        return float(config.get("beta", 1))
