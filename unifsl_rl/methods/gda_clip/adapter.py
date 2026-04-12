import torch

from unifsl_rl.core.method_wrapper import MethodWrapper
from unifsl_rl.core.access_guard import GuardedMapping, ensure_probe_reward_safe
from models import GDA
from utils import loda_val_test_feature
from .slots import GDAAlphaSlot, GDADummyStage0Slot


class GDAClipAdapter(MethodWrapper):
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self._cache = None

    def build_cache(self, cfg=None):
        if self._cache is not None:
            return self._cache
        cfg = cfg or self.cfg
        clip_weights_all = torch.load(cfg["cache_dir"] + "/text_weights_cupl_t_all.pt", weights_only=False).float().to(self.device)
        clip_weights = clip_weights_all.mean(dim=1).t()
        clip_weights = clip_weights / clip_weights.norm(dim=0, keepdim=True)

        vecs = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float().to(self.device)
        labels = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float().to(self.device)
        val_features, val_labels = loda_val_test_feature(cfg, "val")
        val_features, val_labels = val_features.to(self.device).float(), val_labels.to(self.device)
        self._cache = {
            "support_vecs": vecs,
            "support_labels": labels,
            "clip_weights": clip_weights,
            "val_features": val_features,
            "val_labels": val_labels,
            "prompt_num": 1,
            "state_stats": {"alpha_stats": [1.0, 0.0]},
            "meta": {"dataset": cfg["dataset"]},
        }
        return self._cache

    def build_protocol_view(self, protocol, cache):
        return GuardedMapping(dict(cache, protocol=protocol), protocol=protocol, stage="ctx")

    def get_slots(self):
        return [GDADummyStage0Slot(), GDAAlphaSlot()]

    def materialize(self, prefix_action, ctx):
        return dict(ctx)

    def run_with_action(self, ctx, action):
        alpha = float(action.get("alpha", 1.0))
        _, W, b, _ = GDA(ctx["support_vecs"], ctx["support_labels"], ctx["clip_weights"], ctx["val_features"], ctx["val_labels"], alpha_shift=False)
        logits = 100.0 * ctx["val_features"].float() @ ctx["clip_weights"].float() + alpha * (ctx["val_features"].float() @ W + b)
        preds = logits.argmax(dim=-1)
        acc = (preds == ctx["val_labels"]).float().mean().item()
        return {"acc": acc, "diagnostics": {"alpha": alpha}}

    def evaluate(self, ctx, run_output, protocol):
        ensure_probe_reward_safe(protocol, protocol.selection_split)
        return {"base_acc": float(run_output["acc"]), "align_gain": 0.0, "anom_risk": 0.0}

    def estimate_cost(self, action):
        return 1.0
