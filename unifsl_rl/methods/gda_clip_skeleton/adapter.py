"""Minimal second-method skeleton to prove unifsl core is method-agnostic."""


class GDAClipSkeletonAdapter:
    def build_cache(self, cfg):
        return {}

    def build_context(self, cache, protocol, split="val"):
        return {"protocol": protocol}
