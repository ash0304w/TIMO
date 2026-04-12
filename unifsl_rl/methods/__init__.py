from .timo.adapter import TIMOAdapter
from .gda_clip.adapter import GDAClipAdapter


_METHOD_REGISTRY = {}


def register_method(name, adapter_cls):
    _METHOD_REGISTRY[name] = adapter_cls


def build_adapter(method_name, cfg, device):
    if method_name not in _METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{method_name}', available={list(_METHOD_REGISTRY.keys())}")
    return _METHOD_REGISTRY[method_name](cfg, device=device)


register_method("timo", TIMOAdapter)
register_method("gda_clip", GDAClipAdapter)
