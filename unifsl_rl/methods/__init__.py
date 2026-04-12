import importlib


_METHOD_REGISTRY = {}


def register_method(name, adapter_path: str):
    """Register by import path to avoid eager imports and circular deps."""
    _METHOD_REGISTRY[name] = adapter_path


def _load_adapter_cls(adapter_path: str):
    module_path, cls_name = adapter_path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def build_adapter(method_name, cfg, device):
    if method_name not in _METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{method_name}', available={list(_METHOD_REGISTRY.keys())}")
    cls = _load_adapter_cls(_METHOD_REGISTRY[method_name])
    return cls(cfg, device=device)


register_method("timo", "unifsl_rl.methods.timo.adapter:TIMOAdapter")
register_method("gda_clip", "unifsl_rl.methods.gda_clip.adapter:GDAClipAdapter")
