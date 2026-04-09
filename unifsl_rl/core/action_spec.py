from dataclasses import dataclass
from typing import Dict, List

import torch


class ActionSpec:
    requires_projection: bool = False

    def project(self, action: torch.Tensor):
        return action


@dataclass
class ContinuousBox(ActionSpec):
    low: float
    high: float
    shape: List[int]
    requires_projection: bool = True

    def project(self, action: torch.Tensor):
        return torch.clamp(action, self.low, self.high)


@dataclass
class Discrete(ActionSpec):
    values: List[float]

    @property
    def n(self):
        return len(self.values)

    def index_to_value(self, idx: torch.Tensor):
        idx = idx.clamp(0, self.n - 1).long()
        return torch.tensor(self.values, device=idx.device)[idx]


@dataclass
class SubsetK(ActionSpec):
    n_items: int
    per_class: bool = False


@dataclass
class Simplex(ActionSpec):
    dim: int
    requires_projection: bool = True

    def project(self, action: torch.Tensor):
        x = torch.clamp(action, min=1e-12)
        z = x / x.sum(dim=-1, keepdim=True)
        return z


@dataclass
class CompositeActionSpec(ActionSpec):
    specs: Dict[str, ActionSpec]

    def keys(self):
        return list(self.specs.keys())

    def project(self, action_dict: Dict[str, torch.Tensor]):
        out = {}
        for k, spec in self.specs.items():
            v = action_dict[k]
            out[k] = spec.project(v) if getattr(spec, "requires_projection", False) else v
        return out
