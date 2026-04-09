from dataclasses import dataclass
from typing import Dict, List

from .action_spec import ActionSpec


@dataclass
class Slot:
    id: str
    owns: List[str]
    action_spec: ActionSpec

    def observe(self, ctx: Dict):
        raise NotImplementedError

    def apply(self, ctx: Dict, action):
        raise NotImplementedError

    def local_constraints(self, ctx: Dict):
        return []
