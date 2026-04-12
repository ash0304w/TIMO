from abc import ABC, abstractmethod
from typing import Dict, List

from .action_spec import ActionSpec


class Slot(ABC):
    id: str
    stage: int = 0

    @property
    @abstractmethod
    def owns(self) -> List[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        raise NotImplementedError

    @abstractmethod
    def observe(self, ctx: Dict):
        raise NotImplementedError

    @abstractmethod
    def apply(self, ctx: Dict, action):
        raise NotImplementedError

    def local_constraints(self, ctx: Dict):
        return []
