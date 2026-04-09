from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CandidateConfig:
    alpha_idx: int
    alpha_value: float
    beta: int
    gamma_idx: int
    gamma_value: float
    subset_indices: List[int]
    source_tag: str
    mode_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult(CandidateConfig):
    selection_score: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    violation: float = 0.0
    repair_flag: bool = False
    raw_accuracy: Optional[float] = None


@dataclass
class VerificationResult:
    ranked: List[CandidateResult]
    protocol: str
