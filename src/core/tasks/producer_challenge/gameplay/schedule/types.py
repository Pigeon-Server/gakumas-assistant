from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScheduleActionCandidate:
    index: int
    title: str
    kind: str
    recommended: bool
    selected: bool
    box: Any = field(repr=False, default=None)
    action_id: str = ""
    db_id: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleStepResult:
    status: str
    candidate: ScheduleActionCandidate
