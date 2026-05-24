"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class IdolCardLevelLimitStatusUp:
    id: str = None
    rank: str = None
    effectTypes: List[str] = field(default_factory=list)
    effectValue: int = None
    produceVocal: int = None
    produceDance: int = None
    produceVisual: int = None
    isIllustrationChange: bool = None
