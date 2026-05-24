"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class HomeMotion:
    characterId: str = None
    locationType: str = None
    motionType: str = None
    number: int = None
    conditionSetId: str = None
    facialAssetIds: List[str] = field(default_factory=list)
    bodyAssetIds: List[str] = field(default_factory=list)
    voiceAssetId: str = None
    isPrioritized: bool = None
