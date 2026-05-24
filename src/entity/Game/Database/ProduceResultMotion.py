"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceResultMotion:
    characterId: str = None
    liveType: str = None
    number: int = None
    motionAssetId: str = None
    facialAssetId: str = None
    voiceAssetId: str = None
    produceGroupIds: List[str] = field(default_factory=list)
