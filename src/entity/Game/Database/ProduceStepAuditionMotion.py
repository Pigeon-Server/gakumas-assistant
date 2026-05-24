"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceStepAuditionMotion:
    characterId: str = None
    stepType: str = None
    motionType: str = None
    number: int = None
    facialAssetId: str = None
    bodyAssetId: str = None
    voiceAssetId: str = None
    sceneLayoutId: str = None
    cameraId: str = None
    produceGroupIds: List[str] = field(default_factory=list)
    auditionType: str = None
