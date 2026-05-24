"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class Tutorial:
    tutorialType: str = None
    idolCardId: str = None
    step: int = None
    subStep: int = None
    navigationType: str = None
    navigationPositionType: str = None
    title: str = None
    texts: List[str] = field(default_factory=list)
    assetIds: List[str] = field(default_factory=list)
    advAssetId: str = None
    tutorialProduceCommandType: str = None
    localization: TutorialLocalization = None

@dataclass(slots=True)
class TutorialLocalization:
    tutorialType: str = None
    step: int = None
    subStep: int = None
    texts: List[str] = field(default_factory=list)
