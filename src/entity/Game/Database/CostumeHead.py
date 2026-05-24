"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class CostumeHead:
    id: str = None
    characterId: str = None
    name: str = None
    hairAssetId: str = None
    faceAssetId: str = None
    description: str = None
    resourceOriginType: str = None
    targetId: str = None
    isTraining: bool = None
    noGashaAppeal: bool = None
    viewConditionSetId: str = None
    viewStartTime: str = None
    order: int = None
    localization: CostumeHeadLocalization = None

@dataclass(slots=True)
class CostumeHeadLocalization:
    id: str = None
    name: str = None
    description: str = None
