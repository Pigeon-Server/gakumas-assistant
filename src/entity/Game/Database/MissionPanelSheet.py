"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class MissionPanelSheet:
    missionPanelSheetGroupId: str = None
    number: int = None
    name: str = None
    missionGroupId: str = None
    iconAssetId: str = None
    backgroundAssetId: str = None
    rewardAssetId: str = None
    backgroundGradientColor1: str = None
    backgroundGradientColor2: str = None
    panelGradientColors1: List[str] = field(default_factory=list)
    panelGradientColors2: List[str] = field(default_factory=list)
    localization: MissionPanelSheetLocalization = None

@dataclass(slots=True)
class MissionPanelSheetLocalization:
    missionPanelSheetGroupId: str = None
    number: int = None
    name: str = None
