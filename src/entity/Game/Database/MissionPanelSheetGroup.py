"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class MissionPanelSheetGroup:
    id: str = None
    name: str = None
    bannerAssetId: str = None
    conditionSetId: str = None
    dearnessCharacterId: str = None
    localization: MissionPanelSheetGroupLocalization = None

@dataclass(slots=True)
class MissionPanelSheetGroupLocalization:
    id: str = None
    name: str = None
