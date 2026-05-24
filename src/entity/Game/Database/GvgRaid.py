"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class GvgRaid:
    id: str = None
    name: str = None
    titleAssetId: str = None
    bannerAssetId: str = None
    storyGroupId: str = None
    examSettingId: str = None
    order: int = None
    localization: GvgRaidLocalization = None

@dataclass(slots=True)
class GvgRaidLocalization:
    id: str = None
    order: int = None
    name: str = None
