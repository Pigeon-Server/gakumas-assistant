"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class MeishiIllustrationAsset:
    id: str = None
    type: str = None
    isDefault: bool = None
    leftTopPositionX: int = None
    leftTopPositionY: int = None
    rightBottomPositionX: int = None
    rightBottomPositionY: int = None
    name: str = None
    weight: int = None
    height: int = None
    order: int = None
    localization: MeishiIllustrationAssetLocalization = None

@dataclass(slots=True)
class MeishiIllustrationAssetLocalization:
    id: str = None
    name: str = None
