"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class MeishiBaseAsset:
    id: str = None
    name: str = None
    isDefault: bool = None
    meishiBaseAssetType: str = None
    order: int = None
    localization: MeishiBaseAssetLocalization = None

@dataclass(slots=True)
class MeishiBaseAssetLocalization:
    id: str = None
    name: str = None
