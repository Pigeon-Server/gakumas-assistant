"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class PhotoFacialMotionGroup:
    id: str = None
    number: int = None
    facialAssetId: str = None
    name: str = None
    disableAutoBlink: bool = None
    localization: PhotoFacialMotionGroupLocalization = None

@dataclass(slots=True)
class PhotoFacialMotionGroupLocalization:
    id: str = None
    number: int = None
    name: str = None
