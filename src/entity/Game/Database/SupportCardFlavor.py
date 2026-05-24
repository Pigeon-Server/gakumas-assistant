"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class SupportCardFlavor:
    supportCardId: str = None
    number: int = None
    characterIds: List[str] = field(default_factory=list)
    text: str = None
    voiceAssetId: str = None
    localization: SupportCardFlavorLocalization = None

@dataclass(slots=True)
class SupportCardFlavorLocalization:
    supportCardId: str = None
    number: int = None
    text: str = None
