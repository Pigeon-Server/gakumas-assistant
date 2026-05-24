"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceHighScore:
    id: str = None
    name: str = None
    produceHighScoreEventType: str = None
    bannerAssetId: str = None
    order: int = None
    localization: ProduceHighScoreLocalization = None

@dataclass(slots=True)
class ProduceHighScoreLocalization:
    id: str = None
    name: str = None
