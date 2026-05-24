"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceExamBattleConfig:
    id: str = None
    turn: int = None
    vocal: int = None
    dance: int = None
    visual: int = None
    produceExamBattleScoreConfigId: str = None
    vocalExcellent: int = None
    danceExcellent: int = None
    visualExcellent: int = None
    vocalBad: int = None
    danceBad: int = None
    visualBad: int = None
