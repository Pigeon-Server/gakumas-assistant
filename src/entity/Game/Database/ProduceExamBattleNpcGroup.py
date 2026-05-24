"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceExamBattleNpcGroup:
    id: str = None
    number: int = None
    characterId: str = None
    produceExamBattleNpcMobId: str = None
    scoreMin: int = None
    scoreMax: int = None
    vocalPermil: int = None
    dancePermil: int = None
    visualPermil: int = None
    opScorePermil: int = None
    midScorePermil: int = None
    edScorePermil: int = None
