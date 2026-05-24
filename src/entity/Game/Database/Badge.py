"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class Badge:
    id: str = None
    characterId: str = None
    name: str = None
    description: str = None
    type: str = None
    targetId: str = None
    grade: str = None
    producerRankingId: str = None
    targetIdentityName: str = None
    targetContentName: str = None
    targetGradeName: str = None
    order: int = None
