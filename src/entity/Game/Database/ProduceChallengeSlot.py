"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceChallengeSlot:
    id: str = None
    produceId: str = None
    number: int = None
    produceItemChallengeGroupId: str = None
    unlockDescription: str = None
    localization: ProduceChallengeSlotLocalization = None

@dataclass(slots=True)
class ProduceChallengeSlotLocalization:
    id: str = None
    produceId: str = None
    number: int = None
    unlockDescription: str = None
