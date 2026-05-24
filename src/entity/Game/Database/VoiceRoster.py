"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class VoiceRoster:
    characterId: str = None
    assetId: str = None
    title: str = None
    type: str = None
    conditionSetId: str = None
    produceGroupId: str = None
    order: int = None
    localization: VoiceRosterLocalization = None

@dataclass(slots=True)
class VoiceRosterLocalization:
    characterId: str = None
    assetId: str = None
    title: str = None
