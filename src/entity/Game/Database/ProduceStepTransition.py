"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceStepTransition:
    characterId: str = None
    stepType: str = None
    stepPhaseType: str = None
    number: int = None
    costumeHeadId: str = None
    costumeId: str = None
    advAssetId: str = None
    voiceAssetId: str = None
    produceGroupId: str = None
