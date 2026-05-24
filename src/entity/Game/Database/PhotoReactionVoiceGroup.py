"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class PhotoReactionVoiceGroup:
    id: str = None
    number: int = None
    poseVoiceAssetId: str = None
    poseVoiceDelayMilliseconds: int = None
    reactionVoiceAssetId: str = None
    reactionVoiceDelayMilliseconds: int = None
