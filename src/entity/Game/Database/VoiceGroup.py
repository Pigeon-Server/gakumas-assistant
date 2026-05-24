"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class VoiceGroup:
    id: str = None
    voiceAssetId: str = None
    title: str = None
    facialAssetId: str = None
    bodyAssetId: str = None
    order: int = None
    localization: VoiceGroupLocalization = None

@dataclass(slots=True)
class VoiceGroupLocalization:
    id: str = None
    voiceAssetId: str = None
    title: str = None
