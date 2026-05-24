"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceAdv:
    produceType: str = None
    type: str = None
    title: str = None
    assetId: str = None
    localization: ProduceAdvLocalization = None

@dataclass(slots=True)
class ProduceAdvLocalization:
    produceType: str = None
    type: str = None
    title: str = None
