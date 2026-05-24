"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class CoinGashaButton:
    id: str = None
    name: str = None
    description: str = None
    resourceType: str = None
    resourceId: str = None
    quantity: int = None
    maxDrawCount: int = None
    localization: CoinGashaButtonLocalization = None

@dataclass(slots=True)
class CoinGashaButtonLocalization:
    id: str = None
    name: str = None
    description: str = None
