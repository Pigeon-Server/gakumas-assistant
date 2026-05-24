"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class IdolCardPieceExchangeReward:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class IdolCardPiece:
    idolCardId: str = None
    itemId: str = None
    releaseConsumptionQuantity: int = None
    exchangeReward: IdolCardPieceExchangeReward = None
