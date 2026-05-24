"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ShopProduct:
    id: str = None
    appStoreProductId: str = None
    googlePlayStoreProductId: str = None
    dmmGamesProductId: str = None
    jewel: int = None
    priceJpy: int = None
    recoverName: str = None
