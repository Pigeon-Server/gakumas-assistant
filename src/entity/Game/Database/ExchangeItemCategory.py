"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ExchangeItemCategory:
    exchangeId: str = None
    number: int = None
    name: str = None
    categoryType: str = None
    resourceType: str = None
    itemType: str = None
    localization: ExchangeItemCategoryLocalization = None

@dataclass(slots=True)
class ExchangeItemCategoryLocalization:
    exchangeId: str = None
    number: int = None
    categoryType: str = None
    resourceType: str = None
    itemType: str = None
    name: str = None
