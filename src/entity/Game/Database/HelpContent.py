"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class HelpContent:
    helpCategoryId: str = None
    id: str = None
    name: str = None
    order: int = None
    detailUrl: str = None
    localization: HelpContentLocalization = None

@dataclass(slots=True)
class HelpContentLocalization:
    helpCategoryId: str = None
    id: str = None
    order: int = None
    name: str = None
