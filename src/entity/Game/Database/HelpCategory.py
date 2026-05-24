"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class HelpCategory:
    id: str = None
    name: str = None
    assetIds: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    hiddenHelpList: bool = None
    order: int = None
    localization: HelpCategoryLocalization = None

@dataclass(slots=True)
class HelpCategoryLocalization:
    id: str = None
    order: int = None
    name: str = None
    texts: List[str] = field(default_factory=list)
