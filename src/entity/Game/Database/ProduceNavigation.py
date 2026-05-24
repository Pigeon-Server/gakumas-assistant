"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceNavigation:
    id: str = None
    number: int = None
    description: str = None
    localization: ProduceNavigationLocalization = None

@dataclass(slots=True)
class ProduceNavigationLocalization:
    id: str = None
    number: int = None
    description: str = None
