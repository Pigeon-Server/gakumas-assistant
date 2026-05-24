"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceCardTag:
    id: str = None
    name: str = None
    localization: ProduceCardTagLocalization = None

@dataclass(slots=True)
class ProduceCardTagLocalization:
    id: str = None
    name: str = None
