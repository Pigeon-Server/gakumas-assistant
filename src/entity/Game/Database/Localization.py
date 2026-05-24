"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class Localization:
    id: str = None
    description: str = None
    localization: LocalizationLocalization = None

@dataclass(slots=True)
class LocalizationLocalization:
    id: str = None
    description: str = None
