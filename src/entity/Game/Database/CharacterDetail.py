"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class CharacterDetail:
    characterId: str = None
    type: str = None
    content: str = None
    order: int = None
    localization: CharacterDetailLocalization = None

@dataclass(slots=True)
class CharacterDetailLocalization:
    characterId: str = None
    type: str = None
    order: int = None
    content: str = None
