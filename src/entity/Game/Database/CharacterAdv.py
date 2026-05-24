"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class CharacterAdv:
    characterId: str = None
    name: str = None
    regexp: str = None
    notIdol: bool = None
    localization: CharacterAdvLocalization = None

@dataclass(slots=True)
class CharacterAdvLocalization:
    characterId: str = None
    name: str = None
    regexp: str = None
