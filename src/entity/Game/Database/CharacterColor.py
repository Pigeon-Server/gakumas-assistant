"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class CharacterColor:
    characterId: str = None
    mainColor: str = None
    gradientColor1: str = None
    gradientColor2: str = None
    textColor: str = None
    labelTextColor: str = None
    transitionGradientColor1: str = None
    transitionGradientColor2: str = None
