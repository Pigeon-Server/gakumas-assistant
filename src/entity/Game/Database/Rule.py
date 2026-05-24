"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class Rule:
    type: str = None
    platformType: str = None
    number: int = None
    html: str = None
    localization: RuleLocalization = None

@dataclass(slots=True)
class RuleLocalization:
    type: str = None
    platformType: str = None
    number: int = None
    html: str = None
