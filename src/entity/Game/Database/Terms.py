"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class Terms:
    type: str = None
    name: str = None
    url: str = None
    localization: TermsLocalization = None

@dataclass(slots=True)
class TermsLocalization:
    type: str = None
    name: str = None
