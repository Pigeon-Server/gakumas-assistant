"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class EventLabel:
    eventType: str = None
    name: str = None
    color: str = None
    localization: EventLabelLocalization = None

@dataclass(slots=True)
class EventLabelLocalization:
    eventType: str = None
    name: str = None
