"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class MemoryTag:
    id: str = None
    defaultName: str = None
    assetId: str = None
    order: int = None
    localization: MemoryTagLocalization = None

@dataclass(slots=True)
class MemoryTagLocalization:
    id: str = None
    defaultName: str = None
