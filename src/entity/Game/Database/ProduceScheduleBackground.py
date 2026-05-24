"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceScheduleBackground:
    locationType: str = None
    backgroundAssetId: str = None
    sceneLayoutId: str = None
    monitorMovieId: str = None
    produceGroupIds: List[str] = field(default_factory=list)
