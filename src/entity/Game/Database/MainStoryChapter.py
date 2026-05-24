"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class MainStoryChapter:
    mainStoryPartId: str = None
    id: str = None
    title: str = None
    description: str = None
    storyAssetId: str = None
    mainStoryGroupId: str = None
    order: int = None
    localization: MainStoryChapterLocalization = None

@dataclass(slots=True)
class MainStoryChapterLocalization:
    mainStoryPartId: str = None
    id: str = None
    title: str = None
    description: str = None
