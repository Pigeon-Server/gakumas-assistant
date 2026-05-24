"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceExamGimmickEffectGroupProduceDescriptionsItem:
    produceDescriptionType: str = None
    examDescriptionType: str = None
    examEffectType: str = None
    produceCardGrowEffectType: str = None
    produceCardCategory: str = None
    produceCardMovePositionType: str = None
    produceStepType: str = None
    produceStepBusinessType: str = None
    text: str = None
    targetId: str = None
    targetLevel: int = None
    effectValue1: int = None
    effectValue2: int = None
    effectCount: int = None
    turn: int = None
    costValue: int = None
    produceDescriptionSwapId: str = None
    originProduceExamTriggerId: str = None
    originProduceExamEffectId: str = None
    originProduceCardStatusEnchantId: str = None
    isCost: bool = None
    isOnlyOutGame: bool = None
    changeColor: bool = None

@dataclass(slots=True)
class ProduceExamGimmickEffectGroup:
    id: str = None
    priority: int = None
    remainingTurnPermil: int = None
    startTurn: int = None
    remainingTurn: int = None
    fieldStatusType: str = None
    fieldStatusValue: int = None
    fieldStatusCheckType: str = None
    produceExamEffectId: str = None
    fieldStatusProduceCardSearchId: str = None
    isPositive: bool = None
    produceDescriptions: List[ProduceExamGimmickEffectGroupProduceDescriptionsItem] = field(default_factory=list)
    localization: ProduceExamGimmickEffectGroupLocalization = None

@dataclass(slots=True)
class ProduceExamGimmickEffectGroupLocalizationProduceDescriptionsItem:
    produceDescriptionType: str = None
    examDescriptionType: str = None
    examEffectType: str = None
    produceCardCategory: str = None
    produceCardMovePositionType: str = None
    produceStepType: str = None
    targetId: str = None
    text: str = None

@dataclass(slots=True)
class ProduceExamGimmickEffectGroupLocalization:
    id: str = None
    priority: int = None
    produceDescriptions: List[ProduceExamGimmickEffectGroupLocalizationProduceDescriptionsItem] = field(default_factory=list)
