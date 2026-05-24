"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceStepEventSuggestionProduceDescriptionsItem:
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
class ProduceStepEventSuggestion:
    id: str = None
    producePoint: int = None
    stamina: int = None
    produceCardId: str = None
    produceCardUpgradeCount: int = None
    produceEffectIds: List[str] = field(default_factory=list)
    stepType: str = None
    stepId: str = None
    successProbabilityPermyriad: int = None
    successProduceEffectIds: List[str] = field(default_factory=list)
    successStepType: str = None
    successStepId: str = None
    failProduceEffectIds: List[str] = field(default_factory=list)
    failStepType: str = None
    failStepId: str = None
    alwaysSuccessful: bool = None
    produceEffectFireStep: int = None
    isCampaign: bool = None
    produceDescriptions: List[ProduceStepEventSuggestionProduceDescriptionsItem] = field(default_factory=list)
    localization: ProduceStepEventSuggestionLocalization = None

@dataclass(slots=True)
class ProduceStepEventSuggestionLocalizationProduceDescriptionsItem:
    produceDescriptionType: str = None
    examDescriptionType: str = None
    examEffectType: str = None
    produceCardCategory: str = None
    produceCardMovePositionType: str = None
    produceStepType: str = None
    targetId: str = None
    text: str = None

@dataclass(slots=True)
class ProduceStepEventSuggestionLocalization:
    id: str = None
    produceDescriptions: List[ProduceStepEventSuggestionLocalizationProduceDescriptionsItem] = field(default_factory=list)
