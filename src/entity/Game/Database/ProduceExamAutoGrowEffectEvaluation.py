"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceExamAutoGrowEffectEvaluation:
    type: str = None
    examEffectType: str = None
    remainingTerm: int = None
    growEffectType: str = None
    evaluation: int = None
    examStatusEnchantCoefficientPermil: int = None
