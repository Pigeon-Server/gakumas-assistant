"""培育系统相关的枚举常量定义。

所有游戏数据库中的枚举值统一在此定义，禁止在业务代码中使用字符串常量。
"""

from enum import Enum


class ProducePlanType(str, Enum):
    """培育计划类型枚举（对应IdolCard.planType）"""

    PLAN1 = "ProducePlanType_Plan1"  # 感性（センス）
    PLAN2 = "ProducePlanType_Plan2"  # 理性（ロジック）
    PLAN3 = "ProducePlanType_Plan3"  # 非凡（アノマリー）

    @property
    def label(self) -> str:
        """返回计划标签"""
        return {
            self.PLAN1: "sense",
            self.PLAN2: "logic",
            self.PLAN3: "extraordinary",
        }.get(self, "unknown")

    @property
    def display_name(self) -> str:
        """返回显示名称"""
        return {
            self.PLAN1: "センス",
            self.PLAN2: "ロジック",
            self.PLAN3: "アノマリー",
        }.get(self, "")


class ProduceCardCategory(str, Enum):
    """技能卡类型枚举（对应ProduceCard.category）"""

    ACTIVE_SKILL = "ProduceCardCategory_ActiveSkill"
    MENTAL_SKILL = "ProduceCardCategory_MentalSkill"
    TROUBLE = "ProduceCardCategory_Trouble"
    UNKNOWN = "ProduceCardCategory_Unknown"


class ProduceCardRarity(str, Enum):
    """技能卡稀有度枚举"""

    N = "N"
    R = "R"
    SR = "SR"
    SSR = "SSR"


class ProduceItemRarity(str, Enum):
    """P物品稀有度枚举"""

    N = "N"
    R = "R"
    SR = "SR"
    SSR = "SSR"


class ProduceDrinkRarity(str, Enum):
    """P饮料稀有度枚举"""

    N = "N"
    R = "R"
    SR = "SR"
    SSR = "SSR"


class AttributeType(str, Enum):
    """属性类型枚举"""

    VOCAL = "vocal"
    DANCE = "dance"
    VISUAL = "visual"


class ExamEffectType(str, Enum):
    """考试效果类型枚举（对应EffectGroup.examEffectType）"""

    # 即时输出类
    SCORE = "ProduceExamEffectType_Score"  # 得分
    VOCAL = "ProduceExamEffectType_Vocal"  # Vo参数
    DANCE = "ProduceExamEffectType_Dance"  # Da参数
    VISUAL = "ProduceExamEffectType_Visual"  # Vi参数

    # 状态类
    PARAMETER_BUFF = "ProduceExamEffectType_ParameterBuff"  # 好調
    CONCENTRATION = "ProduceExamEffectType_Concentration"  # 集中
    REVIEW = "ProduceExamEffectType_Review"  # 好印象
    AGGRESSIVE = "ProduceExamEffectType_Aggressive"  # やる気
    BLOCK = "ProduceExamEffectType_Block"  # 元気
    ENTHUSIASTIC = "ProduceExamEffectType_Enthusiastic"  # 熱意
    FULL_POWER_POINT = "ProduceExamEffectType_FullPowerPoint"  # 全力値
    FULL_POWER = "ProduceExamEffectType_FullPower"  # 全力

    # 特殊效果
    PLAY_COUNT_UP = "ProduceExamEffectType_PlayCountUp"  # 使用回数追加
    PARAMETER_UP_INCREASE = "ProduceExamEffectType_ParameterUpIncrease"  # 参数上升量增加


__all__ = [
    "ProducePlanType",
    "ProduceCardCategory",
    "ProduceCardRarity",
    "ProduceItemRarity",
    "ProduceDrinkRarity",
    "AttributeType",
    "ExamEffectType",
]
