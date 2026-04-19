from src.core.tasks.producer_challenge.gameplay.handler_base import (
    GameplayDispatcher,
    GameplayHandler,
    HandlerResult,
    ResultHandler,
    AdvanceHandler,
)
from src.core.tasks.producer_challenge.gameplay.lesson import (
    LessonCardCandidate,
    LessonHandler,
    LessonStepResult,
    collect_lesson_card_candidates,
    decide_lesson_card,
    execute_lesson_step,
)
from src.core.tasks.producer_challenge.gameplay.schedule import (
    ScheduleActionCandidate,
    ScheduleHandler,
    ScheduleStepResult,
    collect_schedule_action_candidates,
    decide_schedule_action,
    execute_schedule_step,
)
from src.core.tasks.producer_challenge.gameplay.dialogue import (
    DialogueHandler,
    DialogueOptionCandidate,
    DialogueStepResult,
    collect_dialogue_option_candidates,
    decide_dialogue_option,
    execute_dialogue_step,
)
from src.core.tasks.producer_challenge.gameplay.p_drink import (
    PDrinkCandidate,
    PDrinkHandler,
    PDrinkStepResult,
    collect_p_drink_candidates,
    decide_p_drink,
    execute_p_drink_step,
)
from src.core.tasks.producer_challenge.gameplay.skill_reward import (
    SkillRewardCandidate,
    SkillRewardHandler,
    SkillRewardStepResult,
    collect_skill_reward_candidates,
    decide_skill_reward,
    execute_skill_reward_step,
)
from src.core.tasks.producer_challenge.gameplay.exam import ExamHandler
from src.core.tasks.producer_challenge.gameplay.consult import ConsultHandler
from src.core.tasks.producer_challenge.gameplay.modal import ModalHandler
from src.core.tasks.producer_challenge.gameplay.effect_chain import EffectChainHandler
from src.core.tasks.producer_challenge.gameplay.live_performance import LivePerformanceHandler
from src.core.tasks.producer_challenge.gameplay.item_select import ItemSelectHandler
from src.core.tasks.producer_challenge.gameplay.rl_strategy import RLStrategy, inject_rl_strategy


def build_default_dispatcher() -> GameplayDispatcher:
    """构建默认的 gameplay handler 调度器。

    注册所有内置 handler，按 priority 降序排列。
    扩展新剧本时只需追加 register() 即可，无需改动已有逻辑。

    使用方法::

        dispatcher = build_default_dispatcher()
        # 追加 NIA 专属 handler
        dispatcher.register(MyNiaAuditionHandler())
        # 替换默认 handler
        dispatcher.unregister(ScheduleHandler)
        dispatcher.register(MyCustomScheduleHandler())
    """
    dispatcher = GameplayDispatcher()
    # 高优先级：结果画面退出、弹窗覆盖
    dispatcher.register(ResultHandler())
    dispatcher.register(ModalHandler())
    # ライブ演出（横画面リズムゲーム）
    dispatcher.register(LivePerformanceHandler())
    # 普通 gameplay 阶段
    dispatcher.register(ExamHandler())
    dispatcher.register(ScheduleHandler())
    dispatcher.register(LessonHandler())
    dispatcher.register(DialogueHandler())
    dispatcher.register(PDrinkHandler())
    dispatcher.register(SkillRewardHandler())
    dispatcher.register(ConsultHandler())
    dispatcher.register(ItemSelectHandler())
    # 低优先级：过场效果链、兜底推进
    dispatcher.register(EffectChainHandler())
    dispatcher.register(AdvanceHandler())
    return dispatcher


__all__ = [
    # 基础设施
    "GameplayDispatcher",
    "GameplayHandler",
    "HandlerResult",
    "ResultHandler",
    "AdvanceHandler",
    "build_default_dispatcher",
    # schedule
    "ScheduleActionCandidate",
    "ScheduleHandler",
    "ScheduleStepResult",
    "collect_schedule_action_candidates",
    "decide_schedule_action",
    "execute_schedule_step",
    # lesson
    "LessonCardCandidate",
    "LessonHandler",
    "LessonStepResult",
    "collect_lesson_card_candidates",
    "decide_lesson_card",
    "execute_lesson_step",
    # dialogue
    "DialogueHandler",
    "DialogueOptionCandidate",
    "DialogueStepResult",
    "collect_dialogue_option_candidates",
    "decide_dialogue_option",
    "execute_dialogue_step",
    # p_drink
    "PDrinkCandidate",
    "PDrinkHandler",
    "PDrinkStepResult",
    "collect_p_drink_candidates",
    "decide_p_drink",
    "execute_p_drink_step",
    # skill_reward
    "SkillRewardCandidate",
    "SkillRewardHandler",
    "SkillRewardStepResult",
    "collect_skill_reward_candidates",
    "decide_skill_reward",
    "execute_skill_reward_step",
    # exam
    "ExamHandler",
    # consult
    "ConsultHandler",
    # modal
    "ModalHandler",
    # effect_chain
    "EffectChainHandler",
    # live_performance
    "LivePerformanceHandler",
    # item_select
    "ItemSelectHandler",
    # rl
    "RLStrategy",
    "inject_rl_strategy",
]
