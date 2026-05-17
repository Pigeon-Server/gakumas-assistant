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
from src.core.tasks.producer_challenge.gameplay.strategy.rl_strategy import (
    RLStrategy,
    ScheduleRLStrategy,
    BattleRLStrategy,
    OtherRLStrategy,
    inject_rl_strategy,
)
from src.core.tasks.producer_challenge.gameplay.strategy.algo_strategy import (
    AlgoStrategy,
    BattleAlgoStrategy,
    ScheduleAlgoStrategy,
    OtherAlgoStrategy,
    inject_algo_strategy,
)
from src.core.tasks.producer_challenge.gameplay.strategy.llm_strategy import (
    LLMStrategy,
    ScheduleLLMStrategy,
    BattleLLMStrategy,
    OtherLLMStrategy,
)


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
    # Live演出（横屏节奏玩法）。
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
    # 日程
    "ScheduleActionCandidate",
    "ScheduleHandler",
    "ScheduleStepResult",
    "collect_schedule_action_candidates",
    "decide_schedule_action",
    "execute_schedule_step",
    # 课程
    "LessonCardCandidate",
    "LessonHandler",
    "LessonStepResult",
    "collect_lesson_card_candidates",
    "decide_lesson_card",
    "execute_lesson_step",
    # 对话
    "DialogueHandler",
    "DialogueOptionCandidate",
    "DialogueStepResult",
    "collect_dialogue_option_candidates",
    "decide_dialogue_option",
    "execute_dialogue_step",
    # P饮料
    "PDrinkCandidate",
    "PDrinkHandler",
    "PDrinkStepResult",
    "collect_p_drink_candidates",
    "decide_p_drink",
    "execute_p_drink_step",
    # 技能奖励
    "SkillRewardCandidate",
    "SkillRewardHandler",
    "SkillRewardStepResult",
    "collect_skill_reward_candidates",
    "decide_skill_reward",
    "execute_skill_reward_step",
    # 考试
    "ExamHandler",
    # 咨询
    "ConsultHandler",
    # 弹窗
    "ModalHandler",
    # 效果链
    "EffectChainHandler",
    # Live演出
    "LivePerformanceHandler",
    # 道具选择
    "ItemSelectHandler",
    # LLM策略
    "LLMStrategy",
    "ScheduleLLMStrategy",
    "BattleLLMStrategy",
    "OtherLLMStrategy",
    # RL策略
    "RLStrategy",
    "ScheduleRLStrategy",
    "BattleRLStrategy",
    "OtherRLStrategy",
    # 算法决策策略
    "AlgoStrategy",
    "BattleAlgoStrategy",
    "ScheduleAlgoStrategy",
    "OtherAlgoStrategy",
    # 注入工厂
    "inject_rl_strategy",
    "inject_algo_strategy",
]
