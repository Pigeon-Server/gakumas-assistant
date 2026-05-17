from enum import Enum


class GameplayPhase(str, Enum):
    """培育 gameplay 的一级阶段常量。

    这些值用于：
    1. `ui.py` 的画面分类返回值
    2. handler dispatcher 的 phase 分流
    3. `ProduceContext.gameplay_phase` 的持久化状态

    命名保持和现有运行时字符串一致，避免影响已有存档、测试和调试日志。
    """

    NONE = ""  # 尚未进入 gameplay
    STARTUP_MODALS = "startup_modals"  # 开局设置弹窗序列
    SCHEDULE = "schedule"  # 周行动选择
    LESSON = "lesson"  # レッスン战斗
    SKILL_REWARD = "skill_reward"  # 技能卡奖励选择
    DIALOGUE = "dialogue"  # 对话 / コミュ
    P_DRINK = "p_drink"  # P 饮料选择
    EXAM = "exam"  # 試験 / オーディション战斗
    CONSULT = "consult"  # 相談
    ITEM_SELECT = "item_select"  # Pアイテム選択
    MODAL = "modal"  # 通用弹窗
    LIVE_PERFORMANCE = "live_performance"  # ライブ演出（横画面リズムゲーム）
    RESULT = "result"  # 结果/结算链路
    LOADING = "loading"  # 加载/过场
    UNKNOWN = "unknown"  # 无法判定


class GameplayPosition:
    """培育 gameplay 的二级页面位置常量。

    phase 负责“这是什么大类画面”，position 负责“当前在该类画面的哪一步”。
    命名规则统一为 `<phase>_<substate>`，便于：
    - handler 内做两步确认/重试
    - 测试里直接断言
    - 外部无状态策略按 position 分支
    """

    UNKNOWN = "unknown"
    FINAL_CONFIRM = "final_confirm"

    STARTUP_MODAL_VOICE = "startup_modal_voice"
    STARTUP_MODAL_FAST_FORWARD = "startup_modal_fast_forward"
    STARTUP_MODAL_SKIP_SETTINGS = "startup_modal_skip_settings"
    GAMEPLAY_MODAL = "gameplay_modal"
    P_DRINK_DETAIL = "p_drink_detail"
    DETAIL_MODAL = "detail_modal"
    CONSULT_ENHANCEMENT_CONFIRM_MODAL = "consult_enhancement_confirm_modal"
    EXAM_END_TURN_CONFIRM_MODAL = "exam_end_turn_confirm_modal"
    EXAM_RETRY_CONFIRM_MODAL = "exam_retry_confirm_modal"
    FAST_FORWARD_CONFIRM_MODAL = "fast_forward_confirm_modal"
    MEMORY_REGEN_CONFIRM_MODAL = "memory_regen_confirm_modal"
    MEMORY_CONFIRM_MODAL = "memory_confirm_modal"

    SCHEDULE_SELECTED = "schedule_selected"
    SCHEDULE_RECOMMEND = "schedule_recommend"
    SCHEDULE_IDLE = "schedule_idle"
    SCHEDULE_PRESENT_SUPPORT = "schedule_present_support"
    SCHEDULE_PRESENT_SUPPORT_SHOWCASE = "schedule_present_support_showcase"
    SCHEDULE_EVENT_OPTIONS = "schedule_event_options"    # 行程事件对话选项（おでかけ等）
    SCHEDULE_EVENT_DIALOGUE = "schedule_event_dialogue"  # 行程事件对话文本推进
    SCHEDULE_LESSON_OPTIONS = "schedule_lesson_options"  # 授業課程選項（ボーカル/ダンス/ビジュアル）
    SCHEDULE_LESSON_SELECTED = "schedule_lesson_selected"  # 授業選項已選中（信息面板表示中）

    LESSON_SELECTED = "lesson_selected"
    LESSON_IDLE = "lesson_idle"
    LESSON_SUMMARY_SHOWCASE = "lesson_summary_showcase"

    DIALOGUE_OPTIONS = "dialogue_options"
    DIALOGUE_CONTINUE = "dialogue_continue"

    P_DRINK_SELECTED = "p_drink_selected"
    P_DRINK_IDLE = "p_drink_idle"
    P_DRINK_SHOWCASE = "p_drink_showcase"

    SKILL_REWARD_SELECTED = "skill_reward_selected"
    SKILL_REWARD_IDLE = "skill_reward_idle"
    SKILL_REWARD_SHOWCASE = "skill_reward_showcase"

    CONSULT_EXCHANGE = "consult_exchange"
    CONSULT_ENHANCEMENT_READY = "consult_enhancement_ready"
    CONSULT_ENHANCEMENT_PREVIEW = "consult_enhancement_preview"
    CONSULT_IDLE = "consult_idle"

    ITEM_SELECT_IDLE = "item_select_idle"
    ITEM_SELECT_SELECTED = "item_select_selected"

    EXAM_SELECTED = "exam_selected"
    EXAM_IDLE = "exam_idle"
    EXAM_PREP = "exam_prep"  # 考试准备页面（显示参数加成倍率，YOLO 无法检测）

    LIVE_TAP_TO_START = "live_tap_to_start"   # ライブ開始待ち
    LIVE_PERFORMING = "live_performing"         # ライブ演出中
    LIVE_FINISHED = "live_finished"             # ライブ終了（縦画面に戻る直前）

    RESULT = "result"
    RESULT_EXAM_FAILURE = "result_exam_failure"
    RESULT_EXAM_SUMMARY_SHOWCASE = "result_exam_summary_showcase"
    RESULT_EXAM_RANKING_SUMMARY = "result_exam_ranking_summary"
    RESULT_MEMORY_GENERATION = "result_memory_generation"
    RESULT_MEMORY_PAGE = "result_memory_page"
    RESULT_FINAL_EVALUATION = "result_final_evaluation"
    RESULT_REWARD_SUMMARY = "result_reward_summary"
    RESULT_ACHIEVEMENT_PROGRESS = "result_achievement_progress"
    RESULT_EVENT_REWARD_PROGRESS = "result_event_reward_progress"
    TRANSITION_EMPTY = "transition_empty"
    TRANSITION_HUD = "transition_hud"
    TRANSITION_RESUME_TITLE = "transition_resume_title"


class ProduceExamType:
    """培育阶段使用的考试类型状态常量。"""

    NONE = ""
    MIDTERM = "midterm"
    FINAL = "final"
    AUDITION = "audition"


CONSULT_POSITION_PREFIX = "consult"
CONSULT_ENHANCEMENT_POSITION_PREFIX = "consult_enhancement"


class MemoryPageState:
    """编成前记忆选择流程的页面状态常量。"""

    SELECTION = "selection"
    DETAIL = "detail"
    CANDIDATE_LIST = "candidate_list"
    FINAL_CONFIRM = GameplayPosition.FINAL_CONFIRM
    UNKNOWN = GameplayPosition.UNKNOWN


GAMEPLAY_MODAL_POSITIONS = frozenset({
    GameplayPosition.STARTUP_MODAL_VOICE,
    GameplayPosition.STARTUP_MODAL_FAST_FORWARD,
    GameplayPosition.STARTUP_MODAL_SKIP_SETTINGS,
    GameplayPosition.GAMEPLAY_MODAL,
    GameplayPosition.P_DRINK_DETAIL,
    GameplayPosition.DETAIL_MODAL,
    GameplayPosition.CONSULT_ENHANCEMENT_CONFIRM_MODAL,
    GameplayPosition.EXAM_END_TURN_CONFIRM_MODAL,
    GameplayPosition.EXAM_RETRY_CONFIRM_MODAL,
    GameplayPosition.FAST_FORWARD_CONFIRM_MODAL,
    GameplayPosition.MEMORY_REGEN_CONFIRM_MODAL,
    GameplayPosition.MEMORY_CONFIRM_MODAL,
})

P_DRINK_SELECTION_POSITIONS = frozenset({
    GameplayPosition.P_DRINK_IDLE,
    GameplayPosition.P_DRINK_SELECTED,
})

SKILL_REWARD_SELECTION_POSITIONS = frozenset({
    GameplayPosition.SKILL_REWARD_IDLE,
    GameplayPosition.SKILL_REWARD_SELECTED,
})

CONSULT_SELECTION_POSITIONS = frozenset({
    GameplayPosition.CONSULT_ENHANCEMENT_READY,
    GameplayPosition.CONSULT_ENHANCEMENT_PREVIEW,
})

TRANSITION_POSITIONS = frozenset({
    GameplayPosition.TRANSITION_EMPTY,
    GameplayPosition.TRANSITION_HUD,
})
