from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.core.tasks.producer_challenge.catalog import resolve_produce_route


class GameplayPhase(str, Enum):
    """培育游戏中 YOLO 可识别的画面阶段。"""
    NONE = ""                     # 尚未进入游戏玩法
    STARTUP_MODALS = "startup_modals"  # 设置弹窗序列（语音/快进/跳过）
    SCHEDULE = "schedule"         # 周行程选择（PC:Action/Recommend + Progress）
    LESSON = "lesson"             # 课程/試験（Skill Card + Score/Remaining）
    SKILL_REWARD = "skill_reward" # 技能卡奖励选择（Skill Card + Button/Disable）
    DIALOGUE = "dialogue"         # 对话/交流事件（Universal Options / Fast Forward）
    P_DRINK = "p_drink"           # P飲料选择画面（P Drink 居中，非底栏图标）
    EXAM = "exam"                 # 試験/试镜（与 lesson 共用手牌机制）
    CONSULT = "consult"           # 相談交换页（Card Item Exchange / 強化 / 削除）
    ITEM_SELECT = "item_select"   # P道具选择画面（Special Item）。
    MODAL = "modal"               # 弹窗（Modal Header）
    LIVE_PERFORMANCE = "live_performance"  # Live演出画面（横屏）
    RESULT = "result"             # 培育结果/跳过画面（Skip Button）
    LOADING = "loading"           # 加载/过场（无可操作元素）
    UNKNOWN = "unknown"           # 无法判定


@dataclass
class GameplayOperation:
    """单次游戏内操作记录，用于断点续行与调试回放。"""

    action: str
    phase: str
    position: str
    target: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProduceContext:
    """
    培育流程上下文容器。

    在整个 Pipeline 执行期间由各 Step 共享，
    前面的 Step 写入选择结果，后续 Step 和挑战决策可以读取。
    """

    # ── 用户配置（从 Config 填入） ──
    scenario: str = "hajime"
    difficulty: str = "regular"
    target_idol_card_id: str = ""
    support_card_mode: str = "auto"
    support_card_preset_index: int = 1
    memory_mode: str = "auto"
    memory_preset_index: int = 1
    use_rental: bool = True
    use_boost_items: bool = False
    schedule_notebook_mode: str = "before_decision"  # P手帳读取策略（disabled / before_decision）
    resume_interrupted: bool = False      # 是否恢复上次中断的培育

    # ── 执行期间填充 ──
    resumed_from_interrupt: bool = False  # 实际恢复了中断培育（跳过编成步骤）
    resume_info: Dict[str, Any] = field(default_factory=dict)  # 恢复弹窗中提取的信息
    selected_idol_card: Optional[Any] = None
    support_cards: List[Any] = field(default_factory=list)
    memories: List[Any] = field(default_factory=list)
    memory_attributes: List[Dict[str, Any]] = field(default_factory=list)
    formation_details: Dict[str, Any] = field(default_factory=dict)
    produce_metadata: Dict[str, Any] = field(default_factory=dict)
    produce_route_error: str = ""
    has_rental_support: bool = False
    has_rental_memory: bool = False

    # ── 游戏玩法期间 ──
    gameplay_phase: str = ""           # GameplayPhase 枚举值
    gameplay_position: str = ""        # 更细粒度的 gameplay 位置
    last_stable_position: str = ""     # 最近一次稳定页面位置
    current_week: int = 0              # 当前周数
    total_loops: int = 0               # 已完成的周行程循环次数
    max_gameplay_loops: int = 800      # 防无限循环安全阈值（完整培育约需500-600循环）

    # ── 断点续行追踪 ──
    last_pipeline_step: str = ""       # 最后执行/完成的 step_name
    last_schedule_action: str = ""     # 上次周行程选择的行动名称
    schedule_history: List[str] = field(default_factory=list)   # 各周选择历史
    lesson_turns_played: int = 0       # 当前 lesson 已打出的回合数
    dialogue_choices_made: int = 0     # 已做出的对话选择次数
    consecutive_unknowns: int = 0      # 连续 unknown 计数（防卡死）
    pending_schedule_index: Optional[int] = None
    pending_schedule_label: str = ""
    pending_dialogue_option_index: Optional[int] = None
    pending_lesson_card_index: Optional[int] = None
    pending_lesson_card_label: str = ""
    pending_skill_reward_index: Optional[int] = None
    pending_skill_reward_label: str = ""
    pending_p_drink_index: Optional[int] = None
    pending_p_drink_label: str = ""
    operation_history: List[GameplayOperation] = field(default_factory=list)
    max_operation_history: int = 200

    # ── 試験 tracking ──
    current_exam_type: str = ""        # "midterm" / "final" / "audition"（考试类型）
    consult_remaining_p_points: int = 0

    # ── 决策快照 / 无状态桥接 ──
    hud_stamina: int = 0
    hud_max_stamina: int = 0
    hud_p_point: int = 0
    hud_target_score: int = 0
    recognized_hand_cards: List[Dict[str, Any]] = field(default_factory=list)
    recognized_p_drinks: List[Dict[str, Any]] = field(default_factory=list)
    recognized_produce_items: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_clip_entities: List[Dict[str, Any]] = field(default_factory=list)
    state_revision: int = 0
    last_sync_reason: str = ""
    economy_state: Dict[str, Any] = field(default_factory=dict)
    parameter_state: Dict[str, Any] = field(default_factory=dict)
    inventory_state: Dict[str, Any] = field(default_factory=dict)
    card_zone_state: Dict[str, Any] = field(default_factory=dict)
    observability_state: Dict[str, Any] = field(default_factory=dict)
    rl_inference_url: str = ""

    # ── 牌组/饮料变更追踪（相談・技能奖励等操作后实时更新） ──
    deck_mutations: List[Dict[str, Any]] = field(default_factory=list)

    # ── 扩展处理器用通用存储 ──
    handler_state: Dict[str, Any] = field(default_factory=dict)

    # ── 自动决策回调（可由外部策略注入） ──
    schedule_strategy: Optional[Callable] = field(default=None, repr=False)
    lesson_strategy: Optional[Callable] = field(default=None, repr=False)
    dialogue_strategy: Optional[Callable] = field(default=None, repr=False)
    skill_reward_strategy: Optional[Callable] = field(default=None, repr=False)
    p_drink_strategy: Optional[Callable] = field(default=None, repr=False)
    exam_strategy: Optional[Callable] = field(default=None, repr=False)
    consult_strategy: Optional[Callable] = field(default=None, repr=False)
    item_select_strategy: Optional[Callable] = field(default=None, repr=False)
    modal_strategy: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self):
        """在dataclass初始化后补充派生状态。"""
        try:
            self.produce_metadata = resolve_produce_route(
                self.scenario,
                self.effective_difficulty,
            ).to_context_dict()
        except ValueError as exc:
            self.produce_metadata = {}
            self.produce_route_error = str(exc)

    @property
    def effective_difficulty(self) -> str:
        """返回当前培育实际使用的难度标识。

        Returns:
            str: 当前上下文生效的难度名称。该值会在解析剧本路线、读取数据库映射
            和后续日志输出时使用。
        """
        return self.difficulty

    @property
    def produce_id(self) -> Optional[str]:
        return self.produce_metadata.get("produce_id")

    @property
    def produce_group_id(self) -> Optional[str]:
        return self.produce_metadata.get("produce_group_id")

    @property
    def parameter_growth_limit(self) -> int:
        return int(self.produce_metadata.get("parameter_growth_limit") or 0)

    # ── 阶段更新辅助 ──

    def set_phase(self, phase: str) -> None:
        """更新当前 gameplay 阶段，并在跨阶段时清理对应的 pending 状态。

        Args:
            phase: 新识别出的 gameplay 阶段，通常来自 `GameplayPhase` 枚举值。
                当传入 `unknown` 时会累加连续未知计数；传入明确阶段时会重置
                unknown 计数并清除效果连锁深度等只应在未知态中保留的临时状态。

        Notes:
            - 当阶段发生变化时，会根据旧阶段调用 `_clear_pending_state`，避免上一阶段
              的待点击索引、待确认标签误污染下一阶段。
            - 该方法只负责上下文状态维护，不执行任何识别或点击动作。
        """
        previous_phase = self.gameplay_phase
        if phase != GameplayPhase.UNKNOWN:
            self.consecutive_unknowns = 0
            # 进入可识别阶段时，重置效果连锁深度
            self.handler_state.pop("effect_chain_depth", None)
        else:
            self.consecutive_unknowns += 1
        self.gameplay_phase = phase
        if previous_phase and phase != previous_phase:
            self._clear_pending_state(previous_phase)

    def set_position(self, position: str) -> None:
        """更新更细粒度的 gameplay 位置。"""
        self.gameplay_position = position
        if position and not position.startswith("transition"):
            self.last_stable_position = position

    def record_operation(
        self,
        action: str,
        *,
        target: str = "",
        position: str | None = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """把一次 gameplay 操作写入历史记录，供断点续行和调试复盘使用。

        Args:
            action: 本次操作的动作名称，例如点击、确认、跳过或策略输出的动作类型。
            target: 被操作的目标名称，可选；通常写入按钮标题、卡牌名或候选标签。
            position: 发生操作时的细粒度页面位置；不传时默认使用当前
                `self.gameplay_position`。
            details: 额外的结构化调试信息，例如候选索引、OCR 文本、策略返回值。

        Notes:
            - 记录会带上当前 `gameplay_phase` 与 `gameplay_position` 快照，便于事后还原
              问题发生时的上下文。
            - 历史长度超过 `max_operation_history` 时会截断旧记录，防止长时间培育导致
              上下文对象无限增长。
        """
        self.operation_history.append(
            GameplayOperation(
                action=action,
                phase=str(self.gameplay_phase or ""),
                position=position or str(self.gameplay_position or ""),
                target=target,
                details=details or {},
            )
        )
        if len(self.operation_history) > self.max_operation_history:
            self.operation_history = self.operation_history[-self.max_operation_history:]

    # ── 牌组变更追踪 ──

    def mutate_deck_enhance(self, card_id: str) -> None:
        """记录一次技能卡強化（upgrade_count +1）。"""
        if not card_id:
            return
        # 查找是否已有对同一张卡的強化记录，累加 upgrade_count
        for m in self.deck_mutations:
            if m["type"] == "enhance" and m["card_id"] == card_id:
                m["upgrade_count"] = min(m.get("upgrade_count", 1) + 1, 3)
                return
        self.deck_mutations.append({
            "type": "enhance",
            "card_id": card_id,
            "upgrade_count": 1,
        })

    def mutate_deck_acquire(
        self,
        card_id: str,
        *,
        kind: str = "produce_card",
        name: str = "",
        source: str = "",
    ) -> None:
        """记录一次新增卡牌或道具的牌组变更。

        Args:
            card_id: 新获得实体的数据库主键。为空时直接忽略，避免写入无效变更。
            kind: 变更实体类型，默认是 `produce_card`，也可用于记录饮料、道具等。
            name: 供日志和调试使用的展示名称，可选。
            source: 产出来源说明，例如 skill_reward、consult、event 等，便于后续分析
                牌组变化来自哪个玩法环节。

        Notes:
            该记录不会立刻改写 `recognized_hand_cards`，而是追加到 `deck_mutations`，
            供后续无状态决策桥接统一消费。
        """
        if not card_id:
            return
        self.deck_mutations.append({
            "type": "acquire",
            "card_id": card_id,
            "kind": kind,
            "name": name,
            "source": source,
        })

    def mutate_deck_remove(self, card_id: str, *, kind: str = "produce_card") -> None:
        """记录一次从牌组中移除实体的变更。

        Args:
            card_id: 被移除实体的数据库主键。为空时直接忽略。
            kind: 被移除实体的类型，默认按技能卡处理，也可用于其他可移除资源。

        Notes:
            该记录通常由咨询页删除卡牌等操作触发，后续会与 acquire/enhance 一起由
            决策层合并，推导当前实际牌组状态。
        """
        if not card_id:
            return
        self.deck_mutations.append({
            "type": "remove",
            "card_id": card_id,
            "kind": kind,
        })

    def clear_schedule_pending(self) -> None:
        """清空行程选择阶段遗留的待执行状态。

        Notes:
            该方法会同时清除 `pending_schedule_*` 字段以及写在 `handler_state` 中的
            行动标识，通常在行程选择完成、阶段切换或恢复异常时调用，避免旧选择残留到下一周。
        """
        self.pending_schedule_index = None
        self.pending_schedule_label = ""
        self.handler_state.pop("pending_schedule_action_id", None)

    def clear_dialogue_pending(self) -> None:
        """清空对话阶段遗留的候选选项索引。"""
        self.pending_dialogue_option_index = None

    def clear_lesson_pending(self) -> None:
        """清空课程阶段遗留的待出牌信息和点击缓存。

        Notes:
            除了重置 `pending_lesson_*` 字段外，还会移除 `handler_state` 中保存的
            点击坐标、动作 ID、数据库 ID，避免下一回合沿用旧候选。
        """
        self.pending_lesson_card_index = None
        self.pending_lesson_card_label = ""
        self.handler_state.pop("pending_lesson_click_point", None)
        self.handler_state.pop("pending_lesson_action_id", None)
        self.handler_state.pop("pending_lesson_db_id", None)

    def clear_skill_reward_pending(self) -> None:
        """清空技能奖励阶段遗留的待选候选信息。"""
        self.pending_skill_reward_index = None
        self.pending_skill_reward_label = ""
        self.handler_state.pop("pending_skill_reward_db_id", None)

    def clear_p_drink_pending(self) -> None:
        """清空 P 饮料选择阶段遗留的待选索引与新增饮料标记。"""
        self.pending_p_drink_index = None
        self.pending_p_drink_label = ""
        self.handler_state.pop("pending_new_p_drink", None)

    def consume_recognized_drink(self, index: int) -> None:
        """课内使用饮料后，从已知库存中移除对应饮料。"""
        if 0 <= index < len(self.recognized_p_drinks):
            removed = self.recognized_p_drinks.pop(index)
            # 同步 inventory_state
            inv_drinks = self.inventory_state.get("p_drinks")
            if isinstance(inv_drinks, list) and 0 <= index < len(inv_drinks):
                inv_drinks.pop(index)

    def clear_consult_pending(self) -> None:
        """清空咨询阶段缓存在 handler_state 中的临时决策状态。

        Notes:
            这里会移除咨询页自动强化、交换重试、最近一次交换动作等中间态，
            但不会清除 `consult_total_op_count`，因为 CONSULT → MODAL → CONSULT 的
            中间过渡仍需要依赖累计操作次数判断流程是否异常。
        """
        self.handler_state.pop("consult_auto_used_enhancement", None)
        self.handler_state.pop("consult_detected_actions", None)
        self.handler_state.pop("consult_enhancement_target", None)
        self.handler_state.pop("consult_enhancement_target_label", None)
        self.handler_state.pop("consult_exchange_progressed", None)
        self.handler_state.pop("consult_exchange_retry_count", None)
        self.handler_state.pop("consult_last_exchange_action_id", None)
        self.handler_state.pop("consult_last_exchange_db_id", None)
        self.handler_state.pop("consult_last_exchange_p_points", None)
        self.handler_state.pop("consult_last_exchange_signature", None)
        self.handler_state.pop("consult_last_subaction", None)
        self.handler_state.pop("consult_pending_mode", None)
        self.handler_state.pop("consult_waiting_exchange_result", None)
        self.handler_state.pop("_consult_is_exchange_retry", None)
        # 注意: consult_total_op_count 不在此清除，
        # 因为 CONSULT→MODAL→CONSULT 过渡期间不应重置

    def _clear_pending_state(self, phase: str) -> None:
        """按阶段清理对应的 pending 状态。"""
        if phase == GameplayPhase.SCHEDULE:
            self.clear_schedule_pending()
        elif phase == GameplayPhase.DIALOGUE:
            self.clear_dialogue_pending()
        elif phase == GameplayPhase.LESSON:
            self.clear_lesson_pending()
        elif phase == GameplayPhase.SKILL_REWARD:
            self.clear_skill_reward_pending()
        elif phase == GameplayPhase.P_DRINK:
            self.clear_p_drink_pending()
        elif phase == GameplayPhase.CONSULT:
            self.clear_consult_pending()

    def record_schedule_choice(self, action_name: str) -> None:
        """记录一次周行程选择结果，并推进周计数。

        Args:
            action_name: 本周最终确认的行程名称，会写入 `last_schedule_action`
                和 `schedule_history`，供恢复中断与策略复盘使用。

        Notes:
            该方法会把 `current_week` 与 `total_loops` 同步加一，并清理 schedule
            阶段的 pending 状态，表示本周行程已经正式落地。
        """
        self.last_schedule_action = action_name
        self.schedule_history.append(action_name)
        self.current_week += 1
        self.total_loops += 1
        self.clear_schedule_pending()

    def __repr__(self):
        """返回上下文的调试字符串表示，便于日志快速查看关键状态。"""
        return (
            f"ProduceContext(scenario={self.scenario!r}, difficulty={self.difficulty!r}, "
            f"produce_id={self.produce_id!r}, "
            f"phase={self.gameplay_phase!r}, position={self.gameplay_position!r}, "
            f"week={self.current_week}, "
            f"loops={self.total_loops}, "
            f"idol_card_id={self.target_idol_card_id!r}, "
            f"support_mode={self.support_card_mode!r}, memory_mode={self.memory_mode!r}, "
            f"schedule_notebook_mode={self.schedule_notebook_mode!r}, "
            f"use_rental={self.use_rental!r}, use_boost_items={self.use_boost_items!r})"
        )
