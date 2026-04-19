from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.core.tasks.producer_challenge.catalog import resolve_produce_route


class GameplayPhase(str, Enum):
    """培育游戏中 YOLO 可识别的画面阶段。"""
    NONE = ""                     # 尚未进入游戏玩法
    STARTUP_MODALS = "startup_modals"  # 设置弹窗序列（语音/快进/跳过）
    SCHEDULE = "schedule"         # 周行程选择（PC:Action/Recommend + Progress）
    LESSON = "lesson"             # レッスン/試験 战斗中（Skill Card + Score/Remaining）
    SKILL_REWARD = "skill_reward" # 技能卡奖励选择（Skill Card + Button/Disable）
    DIALOGUE = "dialogue"         # 对话/コミュ事件（Universal Options / Fast Forward）
    P_DRINK = "p_drink"           # P饮料选择画面（P Drink 居中，非底栏图标）
    EXAM = "exam"                 # 試験/オーディション（与 lesson 共用手牌机制）
    CONSULT = "consult"           # 相談交换页（Card Item Exchange / 強化 / 削除）
    ITEM_SELECT = "item_select"   # Pアイテム選択画面（Special Item）
    MODAL = "modal"               # 弹窗（Modal Header）
    LIVE_PERFORMANCE = "live_performance"  # ライブ演出（横画面リズムゲーム）
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
    gameplay_phase: str = ""           # GameplayPhase value
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
    current_exam_type: str = ""        # "midterm" / "final" / "audition"
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

    # ── 拡張ハンドラ用汎用ストレージ ──
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
        """根据剧本返回实际使用的难度。"""
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
        """更新当前阶段并重置连续未知计数。"""
        previous_phase = self.gameplay_phase
        if phase != GameplayPhase.UNKNOWN:
            self.consecutive_unknowns = 0
            # Reset effect chain depth when entering a recognized phase
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
        """记录一条 gameplay 操作。"""
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
        """记录获取一张新的技能卡/饮料/物品。"""
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
        """记录削除/丢弃一张卡或饮料。"""
        if not card_id:
            return
        self.deck_mutations.append({
            "type": "remove",
            "card_id": card_id,
            "kind": kind,
        })

    def clear_schedule_pending(self) -> None:
        self.pending_schedule_index = None
        self.pending_schedule_label = ""
        self.handler_state.pop("pending_schedule_action_id", None)

    def clear_dialogue_pending(self) -> None:
        self.pending_dialogue_option_index = None

    def clear_lesson_pending(self) -> None:
        self.pending_lesson_card_index = None
        self.pending_lesson_card_label = ""
        self.handler_state.pop("pending_lesson_click_point", None)
        self.handler_state.pop("pending_lesson_action_id", None)
        self.handler_state.pop("pending_lesson_db_id", None)

    def clear_skill_reward_pending(self) -> None:
        self.pending_skill_reward_index = None
        self.pending_skill_reward_label = ""
        self.handler_state.pop("pending_skill_reward_db_id", None)

    def clear_p_drink_pending(self) -> None:
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
        """记录本周行程选择。"""
        self.last_schedule_action = action_name
        self.schedule_history.append(action_name)
        self.current_week += 1
        self.total_loops += 1
        self.clear_schedule_pending()

    def __repr__(self):
        return (
            f"ProduceContext(scenario={self.scenario!r}, difficulty={self.difficulty!r}, "
            f"produce_id={self.produce_id!r}, "
            f"phase={self.gameplay_phase!r}, position={self.gameplay_position!r}, "
            f"week={self.current_week}, "
            f"loops={self.total_loops}, "
            f"idol_card_id={self.target_idol_card_id!r}, "
            f"support_mode={self.support_card_mode!r}, memory_mode={self.memory_mode!r}, "
            f"use_rental={self.use_rental!r}, use_boost_items={self.use_boost_items!r})"
        )
