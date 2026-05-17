"""RL 决策策略——通过无状态 HTTP 推理服务处理战斗阶段决策。

内部结构：
- ScheduleRLStrategy: 周行程决策（暂不支持，返回 None）
- BattleRLStrategy: 战斗决策（lesson/exam）
- OtherRLStrategy: 其他决策（暂不支持，返回 None）
- RLStrategy: 统一入口，内部持有3个子策略，按 phase 路由
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from src.core.services.rl_inference_client import RLInferenceClient
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


# ── 工具函数 ───────────────────────────────────────────────


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_turn_color(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "").replace("_", "")
    _TURN_COLOR_MAP = {
        "vocal": "vocal", "dance": "dance", "visual": "visual",
        "vocalturn": "vocal", "danceturn": "dance", "visualturn": "visual",
        "ボーカル": "vocal", "ダンス": "dance", "ビジュアル": "visual",
        "vocalcolor": "vocal", "dancecolor": "dance", "visualcolor": "visual",
    }
    return _TURN_COLOR_MAP.get(normalized, "")


def _derive_action_kind(payload: dict[str, Any]) -> str:
    action_id = str(payload.get("id") or payload.get("action_id") or "")
    if action_id.startswith("produce_card:") or action_id.startswith("produce_card_unknown"):
        return "card"
    if action_id.startswith("produce_drink:") or action_id.startswith("produce_drink_unknown"):
        return "drink"
    if action_id == "end_turn":
        return "end_turn"
    metadata = dict(payload.get("metadata") or {})
    return str(
        metadata.get("rl_action_type")
        or payload.get("type")
        or payload.get("kind")
        or ""
    )


def _estimate_max_turns(snapshot: dict[str, Any]) -> int:
    explicit = _coerce_int(snapshot.get("max_turns"))
    if explicit > 0:
        return explicit
    turn = max(_coerce_int(snapshot.get("turn"), 1), 1)
    remaining = _coerce_int(snapshot.get("remaining"))
    if remaining > 0:
        return max(turn, turn + remaining - 1)
    return turn


def _build_card_payload(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "db_id": str(card.get("db_id") or card.get("id") or ""),
        "name": str(card.get("name") or ""),
        "category": str(card.get("category") or ""),
        "rarity": str(card.get("rarity") or ""),
        "upgrade_count": _coerce_int(card.get("upgrade_count")),
        "cost": _coerce_int(card.get("cost")),
        "description": str(card.get("description") or ""),
        "effect_types": [
            str(value)
            for value in (card.get("effect_types") or [])
            if str(value or "").strip()
        ],
    }


def _build_drink_payload(drink: dict[str, Any]) -> dict[str, Any]:
    return {
        "db_id": str(drink.get("db_id") or drink.get("id") or ""),
        "name": str(drink.get("name") or ""),
        "description": str(drink.get("description") or ""),
        "rarity": str(drink.get("rarity") or ""),
        "effect_types": [
            str(value)
            for value in (drink.get("effect_types") or [])
            if str(value or "").strip()
        ],
    }


def _build_legal_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    return {
        "index": _coerce_int(payload.get("index")),
        "action_id": str(payload.get("id") or payload.get("action_id") or ""),
        "db_id": str(payload.get("db_id") or ""),
        "kind": _derive_action_kind(payload),
        "available": bool(payload.get("available", True)),
        "selected": bool(payload.get("selected", False)),
        "upgrade_count": _coerce_int(metadata.get("upgrade_count")),
        "effect_types": [
            str(value)
            for value in (metadata.get("effect_types") or [])
            if str(value or "").strip()
        ],
        "candidate_type": str(metadata.get("candidate_type") or ""),
    }


def build_battle_predict_payload(decision_state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """把 producer battle 决策快照转换成 RL 服务需要的无状态请求。"""

    _BATTLE_PHASES = {"lesson", "exam"}
    phase = str(decision_state.get("phase") or "")
    if phase not in _BATTLE_PHASES:
        return None

    snapshot = dict(decision_state.get("llm_snapshot") or {})
    parameter_stats = dict(snapshot.get("parameter_stats") or {})
    resources = dict(snapshot.get("resources") or {})
    candidates = [
        dict(payload)
        for payload in (decision_state.get("candidates") or [])
    ]
    legal_actions = [_build_legal_action_payload(payload) for payload in candidates]

    hand_cards = [
        _build_card_payload(card)
        for card in (snapshot.get("hand") or [])
    ]
    drinks = [
        _build_drink_payload(drink)
        for drink in (snapshot.get("drinks") or [])
    ]

    state = {
        "battle_kind": phase,
        "scenario": str(snapshot.get("scenario") or ""),
        "difficulty": str(snapshot.get("difficulty") or ""),
        "vocal": _coerce_int(parameter_stats.get("vocal")),
        "dance": _coerce_int(parameter_stats.get("dance")),
        "visual": _coerce_int(parameter_stats.get("visual")),
        "vocal_max": _coerce_int(parameter_stats.get("vocal_max")),
        "dance_max": _coerce_int(parameter_stats.get("dance_max")),
        "visual_max": _coerce_int(parameter_stats.get("visual_max")),
        "stamina": _coerce_int(snapshot.get("stamina")),
        "max_stamina": max(_coerce_int(snapshot.get("max_stamina"), 1), 1),
        "score": _coerce_int(snapshot.get("score")),
        "target_score": max(_coerce_int(snapshot.get("target"), 1), 1),
        "turn": max(_coerce_int(snapshot.get("turn"), 1), 1),
        "max_turns": max(_estimate_max_turns(snapshot), 1),
        "remaining_turns": _coerce_int(snapshot.get("remaining")),
        "block": _coerce_int(resources.get("block")),
        "review": _coerce_int(resources.get("review")),
        "aggressive": _coerce_int(resources.get("aggressive")),
        "concentration": _coerce_int(resources.get("concentration")),
        "full_power_point": _coerce_int(resources.get("full_power_point")),
        "parameter_buff": _coerce_int(resources.get("parameter_buff")),
        "lesson_buff": _coerce_int(resources.get("lesson_buff")),
        "enthusiastic": _coerce_int(resources.get("enthusiastic")),
        "turn_color": _normalize_turn_color(
            snapshot.get("turn_color_label")
            or snapshot.get("turn_color_display_label")
        ),
        "stance": str(snapshot.get("stance_desc") or "neutral"),
        "hand_cards": hand_cards,
        "deck_count": _coerce_int(snapshot.get("deck_count")),
        "grave_count": _coerce_int((snapshot.get("zone_counts") or {}).get("grave", snapshot.get("grave_count"))),
        "deck_cards": [
            _build_card_payload(card)
            for card in (snapshot.get("deck_cards") or [])
        ],
        "grave_cards": [
            _build_card_payload(card)
            for card in (snapshot.get("grave_cards") or [])
        ],
        "hold_cards": [
            _build_card_payload(card)
            for card in (snapshot.get("hold_cards") or [])
        ],
        "lost_cards": [
            _build_card_payload(card)
            for card in (snapshot.get("lost_cards") or [])
        ],
        "drinks": drinks,
        "status_enchants": [
            str(value)
            for value in (snapshot.get("active_enchants") or [])
            if str(value or "").strip()
        ],
        "p_items": [
            str(item.get("id") or item.get("db_id") or "")
            for item in (snapshot.get("p_items") or [])
            if str(item.get("id") or item.get("db_id") or "").strip()
        ],
        "formation_abilities": [
            str(item.get("id") or item.get("ability_id") or "")
            for item in (snapshot.get("formation_abilities") or [])
            if str(item.get("id") or item.get("ability_id") or "").strip()
        ],
        "fan_votes": _coerce_int(snapshot.get("fan_votes")) or None,
    }
    return state, legal_actions


# ── Phase 路由映射 ───────────────────────────────────────────


_PHASE_TO_STRATEGY = {
    "schedule": "schedule",
    "lesson": "battle",
    "exam": "battle",
    "dialogue": "other",
    "p_drink": "other",
    "skill_reward": "other",
    "consult": "other",
    "item_select": "other",
    "modal": "other",
}


# ── Schedule RL 策略 ─────────────────────────────────────────


class ScheduleRLStrategy:
    """周行程 RL 决策（暂不支持）。"""

    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        phase = str(decision_state.get("phase") or "")
        if phase != "schedule":
            return None
        return None


# ── Battle RL 策略 ───────────────────────────────────────────


class BattleRLStrategy:
    """战斗 RL 决策（lesson/exam）。"""

    def __init__(
        self,
        parent: "RLStrategy",
    ) -> None:
        self._parent = parent

    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        phase = str(decision_state.get("phase") or "")
        if phase not in ("lesson", "exam"):
            return None

        if not candidates or decision_state is None:
            return None

        payload = build_battle_predict_payload(decision_state)
        if payload is None:
            return None
        state, legal_actions = payload
        client = self._parent._client_for_phase(phase)
        result = client.predict(
            state,
            legal_actions,
            deterministic=self._parent.deterministic,
        )
        if not result:
            return None

        candidate_count = len(candidates)
        action_index = result.get("action_index")
        if isinstance(action_index, int) and 0 <= action_index < candidate_count:
            legal_indexes = {
                _coerce_int(item.get("index"))
                for item in legal_actions
                if bool(item.get("available", True))
            }
            if not legal_indexes or action_index in legal_indexes:
                return result
            logger.warning(
                "[RL][battle] {} 返回非法动作索引 {}，legal={}",
                phase,
                action_index,
                sorted(legal_indexes),
            )

        action_id = str(result.get("action_id") or "")
        db_id = str(result.get("db_id") or "")
        if action_id or db_id:
            for action in legal_actions:
                if not bool(action.get("available", True)):
                    continue
                if action_id and action_id == str(action.get("action_id") or ""):
                    return {
                        "action_index": _coerce_int(action.get("index")),
                        "action_id": action_id,
                        "db_id": db_id,
                    }
                if db_id and db_id == str(action.get("db_id") or ""):
                    return {
                        "action_index": _coerce_int(action.get("index")),
                        "action_id": str(action.get("action_id") or ""),
                        "db_id": db_id,
                    }

        return None


# ── Other RL 策略 ────────────────────────────────────────────


class OtherRLStrategy:
    """其他 RL 决策（暂不支持）。"""

    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        phase = str(decision_state.get("phase") or "")
        if phase in ("schedule", "lesson", "exam"):
            return None
        return None


# ── 统一 RL 策略入口 ─────────────────────────────────────────


class RLStrategy:
    """通过 RL HTTP 推理服务做游戏决策的统一策略。

    内部持有3个子策略，按 decision_state["phase"] 路由：
    - ScheduleRLStrategy: schedule（暂不支持）
    - BattleRLStrategy: lesson / exam
    - OtherRLStrategy: dialogue / p_drink 等（暂不支持）
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        *,
        lesson_base_url: str | None = None,
        exam_base_url: str | None = None,
        info_timeout: float = 5.0,
        predict_timeout: float = 10.0,
        deterministic: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.lesson_base_url = str(lesson_base_url or self.base_url).rstrip("/")
        self.exam_base_url = str(exam_base_url or self.base_url).rstrip("/")
        self.deterministic = bool(deterministic)
        self._client = RLInferenceClient(
            self.base_url,
            info_timeout=info_timeout,
            predict_timeout=predict_timeout,
        )
        self._lesson_client = RLInferenceClient(
            self.lesson_base_url,
            info_timeout=info_timeout,
            predict_timeout=predict_timeout,
        )
        self._exam_client = RLInferenceClient(
            self.exam_base_url,
            info_timeout=info_timeout,
            predict_timeout=predict_timeout,
        )

        self._schedule = ScheduleRLStrategy()
        self._battle = BattleRLStrategy(self)
        self._other = OtherRLStrategy()

    def _client_for_phase(self, phase: str) -> RLInferenceClient:
        """按 battle phase 选择对应的 RL 推理客户端。"""

        if str(phase or '') == 'lesson':
            return self._lesson_client
        if str(phase or '') == 'exam':
            return self._exam_client
        return self._client

    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if decision_state is None:
            return None

        phase = str(decision_state.get("phase") or "")
        target = _PHASE_TO_STRATEGY.get(phase, "schedule")

        if target == "battle":
            return self._battle(app, ctx, candidates, decision_state)
        elif target == "other":
            return self._other(app, ctx, candidates, decision_state)
        else:
            return self._schedule(app, ctx, candidates, decision_state)


# ── 工厂与注入 ──────────────────────────────────────────────


def inject_rl_strategy(
    ctx: "ProduceContext",
    *,
    base_url: str = "http://127.0.0.1:8001",
    lesson_base_url: str | None = None,
    exam_base_url: str | None = None,
    info_timeout: float = 5.0,
    predict_timeout: float = 10.0,
    deterministic: bool = True,
) -> RLStrategy:
    """把 RL 战斗策略注入到 producer context。"""

    strategy = RLStrategy(
        base_url=base_url,
        lesson_base_url=lesson_base_url,
        exam_base_url=exam_base_url,
        info_timeout=info_timeout,
        predict_timeout=predict_timeout,
        deterministic=deterministic,
    )
    ctx.rl_inference_url = strategy.base_url
    ctx.schedule_strategy = strategy
    ctx.lesson_strategy = strategy
    ctx.exam_strategy = strategy
    ctx.dialogue_strategy = strategy
    ctx.skill_reward_strategy = strategy
    ctx.p_drink_strategy = strategy
    ctx.consult_strategy = strategy
    ctx.item_select_strategy = strategy
    ctx.modal_strategy = strategy
    return strategy


__all__ = [
    "RLStrategy",
    "ScheduleRLStrategy",
    "BattleRLStrategy",
    "OtherRLStrategy",
    "build_battle_predict_payload",
    "inject_rl_strategy",
]
