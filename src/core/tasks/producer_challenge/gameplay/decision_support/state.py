from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.game.text.produce_text import ProduceText
from src.core.tasks.producer_challenge.gameplay.exam_prep import get_exam_prep_bonuses
from src.core.tasks.producer_challenge.gameplay.exam_wheel import get_exam_wheel_info

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext


def is_produce_card_action_id(action_id: Any) -> bool:
    normalized = str(action_id or "")
    return normalized.startswith("produce_card:") or normalized.startswith("produce_card_unknown")


def is_produce_drink_action_id(action_id: Any) -> bool:
    normalized = str(action_id or "")
    return normalized.startswith("produce_drink:") or normalized.startswith("produce_drink_unknown")


def is_end_turn_action_id(action_id: Any) -> bool:
    return str(action_id or "").strip() == "end_turn"


def _serialize_box(box: Any) -> list[int] | None:
    if box is None:
        return None
    x = int(getattr(box, "x", 0))
    y = int(getattr(box, "y", 0))
    w = int(getattr(box, "w", 0))
    h = int(getattr(box, "h", 0))
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def serialize_candidate(candidate: Any, *, phase: str) -> dict[str, Any]:
    title = getattr(candidate, "title", "") or getattr(candidate, "label", "") or getattr(candidate, "kind", "")
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    payload = {
        "index": int(getattr(candidate, "index", 0)),
        "id": getattr(candidate, "action_id", "") or f"{phase}:{getattr(candidate, 'index', 0)}",
        "db_id": getattr(candidate, "db_id", "") or "",
        "name": title,
        "type": metadata.get("consult_action") or metadata.get("candidate_type") or phase,
        "label": getattr(candidate, "label", "") or getattr(candidate, "kind", "") or title,
        "selected": bool(getattr(candidate, "selected", False)),
        "recommended": bool(getattr(candidate, "recommended", False)),
        "available": bool(metadata.get("available", True)),
        "bbox": _serialize_box(getattr(candidate, "box", None)),
        "source": getattr(candidate, "source", "") or metadata.get("source", ""),
        "confidence": float(getattr(candidate, "confidence", 0.0) or 0.0),
        "metadata": metadata,
    }
    if payload["db_id"]:
        payload["entity_kind"] = (
            "produce_card"
            if is_produce_card_action_id(payload["id"])
            else "produce_drink"
            if is_produce_drink_action_id(payload["id"])
            else "produce_item"
            if payload["id"].startswith("produce_item:")
            else ""
        )
    return payload


def _compute_remaining_weeks(ctx: "ProduceContext") -> int | None:
    """从 P手帳 数据推算剩余可操作周数。"""
    notebook = list(ctx.handler_state.get("p_notebook_schedule") or [])
    if not notebook:
        return None
    remaining = 0
    for entry in notebook:
        if entry.get("completed"):
            continue
        if entry.get("special_event") and not entry.get("actions"):
            continue
        remaining += 1
    return remaining


def _build_stage_context(
    *,
    phase: str,
    position: str,
    hud_state: dict[str, Any],
    candidate_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    phase_key = phase.value if hasattr(phase, "value") else str(phase)
    position_key = position.value if hasattr(position, "value") else str(position)
    has_progress_hud = bool(hud_state.get("has_progress_hud"))
    has_battle_drink_actions = any(
        is_produce_drink_action_id(payload.get("id"))
        for payload in candidate_payloads
    )
    has_end_turn_action = any(
        is_end_turn_action_id(payload.get("id"))
        for payload in candidate_payloads
    )

    stage_id = phase_key or "unknown"
    label = "未知阶段"
    description = "当前画面阶段语义尚未稳定识别。"
    available_action_summary = "优先从合法动作列表中选择当前最稳妥的动作。"
    interaction_hint = ""

    if phase_key == GameplayPhase.SCHEDULE:
        stage_id = "schedule_action_select"
        label = "周行动选择"
        description = "当前处于培育周行程页，需要在本周可执行行动中做选择。"
        available_action_summary = "可从候选周行动中选择一项；若已选中行动，则下一次点击会确认进入该行动。"
        interaction_hint = "周行动通常是先选中，再在下一帧确认。"
        if position_key == GameplayPosition.SCHEDULE_PRESENT_SUPPORT:
            stage_id = "schedule_present_support_options"
            label = "活动支给选项"
            description = "当前已进入「活動支給 / 差し入れ」收益选择页，需要在多个加成候选中选一个。"
            available_action_summary = "可从当前活动支给候选项中选择一个收益分支，这类页面通常单击即可确认。"
            interaction_hint = "活动支给候选通常单击就会直接进入后续奖励链。"
        elif position_key == GameplayPosition.SCHEDULE_SELECTED:
            stage_id = "schedule_action_confirm"
            label = "周行动确认"
            description = "当前已有一个周行动被选中，等待确认进入。"
        elif position_key == GameplayPosition.SCHEDULE_RECOMMEND:
            stage_id = "schedule_action_recommend"
    elif phase_key == GameplayPhase.DIALOGUE:
        if has_progress_hud:
            if position_key in {
                GameplayPosition.DIALOGUE_OPTIONS,
                GameplayPosition.SCHEDULE_EVENT_OPTIONS,
            }:
                stage_id = "schedule_event_options"
                label = "周事件选项"
                description = "当前处于培育流程内的周事件分支选择，不是普通コミュ。"
                available_action_summary = "可从当前周事件选项中选择一个分支。"
                interaction_hint = "事件选项通常是先点一次选中，再点一次确认。"
            else:
                stage_id = "schedule_event_dialogue"
                label = "周事件对话推进"
                description = "当前处于培育流程内的周事件文本推进阶段。"
                available_action_summary = "当前无分支时推进文本；若出现选项则转为选择事件分支。"
                interaction_hint = "无选项时以点击推进为主。"
        elif position_key == GameplayPosition.DIALOGUE_OPTIONS:
            stage_id = "dialogue_options"
            label = "普通对话选项"
            description = "当前是普通コミュ对话分支选择。"
            available_action_summary = "可从当前对话选项中选择一个分支。"
            interaction_hint = "对话选项通常是先点一次选中，再点一次确认。"
        else:
            stage_id = "dialogue_continue"
            label = "普通对话推进"
            description = "当前是无分支的剧情推进阶段。"
            available_action_summary = "当前无选项时推进文本；若可快进则也可切换快进。"
            interaction_hint = "普通对话可推进文本，必要时可以快进。"
    elif phase_key == GameplayPhase.LESSON:
        stage_id = "lesson_card_play"
        label = "课程出牌"
        description = "当前处于レッスン回合，需要决定本回合如何出牌。"
        available_action_summary = "可从当前手牌中选择一张技能卡使用；若已有选中卡，则下一次点击会确认出牌。"
        interaction_hint = "出牌通常是先选中卡牌，再确认使用。"
        if has_battle_drink_actions:
            available_action_summary = (
                "可从当前手牌中选择技能卡，也可以直接使用底栏 P 饮料；"
                "饮料通常点击一次就会打开使用确认。"
            )
            interaction_hint = "技能卡按双击使用；P 饮料通常点击图标后进入确认/详情。"
        if has_end_turn_action:
            available_action_summary = f"{available_action_summary} 也可以选择 SKIP，直接结束本回合。"
            interaction_hint = f"{interaction_hint} 选择 SKIP 前若有残留选中态，先取消选中再结束回合。"
        if position_key == GameplayPosition.LESSON_SELECTED:
            stage_id = "lesson_card_confirm"
            label = "课程出牌确认"
            description = "当前已有技能卡被选中，等待确认出牌。"
    elif phase_key == GameplayPhase.EXAM:
        stage_id = "exam_card_play"
        label = "考试出牌"
        description = "当前处于考试/试演回合，需要决定本回合如何出牌。"
        available_action_summary = "可从当前手牌中选择一张技能卡使用；若已有选中卡，则下一次点击会确认出牌。"
        interaction_hint = "出牌通常是先选中卡牌，再确认使用。"
        if position_key == GameplayPosition.EXAM_RETRY_CONFIRM_MODAL:
            stage_id = "exam_retry_confirm"
            label = "考试失败后的再挑战确认"
            description = "当前考试未通过，需要在「再挑戦」与「プロデュース終了」之间做最终选择。"
            available_action_summary = "可选择消耗一次再挑战机会重打本场考试，或直接结束本次培育并接受失败结果。"
            interaction_hint = "左侧通常是再挑戦，右侧通常是プロデュース終了，这个弹窗点击一次就会立即生效。"
            retry_payload = next(
                (
                    payload
                    for payload in candidate_payloads
                    if str(payload.get("id") or "") == "exam_retry"
                ),
                None,
            )
            remaining_retry_count = None
            if retry_payload is not None:
                remaining_retry_count = retry_payload.get("metadata", {}).get("remaining_retry_count")
            if remaining_retry_count is not None:
                description = f"{description} 当前剩余再挑战次数约为 {remaining_retry_count} 次。"
        elif has_battle_drink_actions:
            available_action_summary = (
                "可从当前手牌中选择技能卡，也可以直接使用底栏 P 饮料；"
                "饮料通常点击一次就会打开使用确认。"
            )
            interaction_hint = "技能卡按双击使用；P 饮料通常点击图标后进入确认/详情。"
        if stage_id != "exam_retry_confirm" and has_end_turn_action:
            available_action_summary = f"{available_action_summary} 也可以选择结束本回合，放弃剩余出牌。"
            interaction_hint = f"{interaction_hint} 结束回合前若有残留选中态，先取消选中再点击按钮。"
        if position_key == GameplayPosition.EXAM_SELECTED:
            stage_id = "exam_card_confirm"
            label = "考试出牌确认"
            description = "当前已有技能卡被选中，等待确认出牌。"
    elif phase_key == GameplayPhase.SKILL_REWARD:
        stage_id = "skill_reward_select"
        label = "技能卡奖励选择"
        description = "当前处于技能卡奖励阶段，需要从候选奖励中选择一张。"
        available_action_summary = "可从候选奖励卡中选择一张；若已有选中卡，则下一次点击会确认领取。"
        interaction_hint = "奖励卡通常是先选中，再确认领取。"
        if position_key == GameplayPosition.SKILL_REWARD_SELECTED:
            stage_id = "skill_reward_confirm"
            label = "技能卡奖励确认"
            description = "当前已有奖励卡被选中，等待确认领取。"
    elif phase_key == GameplayPhase.P_DRINK:
        stage_id = "p_drink_select"
        label = "P饮料选择"
        description = "当前处于 P 饮料选择阶段，需要决定是否使用/领取某个饮料。"
        available_action_summary = "可从当前 P 饮料候选中选择一个；若已有选中饮料，则下一次点击会确认。"
        interaction_hint = "P 饮料通常是先选中，再确认。"
        if position_key == "p_drink_limit":
            stage_id = "p_drink_limit"
            label = "P饮料上限处理"
            description = "当前 P 饮料槽已满，需要决定是放弃新饮料，还是丢弃一瓶旧饮料来保留新饮料。"
            available_action_summary = "可选择放弃新饮料，或丢弃一瓶现有饮料以腾出槽位。"
            interaction_hint = "所持上限页的每个动作都会直接改变保留方案。"
        if position_key == GameplayPosition.P_DRINK_SELECTED:
            stage_id = "p_drink_confirm"
            label = "P饮料确认"
            description = "当前已有 P 饮料被选中，等待确认。"
    elif phase_key == GameplayPhase.CONSULT:
        if position_key == GameplayPosition.CONSULT_EXCHANGE:
            stage_id = "consult_exchange"
            label = "咨询兑换"
            description = "当前处于相談兑换页，可执行多个操作后再退出。"
            available_action_summary = "可兑换物品（多次）、打开強化（限1次）、打开削除（限1次）、或退出。"
            interaction_hint = "兑换类候选点选后立即进入下一步；每次操作后会再次询问。"
        else:
            stage_id = "consult_card_select"
            label = "咨询卡牌处理"
            description = "当前处于相談后的卡牌预览/确认页，需要决定强化或删除对象。"
            available_action_summary = "可从当前可见卡牌中选择要强化/删除的目标。"
            interaction_hint = "卡牌目标通常是先选中，再确认。"
    elif phase_key == GameplayPhase.ITEM_SELECT:
        stage_id = "item_select"
        label = "P物品选择"
        description = "当前处于 P 物品选择阶段，需要从候选物品中选择一个。"
        available_action_summary = "可从当前 P 物品候选中选择一个；若已有选中物品，则下一次点击会确认。"
        interaction_hint = "P 物品通常是先选中，再确认。"
        if position_key == GameplayPosition.ITEM_SELECT_SELECTED:
            stage_id = "item_confirm"
            label = "P物品确认"
            description = "当前已有 P 物品被选中，等待确认。"

    candidate_names = [
        payload.get("name") or payload.get("label") or f"动作{payload.get('index', 0)}"
        for payload in candidate_payloads
    ]
    recommended_names = [
        payload.get("name") or payload.get("label") or f"动作{payload.get('index', 0)}"
        for payload in candidate_payloads
        if payload.get("recommended")
    ]

    system_recommendation = ""
    if recommended_names:
        system_recommendation = f"系统当前推荐优先考虑：{' / '.join(recommended_names[:3])}"
        if len(recommended_names) > 3:
            system_recommendation += f" 等{len(recommended_names)}项"
    else:
        recommend_kind = str(hud_state.get("recommend_action_kind") or "").strip()
        recommend_text = str(hud_state.get("recommend_action_text") or "").strip()
        if recommend_kind and recommend_kind != "unknown":
            system_recommendation = f"系统当前推荐优先考虑 {recommend_kind} 系行动"
        elif recommend_text:
            system_recommendation = f"系统当前推荐提示：{recommend_text}"

    return {
        "id": stage_id,
        "label": label,
        "description": description,
        "available_action_summary": available_action_summary,
        "interaction_hint": interaction_hint,
        "candidate_count": len(candidate_payloads),
        "candidate_names": candidate_names,
        "system_recommendation": system_recommendation,
        "is_schedule_context": has_progress_hud,
    }


def _describe_candidate_operation(
    payload: dict[str, Any],
    *,
    phase: str,
    position: str,
    stage_context: dict[str, Any],
) -> str:
    phase_key = phase.value if hasattr(phase, "value") else str(phase)
    position_key = position.value if hasattr(position, "value") else str(position)
    label = str(payload.get("name") or payload.get("label") or f"动作{payload.get('index', 0)}")
    stage_id = str(stage_context.get("id") or "")

    if phase_key == GameplayPhase.SCHEDULE:
        metadata = dict(payload.get("metadata") or {})
        readable = str(
            metadata.get("display_name")
            or payload.get("name")
            or payload.get("label")
            or f"动作{payload.get('index', 0)}"
        )
        if stage_id == "schedule_present_support_options":
            return f"点击后会直接选择这项活動支給收益：「{readable}」。"
        if position_key == GameplayPosition.SCHEDULE_SELECTED:
            return f"点击后会确认进入「{readable}」这个周行动。"
        return f"点击后会选中「{readable}」这个周行动，下一帧再次点击会确认进入。"
    if stage_id == "schedule_event_options":
        metadata = dict(payload.get("metadata") or {})
        p_cost = metadata.get("p_cost")
        cost_hint = f"（消耗{p_cost}Pポイント）" if p_cost is not None else ""
        return f"点击后会选中「{label}」这个周事件分支{cost_hint}，下一帧再次点击会确认该分支。"
    if phase_key == GameplayPhase.DIALOGUE and position_key == GameplayPosition.DIALOGUE_OPTIONS:
        return f"点击后会选中「{label}」这个对话分支，下一帧再次点击会确认该分支。"
    if phase_key == GameplayPhase.LESSON:
        if is_end_turn_action_id(payload.get("id")):
            if position_key == GameplayPosition.LESSON_SELECTED:
                return "点击后会先取消当前选中的技能卡，再执行 SKIP 结束本回合。"
            return "点击后会执行 SKIP，放弃本回合剩余出牌并直接进入下一回合。"
        if is_produce_drink_action_id(payload.get("id")):
            return f"点击后会确认使用这瓶 P 饮料「{label}」。"
        if position_key == GameplayPosition.LESSON_SELECTED:
            return f"点击后会确认使用这张技能卡：「{label}」。"
        return f"点击后会选中技能卡「{label}」，下一帧再次点击会确认使用。"
    if phase_key == GameplayPhase.EXAM:
        if position_key == GameplayPosition.EXAM_RETRY_CONFIRM_MODAL:
            action_id = str(payload.get("id") or "")
            if action_id == "exam_retry":
                return "点击后会消耗一次再挑战机会，重新开始当前这场考试，不会直接结束本次培育。"
            if action_id == "produce_end":
                return "点击后会确认结束本次培育，本场考试将按失败处理并退出本次挑战。"
            return f"点击后会在考试失败后的确认弹窗里执行「{label}」。"
        if is_end_turn_action_id(payload.get("id")):
            if position_key == GameplayPosition.EXAM_SELECTED:
                return "点击后会先取消当前选中的技能卡，再结束本回合。"
            return "点击后会结束本回合，放弃当前剩余出牌并推进到下一回合。"
        if is_produce_drink_action_id(payload.get("id")):
            return f"点击后会确认在考试中使用这瓶 P 饮料「{label}」。"
        if position_key == GameplayPosition.EXAM_SELECTED:
            return f"点击后会确认在考试中使用这张技能卡：「{label}」。"
        return f"点击后会选中考试用技能卡「{label}」，下一帧再次点击会确认使用。"
    if phase_key == GameplayPhase.SKILL_REWARD:
        metadata = dict(payload.get("metadata") or {})
        if metadata.get("is_redraw"):
            remaining = metadata.get("redraw_remaining", 0)
            return f"点击后会消耗一次再抽選机会（剩余{remaining}回），刷新全部候选技能卡。使用后不可撤销。"
        if position_key == GameplayPosition.SKILL_REWARD_SELECTED:
            return f"点击后会确认领取奖励卡「{label}」。"
        return f"点击后会选中奖励卡「{label}」，下一帧再次点击会确认领取。"
    if phase_key == GameplayPhase.P_DRINK:
        if position_key == "p_drink_limit":
            kind = str(payload.get("kind") or "")
            if kind == "skip_new_drink":
                return f"点击后会放弃新饮料「{label}」，保留当前饮料槽配置。"
            if kind == "discard_existing_drink":
                return f"点击后会丢弃当前库存中的一瓶旧饮料，并保留新饮料「{label}」。"
        if position_key == GameplayPosition.P_DRINK_SELECTED:
            return f"点击后会确认当前选择的 P 饮料「{label}」。"
        return f"点击后会选中 P 饮料「{label}」，下一帧再次点击会确认。"
    if phase_key == GameplayPhase.CONSULT:
        consult_action = str(
            payload.get("type")
            or (payload.get("metadata") or {}).get("consult_action")
            or payload.get("label")
            or ""
        )
        metadata = dict(payload.get("metadata") or {})
        display_name = str(
            metadata.get("display_name")
            or metadata.get("raw_name")
            or label
        )
        if position_key == GameplayPosition.CONSULT_EXCHANGE:
            if consult_action == "consult_open_enhancement":
                return "点击后会进入技能卡強化页面，可以选择一张技能卡进行強化。"
            if consult_action == "consult_open_remove":
                return "点击后会进入技能卡削除页面，可以选择一张技能卡进行削除。"
            if consult_action == "consult_exit":
                return "点击后会退出相談，结束本次相談环节。"
            price = str(metadata.get("price") or "")
            price_part = f"消耗 {price}P" if price else "消耗对应 P ポイント"
            return f"点击后会尝试兑换「{display_name}」，{price_part}。"
        if consult_action == "consult_confirm_enhancement":
            return f"点击后会确认強化选中的技能卡「{display_name}」。"
        if consult_action == "consult_confirm_remove":
            return f"点击后会确认削除选中的技能卡「{display_name}」。"
        if consult_action in {"consult_select_enhancement_target", "consult_select_remove_target"}:
            return f"点击后会选中「{display_name}」作为相談处理目标，下一帧再确认。"
        return f"点击后会选中「{display_name}」作为相談处理目标，下一帧再确认。"
    if phase_key == GameplayPhase.ITEM_SELECT:
        item_meta = dict(payload.get("metadata") or {})
        display_name = str(
            item_meta.get("display_name")
            or item_meta.get("raw_name")
            or label
        )
        if position_key == GameplayPosition.ITEM_SELECT_SELECTED:
            return f"点击后会确认领取/选择 P 物品「{display_name}」。"
        return f"点击后会选中 P 物品「{display_name}」，下一帧再次点击会确认。"
    return str(stage_context.get("available_action_summary") or "")


def register_realtime_resource_snapshot(ctx: "ProduceContext", **values: Any) -> None:
    """注册实时资源观测值，供虚拟状态估算结果覆写。"""
    realtime = ctx.handler_state.setdefault("realtime_battle_state", {})
    resources = realtime.setdefault("resources", {})
    for key, value in values.items():
        if value is not None:
            resources[key] = value


def register_realtime_zone_snapshot(ctx: "ProduceContext", **zones: Any) -> None:
    """注册实时牌区观测值，供虚拟状态估算结果覆写。"""
    realtime = ctx.handler_state.setdefault("realtime_battle_state", {})
    zone_payload = realtime.setdefault("zones", {})
    for key, value in zones.items():
        if value is not None:
            zone_payload[key] = value


_SIM_RESOURCE_KEYS = (
    "parameter_buff",
    "review",
    "aggressive",
    "block",
    "enthusiastic",
    "full_power_point",
    "lesson_buff",
)
_SIM_DECAY_KEYS = ("parameter_buff", "aggressive")
_SIM_DESTINATION_HOLD_KEYWORDS = ("保留",)
_SIM_DESTINATION_LOST_KEYWORDS = ("除外", "削除", "消去")


def _default_virtual_battle_state() -> dict[str, Any]:
    return {
        "version": 1,
        "initialized": False,
        "instance_seq": 0,
        "last_operation_count": 0,
        "last_remaining_turns": None,
        "turn_index": 1,
        "play_limit_total_current": 1,
        "play_limit_remaining": 1,
        "resources": {key: 0 for key in _SIM_RESOURCE_KEYS},
        "resource_source": {key: "simulated" for key in _SIM_RESOURCE_KEYS},
        "zones": {
            "deck": [],
            "hand": [],
            "grave": [],
            "hold": [],
            "lost": [],
        },
        "zone_source": {
            "deck": "simulated",
            "hand": "simulated",
            "grave": "simulated",
            "hold": "simulated",
            "lost": "simulated",
        },
    }


def _get_virtual_battle_state(ctx: "ProduceContext") -> dict[str, Any]:
    state = ctx.handler_state.get("virtual_battle_state")
    if not isinstance(state, dict) or state.get("version") != 1:
        state = _default_virtual_battle_state()
        ctx.handler_state["virtual_battle_state"] = state
    return state


def _new_virtual_card_instance(state: dict[str, Any], entry: dict[str, Any], *, source: str) -> dict[str, Any]:
    state["instance_seq"] += 1
    return {
        "instance_key": f"{entry.get('id') or entry.get('name') or 'card'}#{state['instance_seq']}",
        "id": str(entry.get("id") or ""),
        "name": str(entry.get("name") or ""),
        "description": str(entry.get("description") or ""),
        "category": str(entry.get("category") or ""),
        "upgrade_count": int(entry.get("upgrade_count") or 0),
        "source": source,
    }


def _bootstrap_virtual_deck(state: dict[str, Any], known_deck: list[dict[str, Any]]) -> None:
    if state["initialized"]:
        return
    state["zones"]["deck"] = [
        _new_virtual_card_instance(state, entry, source="formation")
        for entry in known_deck
    ]
    state["initialized"] = True


def _normalize_card_identity(card: dict[str, Any]) -> str:
    return str(card.get("id") or card.get("name") or "").strip()


def _find_virtual_card(
    state: dict[str, Any],
    observed: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    observed_id = _normalize_card_identity(observed)
    observed_name = str(observed.get("name") or "").strip()
    for zone_name in ("hand", "deck", "grave", "hold", "lost"):
        for card in state["zones"][zone_name]:
            if observed_id and observed_id == _normalize_card_identity(card):
                return zone_name, card
            if observed_name and observed_name == str(card.get("name") or "").strip():
                return zone_name, card
    return None


def _remove_virtual_card_from_all_zones(state: dict[str, Any], instance_key: str) -> None:
    for zone_name in ("deck", "hand", "grave", "hold", "lost"):
        state["zones"][zone_name] = [
            card
            for card in state["zones"][zone_name]
            if card.get("instance_key") != instance_key
        ]


def _sync_virtual_hand(
    state: dict[str, Any],
    observed_hand: list[dict[str, Any]],
) -> None:
    if not observed_hand:
        state["zones"]["hand"] = []
        return

    current_hand: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for observed in observed_hand:
        found = _find_virtual_card(state, observed)
        if found is None:
            card = _new_virtual_card_instance(
                state,
                {
                    "id": observed.get("db_id") or "",
                    "name": observed.get("name") or "",
                    "description": observed.get("description") or "",
                    "category": observed.get("category") or "",
                    "upgrade_count": observed.get("upgrade_count") or 0,
                },
                source="observed",
            )
        else:
            _, card = found
            _remove_virtual_card_from_all_zones(state, card["instance_key"])
        if card["instance_key"] in seen_keys:
            continue
        seen_keys.add(card["instance_key"])
        current_hand.append(card)

    previous_hand = list(state["zones"]["hand"])
    state["zones"]["hand"] = current_hand
    for card in previous_hand:
        if card.get("instance_key") not in seen_keys:
            state["zones"]["grave"].append(card)


def _extract_simulated_delta(text: str, keyword: str, *, allow_turn_suffix: bool = True) -> int:
    raw = str(text or "")
    if not raw or keyword not in raw:
        return 0
    patterns = [
        rf"{re.escape(keyword)}\s*[+＋]\s*(\d+)",
        rf"{re.escape(keyword)}\s*(\d+){'(?:ターン|回合)' if allow_turn_suffix else ''}",
        rf"[+＋]\s*(\d+)\s*{re.escape(keyword)}",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return int(match.group(1))
    return 0


def _infer_virtual_destination(card: dict[str, Any]) -> str:
    description = str(card.get("description") or "")
    if any(keyword in description for keyword in _SIM_DESTINATION_HOLD_KEYWORDS):
        return "hold"
    if any(keyword in description for keyword in _SIM_DESTINATION_LOST_KEYWORDS):
        return "lost"
    return "grave"


def _apply_virtual_card_effects(state: dict[str, Any], card: dict[str, Any]) -> None:
    description = str(card.get("description") or "")
    state["resources"]["parameter_buff"] += _extract_simulated_delta(
        description,
        ProduceText.GOOD_CONDITION,
    )
    state["resources"]["review"] += _extract_simulated_delta(
        description,
        ProduceText.CONCENTRATION,
        allow_turn_suffix=False,
    )
    state["resources"]["aggressive"] += _extract_simulated_delta(
        description,
        ProduceText.GOOD_IMPRESSION,
    )
    state["resources"]["block"] += _extract_simulated_delta(
        description,
        ProduceText.GENKI,
        allow_turn_suffix=False,
    )
    state["resources"]["enthusiastic"] += _extract_simulated_delta(
        description,
        ProduceText.ENTHUSIASM,
        allow_turn_suffix=False,
    )
    state["resources"]["full_power_point"] += _extract_simulated_delta(
        description,
        ProduceText.FULL_POWER_POINT,
        allow_turn_suffix=False,
    )
    state["resources"]["lesson_buff"] += _extract_simulated_delta(
        description,
        ProduceText.PARAMETER_UP_INCREASE,
        allow_turn_suffix=False,
    )

    bonus_plays = (
        _extract_simulated_delta(
            description,
            ProduceText.SKILL_CARD_USE_COUNT_UP,
            allow_turn_suffix=False,
        )
        or _extract_simulated_delta(
            description,
            ProduceText.SKILL_CARD_USE_COUNT_UP_SHORT,
            allow_turn_suffix=False,
        )
    )
    if bonus_plays > 0:
        state["play_limit_total_current"] += bonus_plays
        state["play_limit_remaining"] += bonus_plays


def _find_card_in_hand_by_operation(state: dict[str, Any], operation: Any) -> dict[str, Any] | None:
    details = dict(getattr(operation, "details", {}) or {})
    target = str(getattr(operation, "target", "") or "")
    db_id = str(details.get("db_id") or "")
    for card in state["zones"]["hand"]:
        if db_id and card.get("id") == db_id:
            return card
        if target and target == str(card.get("name") or ""):
            return card
    return None


def _apply_virtual_operations(ctx: "ProduceContext", state: dict[str, Any]) -> None:
    operations = list(ctx.operation_history)
    start_index = int(state.get("last_operation_count", 0) or 0)
    for operation in operations[start_index:]:
        action = str(getattr(operation, "action", "") or "")
        if action == "use_lesson_card":
            card = _find_card_in_hand_by_operation(state, operation)
            if card is None:
                continue
            _remove_virtual_card_from_all_zones(state, card["instance_key"])
            destination = _infer_virtual_destination(card)
            state["zones"][destination].append(card)
            state["play_limit_remaining"] = max(int(state["play_limit_remaining"]) - 1, 0)
            _apply_virtual_card_effects(state, card)
    state["last_operation_count"] = len(operations)


def _advance_virtual_turn(state: dict[str, Any], turns: int = 1) -> None:
    for _ in range(max(int(turns), 0)):
        for key in _SIM_DECAY_KEYS:
            state["resources"][key] = max(int(state["resources"].get(key, 0) or 0) - 1, 0)
        state["turn_index"] = int(state.get("turn_index", 1) or 1) + 1
        state["play_limit_total_current"] = 1
        state["play_limit_remaining"] = 1


def _sync_virtual_turn_boundary(state: dict[str, Any], hud_state: dict[str, Any]) -> None:
    current_remaining = int(hud_state.get("remaining_turns") or 0)
    last_remaining = state.get("last_remaining_turns")
    if last_remaining is None:
        state["last_remaining_turns"] = current_remaining
        return
    if current_remaining <= 0:
        return
    if current_remaining < int(last_remaining):
        _advance_virtual_turn(state, int(last_remaining) - current_remaining)
    state["last_remaining_turns"] = current_remaining


def _merge_realtime_virtual_overrides(ctx: "ProduceContext", state: dict[str, Any]) -> None:
    realtime = ctx.handler_state.get("realtime_battle_state", {})
    for key, value in dict(realtime.get("resources", {}) or {}).items():
        if key in state["resources"] and value is not None:
            state["resources"][key] = value
            state["resource_source"][key] = "realtime"
    for zone_name, payload in dict(realtime.get("zones", {}) or {}).items():
        if zone_name in state["zones"] and payload is not None:
            state["zones"][zone_name] = list(payload)
            state["zone_source"][zone_name] = "realtime"


def _sync_virtual_battle_state(
    ctx: "ProduceContext",
    *,
    hud_state: dict[str, Any],
    known_deck: list[dict[str, Any]],
    observed_hand: list[dict[str, Any]],
) -> dict[str, Any]:
    state = _get_virtual_battle_state(ctx)
    _bootstrap_virtual_deck(state, known_deck)
    _apply_virtual_operations(ctx, state)
    _sync_virtual_turn_boundary(state, hud_state)
    _sync_virtual_hand(state, observed_hand)
    _merge_realtime_virtual_overrides(ctx, state)
    return state


def _append_exam_snapshot_details(
    snapshot: dict[str, Any],
    ctx: "ProduceContext",
) -> None:
    wheel_info = get_exam_wheel_info(ctx)
    if wheel_info:
        snapshot["exam_wheel"] = {
            "queue": wheel_info.get("queue", []),
            "remaining_turns": wheel_info.get("remaining_turns"),
            "current_param": wheel_info.get("current_param", ""),
            "bonus_pct": wheel_info.get("current_bonus_pct"),
            "confidence": wheel_info.get("confidence", "low"),
        }
    prep_bonuses = get_exam_prep_bonuses(ctx)
    if prep_bonuses:
        snapshot["exam_prep_bonuses"] = {
            "vocal": prep_bonuses.get("vocal_bonus_pct", 0),
            "dance": prep_bonuses.get("dance_bonus_pct", 0),
            "visual": prep_bonuses.get("visual_bonus_pct", 0),
        }


__all__ = [
    "_build_stage_context",
    "_compute_remaining_weeks",
    "_describe_candidate_operation",
    "_serialize_box",
    "_sync_virtual_battle_state",
    "is_end_turn_action_id",
    "is_produce_card_action_id",
    "is_produce_drink_action_id",
    "register_realtime_resource_snapshot",
    "register_realtime_zone_snapshot",
    "serialize_candidate",
    "_append_exam_snapshot_details",
]
