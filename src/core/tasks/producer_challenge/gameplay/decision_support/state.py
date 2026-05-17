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
    """判断 action_id 是否表示出牌操作。

    Args:
        action_id: 业务对象标识符，用于索引或匹配目标实体。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    normalized = str(action_id or "")
    return normalized.startswith("produce_card:") or normalized.startswith("produce_card_unknown")


def is_produce_drink_action_id(action_id: Any) -> bool:
    """判断 action_id 是否表示使用饮料操作。

    Args:
        action_id: 业务对象标识符，用于索引或匹配目标实体。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    normalized = str(action_id or "")
    return normalized.startswith("produce_drink:") or normalized.startswith("produce_drink_unknown")


def is_end_turn_action_id(action_id: Any) -> bool:
    """判断 action_id 是否表示回合结束操作。

    Args:
        action_id: 业务对象标识符，用于索引或匹配目标实体。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    return str(action_id or "").strip() == "end_turn"


def _serialize_box(box: Any) -> list[int] | None:
    """序列化`serialize_box`。"""
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
    """处理serialize、候选项并返回结果。

    Args:
        candidate: 单个候选项对象。
        phase: 当前 gameplay 阶段标识。

    Returns:
        dict: 结构化结果字典。
    """
    title = getattr(candidate, "title", "") or getattr(candidate, "label", "") or getattr(candidate, "kind", "")
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    payload = {
        "index": int(getattr(candidate, "index", 0)),
        "id": getattr(candidate, "action_id", "") or f"{phase}:{getattr(candidate, 'index', 0)}",
        "db_id": getattr(candidate, "db_id", "") or "",
        "name": title,
        "type": metadata.get("consult_action") or metadata.get("candidate_type") or phase,
        "label": getattr(candidate, "label", "") or getattr(candidate, "kind", "") or title,
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
    """处理compute、remaining、weeks并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        int | None: 返回值类型见注解。
    """
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
    """构建stage、context并返回结果。

    Args:
        phase: 当前 gameplay 阶段标识。
        position: 当前阶段下的细分画面位置标识。
        hud_state: 用于提供HUD、状态相关输入。
        candidate_payloads: 用于提供候选项、payloads相关输入。

    Returns:
        dict: 结构化结果字典。
    """
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
        elif position_key == GameplayPosition.SCHEDULE_PRESENT_SUPPORT_SHOWCASE:
            stage_id = "schedule_present_support_showcase"
            label = "活动支给结果展示"
            description = "当前是活動支給结果展示页，通常用于展示刚获得的收益或演出结果。"
            available_action_summary = "这类页面通常以点击空白处推进为主，不需要重新选择收益候选。"
            interaction_hint = "确认展示内容后推进到下一页。"
        elif position_key == GameplayPosition.SCHEDULE_LESSON_OPTIONS:
            stage_id = "schedule_lesson_options"
            label = "授业课程选择"
            description = "当前已进入授業分支，需要在ボーカル / ダンス / ビジュアル等课程中选一项。"
            available_action_summary = "可从当前授業候选中选择一项课程；通常先选中，再在下一帧确认进入。"
            interaction_hint = "授業候选一般需要先点一次选中，再点一次确认。"
        elif position_key == GameplayPosition.SCHEDULE_LESSON_SELECTED:
            stage_id = "schedule_lesson_confirm"
            label = "授业课程确认"
            description = "当前已有授業课程被选中，等待确认进入该课程。"
            available_action_summary = "再次点击当前已选授業课程即可确认进入。"
            interaction_hint = "确认前先检查是否确实选中了目标课程。"
        elif position_key == GameplayPosition.SCHEDULE_SELECTED:
            stage_id = "schedule_action_confirm"
            label = "周行动确认"
            description = "当前已有一个周行动被选中，等待确认进入。"
            available_action_summary = "再次点击当前已选周行动即可确认进入。"
            interaction_hint = "确认前先检查是否确实选中了目标周行动。"
        elif position_key == GameplayPosition.SCHEDULE_RECOMMEND:
            stage_id = "schedule_action_preview"
            label = "周行动预览"
            description = "当前停留在周行动预览态，仍然需要在候选周行动中自行做选择。"
            available_action_summary = "可从候选周行动中选择一项，再进入后续确认。"
            interaction_hint = "预览态本身不是最终动作，仍要按候选项完成选择。"
        elif position_key == GameplayPosition.SCHEDULE_EVENT_DIALOGUE:
            stage_id = "schedule_event_dialogue"
            label = "周事件对话推进"
            description = "当前处于行程内事件文本推进阶段，尚未出现分支选项。"
            available_action_summary = "当前以推进文本为主；若后续出现选项则转为周事件分支选择。"
            interaction_hint = "无分支时点击推进文本。"
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
        description = "当前处于 P 饮料领取/选择阶段，需要决定保留哪一瓶饮料。"
        available_action_summary = "可从当前 P 饮料候选中选择一个；若已有选中饮料，则下一次点击会确认。"
        interaction_hint = "P 饮料通常是先选中，再确认。"
        if position_key == "p_drink_limit":
            stage_id = "p_drink_limit"
            label = "P饮料上限处理"
            description = "当前 P 饮料槽已满，需要决定是放弃新饮料，还是丢弃一瓶旧饮料来保留新饮料。"
            available_action_summary = "可选择放弃新饮料，或丢弃一瓶现有饮料以腾出槽位。"
            interaction_hint = "所持上限页的每个动作都会直接改变保留方案。"
        elif position_key == GameplayPosition.P_DRINK_SELECTED:
            stage_id = "p_drink_confirm"
            label = "P饮料确认"
            description = "当前已有一瓶 P 饮料被选中，等待点击确认领取。"
            available_action_summary = "再次点击当前已选 P 饮料或确认按钮即可完成领取。"
            interaction_hint = "确认前先检查是否确实选中了目标饮料。"
    elif phase_key == GameplayPhase.CONSULT:
        if position_key == GameplayPosition.CONSULT_EXCHANGE:
            stage_id = "consult_exchange"
            label = "咨询兑换"
            description = "当前处于相談兑换页，可执行多个操作后再退出。"
            available_action_summary = "可兑换物品（多次）、打开強化（限1次）、打开削除（限1次）、或退出。"
            interaction_hint = "兑换类候选点选后立即进入下一步；每次操作后会再次询问。"
        elif position_key == GameplayPosition.CONSULT_ENHANCEMENT_PREVIEW:
            pending_mode = ""
            if candidate_payloads:
                candidate_type = str(candidate_payloads[0].get("type") or "")
                if "remove" in candidate_type:
                    pending_mode = "remove"
                elif "enhancement" in candidate_type:
                    pending_mode = "enhancement"
            stage_id = "consult_remove_preview" if pending_mode == "remove" else "consult_enhancement_preview"
            label = "咨询削除预览" if pending_mode == "remove" else "咨询强化预览"
            description = "当前处于相談卡牌预览页，需要先浏览并选定目标卡牌。"
            if pending_mode == "remove":
                available_action_summary = "可从当前可见卡牌中选择一张作为削除目标，选中后进入确认。"
                interaction_hint = "削除目标通常先点卡牌，再进入确认页。"
            else:
                available_action_summary = "可从当前可见卡牌中选择一张作为強化目标，选中后进入确认。"
                interaction_hint = "強化目标通常先点卡牌，再进入确认页。"
        elif position_key == GameplayPosition.CONSULT_ENHANCEMENT_READY:
            pending_mode = ""
            if candidate_payloads:
                candidate_type = str(candidate_payloads[0].get("type") or "")
                if "remove" in candidate_type:
                    pending_mode = "remove"
                elif "enhancement" in candidate_type:
                    pending_mode = "enhancement"
            stage_id = "consult_remove_confirm" if pending_mode == "remove" else "consult_enhancement_confirm"
            label = "咨询削除确认" if pending_mode == "remove" else "咨询强化确认"
            description = "当前已有相談处理目标被选中，等待最终确认。"
            if pending_mode == "remove":
                available_action_summary = "可确认削除当前选中的技能卡。"
                interaction_hint = "确认后会永久移除该卡。"
            else:
                available_action_summary = "可确认強化当前选中的技能卡。"
                interaction_hint = "确认后会直接完成強化。"
        else:
            stage_id = "consult_idle"
            label = "咨询处理中"
            description = "当前仍处于相談流程中，但尚未稳定识别为兑换页或选卡页。"
            available_action_summary = "优先根据当前合法动作判断是继续选卡、确认，还是返回兑换页。"
            interaction_hint = "若画面刚切换，先等待一帧稳定。"
    elif phase_key == GameplayPhase.ITEM_SELECT:
        has_unresolved = any(not str(payload.get("db_id") or "").strip() for payload in candidate_payloads)
        stage_id = "item_select_probe" if has_unresolved else "item_select"
        label = "P物品探查选择" if has_unresolved else "P物品选择"
        description = (
            "当前有未完成识别的 P 物品，应该先探查清楚候选内容再决定领取。"
            if has_unresolved
            else "当前处于 P 物品选择阶段，需要从候选物品中选择一个。"
        )
        available_action_summary = (
            "可先逐个探查未识别物品，再在已识别候选中选择一个；若已有选中物品，则下一次点击会确认。"
            if has_unresolved
            else "可从当前 P 物品候选中选择一个；若已有选中物品，则下一次点击会确认。"
        )
        interaction_hint = "P 物品通常是先选中，再确认。"
        if position_key == GameplayPosition.ITEM_SELECT_SELECTED:
            stage_id = "item_confirm"
            label = "P物品确认"
            description = "当前已有 P 物品被选中，等待确认领取。"
            available_action_summary = "点击确认按钮即可领取当前已选 P 物品。"
            interaction_hint = "确认前先检查是否确实选中了目标物品。"
        elif has_unresolved:
            interaction_hint = "探查过程中可能先点物品查看详情，再回到领取决策。"

    candidate_names = [
        payload.get("name") or payload.get("label") or f"动作{payload.get('index', 0)}"
        for payload in candidate_payloads
    ]

    return {
        "id": stage_id,
        "label": label,
        "description": description,
        "available_action_summary": available_action_summary,
        "interaction_hint": interaction_hint,
        "candidate_count": len(candidate_payloads),
        "candidate_names": candidate_names,
        "is_schedule_context": has_progress_hud,
    }


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
_SIM_DESTINATION_HOLD_KEYWORDS = (ProduceText.CARD_HOLD,)
_SIM_DESTINATION_LOST_KEYWORDS = (ProduceText.CARD_EXCLUDE, ProduceText.SKILL_CARD_REMOVE, ProduceText.CARD_ERASE)


def _default_virtual_battle_state() -> dict[str, Any]:
    """构建默认`default_virtual_battle_state`。"""
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
    """获取virtual、battle、状态并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        dict: 结构化结果字典。
    """
    state = ctx.handler_state.get("virtual_battle_state")
    if not isinstance(state, dict) or state.get("version") != 1:
        state = _default_virtual_battle_state()
        ctx.handler_state["virtual_battle_state"] = state
    return state


def _new_virtual_card_instance(state: dict[str, Any], entry: dict[str, Any], *, source: str) -> dict[str, Any]:
    """处理new、virtual、卡牌、instance并返回结果。

    Args:
        state: 用于提供状态相关输入。
        entry: 用于提供entry相关输入。
        source: 用于提供source相关输入。

    Returns:
        dict: 结构化结果字典。
    """
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
    """处理bootstrap、virtual、deck并返回结果。

    Args:
        state: 用于提供状态相关输入。
        known_deck: 用于提供known、deck相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    if state["initialized"]:
        return
    state["zones"]["deck"] = [
        _new_virtual_card_instance(state, entry, source="formation")
        for entry in known_deck
    ]
    state["initialized"] = True


def _normalize_card_identity(card: dict[str, Any]) -> str:
    """规范化`card_identity`。"""
    return str(card.get("id") or card.get("name") or "").strip()


def _find_virtual_card(
    state: dict[str, Any],
    observed: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """查找virtual、卡牌并返回结果。

    Args:
        state: 用于提供状态相关输入。
        observed: 用于提供observed相关输入。

    Returns:
        tuple[str, dict[str, Any]] | None: 返回值类型见注解。
    """
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
    """处理remove、virtual、卡牌、from、all、zones并返回结果。

    Args:
        state: 用于提供状态相关输入。
        instance_key: 用于提供instance、key相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
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
    """同步virtual、hand并返回结果。

    Args:
        state: 用于提供状态相关输入。
        observed_hand: 用于提供observed、hand相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
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
    """提取simulated、delta并返回结果。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。
        keyword: 用于提供keyword相关输入。
        allow_turn_suffix: 用于提供allow、turn、suffix相关输入。

    Returns:
        int: 计算得到的数值结果。
    """
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
    """推断`virtual_destination`。"""
    description = str(card.get("description") or "")
    if any(keyword in description for keyword in _SIM_DESTINATION_HOLD_KEYWORDS):
        return "hold"
    if any(keyword in description for keyword in _SIM_DESTINATION_LOST_KEYWORDS):
        return "lost"
    return "grave"


def _apply_virtual_card_effects(state: dict[str, Any], card: dict[str, Any]) -> None:
    """应用`virtual_card_effects`。"""
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
    """查找卡牌、in、hand、by、operation并返回结果。

    Args:
        state: 用于提供状态相关输入。
        operation: 用于提供operation相关输入。

    Returns:
        dict: 结构化结果字典。
    """
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
    """应用`virtual_operations`。"""
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
    """推进`virtual_turn`流程。"""
    for _ in range(max(int(turns), 0)):
        for key in _SIM_DECAY_KEYS:
            state["resources"][key] = max(int(state["resources"].get(key, 0) or 0) - 1, 0)
        state["turn_index"] = int(state.get("turn_index", 1) or 1) + 1
        state["play_limit_total_current"] = 1
        state["play_limit_remaining"] = 1


def _sync_virtual_turn_boundary(state: dict[str, Any], hud_state: dict[str, Any]) -> None:
    """同步virtual、turn、boundary并返回结果。

    Args:
        state: 用于提供状态相关输入。
        hud_state: 用于提供HUD、状态相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
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
    """合并`realtime_virtual_overrides`。"""
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
    """同步virtual、battle、状态并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        hud_state: 用于提供HUD、状态相关输入。
        known_deck: 用于提供known、deck相关输入。
        observed_hand: 用于提供observed、hand相关输入。

    Returns:
        dict: 结构化结果字典。
    """
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
    """追加`exam_snapshot_details`。"""
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
