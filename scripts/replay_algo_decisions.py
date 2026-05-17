#!/usr/bin/env python3
"""离线回放决策 dump，验证算法策略在历史样本上的输出。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.core.tasks.producer_challenge.gameplay.strategy.algo_strategy import AlgoStrategy


def _load_runtime_types() -> tuple[type["ProduceContext"], type["AlgoStrategy"], Any]:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.core.tasks.producer_challenge.gameplay.strategy.algo_strategy import AlgoStrategy
    from src.utils.game_database_tools import GakumasDatabase_IdolCardDataUtils

    return ProduceContext, AlgoStrategy, GakumasDatabase_IdolCardDataUtils


def _load_record_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            paths.append(path)
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    return unique_paths


def _build_ctx(record: dict[str, Any], idol_card_id: str) -> "ProduceContext":
    ProduceContext, _, IdolCardDb = _load_runtime_types()
    llm_snapshot = dict(record.get("llm_snapshot") or {})
    stage_context = dict(record.get("stage_context") or {})
    ctx = ProduceContext(
        scenario=str(llm_snapshot.get("scenario") or "hajime"),
        difficulty=str(llm_snapshot.get("difficulty") or "regular"),
        target_idol_card_id=idol_card_id,
    )
    ctx.current_week = int(record.get("week") or llm_snapshot.get("week") or 0)
    ctx.consult_remaining_p_points = int(
        llm_snapshot.get("p_point")
        or stage_context.get("p_point")
        or 0
    )
    ctx.hud_stamina = int(llm_snapshot.get("stamina") or 0)
    ctx.hud_max_stamina = int(llm_snapshot.get("max_stamina") or 0)
    ctx.hud_p_point = int(llm_snapshot.get("p_point") or 0)
    ctx.hud_target_score = int(llm_snapshot.get("target") or 0)
    ctx.parameter_state = {
        "vocal": llm_snapshot.get("parameter_stats", {}).get("vocal", 0),
        "dance": llm_snapshot.get("parameter_stats", {}).get("dance", 0),
        "visual": llm_snapshot.get("parameter_stats", {}).get("visual", 0),
        "vocal_max": llm_snapshot.get("parameter_stats", {}).get("vocal_max", 0),
        "dance_max": llm_snapshot.get("parameter_stats", {}).get("dance_max", 0),
        "visual_max": llm_snapshot.get("parameter_stats", {}).get("visual_max", 0),
        "remaining_turns": llm_snapshot.get("remaining", 0),
    }
    ctx.inventory_state = {
        "p_drinks": list(llm_snapshot.get("drinks") or []),
        "p_items": list(llm_snapshot.get("p_items") or []),
    }
    ctx.card_zone_state = {
        "hand": list(llm_snapshot.get("hand") or []),
        "deck": list(llm_snapshot.get("deck_cards") or []),
        "grave": list(llm_snapshot.get("grave_cards") or []),
        "hold": list(llm_snapshot.get("hold_cards") or []),
        "lost": list(llm_snapshot.get("lost_cards") or []),
    }
    idol_card_db = IdolCardDb()
    ctx.selected_idol_card = idol_card_db.get_by_id(idol_card_id)
    if ctx.selected_idol_card is None:
        ctx.selected_idol_card = idol_card_db.get_by_raw_id(idol_card_id)
    if ctx.selected_idol_card is None:
        raise ValueError(f"找不到偶像卡: {idol_card_id}")
    return ctx


def _build_decision_state(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": str(record.get("phase") or ""),
        "position": str(record.get("position") or ""),
        "llm_snapshot": dict(record.get("llm_snapshot") or {}),
        "stage_context": dict(record.get("stage_context") or {}),
        "decision_explanation": dict(record.get("decision_explanation") or {}),
        "candidates": list(record.get("candidates") or []),
        "legal_actions": list(record.get("legal_actions") or []),
        "reason": "offline_replay",
    }


def _resolved_index(record: dict[str, Any]) -> int | None:
    decision = dict(record.get("decision") or {})
    value = decision.get("resolved_index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def replay_records(paths: list[Path], idol_card_id: str, phase_filter: set[str]) -> int:
    _, AlgoStrategy, _ = _load_runtime_types()
    strategy = AlgoStrategy()
    app_stub = SimpleNamespace()
    total = 0
    matched = 0
    skipped = 0
    mismatches: list[str] = []

    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        phase = str(record.get("phase") or "")
        if phase_filter and phase not in phase_filter:
            continue
        decision_state = _build_decision_state(record)
        expected_index = _resolved_index(record)
        if expected_index is None:
            skipped += 1
            continue
        ctx = _build_ctx(record, idol_card_id)
        result = strategy(app_stub, ctx, decision_state["candidates"], decision_state)
        total += 1
        actual_index = result.selected_index if result is not None else None
        if actual_index == expected_index:
            matched += 1
            continue
        mismatches.append(
            f"{path.name}\tphase={phase}\texpected={expected_index}\tactual={actual_index}\treason={getattr(result, 'reason', '')}"
        )

    print(f"总样本: {total}")
    print(f"命中: {matched}")
    print(f"未命中: {total - matched}")
    print(f"跳过(无 resolved_index): {skipped}")
    if total > 0:
        print(f"命中率: {matched / total:.2%}")
    if mismatches:
        print("\n未命中样本:")
        for line in mismatches[:50]:
            print(line)
    return 0 if not mismatches else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="离线回放算法策略决策 dump")
    parser.add_argument("inputs", nargs="+", help="决策 dump 文件或目录")
    parser.add_argument("--idol-card-id", required=True, help="本次回放使用的偶像卡 ID")
    parser.add_argument("--phase", action="append", default=[], help="仅回放指定 phase，可重复传入")
    args = parser.parse_args()

    paths = _load_record_paths(args.inputs)
    if not paths:
        print("没有找到可回放的 JSON 文件", file=sys.stderr)
        return 2
    return replay_records(paths, args.idol_card_id, {str(item) for item in args.phase if str(item).strip()})


if __name__ == "__main__":
    raise SystemExit(main())
