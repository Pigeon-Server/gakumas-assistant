"""試験 / オーディション handler。

試験与レッスン共用手牌交互机制: 每回合从手牌中出片，
目标是分数/排名。主要区别:
  - 試験显示排名 vs 对手，而非参数成长
  - 中间試験 (中間試験) / 最终試験 (最終試験)
  - NIA 剧本有オーディション（审核）

当前 YOLO 模型无法可靠区分試験与レッスン —
两者都显示技能卡 + 训练分数/剩余回合。本 handler 在检测到
「exam」阶段时激活（需要后续补充检测逻辑），并回退到 lesson 出牌逻辑。

考试准备页面（审查基准 + 参数加成倍率预览）也由本 handler 处理：
  - 通过 OCR 提取三个参数加成百分比
  - 存入 ctx.handler_state["exam_prep_bonuses"]
  - 点击画面继续进入考试

考试进行中：
  - 每回合提取轮盘队列信息（色段、指针、参数名、加成倍率）
  - 存入 ctx.handler_state["exam_wheel_info"]
  - 供 LLM 决策使用（队列规划 + 爆发回合选择）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.gameplay.exam_prep import (
    extract_exam_prep_bonuses,
    store_exam_prep_bonuses,
)
from src.core.tasks.producer_challenge.gameplay.exam_ranking import (
    extract_exam_ranking,
    store_exam_ranking,
)
from src.core.tasks.producer_challenge.gameplay.exam_wheel import (
    extract_exam_wheel_info,
    store_exam_wheel_info,
)
from src.core.tasks.producer_challenge.gameplay.handler_base import (
    GameplayHandler,
    HandlerResult,
)
from src.core.tasks.producer_challenge.gameplay.lesson import (
    _click_battle_end_turn,
    _try_resolve_empty_hand_action,
    execute_lesson_step,
)
from src.core.tasks.producer_challenge.shared.common import click_relative_point
from src.core.tasks.producer_challenge.gameplay.decision import is_produce_drink_action_id
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


def _try_click_skip(app: "AppProcessor") -> bool:
    """尝试点击試験画面的「スキップ」按钮（所有卡片不可用时跳过回合）。"""
    if _click_battle_end_turn(app):
        logger.info("exam: 点击スキップ按钮跳过回合")
        return True
    return False


class ExamHandler(GameplayHandler):
    """試験 / オーディション画面处理。

    当前委托给 lesson 出牌逻辑，因为 UI 机制相同。
    当試験专用检测可用时，可改用 ctx.exam_strategy。

    考试准备页面 (EXAM_PREP position):
      - 提取参数加成倍率（Vocal/Dance/Visual）
      - 存入上下文供后续考试决策使用
      - 点击画面继续
    """

    phase_tag = "exam"
    priority = 55  # 略高于 lesson，确保可检测时优先处理

    def can_handle(self, app, ctx, phase, position):
        return phase == "exam"

    def handle(self, app, ctx, phase, position):
        # ── 考试准备页面：提取加成倍率后点击继续 ──
        if position == GameplayPosition.EXAM_PREP:
            return self._handle_exam_prep(app, ctx)

        # ── 考试进行中：每回合提取轮盘队列 + 排名信息 ──
        self._update_wheel_info(app, ctx)
        self._update_ranking(app, ctx)

        result = execute_lesson_step(app, ctx, position=position, phase="exam")
        if result is None:
            logger.info("exam: 检测到无手牌（0枚），改为重新决策饮料 / 结束回合")
            empty_hand_result = _try_resolve_empty_hand_action(
                app,
                ctx,
                phase="exam",
                position=position,
            )
            if empty_hand_result is not None:
                return empty_hand_result
            # 仅在 fallback 无法构造候选时，才退回本地点击跳过
            if _try_click_skip(app):
                return HandlerResult.ok("exam: skip (no cards)", sleep_after=1.0)
            return HandlerResult.no_action("exam: no playable cards")
        if result.status == "used":
            if not is_produce_drink_action_id(result.candidate.action_id):
                ctx.lesson_turns_played += 1
            return HandlerResult.ok(f"exam {result.status}", sleep_after=0.5)
        if result.status == "end_turn":
            return HandlerResult.ok("exam: end_turn", sleep_after=0.8)
        if result.status == "all_unplayable":
            # 所有卡片不可用 → 尝试点击スキップ按钮
            if _try_click_skip(app):
                return HandlerResult.ok("exam: skip (all_unplayable)", sleep_after=1.0)
            return HandlerResult.ok("exam: all_unplayable", sleep_after=0.5)
        return HandlerResult.ok(f"exam {result.status}", sleep_after=0.5)

    def _handle_exam_prep(self, app: "AppProcessor", ctx: "ProduceContext") -> HandlerResult:
        """处理考试准备页面：提取加成倍率并点击继续。"""
        frame = getattr(app.latest_results, "frame", None)
        if frame is None:
            logger.warning("[考试准备] 无法获取帧画面")
            click_relative_point(app, x_ratio=0.5, y_ratio=0.5, label="exam_prep_no_frame")
            return HandlerResult.ok("exam_prep: no frame, tap to continue", sleep_after=1.5)

        bonuses = extract_exam_prep_bonuses(frame)
        if bonuses:
            store_exam_prep_bonuses(ctx, bonuses)
            logger.info(
                f"[考试准备] 提取成功: Vocal={bonuses['vocal_bonus_pct']}%, "
                f"Dance={bonuses['dance_bonus_pct']}%, Visual={bonuses['visual_bonus_pct']}%"
            )
        else:
            logger.warning("[考试准备] 加成倍率提取失败，继续推进")

        # 点击画面继续（「タップして次へ」）
        click_relative_point(app, x_ratio=0.5, y_ratio=0.5, label="exam_prep_continue")
        return HandlerResult.ok("exam_prep: extracted bonuses, tap to continue", sleep_after=2.0)

    def _update_wheel_info(self, app: "AppProcessor", ctx: "ProduceContext") -> None:
        """每回合提取轮盘队列信息（色段、指针、参数、加成倍率）。"""
        frame = getattr(app.latest_results, "frame", None)
        yolo_results = getattr(app, "latest_results", None)
        if frame is None:
            return
        try:
            wheel_info = extract_exam_wheel_info(frame, yolo_results=yolo_results)
            if wheel_info and wheel_info.get("queue"):
                store_exam_wheel_info(ctx, wheel_info)
                logger.debug(
                    f"[考试轮盘] 队列={wheel_info['queue']}, "
                    f"剩余={wheel_info.get('remaining_turns')}回合, "
                    f"加成={wheel_info.get('current_bonus_pct')}%, "
                    f"置信度={wheel_info.get('confidence', '?')}"
                )
        except Exception as e:
            logger.warning(f"[考试轮盘] 提取失败: {e}")

    def _update_ranking(self, app: "AppProcessor", ctx: "ProduceContext") -> None:
        """每回合提取玩家排名（YOLO 锚点 + OCR 序号面积推断）。"""
        results = getattr(app, "latest_results", None)
        if results is None or getattr(results, "frame", None) is None:
            return
        try:
            ranking_info = extract_exam_ranking(results.frame, results)
            if ranking_info and ranking_info.get("rank"):
                store_exam_ranking(ctx, ranking_info)
                logger.debug(
                    f"[考试排名] 第{ranking_info['rank']}位, "
                    f"置信度={ranking_info.get('confidence', '?')}"
                )
        except Exception as e:
            logger.warning(f"[考试排名] 提取失败: {e}")
