"""ライブ演出（リズムゲーム）処理 handler。

ライブ演出は横画面で実行されるリズムゲームフェーズ。
YOLO モデルは縦画面用にトレーニングされているため、
横画面ではラベルが検出されず、OCR でテキストを識別して操作を判断する。

フロー:
  1. "TAP TO START" 画面 → 画面中央をタップして開始
  2. リズムゲーム実行中 → 自動演出を待つ（操作不要）
  3. 終了 → 画面が縦に戻り、結果画面へ遷移
"""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep, time
from typing import TYPE_CHECKING

from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.game.text.produce_text import ProduceText
from src.core.tasks.producer_challenge.shared.common import (
    click_relative_point,
    ocr_text,
)
from src.core.tasks.producer_challenge.gameplay.handler_base import (
    GameplayHandler,
    HandlerResult,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_LIVE_TAP_PROBE_INTERVAL = 5
_LIVE_TAP_EARLY_PROBE_LIMIT = 3
_LIVE_TAP_ROI_X1_RATIO = 0.18
_LIVE_TAP_ROI_X2_RATIO = 0.82
_LIVE_TAP_ROI_Y1_RATIO = 0.68
_LIVE_TAP_ROI_Y2_RATIO = 0.88
_LIVE_TAP_CONFIRMED_KEY = "live_tap_to_start_confirmed"


@dataclass(frozen=True)
class _TapStartProbeWindow:
    """定义 LIVE 开始提示 OCR 探测窗口。"""

    x1: int
    y1: int
    x2: int
    y2: int


def _is_landscape(frame) -> bool:
    """フレームが横画面かどうかを判定（width > height × 1.3）。"""
    if frame is None:
        return False
    return frame.shape[1] > frame.shape[0] * 1.3


def _build_tap_to_start_probe_window(frame) -> _TapStartProbeWindow | None:
    """构建 LIVE 开始提示的 OCR 探测区域。"""
    if frame is None:
        return None
    height, width = frame.shape[:2]
    x1 = max(0, int(width * _LIVE_TAP_ROI_X1_RATIO))
    x2 = min(width, int(width * _LIVE_TAP_ROI_X2_RATIO))
    y1 = max(0, int(height * _LIVE_TAP_ROI_Y1_RATIO))
    y2 = min(height, int(height * _LIVE_TAP_ROI_Y2_RATIO))
    if x2 <= x1 + 10 or y2 <= y1 + 10:
        return None
    return _TapStartProbeWindow(x1=x1, y1=y1, x2=x2, y2=y2)


def _detect_tap_to_start(frame, *, debugger=None) -> bool:
    """OCR 检测「TAP TO START」提示，仅扫描横屏下方中心区域。"""
    probe_window = _build_tap_to_start_probe_window(frame)
    if probe_window is None:
        return False
    if debugger is not None:
        debugger.add_box(
            probe_window.x1,
            probe_window.y1,
            probe_window.x2,
            probe_window.y2,
            label="live_tap_to_start_roi",
            color=(120, 220, 255),
            alpha=0.1,
            duration=2.5,
            font_size=16,
        )
    crop = frame[probe_window.y1:probe_window.y2, probe_window.x1:probe_window.x2]
    text = ocr_text(crop)
    if not text:
        return False
    upper = text.upper()
    for variant in ProduceText.TAP_TO_START_OCR_VARIANTS:
        if variant.upper() in upper:
            if debugger is not None:
                debugger.add_box(
                    probe_window.x1,
                    probe_window.y1,
                    probe_window.x2,
                    probe_window.y2,
                    label=f"live_tap_to_start:{variant}",
                    color=(120, 255, 160),
                    alpha=0.14,
                    duration=2.5,
                    font_size=16,
                )
            return True
    return False


def _should_probe_tap_to_start(wait_count: int) -> bool:
    """判断当前轮是否需要再次 OCR 探测开始提示。"""
    if wait_count < _LIVE_TAP_EARLY_PROBE_LIMIT:
        return True
    return wait_count % _LIVE_TAP_PROBE_INTERVAL == 0


def classify_live_position(frame, *, should_probe_tap: bool = True, debugger=None) -> str:
    """ライブ演出の二级位置判定。

    Returns:
        GameplayPosition の live 系ポジション文字列。
    """
    if frame is None:
        return GameplayPosition.UNKNOWN
    if not _is_landscape(frame):
        # 恢复竖屏，说明 Live 已结束。
        return GameplayPosition.LIVE_FINISHED
    if should_probe_tap and _detect_tap_to_start(frame, debugger=debugger):
        return GameplayPosition.LIVE_TAP_TO_START
    return GameplayPosition.LIVE_PERFORMING


class LivePerformanceHandler(GameplayHandler):
    """ライブ演出（横画面リズムゲーム）handler。

    優先度 80: RESULT (95) や MODAL (90) より低いが、
    通常 gameplay (50) や ADVANCE (-100) より高い。
    """

    phase_tag = GameplayPhase.LIVE_PERFORMANCE
    priority = 80

    def can_handle(self, app, ctx, phase, position):
        """判断当前画面是否应由该处理器接管。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        return phase == GameplayPhase.LIVE_PERFORMANCE

    def handle(self, app, ctx, phase, position):
        """执行处理器主逻辑并返回处理结果。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            返回执行结果对象，具体类型见函数注解。
        """
        frame = app.latest_frame
        if frame is None:
            return HandlerResult.waiting("ライブ: フレーム取得待ち")

        wait_count = int(ctx.handler_state.get("live_wait_count", 0) or 0)
        tap_confirmed = bool(ctx.handler_state.get(_LIVE_TAP_CONFIRMED_KEY, False))
        should_probe_tap = (not tap_confirmed) and _should_probe_tap_to_start(wait_count)
        live_pos = classify_live_position(
            frame,
            should_probe_tap=should_probe_tap,
            debugger=getattr(app, "debug_tools", None),
        )
        logger.info(f"[ライブ演出] position={live_pos}")

        if live_pos == GameplayPosition.LIVE_TAP_TO_START:
            return self._tap_to_start(app, ctx, frame)
        elif live_pos == GameplayPosition.LIVE_PERFORMING:
            return self._wait_performance(app, ctx)
        elif live_pos == GameplayPosition.LIVE_FINISHED:
            return self._handle_finished(app, ctx)
        else:
            return HandlerResult.waiting("ライブ: 不明な状態", sleep_after=2.0)

    def _tap_to_start(self, app, ctx, frame):
        """「TAP TO START」画面 → 中央タップで開始。"""
        logger.info("[ライブ演出] TAP TO START 検出 → タップして開始")
        ctx.handler_state[_LIVE_TAP_CONFIRMED_KEY] = True
        # 横屏时坐标系旋转，点击屏幕中央。
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        app.device.click(cx, cy)
        # 点击后稍作等待（演出开场动画）。
        return HandlerResult.ok("ライブ開始タップ", sleep_after=3.0)

    def _wait_performance(self, app, ctx):
        """リズムゲーム実行中 → 自動演出を待つ。"""
        # 重置连续 unknown 计数（Live 期间不按 unknown 处理）。
        ctx.consecutive_unknowns = 0
        elapsed = ctx.handler_state.get("live_wait_count", 0) + 1
        ctx.handler_state["live_wait_count"] = elapsed
        # 连续进入演出态后，可视为已经越过开始提示，不再做高频 OCR 探测。
        if elapsed >= _LIVE_TAP_EARLY_PROBE_LIMIT:
            ctx.handler_state[_LIVE_TAP_CONFIRMED_KEY] = True
        if elapsed % 10 == 0:
            logger.info(f"[ライブ演出] 演出中... ({elapsed} 回待機)")
        return HandlerResult.ok("ライブ演出待ち", sleep_after=3.0)

    def _handle_finished(self, app, ctx):
        """处理handle、finished并返回结果。

        Args:
            app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
            ctx: 培育上下文对象，保存跨步骤状态与策略配置。

        Returns:
            返回处理结果，具体类型见返回注解。
        """
        logger.success("[ライブ演出] 終了検出（縦画面に復帰）")
        ctx.handler_state["live_wait_count"] = 0
        ctx.handler_state[_LIVE_TAP_CONFIRMED_KEY] = False
        # 等待切换到结果画面。
        return HandlerResult.ok("ライブ終了", sleep_after=2.0)
