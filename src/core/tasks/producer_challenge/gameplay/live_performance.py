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


def _is_landscape(frame) -> bool:
    """フレームが横画面かどうかを判定（width > height × 1.3）。"""
    if frame is None:
        return False
    return frame.shape[1] > frame.shape[0] * 1.3


def _detect_tap_to_start(frame) -> bool:
    """OCR で "TAP TO START" テキストを検出。"""
    if frame is None:
        return False
    text = ocr_text(frame)
    if not text:
        return False
    upper = text.upper()
    for variant in ProduceText.TAP_TO_START_OCR_VARIANTS:
        if variant.upper() in upper:
            return True
    return False


def classify_live_position(frame) -> str:
    """ライブ演出の二級ポジションを判定。

    Returns:
        GameplayPosition の live 系ポジション文字列。
    """
    if frame is None:
        return GameplayPosition.UNKNOWN
    if not _is_landscape(frame):
        # 縦画面に戻った → ライブ終了
        return GameplayPosition.LIVE_FINISHED
    if _detect_tap_to_start(frame):
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
        return phase == GameplayPhase.LIVE_PERFORMANCE

    def handle(self, app, ctx, phase, position):
        frame = app.latest_frame
        if frame is None:
            return HandlerResult.waiting("ライブ: フレーム取得待ち")

        live_pos = classify_live_position(frame)
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
        # 横画面では座標系が回転しているため、中央をタップ
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        app.device.click(cx, cy)
        # タップ後少し待つ（演出開始アニメーション）
        return HandlerResult.ok("ライブ開始タップ", sleep_after=3.0)

    def _wait_performance(self, app, ctx):
        """リズムゲーム実行中 → 自動演出を待つ。"""
        # 連続 unknown カウンタをリセット（ライブ中は unknown 扱いしない）
        ctx.consecutive_unknowns = 0
        elapsed = ctx.handler_state.get("live_wait_count", 0) + 1
        ctx.handler_state["live_wait_count"] = elapsed
        if elapsed % 10 == 0:
            logger.info(f"[ライブ演出] 演出中... ({elapsed} 回待機)")
        return HandlerResult.ok("ライブ演出待ち", sleep_after=3.0)

    def _handle_finished(self, app, ctx):
        """画面が縦に戻った → ライブ終了。"""
        logger.success("[ライブ演出] 終了検出（縦画面に復帰）")
        ctx.handler_state["live_wait_count"] = 0
        # 結果画面への遷移を待つ
        return HandlerResult.ok("ライブ終了", sleep_after=2.0)
