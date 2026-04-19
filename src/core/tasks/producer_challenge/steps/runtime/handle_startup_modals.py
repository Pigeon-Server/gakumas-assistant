"""Step 10: 处理培育开始后的启动弹窗，并确保真正进入 gameplay。

根据 `producer_plan.md` 的 ADB 实测记录，开始确认页存在一个容易踩坑的真实链路：

1. 第一次点击 `プロデュース開始`
2. 依次进入语音 / 快进 / 跳过设置三个启动弹窗
3. 弹窗关闭后回到开始确认页
4. 第二次点击 `プロデュース開始`
5. 才真正进入 producer gameplay

因此本 Step 不能再假设"一次点击开始 -> 直接进入 gameplay"，
而是要支持：

- 直接进入 gameplay
- 回到开始确认页后，执行一次可重入的二次开始

重要：启动弹窗（ボイス再生確認等）属于 BASE_UI 元素，
必须用 BASE_UI 模型处理，处理完毕后再切换到 PRODUCER 模型等待 gameplay。
"""

from time import sleep
from typing import TYPE_CHECKING

from src.constants.game.producer_gameplay import GameplayPhase
from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.constants.yolo.model_type import YoloModelType
from src.core.tasks.producer_challenge.shared.common import click_relative_point
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    click_modal_action_with_retry,
    detect_gameplay_phase,
    is_final_confirm_page,
    wait_for_final_confirm_page,
)
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


class HandleStartupModalsStep(ProduceStep):
    """处理开局设置弹窗，并把模型切换到 gameplay 可识别状态。"""

    step_name = "handle_startup_modals"

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        ctx.set_phase(GameplayPhase.STARTUP_MODALS)

        # ── 恢复模式：跳过启动弹窗，直接切换模型等待 gameplay ──
        if getattr(ctx, "resumed_from_interrupt", False):
            logger.info("[恢复培育] 跳过启动弹窗处理，直接切换到 PRODUCER 模型")
            self._switch_model(app, YoloModelType.PRODUCER)
            detected_phase = self._wait_for_gameplay_phase(app, ctx, timeout=30)
            if not detected_phase:
                raise TimeoutError("[恢复培育] 切换模型后等待 gameplay 首帧超时")
            ctx.set_phase(detected_phase)
            logger.success(f"[恢复培育] 进入游戏玩法阶段: {detected_phase}")
            return True

        # ── 阶段 1：用 BASE_UI 模型处理启动设置弹窗 ──
        # 第一次点击「プロデュース開始」后，会依次弹出语音/快进/跳过三个设置弹窗。
        # 这些弹窗属于 BASE_UI 元素，必须用 BASE_UI 模型检测和处理。
        logger.info("使用 BASE_UI 模型处理启动设置弹窗...")
        self._dismiss_startup_modals_with_base_ui(app, ctx, timeout=25)

        # ── 阶段 2：判断当前是否回到了开始确认页（需要二次点击开始） ──
        retried_start = False
        if is_final_confirm_page(app):
            logger.info("启动弹窗结束后回到开始确认页，执行二次点击「プロデュース開始」")
            retried_start = True
            ctx.record_operation(
                "confirm_produce_start_again",
                target=ButtonText.PRODUCE_START,
                details={"reason": "returned_to_final_confirm_after_startup_modals"},
            )
            app.game_utils.click_button(
                ButtonText.PRODUCE_START,
                match_config=MatchConfig(fuzz_threshold=65),
            )
            sleep(1.0)
            # 二次点击后可能还有弹窗（少见但需兜底）
            self._dismiss_startup_modals_with_base_ui(app, ctx, timeout=15)

        # ── 阶段 3：切换到 PRODUCER 模型，等待 gameplay 首帧 ──
        self._switch_model(app, YoloModelType.PRODUCER)
        logger.info("等待培育 gameplay 首帧...")
        detected_phase = self._wait_for_gameplay_phase(app, ctx, timeout=30)

        if not detected_phase:
            raise TimeoutError("切换到 PRODUCER 模型后等待 gameplay 首帧超时")

        ctx.set_phase(detected_phase)
        ctx.record_operation(
            "startup_sequence_complete",
            target=str(detected_phase),
            details={"retried_start": retried_start},
        )
        logger.success(f"启动弹窗处理完毕，进入游戏玩法阶段: {detected_phase}")
        return True

    @staticmethod
    def _switch_model(app: "AppProcessor", model_type: str) -> None:
        """切换 YOLO 模型，并给首帧推理留出稳定时间。"""
        logger.info(f"切换 YOLO 模型到 {model_type}")
        app.yolo_engine.load_model(model_type)
        sleep(2.0)

    @staticmethod
    def _dismiss_startup_modals_with_base_ui(
        app: "AppProcessor",
        ctx: "ProduceContext",
        *,
        timeout: int = 25,
    ) -> int:
        """用 BASE_UI 模型处理启动设置弹窗（ボイス再生確認/快进/跳过等）。

        不断轮询弹窗并点击确认，直到连续若干秒没有新弹窗出现。
        返回处理的弹窗数量。
        """
        handled = 0
        no_modal_streak = 0
        max_no_modal = 5  # 连续 5 秒无弹窗则认为启动弹窗序列结束
        elapsed = 0

        while elapsed < timeout:
            modal = app.game_utils.try_get_modal(no_body=True)
            if modal is not None:
                handled += 1
                no_modal_streak = 0
                logger.info(f"处理启动弹窗 {handled}: {modal.modal_title!r}")
                ctx.record_operation(
                    "startup_modal_confirm",
                    target=modal.modal_title or f"startup_modal_{handled}",
                    details={"index": handled},
                )
                if not click_modal_action_with_retry(
                    app, modal,
                    prefer_confirm=True,
                    retries=3,
                    timeout=5.0,
                    action_name=f"startup modal {handled}",
                ):
                    # 兜底：点击右下角确认区域
                    logger.warning(f"启动弹窗 {handled} 按钮点击失败，兜底点击右下确认区域")
                    click_relative_point(
                        app, x_ratio=0.7, y_ratio=0.92,
                        label="startup-modal-confirm-fallback",
                    )
                    sleep(1.0)
                sleep(0.5)
                continue

            no_modal_streak += 1
            if no_modal_streak >= max_no_modal:
                logger.debug(f"连续 {max_no_modal} 秒无启动弹窗，结束弹窗处理阶段")
                break

            sleep(1)
            elapsed += 1

        logger.info(f"启动弹窗处理完毕，共处理 {handled} 个弹窗")
        return handled

    @staticmethod
    def _wait_for_gameplay_phase(
        app: "AppProcessor",
        ctx: "ProduceContext",
        timeout: int = 30,
    ) -> str:
        """在 PRODUCER 模型下等待 gameplay 首帧。

        返回检测到的 GameplayPhase 字符串，超时返回空字符串。
        """
        for wait in range(timeout):
            phase = detect_gameplay_phase(app, ctx)
            if phase not in {
                GameplayPhase.UNKNOWN,
                GameplayPhase.MODAL,
                GameplayPhase.STARTUP_MODALS,
            }:
                return str(phase)

            # 如果检测到 MODAL，可能是 gameplay 中的弹窗，尝试确认后继续
            if phase == GameplayPhase.MODAL:
                modal = app.game_utils.try_get_modal(no_body=True)
                if modal is not None:
                    logger.debug(f"gameplay 等待期间检测到弹窗: {modal.modal_title!r}，尝试确认")
                    click_modal_action_with_retry(
                        app, modal,
                        prefer_confirm=True,
                        retries=2,
                        timeout=3.0,
                        action_name="gameplay-wait modal",
                    )
                    continue

            sleep(1)

        logger.warning(f"等待 gameplay 首帧超时 ({timeout}s)")
        return ""
