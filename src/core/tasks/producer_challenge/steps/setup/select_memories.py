"""Step 6: 记忆编成（メモリー選択）。

两种模式：
  - auto: 处理「レンタルを使用」复选框 → 点击「おまかせ」→ 确认弹窗「決定」→ 点击「次へ」
  - preset: 选择用户预设编号

点击「次へ」后可能出现「レンタル可能」弹窗，需要处理。
「レンタルを使用」复选框的勾选状态由用户配置 ctx.use_rental 控制。
"""

from time import sleep
from typing import TYPE_CHECKING

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    click_modal_action_with_retry,
    select_preset_by_horizontal_swipe,
    wait_for_final_confirm_page,
    wait_for_memory_selection_page,
    wait_frame_stable,
)
from src.entity.Game.Components.CheckBox import CheckBox
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig, string_match

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


class SelectMemoriesStep(ProduceStep):
    """完成记忆编成与租借选项同步，并等待最终确认页就绪。"""

    step_name = "select_memories"
    skip_on_resume = True

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """完成记忆编成，并把流程推进到开始确认页。

        Args:
            app: 当前应用处理器，用于点击自动编成、处理租借弹窗和等待页面切换。
            ctx: 培育上下文；函数会读取 `memory_mode`、`memory_preset_index` 和 `use_rental`
                决定采用自动编成、预设编组及租借复选框状态。

        Returns:
            bool: 成功处理记忆编成并进入最终确认页时返回 True。

        Raises:
            ValueError: 配置了未知的记忆编成模式时抛出。
        """
        mode = ctx.memory_mode.lower()

        if mode == "auto":
            return self._auto_select(app, ctx)
        elif mode == "preset":
            return self._preset_select(app, ctx)
        else:
            raise ValueError(f"未知记忆编成模式: {mode!r}")

    def _auto_select(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """おまかせ 自动编成——先同步レンタル复选框，再执行自动编成。"""
        # ── 同步“租借を使用”复选框。──
        self._sync_rental_checkbox(app, ctx)

        # 点击“おまかせ”。
        app.game_utils.click_button(
            ButtonText.AUTO_SELECT,
            match_config=MatchConfig(fuzz_threshold=80),
        )
        sleep(1)

        # 等待弹窗 → 点击「決定」
        modal = app.game_utils.wait_for_modal(None, timeout=5, no_body=True)
        if modal:
            if not click_modal_action_with_retry(
                app,
                modal,
                prefer_confirm=True,
                action_name="memory auto-select confirm",
            ):
                raise TimeoutError("记忆自动编成确认弹窗未能关闭")
        else:
            app.game_utils.click_button(
                ButtonText.CONFIRM,
                match_config=MatchConfig(fuzz_threshold=80),
            )
            sleep(0.5)
            if pending_modal := app.game_utils.try_get_modal(no_body=True):
                if not click_modal_action_with_retry(
                    app,
                    pending_modal,
                    prefer_confirm=True,
                    action_name="memory auto-select fallback confirm",
                ):
                    raise TimeoutError("记忆自动编成确认弹窗未能关闭")
        sleep(1)

        # 点击“次へ”。
        app.game_utils.click_button(
            ButtonText.NEXT,
            match_config=MatchConfig(fuzz_threshold=80),
        )
        app.game_utils.wait_loading()

        # 处理「租借可能」弹窗
        self._handle_rental_modal(app, ctx)

        return self._wait_for_confirm_page(app)

    def _preset_select(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """切换到指定记忆预设编组，并确认仍停留在记忆编成页。

        Returns:
            bool: 成功切换到目标预设且页面仍可识别为记忆编成页时返回 True。
        """
        logger.info(f"使用预设记忆编号: {ctx.memory_preset_index}")
        select_preset_by_horizontal_swipe(
            app,
            ctx.memory_preset_index,
            card_labels=(BaseUILabels.MEMORY_CARD,),
            description="记忆编成",
        )
        return wait_for_memory_selection_page(app, timeout=6.0)

    @staticmethod
    def _sync_rental_checkbox(app: "AppProcessor", ctx: "ProduceContext"):
        """
        同步记忆编成页面的「レンタルを使用」复选框。

        检测当前复选框状态（checked/unchecked），
        如果与用户配置 ctx.use_rental 不一致则点击切换。
        """
        checkbox_boxes = app.latest_results.filter_by_label(BaseUILabels.CHECKBOX)
        if not checkbox_boxes:
            logger.debug("记忆编成页未检测到复选框，跳过 rental 同步")
            return

        # 遍历复选框，通过 OCR 识别包含「租借」文本的项
        for box in checkbox_boxes:
            cb = CheckBox(box)
            cb_text = cb.text
            if string_match(cb_text, ProduceText.RENTAL, MatchConfig(fuzz_threshold=60)):
                current_checked = cb.checked
                if current_checked == ctx.use_rental:
                    logger.debug(f"レンタル复选框已符合配置 (checked={current_checked})")
                else:
                    logger.info(f"レンタル复选框需要切换: {current_checked} → {ctx.use_rental}")
                    app.device.click_element(cb)
                    wait_frame_stable(app, timeout=2.0)
                    ctx.has_rental_memory = ctx.use_rental
                return

        # 复选框文本未匹配——尝试基于位置启发式
        # 记忆编成页“租借を使用”通常是页面唯一复选框。
        if len(checkbox_boxes) == 1:
            cb = CheckBox(checkbox_boxes.first())
            current_checked = cb.checked
            if current_checked != ctx.use_rental:
                logger.info(f"唯一复选框（推断为レンタル）切换: {current_checked} → {ctx.use_rental}")
                app.device.click_element(cb)
                wait_frame_stable(app, timeout=2.0)
            ctx.has_rental_memory = ctx.use_rental
        else:
            logger.warning("未识别出レンタル复选框，跳过")

    @staticmethod
    def _handle_rental_modal(app: "AppProcessor", ctx: "ProduceContext"):
        """
        处理「レンタル可能」/「レンタル確認」弹窗。

        该弹窗在记忆编成完成后可能出现，提示有可租赁的支援卡/记忆。
        可能依次弹出多个（先レンタル可能再レンタル確認），逐个确认。
        弹窗可能需要几秒才出现，因此第一轮需要带超时等待。
        """
        # 第一轮：带超时等待弹窗出现（最多等 5 秒）
        modal = None
        deadline = 5.0
        poll = 0.6
        elapsed = 0.0
        while elapsed < deadline:
            sleep(poll)
            elapsed += poll
            modal = app.game_utils.try_get_modal(no_body=True)
            if modal is not None:
                break
        if modal is None:
            logger.debug("未检测到レンタル弹窗（等待超时）")
            return

        # 逐个处理可能连续出现的 rental 弹窗
        for _ in range(3):
            if modal is None:
                logger.debug("后续无更多レンタル弹窗")
                return

            if modal.modal_title and string_match(
                modal.modal_title,
                [ModalText.TITLE.RENTAL_AVAILABLE, ModalText.TITLE.RENTAL_CONFIRMATION],
                MatchConfig(fuzz_threshold=70),
            ):
                logger.info(f"检测到レンタル弹窗（{modal.modal_title!r}），确认")
                ctx.has_rental_memory = True
                if not click_modal_action_with_retry(
                    app,
                    modal,
                    prefer_confirm=True,
                    action_name="memory rental modal",
                ):
                    raise TimeoutError(f"{modal.modal_title!r} 弹窗未能关闭")
                sleep(1)
                modal = app.game_utils.try_get_modal(no_body=True)
            else:
                logger.debug(f"弹窗标题不匹配レンタル: {modal.modal_title!r}")
                return

    @staticmethod
    def _wait_for_confirm_page(app: "AppProcessor") -> bool:
        """等待最终确认页出现，并在成功时返回 True。"""
        if wait_for_final_confirm_page(app, timeout=15.0):
            logger.debug("成功进入最终确认页")
            return True
        raise TimeoutError("等待最终确认页超时")
