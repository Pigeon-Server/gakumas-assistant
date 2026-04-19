"""Step 5: 支援卡编成（サポート選択）。

两种模式：
  - auto: 点击「おまかせ」→ 确认弹窗「決定」→ 点击「次へ」
  - preset: 按用户预设编号在编组区域横向滑动切换
"""

from time import sleep
from typing import TYPE_CHECKING

from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    click_modal_action_with_retry,
    select_preset_by_horizontal_swipe,
)
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


class SelectSupportCardsStep(ProduceStep):
    """完成支援卡编成，并把流程稳定推进到记忆编成页。"""

    step_name = "select_support_cards"
    skip_on_resume = True

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        mode = ctx.support_card_mode.lower()

        if mode == "auto":
            return self._auto_select(app, ctx)
        elif mode == "preset":
            return self._preset_select(app, ctx)
        else:
            raise ValueError(f"未知支援卡编成模式: {mode!r}")

    def _auto_select(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """おまかせ 自动编成。"""
        # 点击おまかせ
        app.game_utils.click_button(
            ButtonText.AUTO_SELECT,
            match_config=MatchConfig(fuzz_threshold=80),
        )
        sleep(1)

        # 等待弹窗 → 点击「決定」
        modal = app.game_utils.wait_for_modal(None, timeout=5, no_body=True)
        if modal:
            if not click_modal_action_with_retry(app, modal, action_name="support auto-select confirm"):
                raise TimeoutError("支援卡自动编成确认弹窗未能关闭")
        else:
            # 回退：直接点击確定按钮
            app.game_utils.click_button(
                ButtonText.CONFIRM,
                match_config=MatchConfig(fuzz_threshold=80),
            )
            sleep(0.5)
            if pending_modal := app.game_utils.try_get_modal(no_body=True):
                if not click_modal_action_with_retry(app, pending_modal, action_name="support auto-select fallback confirm"):
                    raise TimeoutError("支援卡自动编成确认弹窗未能关闭")
        sleep(1)

        return self._advance_to_memory_selection(app)

    def _advance_to_memory_selection(self, app: "AppProcessor") -> bool:
        # 点击「次へ」
        app.game_utils.click_button(
            ButtonText.NEXT,
            match_config=MatchConfig(fuzz_threshold=80),
        )
        app.game_utils.wait_loading()

        # 等待记忆编成页（等待支援卡标签消失 + 记忆卡标签出现，或空白槽位）
        # 有时页面切换有延迟，等待久一点
        for _ in range(20):
            if app.latest_results.exists_label(BaseUILabels.MEMORY_CARD):
                logger.debug("成功进入记忆编成页")
                return True
            if app.latest_results.exists_label(BaseUILabels.BLANK_SLOT):
                logger.debug("进入记忆编成页（存在空白槽位）")
                return True
            # 如果还在支援卡页面（还有おまかせ和リセット按钮）则尝试等待
            sleep(1)

        raise TimeoutError("等待记忆编成页超时")

    def _preset_select(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """使用预设编号编成。"""
        logger.info(f"使用预设支援卡编号: {ctx.support_card_preset_index}")
        select_preset_by_horizontal_swipe(
            app,
            ctx.support_card_preset_index,
            card_labels=(BaseUILabels.SUPPORT_CARD,),
            description="支援卡编成",
        )
        return self._advance_to_memory_selection(app)
