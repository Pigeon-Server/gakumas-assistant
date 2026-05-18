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
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.i18n_tools import i18n_text
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
        """按配置完成支援卡编成，并推进到记忆编成页。

        Args:
            app: 当前应用处理器，用于点击自动编成、处理确认弹窗或执行预设横滑。
            ctx: 培育上下文；函数会读取 `support_card_mode` 和 `support_card_preset_index`
                决定采用自动编成还是预设编组。

        Returns:
            bool: 成功完成支援卡编成并稳定进入记忆编成页时返回 True。

        Raises:
            ValueError: 配置了未知的支援卡编成模式时抛出。
        """
        app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__SUPPORT_SELECTION)
        mode = ctx.support_card_mode.lower()

        self._select_mode(app, ctx, mode)
        buttons = ButtonList(app.latest_results)
        flag = False
        for _ in range(5):
            next_button = buttons.get_button_by_text(ButtonText.NEXT)
            if not next_button:
                logger.warning(f"find {ButtonText.NEXT} button not found, try again.....")
                sleep(0.5)
                continue
            if next_button.is_disabled():
                logger.warning(f"find {ButtonText.NEXT} button is disabled, switch to automatic grouping")
                self._auto_select(app, ctx)
                sleep(0.5)
                continue
            flag = True
        if not flag:
            raise RuntimeError(i18n_text("backend.task.supportCardSelectionFailed", fallback="未能完成支援卡选择"))
        self._advance_to_memory_selection(app)
        return True



    def _select_mode(self, app: "AppProcessor", ctx: "ProduceContext", mode) -> bool:
        if mode == "auto":
            self._auto_select(app, ctx)
            return True
        elif mode == "preset":
            self._preset_select(app, ctx)
            return True
        else:
            raise ValueError(i18n_text("backend.task.unknownSupportCardMode", fallback=f"未知支援卡编成模式: {mode!r}", mode=mode))

    def _auto_select(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """执行支援卡自动编成，并关闭过程中出现的确认弹窗。

        Returns:
            bool: 自动编成完成后成功推进到记忆编成页时返回 True。

        Notes:
            该函数优先走标准弹窗流程；若等待弹窗失败，会回退到直接点击确认按钮，
            以兼容不同设备上弹窗检测时机不稳定的情况。
        """
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
                action_name="support auto-select confirm",
            ):
                raise TimeoutError("支援卡自动编成确认弹窗未能关闭")
        else:
            # 回退：直接点击確定按钮
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
                    action_name="support auto-select fallback confirm",
                ):
                    raise TimeoutError("支援卡自动编成确认弹窗未能关闭")
        sleep(1)

        return True

    def _advance_to_memory_selection(self, app: "AppProcessor") -> bool:
        """点击“次へ”并等待页面稳定进入记忆编成页。

        Returns:
            bool: 识别到记忆卡槽位或空白槽位时返回 True。

        Raises:
            TimeoutError: 长时间未出现记忆编成页特征时抛出。
        """
        app.game_utils.click_button(
            ButtonText.NEXT,
            match_config=MatchConfig(fuzz_threshold=80),
        )
        app.game_utils.wait_loading()

        # 等待记忆编成页（等待支援卡标签消失 + 记忆卡标签出现，或空白槽位）
        # 有时页面切换有延迟，等待久一点
        for _ in range(20):
            if app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_SELECTION):
                logger.debug("成功进入记忆编成页")
                return True
            # 如果仍在支援卡页面（仍有“おまかせ”“リセット”按钮）则继续等待。
            sleep(1)

        raise TimeoutError("等待记忆编成页超时")

    def _preset_select(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """切换到指定支援卡预设编组，并推进到记忆编成页。

        Args:
            app: 当前应用处理器，用于执行横向滑动与页面推进。
            ctx: 培育上下文，函数会读取 `support_card_preset_index` 作为目标编组编号。

        Returns:
            bool: 成功切换到目标预设并进入记忆编成页时返回 True。
        """
        logger.info(f"使用预设支援卡编号: {ctx.support_card_preset_index}")
        select_preset_by_horizontal_swipe(
            app,
            ctx.support_card_preset_index,
            card_labels=(BaseUILabels.SUPPORT_CARD,),
            description="支援卡编成",
        )
        return True
