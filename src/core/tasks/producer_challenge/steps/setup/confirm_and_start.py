"""Step 7: 确认并点击「プロデュース開始」。

此页面显示完整的编成详情，底部有「プロデュース開始」按钮。
点击前需要：
  1. 处理可能存在的弹窗（如「レンタル可能」）
  2. 根据用户配置決定是否使用加成道具（編成詳細按钮上方的两个加成道具）
"""

from time import sleep
from typing import TYPE_CHECKING

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    click_modal_action_with_retry,
    is_final_confirm_page,
    wait_for_final_confirm_page,
)
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig, string_match

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


class ConfirmAndStartStep(ProduceStep):
    """处理开跑前残留弹窗与加成道具，然后正式启动培育。"""

    step_name = "confirm_and_start"
    skip_on_resume = True

    def validate(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        return is_final_confirm_page(app)

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        # 处理可能残留的弹窗
        self._handle_pending_modals(app, ctx)

        # 处理加成道具
        self._handle_boost_items(app, ctx)

        # 点击「プロデュース開始」
        # OCR 可能读到后缀如 "B 40→25"（AP 消耗），所以用模糊匹配
        app.game_utils.click_button(
            ButtonText.PRODUCE_START,
            match_config=MatchConfig(fuzz_threshold=65),
        )
        sleep(1)

        logger.success("已点击「プロデュース開始」按钮")
        return True

    @staticmethod
    def _handle_boost_items(app: "AppProcessor", ctx: "ProduceContext"):
        """处理加成道具（編成詳細按钮上方）。

        開始確認页面在「プロデュース開始」按钮和「編成詳細」按钮之间
        有两个加成道具槽位（SPECIAL_ITEMS）。
        如果用户配置 use_boost_items=True，则逐个点击并确认选择。
        如果 use_boost_items=False（默认），确保道具未被选中。
        """
        if not ctx.use_boost_items:
            logger.debug("用户配置不使用加成道具，跳过")
            return

        # 检测加成道具槽位
        special_items = app.latest_results.filter_by_label(BaseUILabels.SPECIAL_ITEMS)
        if not special_items:
            logger.debug("未检测到加成道具槽位")
            return

        logger.info(f"检测到 {len(special_items)} 个加成道具槽位，尝试激活")

        for idx, item_box in enumerate(special_items):
            # 点击道具槽位
            app.game_utils.click_element_and_wait_trigger(item_box, retries=2, timeout=2.0)
            sleep(1)

            # 在弹窗中选择道具（通常点击第一个可用道具后确认）
            modal = app.game_utils.try_get_modal(no_body=True)
            if modal:
                if not click_modal_action_with_retry(app, modal, action_name=f"boost item {idx + 1} confirm"):
                    raise TimeoutError(f"加成道具槽位 {idx + 1} 的确认弹窗未能关闭")
                sleep(0.5)
            else:
                # 如果没有弹窗，可能已经选中或没有可用道具
                logger.debug(f"道具槽位 {idx + 1}: 无弹窗响应（可能已选中或无可用道具）")
            wait_for_final_confirm_page(app, timeout=6.0)

    @staticmethod
    def _handle_pending_modals(app: "AppProcessor", ctx: "ProduceContext"):
        """处理确认页可能出现的弹窗。"""
        modal = app.game_utils.try_get_modal(no_body=True)
        if modal is None:
            return

        # レンタル可能 / レンタル確認弹窗
        if modal.modal_title and string_match(
            modal.modal_title,
            [ModalText.TITLE.RENTAL_AVAILABLE, ModalText.TITLE.RENTAL_CONFIRMATION],
            MatchConfig(fuzz_threshold=70),
        ):
            logger.info(f"确认页检测到レンタル弹窗（{modal.modal_title!r}）")
            if not click_modal_action_with_retry(app, modal, action_name="final confirm rental modal"):
                raise TimeoutError("确认页的レンタル可能弹窗未能关闭")
            return

        # 其他弹窗：尝试关闭
        logger.warning(f"确认页出现未知弹窗: {modal.modal_title!r}")
        if not click_modal_action_with_retry(app, modal, action_name="final confirm modal close"):
            raise TimeoutError(f"确认页弹窗未能关闭: {modal.modal_title!r}")
