"""Step 2: 选择剧本（初 / NIA）。"""

from typing import TYPE_CHECKING

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import inertial_swipe
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_HAJIME_LABELS = (
    BaseUILabels.PRODUCER_REGULAR,
    BaseUILabels.PRODUCER_PRO,
    BaseUILabels.PRODUCER_MASTER,
)

MAX_SWIPE_ATTEMPTS = 5


def _is_hajime_visible(app: "AppProcessor") -> bool:
    """判断当前剧本轮播页是否已经显示 HAJIME 剧本标签。"""
    return any(app.latest_results.exists_label(lbl) for lbl in _HAJIME_LABELS)


def _is_nia_visible(app: "AppProcessor") -> bool:
    """判断当前剧本轮播页是否已经显示 NIA 剧本标签。"""
    return app.latest_results.exists_label(BaseUILabels.PRODUCER_NIA)


class SelectScenarioStep(ProduceStep):
    """在剧本轮播页切到目标剧本，确保后续难度选择命中正确页签。"""

    step_name = "select_scenario"
    skip_on_resume = True

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """把剧本轮播页切换到上下文指定的目标剧本。

        Args:
            app: 当前应用处理器，用于读取页面尺寸并执行横向滑动。
            ctx: 培育上下文，函数会读取 `ctx.scenario` 决定目标剧本是 `hajime` 还是 `nia`。

        Returns:
            bool: 成功切换并确认目标剧本已经显示时返回 True。

        Raises:
            ValueError: `ctx.scenario` 不是受支持的剧本标识时抛出。
            TimeoutError: 在允许的滑动次数内仍未看到目标剧本标签时抛出。

        Notes:
            HAJIME 通过向右滑动回到第一页，NIA 通过向左滑动切到对应剧本页；
            函数只负责切换剧本，不点击具体难度按钮。
        """

        app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.PRODUCER)

        target = ctx.scenario.lower()
        h, w = app.latest_frame.shape[:2]
        cy = h // 2

        if target == "hajime":
            if _is_hajime_visible(app):
                logger.debug("HAJIME 剧本已显示")
                return True
            for attempt in range(MAX_SWIPE_ATTEMPTS):
                logger.debug(f"尝试向右滑动切换到 HAJIME ({attempt + 1}/{MAX_SWIPE_ATTEMPTS})")
                inertial_swipe(app, w // 4, cy, w * 3 // 4, cy)
                if _is_hajime_visible(app):
                    return True

        elif target == "nia":
            if _is_nia_visible(app):
                logger.debug("NIA 剧本已显示")
                return True
            for attempt in range(MAX_SWIPE_ATTEMPTS):
                logger.debug(f"尝试向左滑动切换到 NIA ({attempt + 1}/{MAX_SWIPE_ATTEMPTS})")
                inertial_swipe(app, w * 3 // 4, cy, w // 4, cy)
                if _is_nia_visible(app):
                    return True

        else:
            raise ValueError(f"未知剧本: {target!r}，支持 'hajime' 或 'nia'")

        raise TimeoutError(f"滑动 {MAX_SWIPE_ATTEMPTS} 次后仍未找到目标剧本: {target}")
