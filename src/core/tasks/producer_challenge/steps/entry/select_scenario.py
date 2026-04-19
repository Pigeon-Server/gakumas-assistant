"""Step 2: 选择剧本（初 / NIA）。"""

from typing import TYPE_CHECKING

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import inertial_swipe
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
    return any(app.latest_results.exists_label(lbl) for lbl in _HAJIME_LABELS)


def _is_nia_visible(app: "AppProcessor") -> bool:
    return app.latest_results.exists_label(BaseUILabels.PRODUCER_NIA)


class SelectScenarioStep(ProduceStep):
    """在剧本轮播页切到目标剧本，确保后续难度选择命中正确页签。"""

    step_name = "select_scenario"
    skip_on_resume = True

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
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
