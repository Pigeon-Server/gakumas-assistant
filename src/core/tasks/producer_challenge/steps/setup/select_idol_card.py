"""Step 4: 选择偶像卡（P アイドル）。

通过「Pアイドル一覧」网格视图选择目标偶像卡。
支持两种模式：
  - 打开网格视图，逐个缩略图 CLIP 快速匹配 + 点击 OCR 匹配目标偶像卡 ID
  - 默认使用当前选中的卡（不配置 ID 时）

遍历过程中自动将 OCR 已识别的缩略图作为 CLIP 变体学习。
选择完毕后点击「次へ」进入支援卡编成。
"""

from time import sleep
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.base_ui.learn_idol_card_clip import (
    _detect_idol_list_thumbnail_boxes,
    _enter_idol_list_page,
    _extract_idol_list_grid_region,
    _IDOL_LIST_MAX_SCROLLS,
    _ocr_match_current_idol_card,
    _ocr_match_grid_selected_card,
    _scroll_idol_list,
    _try_clip_identify,
)
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import wait_frame_stable
from src.utils.game_database_tools import GakumasDatabase_IdolCardDataUtils
from src.utils.logger import logger
from src.utils.opencv_tools import check_frame_change
from src.utils.string_tools import MatchConfig

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.entity.Game.Database.IdolCard import IdolCard
    from src.main import AppProcessor

idol_card_db = GakumasDatabase_IdolCardDataUtils()


class SelectIdolCardStep(ProduceStep):
    """在偶像卡列表中定位目标卡，并处理滑动中的误识别与回退。"""

    step_name = "select_idol_card"
    skip_on_resume = True

    def validate(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        if app.latest_results.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED):
            return True
        idol_labels = (
            BaseUILabels.PRODUCE_CARD_VOCAL,
            BaseUILabels.PRODUCE_CARD_DANCE,
            BaseUILabels.PRODUCE_CARD_VISUAL,
        )
        return any(app.latest_results.exists_label(lbl) for lbl in idol_labels)

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        if ctx.target_idol_card_id:
            self._select_by_list(app, ctx)
        else:
            logger.info("未配置目标偶像卡 ID，使用默认选中卡")
            self._remember_current_selection(app, ctx)

        return self._advance_to_support_selection(app)

    def _select_by_list(self, app: "AppProcessor", ctx: "ProduceContext"):
        """通过「Pアイドル一覧」网格视图匹配目标偶像卡。

        流程：点击「Pアイドル一覧」按钮 → 逐个缩略图点击 + OCR 头部信息
        → 比对目标 ID → 命中则点击「決定」确认；未命中则滚动继续；
        全部遍历后仍未找到则取消返回，使用当前选中卡。
        """
        target_id = ctx.target_idol_card_id
        logger.info(f"目标偶像卡 ID: {target_id}")

        target_card = idol_card_db.get_by_id(target_id)
        if target_card is None:
            logger.warning(f"目标偶像卡 ID '{target_id}' 在主数据库中未找到，使用默认选中卡")
            return

        if not _enter_idol_list_page(app):
            logger.warning("无法打开 Pアイドル一覧 页面，使用默认选中卡")
            return

        found = self._search_idol_list_grid(app, target_id, ctx)

        if found:
            app.game_utils.click_button(
                ButtonText.CONFIRM,
                match_config=MatchConfig(use_fuzz=True, fuzz_threshold=70),
            )
        else:
            logger.warning(
                f"在 Pアイドル一覧 中未找到目标偶像卡 '{target_id}'，使用当前选中卡"
            )
            app.game_utils.back_next_page()

        sleep(0.8)
        app.game_utils.wait_frame_stable(stable_count=2)

        if not found:
            self._remember_current_selection(app, ctx)

    def _search_idol_list_grid(
            self,
            app: "AppProcessor",
            target_id: str,
            ctx: "ProduceContext",
    ) -> bool:
        """遍历 Pアイドル一覧 网格，逐个缩略图 CLIP/OCR 匹配目标卡。

        对每个缩略图：先尝试 CLIP 快速匹配；未命中再点击后 OCR 网格
        上方区域（取最顶部同侧两行：角色名+偶像卡名）；OCR 成功时
        自动将缩略图作为 CLIP 变体学习。
        """
        has_clip = (
            getattr(app, "clip_manager", None) is not None
            and hasattr(app.clip_manager, "idol_card_clip")
        )
        previous_grid: Optional[np.ndarray] = None

        for scroll_index in range(_IDOL_LIST_MAX_SCROLLS):
            frame = getattr(app, "latest_frame", None)
            if frame is None or frame.size == 0:
                break

            grid_boxes = _detect_idol_list_thumbnail_boxes(frame)
            if not grid_boxes:
                logger.warning(f"未检测到网格缩略图 (scroll {scroll_index})")
                break

            page_frame = frame.copy()
            for thumb_box in grid_boxes:
                thumb_image = page_frame[
                    thumb_box.y1:thumb_box.y2, thumb_box.x1:thumb_box.x2
                ].copy()

                # CLIP 快速识别（原始缩略图）
                if has_clip and thumb_image.size > 0:
                    clip_card = _try_clip_identify(app, thumb_image)
                    if clip_card is not None and clip_card.id == target_id:
                        logger.success(
                            f"[CLIP] 在 Pアイドル一覧 中找到目标偶像卡: "
                            f"{clip_card.name} ({clip_card.id})"
                        )
                        app.device.click(thumb_box.cx, thumb_box.cy, "idol-list-thumbnail")
                        sleep(0.35)
                        app.game_utils.wait_frame_stable(stable_count=2)
                        ctx.selected_idol_card = clip_card
                        return True

                app.device.click(thumb_box.cx, thumb_box.cy, "idol-list-thumbnail")
                sleep(0.35)
                app.game_utils.wait_frame_stable(stable_count=2)

                current_card, texts = _ocr_match_grid_selected_card(app)
                if current_card is not None and current_card.id == target_id:
                    logger.success(
                        f"在 Pアイドル一覧 中找到目标偶像卡: "
                        f"{current_card.name} ({current_card.id})"
                    )
                    ctx.selected_idol_card = current_card
                    self._clip_learn_variant(app, has_clip, thumb_image, current_card)
                    return True

                if current_card is not None:
                    logger.debug(
                        f"网格卡: {current_card.name} ({current_card.id})，非目标卡"
                    )
                    self._clip_learn_variant(app, has_clip, thumb_image, current_card)

            current_grid = _extract_idol_list_grid_region(
                getattr(app, "latest_frame", None)
            )
            if previous_grid is not None and check_frame_change(
                previous_grid, current_grid
            ):
                logger.info("已到达 Pアイドル一覧 列表末尾")
                break
            previous_grid = current_grid.copy()

            _scroll_idol_list(app, boxes=grid_boxes)

        return False

    @staticmethod
    def _clip_learn_variant(
            app: "AppProcessor",
            has_clip: bool,
            image: Optional[np.ndarray],
            card: "IdolCard",
    ) -> None:
        """将缩略图作为 CLIP 变体自动学习。"""
        if not has_clip or image is None or image.size == 0:
            return
        try:
            if app.clip_manager.idol_card_clip.add_variant_to_memory(
                image, card, augment=False,
            ):
                logger.debug(f"[CLIP] 自动学习偶像卡变体: {card.id}")
        except Exception as exc:
            logger.debug(f"[CLIP] 自动学习偶像卡变体失败: {exc}")

    def _remember_current_selection(self, app: "AppProcessor", ctx: "ProduceContext"):
        """通过 OCR 记录当前选中的偶像卡。"""
        if ctx.selected_idol_card is not None:
            return
        matched_card, matched_texts = _ocr_match_current_idol_card(app)
        if matched_card is not None:
            ctx.selected_idol_card = matched_card
            logger.info(f"记录当前偶像卡: {matched_card.name} ({matched_card.id})")
            return
        if matched_texts:
            logger.debug(f"记录当前偶像卡失败，ocr_texts={matched_texts}")

    def _advance_to_support_selection(self, app: "AppProcessor") -> bool:
        """点击「次へ」并等待进入支援卡编成页。"""
        app.game_utils.click_button(
            ButtonText.NEXT,
            match_config=MatchConfig(fuzz_threshold=80),
        )
        app.game_utils.wait_loading()
        return self._wait_for_support_selection_page(app)

    @staticmethod
    def _wait_for_support_selection_page(app: "AppProcessor") -> bool:
        """等待支援卡编成页稳定出现。

        真机上进入支援卡编成页后，画面里仍可能残留一些 Produce Card /
        Skill Card 类缩略图，因此不能再把“偶像卡标签全部消失”当成硬条件。

        实际可依赖的稳定信号是：
        - 支援卡槽位已被 YOLO 识别为 `Support Card`
        - 或页面存在空槽 `Blank Slot`
        """
        for _ in range(15):
            has_support_slot = (
                app.latest_results.exists_label(BaseUILabels.SUPPORT_CARD)
                or app.latest_results.exists_label(BaseUILabels.BLANK_SLOT)
            )
            if has_support_slot:
                wait_frame_stable(app, timeout=2.5)
                logger.debug("成功进入支援卡编成页")
                return True
            sleep(1)
        raise TimeoutError("等待支援卡编成页超时")
