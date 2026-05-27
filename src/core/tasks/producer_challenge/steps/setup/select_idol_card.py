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
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.game_database_tools import GakumasDatabase_IdolCardDataUtils
from src.utils.logger import logger
from src.utils.opencv_tools import check_frame_change, compute_ssim_score
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
        """确认当前页面已经具备选择偶像卡的基本特征。

        Returns:
            bool: 当前已经识别到选中态偶像卡，或页面上存在 Vocal / Dance / Visual 任意偶像卡标签时返回 True。
            该校验用于防止难度选择尚未完成就提前进入偶像卡步骤。
        """
        if app.latest_results.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED):
            return True
        idol_labels = (
            BaseUILabels.PRODUCE_CARD_VOCAL,
            BaseUILabels.PRODUCE_CARD_DANCE,
            BaseUILabels.PRODUCE_CARD_VISUAL,
        )
        return any(app.latest_results.exists_label(lbl) for lbl in idol_labels)

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """选择目标偶像卡，并推进到支援卡编成页面。

        Args:
            app: 当前应用处理器，用于打开列表、执行点击、OCR 识别和等待页面稳定。
            ctx: 培育上下文；当 `target_idol_card_id` 有值时会尝试精确选卡，
                否则直接记录当前默认选中卡。

        Returns:
            bool: 成功处理偶像卡选择并进入支援卡编成页时返回 True。

        Notes:
            该步骤会把最终确认的偶像卡写入 `ctx.selected_idol_card`，供后续编成信息采集与日志使用。
        """
        app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__IDOL_SELECTION)
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
            flag = False
            for _ in range(5):
                try:
                    app.game_utils.click_button(
                        ButtonText.CONFIRM,
                        match_config=MatchConfig(use_fuzz=True, fuzz_threshold=70),
                    )
                    app.game_utils.wait_location_update(
                        GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__IDOL_SELECTION
                    )
                    flag = True
                    break
                except TimeoutError:
                    sleep(1)
                    continue
            if not flag:
                self._try_use_back_button_exit_idol_select_page(app)
        else:
            logger.warning(
                f"在 Pアイドル一覧 中未找到目标偶像卡 '{target_id}'，使用当前选中卡"
            )
            self._try_use_back_button_exit_idol_select_page(app)

        sleep(0.8)
        app.game_utils.wait_frame_stable(stable_count=2)

        if not found:
            self._remember_current_selection(app, ctx)

    def _try_use_back_button_exit_idol_select_page(self, app: "AppProcessor") -> bool:
        flag = False
        for _ in range(5):
            if not self._leave_idol_list_page(app):
                logger.warning("无法稳定退出 Pアイドル一覧 页面，继续使用当前画面")
                sleep(1)
                continue
            if not app.game_utils.wait_location_update(
                    GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__IDOL_SELECTION):
                sleep(1)
                continue
            flag = True
            break
        return flag

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
        previous_grid: Optional[np.ndarray] = None
        previous_selected_card, _ = _ocr_match_grid_selected_card(app)
        previous_selected_id = previous_selected_card.id if previous_selected_card is not None else None

        for scroll_index in range(_IDOL_LIST_MAX_SCROLLS):
            frame = app.latest_frame
            if frame is None or frame.size == 0:
                break

            grid_boxes = _detect_idol_list_thumbnail_boxes(frame)
            if not grid_boxes:
                logger.warning(f"未检测到网格缩略图 (scroll {scroll_index})")
                break

            grid_boxes = self._dedupe_thumbnail_boxes(grid_boxes)
            page_frame = frame.copy()
            for thumb_box in grid_boxes:
                thumb_image = page_frame[
                    thumb_box.y1:thumb_box.y2, thumb_box.x1:thumb_box.x2
                ].copy()

                # CLIP 快速识别（原始缩略图）
                if thumb_image.size > 0:
                    clip_card = _try_clip_identify(app, thumb_image)
                    # 找不到匹配的偶像卡
                    if clip_card is not None and clip_card.id == target_id:
                        app.device.click(thumb_box.cx, thumb_box.cy, "idol-list-thumbnail")
                        sleep(0.35)
                        app.game_utils.wait_frame_stable(stable_count=2)

                        # CLIP 命中后做短时复核，避免切换动画中的旧帧 OCR 导致误判“点过头”。
                        verified_card = self._confirm_grid_target_selected(app, target_id)
                        if verified_card is not None and verified_card.id == target_id:
                            logger.success(
                                f"[CLIP] 在 Pアイドル一覧 中找到目标偶像卡: "
                                f"{verified_card.name} ({verified_card.id})"
                            )
                            ctx.selected_idol_card = verified_card
                            return True
                        if verified_card is not None:
                            logger.debug(
                                f"[CLIP] 命中后校验未通过，当前为 {verified_card.name} ({verified_card.id})，继续搜索"
                            )
                            previous_selected_id = verified_card.id
                        continue

                before_crop = thumb_image
                app.device.click(thumb_box.cx, thumb_box.cy, "idol-list-thumbnail")
                sleep(0.35)
                app.game_utils.wait_frame_stable(stable_count=2)

                current_card = self._confirm_grid_target_selected(app, target_id)
                texts: list[str] = []
                if current_card is None:
                    current_card, texts = _ocr_match_grid_selected_card(app)
                current_frame = app.latest_frame
                region_changed = False
                if current_frame is not None and current_frame.size > 0:
                    after_crop = current_frame[
                        thumb_box.y1:thumb_box.y2, thumb_box.x1:thumb_box.x2
                    ].copy()
                    if before_crop.size > 0 and after_crop.shape == before_crop.shape:
                        region_changed = compute_ssim_score(before_crop, after_crop) < 0.995

                if current_card is not None and current_card.id == previous_selected_id and not region_changed:
                    continue

                if current_card is not None and current_card.id == target_id:
                    logger.success(
                        f"在 Pアイドル一覧 中找到目标偶像卡: "
                        f"{current_card.name} ({current_card.id})"
                    )
                    ctx.selected_idol_card = current_card
                    self._clip_learn_variant(app, thumb_image, current_card)
                    return True

                if current_card is not None:
                    logger.debug(
                        f"网格卡: {current_card.name} ({current_card.id})，非目标卡"
                    )
                    previous_selected_id = current_card.id
                    self._clip_learn_variant(app, thumb_image, current_card)

            current_grid = _extract_idol_list_grid_region(app.latest_frame)
            if previous_grid is not None and check_frame_change(
                previous_grid, current_grid
            ):
                logger.info("已到达 Pアイドル一覧 列表末尾")
                break
            previous_grid = current_grid.copy()

            _scroll_idol_list(app, boxes=grid_boxes)

        return False

    @staticmethod
    def _dedupe_thumbnail_boxes(boxes):
        """按中心点去重，避免同页重复点击同一缩略图。"""
        deduped = []
        seen = set()
        for box in boxes:
            key = (int(box.cx // 12), int(box.cy // 12))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(box)
        return deduped

    @staticmethod
    def _confirm_grid_target_selected(
            app: "AppProcessor",
            target_id: str,
            attempts: int = 4,
    ) -> Optional["IdolCard"]:
        """
        在网格点选后短时复核目标卡是否已选中。

        说明：
        - 首帧 OCR 可能还是上一张卡的顶部信息；
        - 这里连读几帧，任一帧命中目标即视为成功，防止刚命中就误点下一张。
        """
        latest_card: Optional["IdolCard"] = None
        for _ in range(max(1, attempts)):
            card, _texts = _ocr_match_grid_selected_card(app)
            if card is not None:
                latest_card = card
                if card.id == target_id:
                    return card
            sleep(0.2)
        return latest_card

    @staticmethod
    def _leave_idol_list_page(app: "AppProcessor") -> bool:
        """优先使用列表页自身按钮退出，避免误走通用返回逻辑。"""
        for _ in range(3):
            app.game_utils.wait_frame_stable(stable_count=2)
            buttons = ButtonList(app.latest_results)
            for button in buttons:
                text = (button.text or "").strip()
                if text in {"←", "＜", "<"}:
                    if app.game_utils.click_element_and_wait_trigger(button, retries=2, timeout=2.5, interval=0.1):
                        return True
            for button_text in (ButtonText.CANCEL, ButtonText.CLOSE):
                button = buttons.get_button_by_text(
                    button_text,
                    match_config=MatchConfig(use_fuzz=True, fuzz_threshold=70, use_contains=True),
                )
                if button and app.game_utils.click_element_and_wait_trigger(button, retries=2, timeout=2.5, interval=0.1):
                    return True
            sleep(0.3)
        return False

    @staticmethod
    def _clip_learn_variant(
            app: "AppProcessor",
            image: Optional[np.ndarray],
            card: "IdolCard",
    ) -> None:
        """将缩略图作为 CLIP 变体自动学习。"""
        if image is None or image.size == 0:
            return
        try:
            if app.clip_manager.idol_card_clip.add_variant_to_memory(
                image,
                card,
                similarity_threshold=0.94,
                augment=False,
            ):
                logger.debug(f"[CLIP] 自动学习偶像卡变体: {card.id}")
        except Exception as exc:
            logger.debug(f"[CLIP] 自动学习偶像卡变体失败: {exc}")

    def _remember_current_selection(self, app: "AppProcessor", ctx: "ProduceContext"):
        """处理remember、当前、selection并返回结果。

        Args:
            app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
            ctx: 培育上下文对象，保存跨步骤状态与策略配置。

        Returns:
            返回处理结果，具体类型见返回注解。
        """
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
        """推进to、支援卡、selection并返回结果。

        Args:
            app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        app.game_utils.click_button(
            ButtonText.NEXT,
            match_config=MatchConfig(use_fuzz=True, use_contains=True, fuzz_threshold=70, normalize=True),
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
            if app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__SUPPORT_SELECTION):
                wait_frame_stable(app, timeout=2.5)
                logger.debug("成功进入支援卡编成页")
                return True
            sleep(1)
        raise TimeoutError("等待支援卡编成页超时")
