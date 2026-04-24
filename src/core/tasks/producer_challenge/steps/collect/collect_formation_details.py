"""Step 6.8: 采集开始前的「編成詳細」三子页签并映射到主数据库。"""

from __future__ import annotations

from time import sleep, time
from typing import TYPE_CHECKING, Any

import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.catalog import (
    match_card_and_item_entries,
    match_memory_abilities,
    match_memory_tags,
    match_support_abilities,
    match_support_card_names,
)
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    click_top_right_action,
    find_button,
    has_button,
    inertial_swipe,
    is_final_confirm_page,
    wait_for_final_confirm_page,
    wait_frame_stable,
)
from src.utils.debug_tools import DebugTools
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig, string_match

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

ocr_service = OCRService()
_debugger = DebugTools()

_SKILL_CARD_LABELS = frozenset({
    BaseUILabels.SKILL_CARD,
    BaseUILabels.SKILL_CARD_ACTIVE,
    BaseUILabels.SKILL_CARD_MENTAL,
    BaseUILabels.SKILL_CARD_TRAP,
})
_ITEM_LABELS = frozenset({BaseUILabels.SPECIAL_ITEMS})
_ALL_CARD_ITEM_LABELS = _SKILL_CARD_LABELS | _ITEM_LABELS
_CLIP_BOX_MIN_SIZE = 40
_CLIP_BOX_OVERLAP_THRESHOLD = 0.6
_SHOWCASE_DRIFT_TOLERANCE = 0.05  # 占画面高度的比例

_TAB_CARD_ITEM = ProduceText.TAB_CARD_ITEM
_TAB_ABILITY = ProduceText.TAB_ABILITY
_TAB_EVENT = ProduceText.TAB_EVENT
_TAB_FUZZ = 65
_SECTION_FUZZ = 70
_MAX_SCROLL_ROUNDS = 10
_NO_PROGRESS_LIMIT = 1          # 滚动后画面未变化1次即停止
_SCROLL_SSIM_THRESHOLD = 0.992
_CARD_SOURCE_HEADERS = {
    "initial_owned": (ProduceText.OWNED_AT_START, ProduceText.OWNED_AT_START_SHORT),
    "earned_during_produce": (ProduceText.EARNED_DURING_PRODUCE,),
}
_PHASE_KEYWORDS = ProduceText.PHASE_KEYWORDS
_FORMATION_NOISE_TEXTS = (
    ProduceText.FORMATION_DETAILS,
    ProduceText.GUIDE,
    ProduceText.TAB_CARD_ITEM,
    ProduceText.TAB_ABILITY,
    ProduceText.TAB_EVENT,
    ProduceText.SKILL_CARD_SWITCH,
)


class CollectFormationDetailsStep(ProduceStep):
    """读取编成详情页中的成员与加成信息，补全开局阵容快照。"""

    step_name = "collect_formation_details"
    skip_on_resume = True

    def validate(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """确认当前仍停留在可打开「編成詳細」的开始确认页。

        本步骤只能从最终确认页进入；若页面已经进入正式 gameplay 或仍停留在
        记忆编成页，后续的「編成詳細」按钮和三子页签结构都不成立。
        """
        return is_final_confirm_page(app)

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """采集「編成詳細」三子页签，并把结果写回上下文。

        处理流程如下：
        1. 打开开始确认页的「編成詳細」浮层。
        2. 依次采集「カード/アイテム」「アビリティ」「イベント」页签文本与视觉信息。
        3. 用数据库匹配结果构造 `ctx.formation_details`，并从中汇总出
           `ctx.memory_attributes` 作为后续决策的记忆摘要。
        4. 关闭浮层并回到开始确认页。

        Returns:
            bool: 采集完成或在无法打开浮层时安全跳过，均返回 True；真正的失败通过
                浮层打开/页面切换超时等异常向上抛出。
        """
        if not self._open_formation_details(app):
            logger.warning("无法打开編成詳細，跳过开始前详情采集")
            return True

        details: dict[str, Any] = {
            "overlay_model": "three-sub-tab",
            "tabs": {},
        }

        if self._click_tab(app, _TAB_CARD_ITEM, tab_index=0):
            clip_entries, card_texts = self._collect_card_items_with_clip_and_scroll(app)
            card_details = self._build_card_item_details(card_texts, clip_entries=clip_entries)
            details["tabs"]["card_item"] = card_details
            details["cards_and_items"] = card_details
            logger.info(
                f"編成詳細-カード/アイテム CLIP={len(clip_entries)}件, "
                f"OCR={len(card_texts)}条文本"
            )
        else:
            logger.warning("切换到カード/アイテム 子页签失败")

        if self._click_tab(app, _TAB_ABILITY, tab_index=1):
            ability_texts = self._collect_tab_with_scroll(app, section_label="formation:ability")
            ability_details = self._build_ability_details(ability_texts, ctx)
            details["tabs"]["ability"] = ability_details
            details["abilities"] = ability_details
            logger.info(f"編成詳細-アビリティ 收集到 {len(ability_texts)} 条文本")
        else:
            logger.warning("切换到アビリティ 子页签失败")

        if self._click_tab(app, _TAB_EVENT, tab_index=2):
            event_texts = self._collect_tab_with_scroll(app, section_label="formation:event")
            matched_cards = match_support_card_names(event_texts)
            support_card_events: list[dict[str, Any]] = []
            for mc in matched_cards:
                card_events = mc.get("metadata", {}).get("events", [])
                support_card_events.append({
                    "id": mc["id"],
                    "name": mc["name"],
                    "name_ja": mc.get("metadata", {}).get("name_ja", ""),
                    "events": card_events,
                })
            event_details: dict[str, Any] = {
                "support_cards": support_card_events,
                "support_card_ids": [sc["id"] for sc in support_card_events],
            }
            details["tabs"]["event"] = event_details
            details["events"] = event_details
            logger.info(
                f"編成詳細-イベント 识别支援卡 {len(support_card_events)} 张: "
                + ", ".join(sc["name_ja"] or sc["name"] for sc in support_card_events)
            )
        else:
            logger.warning("切换到イベント 子页签失败")

        # 用游戏数据库补全 CLIP 未识别的 P-item。
        card_details = details.get("cards_and_items")
        if card_details is not None:
            support_card_ids = (
                details.get("events", {}).get("support_card_ids")
            )
            self._supplement_produce_items_from_db(
                card_details, ctx, support_card_ids=support_card_ids,
            )

        memory_summary_entries = self._build_memory_fallback(
            details.get("cards_and_items", {}),
            details.get("abilities", {}),
            produce_group_id=ctx.produce_group_id,
        )
        details["memory_summary"] = {
            "entries": memory_summary_entries,
            "entry_count": len(memory_summary_entries),
        }
        if memory_summary_entries:
            ctx.memory_attributes = memory_summary_entries
            logger.info(f"从編成詳細汇总出 {len(ctx.memory_attributes)} 条记忆上下文")
        else:
            logger.warning("編成詳細未能汇总出记忆上下文")

        ctx.formation_details = details
        self._close_overlay(app)
        return True

    @staticmethod
    def _open_formation_details(app: "AppProcessor") -> bool:
        """从开始确认页打开「編成詳細」浮层。

        会重复尝试查找并点击「編成詳細」按钮；只有在点击后成功识别到浮层结构
        （关闭按钮 / 返回按钮 / TAB_BAR）时才视为成功，避免把普通按钮闪烁误判为已进入详情页。
        """
        for _ in range(5):
            if button := find_button(app, ProduceText.FORMATION_DETAILS, fuzz_threshold=68):
                if app.game_utils.click_element_and_wait_trigger(button, retries=2, timeout=2.5, interval=0.1):
                    sleep(0.8)
                    if CollectFormationDetailsStep._wait_for_formation_overlay(app, timeout=5.0):
                        return True
            sleep(0.8)
        return False

    @staticmethod
    def _wait_for_formation_overlay(app: "AppProcessor", timeout: float = 6.0) -> bool:
        """等待「編成詳細」浮层稳定出现。

        Args:
            app: 当前应用处理器。
            timeout: 最长等待时间；超过后由调用方决定是否跳过该步骤。

        Returns:
            bool: 在时限内识别到详情浮层并完成稳帧等待时返回 True，否则返回 False。
        """
        end_time = time() + timeout
        while time() < end_time:
            if CollectFormationDetailsStep._is_formation_overlay_page(app):
                wait_frame_stable(app, timeout=2.5)
                return True
            sleep(0.4)
        return False

    @staticmethod
    def _is_formation_overlay_page(app: "AppProcessor") -> bool:
        """判断当前是否已经进入「編成詳細」浮层。

        判定逻辑分两层：
        1. 若仍能看到开始确认页的「プロデュース開始」或「編成詳細」按钮，说明浮层尚未覆盖成功。
        2. 若上述按钮消失，同时出现关闭按钮、返回按钮或 TAB_BAR 之一，则认为已经进入浮层。
        """
        if has_button(app, ButtonText.PRODUCE_START, fuzz_threshold=65):
            return False
        if has_button(app, ProduceText.FORMATION_DETAILS, fuzz_threshold=68):
            return False
        return bool(
            app.latest_results.exists_label(BaseUILabels.CLOSE_BUTTON)
            or app.latest_results.exists_label(BaseUILabels.BACK_BTN)
            or app.latest_results.exists_label(BaseUILabels.TAB_BAR)
        )

    @staticmethod
    def _click_tab(app: "AppProcessor", tab_name: str, tab_index: int) -> bool:
        """切换「編成詳細」浮层中的目标页签。

        优先使用 TAB_BAR 的几何位置直接点击第 `tab_index` 个页签；这样对 OCR 波动更稳。
        若当前机型没有稳定识别出 TAB_BAR，则回退为 OCR 匹配页签文字并点击对应文本中心。

        Args:
            app: 当前应用处理器。
            tab_name: 目标页签文本，用于 OCR 回退匹配。
            tab_index: 目标页签在三联标签中的序号，从 0 开始。

        Returns:
            bool: 点击后完成稳帧等待时返回 True；若既无法定位 TAB_BAR 也无法匹配页签文本，
                返回 False。
        """
        tab_bars = app.latest_results.filter_by_label(BaseUILabels.TAB_BAR)
        if tab_bars:
            tab_bar = tab_bars.first()
            bar_width = tab_bar.w - tab_bar.x
            tab_cy = int((tab_bar.y + tab_bar.h) / 2)
            tab_cx = int(tab_bar.x + bar_width * (2 * tab_index + 1) / 6)
            app.device.click(tab_cx, tab_cy)
            sleep(0.5)
            wait_frame_stable(app, timeout=2.5)
            return True

        frame = app.latest_frame
        if frame is None or frame.size == 0:
            return False

        ocr_result = ocr_service.ocr(frame)
        for result in ocr_result.results:
            if string_match(result.text, tab_name, MatchConfig(fuzz_threshold=_TAB_FUZZ)):
                app.device.click(result.cx, result.cy)
                sleep(0.5)
                wait_frame_stable(app, timeout=2.5)
                return True
        return False

    def _collect_tab_with_scroll(self, app: "AppProcessor", section_label: str) -> list[str]:
        """滚动采集当前页签中的全部 OCR 文本。

        每一轮都会先抽取当前视口中的去重文本，再执行一次短距离惯性滑动。
        滚动后通过裁剪区域的 SSIM 比较判断画面是否还在变化；若相似度过高，说明已经滚到底部，
        立即停止，避免反复采同一屏内容。

        Args:
            app: 当前应用处理器。
            section_label: 调试日志中的章节标记，便于区分是哪个页签滚动到了底部。

        Returns:
            list[str]: 按出现顺序收集到的去重 OCR 文本列表。
        """
        from src.utils.opencv_tools import compute_ssim_score

        all_texts: list[str] = []
        seen: set[str] = set()

        for scroll_round in range(_MAX_SCROLL_ROUNDS):
            frame = app.latest_frame
            if frame is None or frame.size == 0:
                sleep(0.4)
                continue

            for text in self._extract_unique_texts(frame):
                if text in seen:
                    continue
                seen.add(text)
                all_texts.append(text)

            if scroll_round == _MAX_SCROLL_ROUNDS - 1:
                break

            prev_crop = self._crop_scroll_area(frame)
            height, width = frame.shape[:2]
            # 缩短滑动距离（0.70→0.40 替代 0.78→0.28），配合 hold_end 消除惯性
            swipe_start_y = int(height * 0.70)
            swipe_end_y = int(height * 0.40)
            _debugger.add_box(
                width // 2 - 20, swipe_end_y, width // 2 + 20, swipe_start_y,
                label=f"scroll:{scroll_round}", color=(100, 200, 255),
                alpha=0.3, duration=2.0,
            )
            inertial_swipe(
                app,
                width // 2,
                swipe_start_y,
                width // 2,
                swipe_end_y,
                duration=0.45,
                settle_timeout=4.0,
            )

            next_frame = app.latest_frame
            if next_frame is not None and next_frame.size > 0:
                next_crop = self._crop_scroll_area(next_frame)
                if prev_crop.shape == next_crop.shape:
                    if compute_ssim_score(prev_crop, next_crop) > _SCROLL_SSIM_THRESHOLD:
                        # 滚动后画面未变化，说明已到底，立即停止
                        logger.debug(f"[{section_label}] 滚动后画面未变化，已到底")
                        break

        return all_texts

    def _collect_card_items_with_clip_and_scroll(
        self, app: "AppProcessor", section_label: str = "formation:card_item",
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """使用 CLIP 识别与 OCR 文本联合收集卡片/道具条目。

        返回 ``(clip_entries, ocr_texts)``：
        * ``clip_entries`` 为通过 CLIP 视觉匹配到的数据库条目；
        * ``ocr_texts`` 为原始 OCR 文本列表（格式与 ``_collect_tab_with_scroll`` 一致）。
        """
        from src.utils.opencv_tools import compute_ssim_score

        clip_entries: list[dict[str, Any]] = []
        seen_clip_ids: set[str] = set()
        all_texts: list[str] = []
        seen_texts: set[str] = set()
        showcase_center: tuple[float, float] | None = None

        clip_manager = getattr(app, "clip_manager", None)

        for scroll_round in range(_MAX_SCROLL_ROUNDS):
            frame = app.latest_frame
            if frame is None or frame.size == 0:
                sleep(0.4)
                continue

            # ---- 使用 CLIP 识别当前可见卡片/道具图标 ----
            if clip_manager is not None:
                _new_clip, showcase_center = self._clip_identify_visible_cards(
                    app, frame, clip_manager, clip_entries, seen_clip_ids,
                    showcase_center,
                )

            # ---- 采集 OCR 文本（与 _collect_tab_with_scroll 相同）----
            for text in self._extract_unique_texts(frame):
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                all_texts.append(text)

            if scroll_round == _MAX_SCROLL_ROUNDS - 1:
                break

            # ---- 执行滚动 ----
            prev_crop = self._crop_scroll_area(frame)
            height, width = frame.shape[:2]
            swipe_start_y = int(height * 0.70)
            swipe_end_y = int(height * 0.40)
            _debugger.add_box(
                width // 2 - 20, swipe_end_y, width // 2 + 20, swipe_start_y,
                label=f"clip-scroll:{scroll_round}", color=(100, 200, 255),
                alpha=0.3, duration=2.0,
            )
            inertial_swipe(
                app, width // 2, swipe_start_y, width // 2, swipe_end_y,
                duration=0.45, settle_timeout=4.0,
            )

            next_frame = app.latest_frame
            if next_frame is not None and next_frame.size > 0:
                next_crop = self._crop_scroll_area(next_frame)
                if prev_crop.shape == next_crop.shape:
                    if compute_ssim_score(prev_crop, next_crop) > _SCROLL_SSIM_THRESHOLD:
                        # 滚动后画面未变化，说明已到底，立即停止
                        logger.debug(f"[{section_label}] 滚动后画面未变化，已到底")
                        break

        logger.info(
            f"[{section_label}] CLIP identified {len(clip_entries)} entries, "
            f"OCR collected {len(all_texts)} texts"
        )
        return clip_entries, all_texts

    @staticmethod
    def _clip_identify_visible_cards(
        app: "AppProcessor",
        frame: np.ndarray,
        clip_manager: Any,
        clip_entries: list[dict[str, Any]],
        seen_clip_ids: set[str],
        showcase_center: tuple[float, float] | None = None,
    ) -> tuple[bool, tuple[float, float] | None]:
        """对可见卡片/道具图标执行 YOLO 检测与 CLIP 检索。

        当 CLIP 检索未命中时，会回退到卡面 OCR，并触发自动学习，
        使该卡片在后续流程中可被 CLIP 直接识别。

        该函数会原地更新 ``clip_entries`` 与 ``seen_clip_ids``，
        返回 ``(found_new, showcase_center)``：
        * ``found_new`` 表示本轮是否新增识别条目；
        * ``showcase_center`` 为展示卡中心点缓存（未检测到时为 ``None``）。
        """
        results = app.latest_results
        if results is None:
            return False, showcase_center

        card_item_boxes = results.filter_by_labels(list(_ALL_CARD_ITEM_LABELS))
        if not card_item_boxes:
            return False, showcase_center

        h_frame, w_frame = frame.shape[:2]
        found_new = False
        unmatched_cards: list[dict[str, Any]] = []
        drift_tol = h_frame * _SHOWCASE_DRIFT_TOLERANCE

        # 锚点检查：通过上方地址栏与下方按钮判断是否为展示区。
        # 该区域为展示卡，不是网格卡片。
        anchors_above = results.filter_by_labels([BaseUILabels.CURRENT_LOCATION])
        anchors_below = results.filter_by_labels([BaseUILabels.BUTTON])

        for box in card_item_boxes:
            x1 = max(0, int(box.x))
            y1 = max(0, int(box.y))
            x2 = min(w_frame, int(box.w))
            y2 = min(h_frame, int(box.h))
            bw, bh = x2 - x1, y2 - y1
            if bw < _CLIP_BOX_MIN_SIZE or bh < _CLIP_BOX_MIN_SIZE:
                continue

            cx, cy = float(box.cx), float(box.cy)

            # 快速路径：用缓存的展示区位置直接匹配。
            if showcase_center is not None:
                if (abs(cx - showcase_center[0]) < drift_tol
                        and abs(cy - showcase_center[1]) < drift_tol):
                    continue

            # 慢速路径：首次出现时用锚点规则检测。
            if showcase_center is None:
                has_bar_above = any(int(a.h) <= y1 + bh for a in anchors_above)
                has_btn_below = any(int(b.y) >= y2 - bh for b in anchors_below)
                if has_bar_above and has_btn_below:
                    showcase_center = (cx, cy)
                    continue

            card_img = frame[y1:y2, x1:x2]
            if card_img.size == 0:
                continue

            matched = None
            entry_kind: str | None = None
            if box.label in _SKILL_CARD_LABELS:
                matched = clip_manager.skill_card_clip.retrieve(card_img)
                entry_kind = "produce_card"
            elif box.label in _ITEM_LABELS:
                produce_item_clip = getattr(clip_manager, "produce_item_clip", None)
                if produce_item_clip is not None:
                    matched = produce_item_clip.retrieve(card_img)
                entry_kind = "produce_item"

            if matched is not None and hasattr(matched, "id"):
                if matched.id not in seen_clip_ids:
                    seen_clip_ids.add(matched.id)
                    clip_entries.append({
                        "id": matched.id,
                        "name": getattr(matched, "name", ""),
                        "kind": entry_kind,
                        "source": "clip",
                        "yolo_label": box.label,
                        "position": (box.cx, box.cy),
                    })
                    found_new = True
                    _debugger.add_box(
                        x1, y1, x2, y2,
                        label=f"CLIP:{getattr(matched, 'name', '?')[:12]}",
                        color=(0, 255, 100), alpha=0.4, duration=3.0,
                    )
                    logger.debug(
                        f"[CLIP] {entry_kind}: {getattr(matched, 'name', '')} "
                        f"(id={matched.id}) at ({box.cx},{box.cy})"
                    )
            else:
                # --- 缩略图 OCR 回退 ---
                ocr_entry = CollectFormationDetailsStep._ocr_fallback_and_learn(
                    app, card_img, entry_kind, clip_manager,
                )
                if ocr_entry is not None and ocr_entry["id"] not in seen_clip_ids:
                    seen_clip_ids.add(ocr_entry["id"])
                    ocr_entry["yolo_label"] = box.label
                    ocr_entry["position"] = (box.cx, box.cy)
                    clip_entries.append(ocr_entry)
                    found_new = True
                    _debugger.add_box(
                        x1, y1, x2, y2,
                        label=f"OCR→CLIP:{ocr_entry['name'][:10]}",
                        color=(0, 200, 255), alpha=0.4, duration=3.0,
                    )
                else:
                    # 加入“点击后详情 OCR”回退队列。
                    unmatched_cards.append({
                        "box": box,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "card_img": card_img,
                        "entry_kind": entry_kind,
                    })

        # --- 第二轮：点击未匹配技能卡做详情 OCR ---
        # 此处跳过 P-item（详情 OCR 布局不同）；
        # 改为后续由游戏数据库补全。
        for info in unmatched_cards:
            box = info["box"]
            if info["entry_kind"] == "produce_item":
                _debugger.add_box(
                    info["x1"], info["y1"], info["x2"], info["y2"],
                    label="P-item:DB补全",
                    color=(0, 100, 255), alpha=0.25, duration=2.0,
                )
                continue
            detail_entry = CollectFormationDetailsStep._click_and_ocr_card_detail(
                app, box.cx, box.cy, info["entry_kind"],
                info["card_img"], clip_manager,
            )
            if detail_entry is not None and detail_entry["id"] not in seen_clip_ids:
                seen_clip_ids.add(detail_entry["id"])
                detail_entry["yolo_label"] = box.label
                detail_entry["position"] = (box.cx, box.cy)
                clip_entries.append(detail_entry)
                found_new = True
                _debugger.add_box(
                    info["x1"], info["y1"], info["x2"], info["y2"],
                    label=f"Detail:{detail_entry['id'][:14]}",
                    color=(255, 200, 0), alpha=0.4, duration=3.0,
                )
            else:
                _debugger.add_box(
                    info["x1"], info["y1"], info["x2"], info["y2"],
                    label="CLIP:unmatched",
                    color=(0, 100, 255), alpha=0.25, duration=2.0,
                )

        return found_new, showcase_center

    # ------------------------------------------------------------------
    # OCR 回退辅助函数
    # ------------------------------------------------------------------

    @staticmethod
    def _ocr_fallback_and_learn(
        app: "AppProcessor",
        card_img: np.ndarray,
        entry_kind: str | None,
        clip_manager: Any,
    ) -> dict[str, Any] | None:
        """对卡面执行 OCR，与目录匹配后自动学习到 CLIP。

        成功时返回与 ``clip_entries`` 兼容的条目字典，失败时返回 ``None``。
        """
        try:
            ocr_result = ocr_service.ocr(card_img)
        except Exception:  # noqa: BLE001
            return None

        texts = [
            item.text.strip()
            for item in ocr_result.results
            if item.text.strip() and (item.confidence is None or item.confidence >= 0.25)
        ]
        if not texts:
            return None

        catalog_match = match_card_and_item_entries(texts, threshold=72)
        if entry_kind is not None:
            catalog_match = [e for e in catalog_match if e["kind"] == entry_kind]
        if not catalog_match:
            return None

        # 选择得分最高的目录匹配项
        catalog_match.sort(key=lambda e: float(e.get("score") or 0), reverse=True)
        best = catalog_match[0]
        card_id = str(best["id"])
        card_name = str(best.get("name") or card_id)

        # 自动写入 CLIP 记忆库
        CollectFormationDetailsStep._learn_clip_from_ocr(
            app, card_img, card_id, best["kind"], clip_manager,
        )

        logger.info(
            f"[formation] CLIP未命中 → OCR回退成功: {card_name} "
            f"(id={card_id}, kind={best['kind']}, score={best.get('score')})"
        )

        return {
            "id": card_id,
            "name": card_name,
            "kind": best["kind"],
            "source": "ocr_learned",
        }

    @staticmethod
    def _learn_clip_from_ocr(
        app: "AppProcessor",
        card_img: np.ndarray,
        card_id: str,
        kind: str,
        clip_manager: Any,
    ) -> None:
        """将卡面注册到 CLIP 记忆库，后续可直接命中。"""
        try:
            if kind == "produce_card":
                from src.utils.game_database_tools import GakumasDatabase_ProduceCardDataUtils
                payload = GakumasDatabase_ProduceCardDataUtils().get_by_id(f"{card_id}.0")
                if payload is None:
                    return
                skill_card_clip = getattr(clip_manager, "skill_card_clip", None)
                if skill_card_clip is not None:
                    skill_card_clip.add_to_memory(card_img, payload, similarity_threshold=0.98)
                    logger.debug(f"[formation] CLIP自动学习 produce_card: {card_id}")
            elif kind == "produce_item":
                from src.utils.game_database_tools import GakumasDatabase_ProduceItemDataUtils
                payload = GakumasDatabase_ProduceItemDataUtils().get_by_id(str(card_id))
                if payload is None:
                    return
                produce_item_clip = getattr(clip_manager, "produce_item_clip", None)
                if produce_item_clip is not None:
                    produce_item_clip.add_to_memory(card_img, payload, similarity_threshold=0.98)
                    logger.debug(f"[formation] CLIP自动学习 produce_item: {card_id}")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[formation] CLIP自动学习失败 {kind}:{card_id}: {exc}")

    @staticmethod
    def _click_and_ocr_card_detail(
        app: "AppProcessor",
        cx: float,
        cy: float,
        entry_kind: str | None,
        card_img: np.ndarray,
        clip_manager: Any,
    ) -> dict[str, Any] | None:
        """点击卡片缩略图并 OCR 详情区域完成识别。

        After clicking, uses the topmost YOLO-detected card/item icon as an
        anchor (the detail-area card icon), then OCRs the rectangular region
        to its right (where the card name is displayed).  Falls back to a
        fixed region when no YOLO anchor is found.
        """
        app.device.click(cx, cy, "formation-card-detail")
        sleep(0.4)
        app.game_utils.wait_frame_stable(stable_count=1, timeout=2.0)

        frame = app.latest_frame
        results = app.latest_results
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]

        # 查找详情卡片图标：取上半区域最靠上的 YOLO 卡片/道具框。
        # 该区域位于网格上方的详情区。
        anchor_box = None
        top_limit = h * 0.28
        if results is not None:
            detail_boxes = results.filter_by_labels(list(_ALL_CARD_ITEM_LABELS))
            for box in sorted(detail_boxes, key=lambda b: b.y):
                # 要求框的上下边都位于详情区内。
                if box.y < top_limit and box.h < top_limit:
                    anchor_box = box
                    break

        if anchor_box is not None:
            # 对卡片图标右侧区域执行 OCR。
            # w 轴：图标右边界到屏宽 80%。
            # y 轴：图标上边界到下边界。
            ocr_x1 = min(w, int(anchor_box.w))
            ocr_y1 = max(0, int(anchor_box.y))
            ocr_x2 = int(w * 0.80)
            ocr_y2 = min(h, int(anchor_box.h))
        else:
            # 回退：使用固定的详情卡名区域。
            ocr_x1 = int(w * 0.16)
            ocr_y1 = int(h * 0.04)
            ocr_x2 = int(w * 0.65)
            ocr_y2 = int(h * 0.14)

        detail_region = frame[ocr_y1:ocr_y2, ocr_x1:ocr_x2]
        if detail_region.size == 0:
            return None

        _debugger.add_box(
            ocr_x1, ocr_y1, ocr_x2, ocr_y2,
            label="detail-ocr-region",
            color=(255, 255, 0), alpha=0.3, duration=2.0,
        )

        try:
            ocr_result = ocr_service.ocr(detail_region)
        except Exception:  # noqa: BLE001
            return None

        # 按置信度收集文本（从高到低）。
        candidates = [
            (item.text.strip(), item.confidence or 0)
            for item in ocr_result.results
            if item.text.strip() and (item.confidence is None or item.confidence >= 0.4)
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        texts = [c[0] for c in candidates]

        catalog_match = match_card_and_item_entries(texts, threshold=72)
        if entry_kind is not None:
            filtered = [e for e in catalog_match if e["kind"] == entry_kind]
            if filtered:
                catalog_match = filtered
        if not catalog_match:
            logger.debug(f"[formation] 详情OCR文本无法匹配: {texts}")
            return None

        catalog_match.sort(key=lambda e: float(e.get("score") or 0), reverse=True)
        best = catalog_match[0]
        card_id = str(best["id"])

        # 将原始缩略图自动学习进 CLIP。
        CollectFormationDetailsStep._learn_clip_from_ocr(
            app, card_img, card_id, best["kind"], clip_manager,
        )

        logger.info(
            f"[formation] 点击详情OCR成功: {card_id} "
            f"(kind={best['kind']}, score={best.get('score')})"
        )

        return {
            "id": card_id,
            "name": str(best.get("name") or card_id),
            "kind": best["kind"],
            "source": "detail_ocr",
        }

    @staticmethod
    def _build_card_item_details(
        texts: list[str],
        clip_entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """合并 OCR 与 CLIP 识别结果，构造卡片/道具页签的结构化摘要。

        该摘要既保留原始 OCR 文本，也保留数据库匹配条目，并按类型拆出
        `produce_card_ids`、`produce_item_ids`、`produce_drink_ids`，供后续记忆汇总与
        gameplay 决策直接消费。若同一条目同时被 CLIP 和 OCR 识别到，会优先保留 CLIP 结果。
        """
        ocr_matched_entries = match_card_and_item_entries(texts)

        # 合并 CLIP 与 OCR 结果，去重时优先 CLIP。
        if clip_entries:
            clip_ids = {e["id"] for e in clip_entries}
            merged: list[dict[str, Any]] = [
                {
                    "id": e["id"],
                    "name": e.get("name", e["id"]),
                    "kind": e["kind"],
                    "matched_text": f"[CLIP] {e.get('name', e['id'])}",
                    "match_score": 100,
                    "source": e.get("source", "clip"),
                }
                for e in clip_entries
            ]
            for entry in ocr_matched_entries:
                if entry.get("id") not in clip_ids:
                    merged.append(entry)
            matched_entries = merged
        else:
            matched_entries = ocr_matched_entries

        skill_card_summaries = CollectFormationDetailsStep._extract_memory_skill_card_summaries(texts)
        return {
            "raw_texts": texts,
            "matched_entries": matched_entries,
            "clip_entries": clip_entries or [],
            "clip_count": len(clip_entries) if clip_entries else 0,
            "ocr_matched_count": len(ocr_matched_entries),
            "entry_ids": CollectFormationDetailsStep._collect_entry_ids(matched_entries),
            "produce_card_ids": CollectFormationDetailsStep._collect_entry_ids(matched_entries, kind="produce_card"),
            "produce_item_ids": CollectFormationDetailsStep._collect_entry_ids(matched_entries, kind="produce_item"),
            "produce_drink_ids": CollectFormationDetailsStep._collect_entry_ids(matched_entries, kind="produce_drink"),
            "skill_card_summaries": skill_card_summaries,
        }

    @staticmethod
    def _supplement_produce_items_from_db(
        card_details: dict[str, Any],
        ctx: "ProduceContext",
        support_card_ids: list[str] | None = None,
    ) -> None:
        """用数据库推导出的 P-item 补全 produce_item_ids。

        来源：
        1. 偶像卡：``beforeProduceItemId``、``afterProduceItemId``
        2. 支援卡事件：事件奖励的 P-item
        """
        from src.utils.game_database_tools import (
            GakumasDatabase_IdolCardDataUtils,
            GakumasDatabase_ProduceItemDataUtils,
            build_support_card_event_items,
        )

        existing_ids: set[str] = set(card_details.get("produce_item_ids", []))
        added: list[str] = []

        # --- 1. 偶像卡来源 P-item ---
        idol_card_id = ctx.target_idol_card_id
        if idol_card_id:
            idol_db = GakumasDatabase_IdolCardDataUtils()
            idol_card = idol_db.get_by_id(idol_card_id)
            if idol_card is not None:
                for attr in ("beforeProduceItemId", "afterProduceItemId"):
                    pid = getattr(idol_card, attr, None)
                    if pid and pid not in existing_ids:
                        existing_ids.add(pid)
                        added.append(pid)

        # --- 2. 支援卡事件来源 P-item ---
        if support_card_ids:
            event_items = build_support_card_event_items()
            for sc_id in support_card_ids:
                for item_entry in event_items.get(sc_id, []):
                    if item_entry.get("kind") == "item":
                        pid = str(item_entry["id"])
                        if pid not in existing_ids:
                            existing_ids.add(pid)
                            added.append(pid)

        if added:
            # 通过数据库补齐 matched_entries 的名称。
            item_db = GakumasDatabase_ProduceItemDataUtils()
            for pid in added:
                card_details["produce_item_ids"].append(pid)
                item = item_db.get_by_id(pid)
                name = pid
                if item is not None:
                    if item.localization and getattr(item.localization, "name", None):
                        name = item.localization.name
                    elif item.name:
                        name = item.name
                card_details.setdefault("matched_entries", []).append({
                    "id": pid,
                    "name": name,
                    "kind": "produce_item",
                    "matched_text": f"[DB] {name}",
                    "match_score": 100,
                    "source": "db_supplement",
                })
                card_details.setdefault("entry_ids", []).append(pid)
            logger.info(
                f"[formation] DB补全 P物品 {len(added)} 件: "
                + ", ".join(added)
            )

    @staticmethod
    def _build_ability_details(texts: list[str], ctx: "ProduceContext") -> dict[str, Any]:
        """解析「アビリティ」页签文本，并按来源分组映射到数据库。

        该页签会混合出现 Pアイドル 自身能力、授课支援、支援卡能力以及记忆能力。
        函数先按章节标题切段，再分别调用对应目录匹配函数；其中记忆能力匹配会在
        分段失败时回退到全量文本，提高 OCR 噪声下的召回率。

        Args:
            texts: 当前页签滚动收集到的全部 OCR 文本。
            ctx: 培育上下文；主要读取 `produce_group_id`，用于约束记忆能力匹配范围。

        Returns:
            dict[str, Any]: 包含各章节原始文本、匹配条目与条目 ID 的结构化结果。
        """
        sections = {
            "p_idol_abilities": [],
            "lesson_support": [],
            "support_abilities": [],
            "memory_abilities": [],
        }
        current_section: str | None = None

        # 章节标题采用严格匹配：禁用子串回退。
        # 防止诸如“...スキルカードサポート発生率...”这类正文内容被
        # 防止正文被误识别为章节标题。
        _header_cfg = MatchConfig(fuzz_threshold=_SECTION_FUZZ, use_contains=False)

        for text in texts:
            if string_match(text, ProduceText.P_IDOL_ABILITY, _header_cfg):
                current_section = "p_idol_abilities"
                continue
            if string_match(
                text,
                [ProduceText.SKILL_CARD_SUPPORT, ProduceText.LESSON_SUPPORT],
                _header_cfg,
            ):
                current_section = "lesson_support"
                continue
            if string_match(text, ProduceText.SUPPORT_ABILITY, _header_cfg):
                current_section = "support_abilities"
                continue
            if string_match(text, ProduceText.MEMORY_ABILITY, _header_cfg):
                current_section = "memory_abilities"
                continue
            if current_section is not None:
                sections[current_section].append(text)

        memory_match_scope = "section"
        memory_texts = sections["memory_abilities"]
        memory_matches = match_memory_abilities(
            memory_texts,
            produce_group_id=ctx.produce_group_id,
        )
        if not memory_matches:
            memory_matches = match_memory_abilities(
                texts,
                produce_group_id=ctx.produce_group_id,
                threshold=78,
            )
            if memory_matches:
                memory_match_scope = "raw_texts_fallback"
                memory_texts = [match["matched_text"] for match in memory_matches]
        if not memory_texts:
            memory_texts = [
                text
                for text in texts
                if not CollectFormationDetailsStep._is_formation_noise_text(text)
            ]
            if memory_texts:
                memory_match_scope = "raw_texts_unmatched"
        lesson_support_matches = match_support_abilities(sections["lesson_support"])
        support_ability_matches = match_support_abilities(sections["support_abilities"])
        p_idol_matches = match_support_abilities(sections["p_idol_abilities"])

        return {
            "raw_texts": texts,
            "p_idol_abilities": {
                "raw_texts": sections["p_idol_abilities"],
                "matched_entries": p_idol_matches,
                "entry_ids": CollectFormationDetailsStep._collect_entry_ids(p_idol_matches),
            },
            "lesson_support": {
                "raw_texts": sections["lesson_support"],
                "matched_entries": lesson_support_matches,
                "entry_ids": CollectFormationDetailsStep._collect_entry_ids(lesson_support_matches),
            },
            "support_abilities": {
                "raw_texts": sections["support_abilities"],
                "matched_entries": support_ability_matches,
                "entry_ids": CollectFormationDetailsStep._collect_entry_ids(support_ability_matches),
            },
            "memory_abilities": {
                "raw_texts": memory_texts,
                "matched_entries": memory_matches,
                "match_scope": memory_match_scope,
                "entry_ids": CollectFormationDetailsStep._collect_entry_ids(memory_matches),
            },
        }

    @staticmethod
    def _build_memory_fallback(
        card_details: dict[str, Any],
        ability_details: dict[str, Any],
        *,
        produce_group_id: str | None,
    ) -> list[dict[str, Any]]:
        """构建记忆、fallback并返回结果。

        Args:
            card_details: 用于提供卡牌、details相关输入。
            ability_details: 用于提供ability、details相关输入。
            produce_group_id: 业务对象标识符，用于索引或匹配目标实体。

        Returns:
            list: 结果列表，元素类型见返回注解。
        """
        fallback: list[dict[str, Any]] = []
        matched_entries = ability_details.get("memory_abilities", {}).get("matched_entries", [])
        memory_raw_texts = ability_details.get("memory_abilities", {}).get("raw_texts", [])
        skill_card_summaries = card_details.get("skill_card_summaries", [])
        total_entries = max(len(matched_entries), len(skill_card_summaries))
        if total_entries == 0:
            raw_texts = CollectFormationDetailsStep._dedupe_texts(
                [
                    *memory_raw_texts,
                    *[
                        text
                        for summary in skill_card_summaries
                        for text in summary.get("raw_texts", [])
                    ],
                ]
            )
            if not raw_texts:
                return []
            return [
                {
                    "slot_index": 1,
                    "source": "formation-details-summary",
                    "raw_texts": raw_texts,
                    "detail_texts": memory_raw_texts,
                    "stats": {},
                    "tags": match_memory_tags(raw_texts),
                    "abilities": [],
                    "evaluation_candidates": [],
                    "skill_cards": skill_card_summaries,
                    "skill_card_count": len(skill_card_summaries),
                    "gain_timing": next(
                        (summary.get("gain_timing") for summary in skill_card_summaries if summary.get("gain_timing")),
                        None,
                    ),
                    "card_source": next(
                        (summary.get("source_kind") for summary in skill_card_summaries if summary.get("source_kind")),
                        None,
                    ),
                    "card_acquisition_texts": CollectFormationDetailsStep._dedupe_texts(
                        [
                            *[
                                text
                                for summary in skill_card_summaries
                                for text in (
                                    summary.get("phase_texts")
                                    or ([summary.get("source_text")] if summary.get("source_text") else [])
                                )
                            ],
                        ]
                    ),
                    "produce_group_id": produce_group_id,
                }
            ]
        for index in range(total_entries):
            ability_entry = matched_entries[index] if index < len(matched_entries) else None
            skill_summary = skill_card_summaries[index] if index < len(skill_card_summaries) else None
            unmatched_memory_texts = memory_raw_texts if ability_entry is None and index == 0 else []
            tags = match_memory_tags(
                CollectFormationDetailsStep._dedupe_texts(
                    [
                        *((skill_summary or {}).get("phase_texts", []) or []),
                        *((skill_summary or {}).get("raw_texts", []) or []),
                        *unmatched_memory_texts,
                        (ability_entry or {}).get("matched_text", ""),
                    ]
                )
            )
            evaluation_candidates = sorted(
                {
                    candidate["evaluation"]
                    for candidate in (ability_entry or {}).get("metadata", {}).get("candidates", [])
                }
            )
            raw_texts = CollectFormationDetailsStep._dedupe_texts(
                [
                    *((skill_summary or {}).get("phase_texts", []) or []),
                    *((skill_summary or {}).get("raw_texts", []) or []),
                    *unmatched_memory_texts,
                    (ability_entry or {}).get("matched_text", ""),
                ]
            )
            fallback.append(
                {
                    "slot_index": index + 1,
                    "source": "formation-details-summary",
                    "raw_texts": raw_texts,
                    "detail_texts": [
                        text
                        for text in [((ability_entry or {}).get("matched_text", "")), *unmatched_memory_texts]
                        if text
                    ],
                    "stats": {},
                    "tags": tags,
                    "memory_tag_ids": [tag["id"] for tag in tags],
                    "abilities": [ability_entry] if ability_entry is not None else [],
                    "memory_ability_ids": [ability_entry["id"]] if ability_entry is not None else [],
                    "evaluation_candidates": evaluation_candidates,
                    "skill_cards": [skill_summary] if skill_summary is not None else [],
                    "skill_card_count": 1 if skill_summary is not None else 0,
                    "skill_card_ids": (
                        [
                            skill_summary.get("matched_entry_id")
                            or ((skill_summary.get("matched_entry") or {}).get("id"))
                        ]
                        if skill_summary is not None
                        and (
                            skill_summary.get("matched_entry_id")
                            or (skill_summary.get("matched_entry") or {}).get("id")
                        )
                        else []
                    ),
                    "gain_timing": (skill_summary or {}).get("gain_timing"),
                    "card_source": (skill_summary or {}).get("source_kind"),
                    "card_acquisition_texts": (
                        (skill_summary or {}).get("phase_texts")
                        or ([skill_summary.get("source_text")] if skill_summary and skill_summary.get("source_text") else [])
                    ),
                    "produce_group_id": produce_group_id,
                }
            )
        return fallback

    @staticmethod
    def _extract_memory_skill_card_summaries(texts: list[str]) -> list[dict[str, Any]]:
        """提取记忆、skill、卡牌、summaries并返回结果。

        Args:
            texts: 用于提供texts相关输入。

        Returns:
            list: 结果列表，元素类型见返回注解。
        """
        summaries: list[dict[str, Any]] = []
        current_source_kind: str | None = None
        current_source_text = ""
        pending_phase_texts: list[str] = []
        current_section_raw_texts: list[str] = []
        seen_card_ids: set[str] = set()
        generic_sections: list[dict[str, Any]] = []

        for text in texts:
            text = text.strip()
            if not text or CollectFormationDetailsStep._is_formation_noise_text(text):
                continue

            source_kind = CollectFormationDetailsStep._classify_card_source_header(text)
            if source_kind is not None:
                if current_source_kind is not None and current_section_raw_texts:
                    generic_sections.append(
                        {
                            "source_kind": current_source_kind,
                            "source_text": current_source_text,
                            "phase_texts": pending_phase_texts[:],
                            "raw_texts": CollectFormationDetailsStep._dedupe_texts(current_section_raw_texts),
                        }
                    )
                current_source_kind = source_kind
                current_source_text = text
                pending_phase_texts = []
                current_section_raw_texts = [text]
                continue

            produce_card_matches = [
                entry
                for entry in match_card_and_item_entries([text], threshold=78)
                if entry["kind"] == "produce_card"
            ]
            if produce_card_matches:
                entry = produce_card_matches[0]
                if entry["id"] in seen_card_ids:
                    pending_phase_texts = []
                    current_section_raw_texts = []
                    continue
                seen_card_ids.add(entry["id"])
                phase_texts = pending_phase_texts[:]
                summaries.append(
                    {
                        "page_index": len(summaries) + 1,
                        "total_pages": None,
                        "phase_texts": phase_texts,
                        "title": entry["name"],
                        "raw_texts": CollectFormationDetailsStep._dedupe_texts(
                            [
                                *([current_source_text] if current_source_text else []),
                                *phase_texts,
                                entry["matched_text"],
                            ]
                        ),
                        "effect_texts": [],
                        "matched_entry": entry,
                        "matched_entry_id": entry["id"],
                        "matched_entry_kind": entry["kind"],
                        "db_description": "",
                        "description_match_score": 0.0,
                        "source_kind": current_source_kind,
                        "source_text": current_source_text,
                        "gain_timing": CollectFormationDetailsStep._classify_gain_timing(
                            phase_texts,
                            current_source_kind,
                        ),
                    }
                )
                pending_phase_texts = []
                current_section_raw_texts = []
                continue

            if CollectFormationDetailsStep._is_phase_like_text(text):
                if text not in pending_phase_texts:
                    pending_phase_texts.append(text)
            if current_source_kind is not None:
                current_section_raw_texts.append(text)

        if current_source_kind is not None and current_section_raw_texts:
            generic_sections.append(
                {
                    "source_kind": current_source_kind,
                    "source_text": current_source_text,
                    "phase_texts": pending_phase_texts[:],
                    "raw_texts": CollectFormationDetailsStep._dedupe_texts(current_section_raw_texts),
                }
            )

        generic_summaries: list[dict[str, Any]] = []
        if generic_sections:
            for section in generic_sections:
                generic_summaries.append(
                    {
                        "page_index": len(generic_summaries) + 1,
                        "total_pages": None,
                        "phase_texts": section["phase_texts"],
                        "title": "",
                        "raw_texts": section["raw_texts"],
                        "effect_texts": [],
                        "matched_entry": None,
                        "matched_entry_id": None,
                        "matched_entry_kind": None,
                        "db_description": "",
                        "description_match_score": 0.0,
                        "source_kind": section["source_kind"],
                        "source_text": section["source_text"],
                        "gain_timing": CollectFormationDetailsStep._classify_gain_timing(
                            section["phase_texts"],
                            section["source_kind"],
                        ),
                    }
                )

        if summaries or generic_summaries:
            combined = [*summaries, *generic_summaries]
            total_pages = len(combined)
            for index, summary in enumerate(combined, start=1):
                summary["page_index"] = index
                summary["total_pages"] = total_pages
            return combined

        fallback_summaries: list[dict[str, Any]] = []
        for entry in match_card_and_item_entries(texts):
            if entry["kind"] != "produce_card":
                continue
            fallback_summaries.append(
                {
                    "page_index": len(fallback_summaries) + 1,
                    "total_pages": None,
                    "phase_texts": [],
                    "title": entry["name"],
                    "raw_texts": [entry["matched_text"]],
                    "effect_texts": [],
                    "matched_entry": entry,
                    "matched_entry_id": entry["id"],
                    "matched_entry_kind": entry["kind"],
                    "db_description": "",
                    "description_match_score": 0.0,
                    "source_kind": None,
                    "source_text": "",
                    "gain_timing": None,
                }
            )
        total_pages = len(fallback_summaries)
        for index, summary in enumerate(fallback_summaries, start=1):
            summary["page_index"] = index
            summary["total_pages"] = total_pages
        return fallback_summaries

    @staticmethod
    def _classify_card_source_header(text: str) -> str | None:
        """分类判定`card_source_header`。"""
        for source_kind, candidates in _CARD_SOURCE_HEADERS.items():
            if any(string_match(text, candidate, MatchConfig(fuzz_threshold=68)) for candidate in candidates):
                return source_kind
        return None

    @staticmethod
    def _is_phase_like_text(text: str) -> bool:
        """判断文本是否像阶段标识文本。

        Args:
            text: 待处理文本，通常来源于 OCR 或配置。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        if len(text) < 4:
            return False
        return any(keyword in text for keyword in _PHASE_KEYWORDS)

    @staticmethod
    def _is_formation_noise_text(text: str) -> bool:
        """判断文本是否属于编队识别噪声。

        Args:
            text: 待处理文本，通常来源于 OCR 或配置。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        return any(string_match(text, noise, MatchConfig(fuzz_threshold=70)) for noise in _FORMATION_NOISE_TEXTS)

    @staticmethod
    def _classify_gain_timing(phase_texts: list[str], source_kind: str | None) -> str | None:
        """判定gain、timing并返回结果。

        Args:
            phase_texts: 用于提供phase、texts相关输入。
            source_kind: 用于提供source、kind相关输入。

        Returns:
            str | None: 返回值类型见注解。
        """
        joined = "".join(phase_texts)
        if ProduceText.MID_EXAM in joined or ProduceText.MID_REVIEW in joined:
            if ProduceText.FIRST_AUDITION in joined:
                return "mid_exam_or_first_audition"
            return "mid_exam"
        if ProduceText.FIRST_AUDITION in joined:
            return "first_audition"
        if source_kind == "initial_owned":
            return "initial_owned"
        if source_kind == "earned_during_produce":
            return "earned_during_produce"
        return None

    @staticmethod
    def _dedupe_texts(texts: list[str]) -> list[str]:
        """处理dedupe、texts并返回结果。

        Args:
            texts: 用于提供texts相关输入。

        Returns:
            list: 结果列表，元素类型见返回注解。
        """
        deduped: list[str] = []
        seen: set[str] = set()
        for text in texts:
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    @staticmethod
    def _collect_entry_ids(entries: list[dict[str, Any]], kind: str | None = None) -> list[str]:
        """收集entry、ids并返回结果。

        Args:
            entries: 用于提供entries相关输入。
            kind: 用于提供kind相关输入。

        Returns:
            list: 结果列表，元素类型见返回注解。
        """
        ids: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            if kind is not None and entry.get("kind") != kind:
                continue
            entry_id = entry.get("id")
            if not entry_id or entry_id in seen:
                continue
            seen.add(entry_id)
            ids.append(entry_id)
        return ids

    @staticmethod
    def _extract_unique_texts(frame) -> list[str]:
        """提取unique、texts并返回结果。

        Args:
            frame: 待识别图像帧。

        Returns:
            list: 结果列表，元素类型见返回注解。
        """
        ocr_result = ocr_service.ocr(frame)
        texts: list[str] = []
        seen: set[str] = set()
        for item in ocr_result.results:
            text = item.text.strip()
            if not text or text in seen:
                continue
            if item.confidence is not None and item.confidence < 0.25:
                continue
            seen.add(text)
            texts.append(text)
        return texts

    @staticmethod
    def _crop_scroll_area(frame):
        """裁剪`crop_scroll_area`。"""
        height, width = frame.shape[:2]
        top = int(height * 0.18)
        bottom = int(height * 0.86)
        left = int(width * 0.04)
        right = int(width * 0.96)
        return frame[top:bottom, left:right]

    @staticmethod
    def _close_overlay(app: "AppProcessor") -> None:
        """关闭overlay并返回结果。

        Args:
            app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

        Returns:
            None: 仅产生副作用，不返回业务值。
        """
        close_boxes = app.latest_results.filter_by_label(BaseUILabels.CLOSE_BUTTON)
        if close_boxes:
            app.game_utils.click_element_and_wait_trigger(close_boxes.first(), retries=2, timeout=2.0)
        else:
            back_boxes = app.latest_results.filter_by_label(BaseUILabels.BACK_BTN)
            if back_boxes:
                app.game_utils.click_element_and_wait_trigger(back_boxes.first(), retries=2, timeout=2.0)
            else:
                click_top_right_action(app, timeout=2.0)

        wait_for_final_confirm_page(app, timeout=8.0)
