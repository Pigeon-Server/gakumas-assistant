"""Step 6.5: 确认记忆编成后进入开始确认页，记忆属性主采集改走「編成詳細」。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import sleep, time
from typing import TYPE_CHECKING, Any, Iterable

from rapidfuzz import fuzz

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ocr_engine import OCRService, OCR_Result
from src.entity.Game.Components.Button import ButtonList
from src.core.tasks.producer_challenge.catalog import (
    match_card_and_item_entries,
    match_memory_abilities,
    match_memory_tags,
)
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    click_modal_action_with_retry,
    click_top_right_action,
    find_button,
    has_button,
    is_final_confirm_page,
    is_memory_selection_page,
    wait_for_final_confirm_page,
    wait_for_memory_selection_page,
    wait_frame_stable,
)
from src.utils.game_database_tools import GakumasDatabase_ProduceCardDataUtils, _concat_produce_descriptions
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig, fullwidth_to_halfwidth, normalize_ocr_jp, string_match

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

ocr_service = OCRService()

_STAT_LABELS = {
    "vocal": ProduceText.VOCAL_OCR_VARIANTS,
    "dance": ProduceText.DANCE_OCR_VARIANTS,
    "visual": ProduceText.VISUAL_OCR_VARIANTS,
    "stamina": ProduceText.STAMINA_OCR_VARIANTS,
}
_PAGE_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


@dataclass(frozen=True)
class MemorySlotTarget:
    slot_index: int
    cx: int
    cy: int
    synthetic: bool = False


class CollectMemoryAttributesStep(ProduceStep):
    """采集已选记忆的属性信息，补齐后续决策所需的基础上下文。"""

    step_name = "collect_memory_attributes"
    skip_on_resume = True

    def validate(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        return self._get_memory_page_state(app) in {"selection", "candidate_list", "detail", "final_confirm"}

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        if self._get_memory_page_state(app) == "final_confirm":
            logger.debug("记忆属性步骤进入时已处于開始確認页，跳过记忆页过渡")
            return True

        if not self._ensure_memory_selection_page(app, timeout=10.0):
            raise TimeoutError("采集记忆卡前未处于メモリー選択页")

        slot_targets = self._infer_slot_targets(app)
        if not slot_targets:
            logger.warning("记忆编成页未识别到可点击记忆槽位，直接进入开始确认页")
            return self._advance_to_final_confirm(app, ctx)

        ctx.memories = [
            {
                "slot_index": target.slot_index,
                "selected_cx": target.cx,
                "selected_cy": target.cy,
                "synthetic": target.synthetic,
            }
            for target in sorted(slot_targets, key=lambda target: target.slot_index)
        ]
        logger.info(
            "记忆卡属性主采集已切换到開始確認页的「編成詳細」汇总视图，"
            f"当前仅记录 {len(ctx.memories)} 个记忆槽位后进入开始确认页"
        )

        return self._advance_to_final_confirm(app, ctx)

    @staticmethod
    def _infer_slot_targets(app: "AppProcessor") -> list[MemorySlotTarget]:
        frame = app.latest_frame
        if frame is None or frame.size == 0:
            return []

        height = frame.shape[0]
        memory_boxes = [
            box
            for box in app.latest_results.filter_by_label(BaseUILabels.MEMORY_CARD)
            if box.cy >= int(height * 0.58)
        ]
        if not memory_boxes:
            memory_boxes = list(app.latest_results.filter_by_label(BaseUILabels.MEMORY_CARD))
        if not memory_boxes:
            return []

        memory_boxes.sort(key=lambda box: (box.cy, box.cx))
        x_centers = CollectMemoryAttributesStep._cluster_centers((box.cx for box in memory_boxes), tolerance=120)
        y_centers = CollectMemoryAttributesStep._cluster_centers((box.cy for box in memory_boxes), tolerance=120)

        if len(x_centers) == 2 and len(y_centers) == 2 and len(memory_boxes) >= 3:
            slot_map: dict[tuple[int, int], Any] = {}
            for box in memory_boxes:
                key = (
                    CollectMemoryAttributesStep._nearest_center(box.cx, x_centers),
                    CollectMemoryAttributesStep._nearest_center(box.cy, y_centers),
                )
                slot_map[key] = box

            targets: list[MemorySlotTarget] = []
            for row_index, y_center in enumerate(sorted(y_centers)):
                for col_index, x_center in enumerate(sorted(x_centers)):
                    box = slot_map.get((x_center, y_center))
                    targets.append(
                        MemorySlotTarget(
                            slot_index=row_index * 2 + col_index + 1,
                            cx=int(box.cx if box is not None else x_center),
                            cy=int(box.cy if box is not None else y_center),
                            synthetic=box is None,
                        )
                    )
            return targets

        return [
            MemorySlotTarget(
                slot_index=index,
                cx=int(box.cx),
                cy=int(box.cy),
            )
            for index, box in enumerate(memory_boxes[:4], start=1)
        ]

    @staticmethod
    def _cluster_centers(values: Iterable[int], tolerance: int) -> list[int]:
        sorted_values = sorted(int(value) for value in values)
        if not sorted_values:
            return []

        clusters: list[list[int]] = [[sorted_values[0]]]
        for value in sorted_values[1:]:
            if abs(value - clusters[-1][-1]) <= tolerance:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        return [round(sum(cluster) / len(cluster)) for cluster in clusters]

    @staticmethod
    def _nearest_center(value: int, centers: list[int]) -> int:
        return min(centers, key=lambda center: abs(center - value))

    def _get_memory_page_state(self, app: "AppProcessor") -> str:
        if is_memory_selection_page(app):
            return "selection"
        if self._is_memory_detail_page(app):
            return "detail"
        if self._is_memory_candidate_list_page(app):
            return "candidate_list"
        if is_final_confirm_page(app):
            return "final_confirm"
        return "unknown"

    def _wait_for_memory_page_state(
        self,
        app: "AppProcessor",
        allowed_states: set[str],
        *,
        timeout: float,
        settle_timeout: float = 2.5,
    ) -> str | None:
        end_time = time() + timeout
        while time() < end_time:
            state = self._get_memory_page_state(app)
            if state in allowed_states:
                wait_frame_stable(app, timeout=settle_timeout)
                return state
            sleep(0.4)
        return None

    def _dismiss_memory_detail_overlay(self, app: "AppProcessor") -> str | None:
        if self._get_memory_page_state(app) != "detail":
            return self._get_memory_page_state(app)

        cancel_button = find_button(app, ButtonText.CANCEL, fuzz_threshold=60)
        if cancel_button is not None and app.game_utils.click_element_and_wait_trigger(
            cancel_button,
            retries=2,
            timeout=2.5,
            interval=0.1,
        ):
            return self._wait_for_memory_page_state(
                app,
                {"selection", "candidate_list"},
                timeout=5.0,
            )

        close_boxes = app.latest_results.filter_by_label(BaseUILabels.CLOSE_BUTTON)
        if close_boxes and app.game_utils.click_element_and_wait_trigger(
            close_boxes.first(),
            retries=2,
            timeout=2.0,
            interval=0.1,
        ):
            return self._wait_for_memory_page_state(
                app,
                {"selection", "candidate_list"},
                timeout=5.0,
            )

        back_boxes = app.latest_results.filter_by_label(BaseUILabels.BACK_BTN)
        if back_boxes and app.game_utils.click_element_and_wait_trigger(
            back_boxes.first(),
            retries=2,
            timeout=2.0,
            interval=0.1,
        ):
            return self._wait_for_memory_page_state(
                app,
                {"selection", "candidate_list"},
                timeout=5.0,
            )

        if click_top_right_action(app, timeout=2.0):
            return self._wait_for_memory_page_state(
                app,
                {"selection", "candidate_list"},
                timeout=5.0,
            )

        return None

    def _dismiss_memory_candidate_list(self, app: "AppProcessor") -> bool:
        if self._get_memory_page_state(app) == "selection":
            return True
        if self._get_memory_page_state(app) != "candidate_list":
            return False

        close_boxes = app.latest_results.filter_by_label(BaseUILabels.CLOSE_BUTTON)
        if close_boxes and app.game_utils.click_element_and_wait_trigger(
            close_boxes.first(),
            retries=2,
            timeout=2.5,
            interval=0.1,
        ):
            return wait_for_memory_selection_page(app, timeout=6.0)

        back_boxes = app.latest_results.filter_by_label(BaseUILabels.BACK_BTN)
        if back_boxes and app.game_utils.click_element_and_wait_trigger(
            back_boxes.first(),
            retries=2,
            timeout=2.0,
            interval=0.1,
        ):
            return wait_for_memory_selection_page(app, timeout=6.0)

        if click_top_right_action(app, timeout=2.0):
            return wait_for_memory_selection_page(app, timeout=6.0)

        return False

    def _ensure_memory_selection_page(
        self,
        app: "AppProcessor",
        *,
        timeout: float = 10.0,
        recovery_rounds: int = 4,
    ) -> bool:
        if wait_for_memory_selection_page(app, timeout=min(timeout, 1.5)):
            return True

        deadline = time() + timeout
        attempt = 0
        while time() < deadline and attempt < recovery_rounds:
            attempt += 1
            state = self._get_memory_page_state(app)
            if state == "selection":
                wait_frame_stable(app, timeout=2.0)
                return True
            if state == "final_confirm":
                return False

            if state == "detail":
                exit_state = self._dismiss_memory_detail_overlay(app)
                if exit_state == "selection":
                    return True
                if exit_state == "candidate_list" and self._dismiss_memory_candidate_list(app):
                    return True
            elif state == "candidate_list":
                if self._dismiss_memory_candidate_list(app):
                    return True
            else:
                logger.debug(f"记忆页恢复时遇到未知页面状态: {state}")
                sleep(0.6)

            if wait_for_memory_selection_page(app, timeout=1.2):
                return True

        return wait_for_memory_selection_page(app, timeout=1.0)

    def _select_memory_slot(self, app: "AppProcessor", target: MemorySlotTarget) -> bool:
        if not self._ensure_memory_selection_page(app, timeout=8.0):
            logger.warning("记忆槽位切换前未回到メモリー選択页")
            return False

        if target.synthetic:
            logger.debug(f"第 {target.slot_index} 个记忆槽位使用当前已选中项")
            return True

        app.device.click(target.cx, target.cy)
        sleep(0.6)
        wait_frame_stable(app, timeout=2.5)
        if wait_for_memory_selection_page(app, timeout=4.0):
            return True

        logger.warning(f"第 {target.slot_index} 个记忆槽位未能切换到选中状态")
        return False

    def _open_memory_candidate_list(self, app: "AppProcessor") -> bool:
        if not self._ensure_memory_selection_page(app, timeout=8.0):
            return False

        indicator_button = self._find_page_indicator_button(app)
        if indicator_button is None:
            logger.warning("未识别到记忆页的 N/20 入口按钮")
            return False

        if app.game_utils.click_element_and_wait_trigger(indicator_button, retries=2, timeout=2.5, interval=0.1):
            if self._wait_for_memory_candidate_list_page(app, timeout=6.0):
                return True

        app.device.click(indicator_button.cx, indicator_button.cy)
        sleep(0.8)
        return self._wait_for_memory_candidate_list_page(app, timeout=6.0)

    @staticmethod
    def _find_page_indicator_button(app: "AppProcessor"):
        for button in ButtonList(app.latest_results):
            text = getattr(button, "text", "") or ""
            if CollectMemoryAttributesStep._parse_page_indicator(text) is not None:
                return button
        return None

    def _wait_for_memory_candidate_list_page(self, app: "AppProcessor", timeout: float = 6.0) -> bool:
        return self._wait_for_memory_page_state(app, {"candidate_list"}, timeout=timeout) == "candidate_list"

    def _is_memory_candidate_list_page(self, app: "AppProcessor") -> bool:
        if is_memory_selection_page(app) or is_final_confirm_page(app) or self._is_memory_detail_page(app):
            return False

        if (
            app.latest_results.exists_label(BaseUILabels.CLOSE_BUTTON)
            and app.latest_results.exists_label(BaseUILabels.BLANK_SLOT)
        ):
            return True

        texts = self._extract_unique_texts(app.latest_frame)
        return any(
            string_match(text, ProduceText.MEMORY_FORMATION, MatchConfig(fuzz_threshold=65))
            for text in texts
        )

    def _open_current_memory_detail_from_list(self, app: "AppProcessor") -> bool:
        if not self._wait_for_memory_candidate_list_page(app, timeout=2.0):
            logger.warning("打开记忆详情前未进入メモリー編成一覧")
            return False

        hotspots = self._get_memory_detail_hotspots(app)
        for index, (tap_x, tap_y) in enumerate(hotspots, start=1):
            app.device.click(tap_x, tap_y)
            sleep(0.6)
            page_state = self._wait_for_memory_page_state(
                app,
                {"detail", "candidate_list", "selection"},
                timeout=4.0,
            )
            if page_state == "detail":
                return True
            if page_state == "selection":
                logger.warning("点击记忆详情热点后回到了メモリー選択页，尝试重新打开メモリー編成一覧")
                if index >= len(hotspots):
                    return False
                if not self._open_memory_candidate_list(app):
                    return False

        logger.warning("未能从メモリー編成一覧进入所持メモリー详情页")
        return False

    def _get_memory_detail_hotspots(self, app: "AppProcessor") -> list[tuple[int, int]]:
        frame = app.latest_frame
        if frame is None or frame.size == 0:
            default_width = 1080
            default_height = 2340
            return self._build_default_memory_detail_hotspots(default_width, default_height)

        height, width = frame.shape[:2]
        lines = self._extract_ocr_lines(frame)
        indicators = [
            line
            for line in lines
            if self._parse_page_indicator(line.text) is not None and line.y < int(height * 0.45)
        ]
        indicators.sort(key=lambda line: line.y)
        if indicators:
            first = indicators[0]
            second_y = indicators[1].y if len(indicators) >= 2 else min(int(height * 0.46), first.y + int(height * 0.18))
            line_right = first.x + first.w
            y_candidates = [
                int(first.cy),
                int(min(second_y - 60, first.cy + 18)),
                int(min(second_y - 50, first.cy + 70)),
                int(min(second_y - 35, first.cy + 110)),
            ]
            x_candidates = [
                int(min(width - 60, line_right - 14)),
                int(min(width - 44, line_right + 24)),
                int(min(width - 72, max(first.cx + 36, width * 0.90))),
                int(min(width - 96, max(first.cx + 4, width * 0.86))),
            ]
            hotspots: list[tuple[int, int]] = []
            for tap_x, tap_y in zip(x_candidates, y_candidates, strict=False):
                hotspot = (max(40, tap_x), max(40, tap_y))
                if hotspot not in hotspots:
                    hotspots.append(hotspot)
            if hotspots:
                return hotspots

        return self._build_default_memory_detail_hotspots(width, height)

    @staticmethod
    def _build_default_memory_detail_hotspots(width: int, height: int) -> list[tuple[int, int]]:
        return [
            (int(width * 0.91), int(height * 0.14)),
            (int(width * 0.95), int(height * 0.15)),
            (int(width * 0.89), int(height * 0.17)),
            (int(width * 0.86), int(height * 0.19)),
        ]

    def _wait_for_memory_detail_page(self, app: "AppProcessor", timeout: float = 6.0) -> bool:
        return self._wait_for_memory_page_state(app, {"detail"}, timeout=timeout) == "detail"

    def _is_memory_detail_page(self, app: "AppProcessor") -> bool:
        if is_memory_selection_page(app) or is_final_confirm_page(app):
            return False

        has_confirm = has_button(app, ButtonText.CONFIRM, fuzz_threshold=75)
        has_cancel = has_button(app, ButtonText.CANCEL, fuzz_threshold=60)
        if has_confirm and has_cancel:
            return True

        if not has_confirm:
            return False

        texts = self._extract_unique_texts(app.latest_frame)
        return any(
            string_match(text, ProduceText.OWNED_MEMORY, MatchConfig(fuzz_threshold=68))
            for text in texts
        )

    def _collect_current_memory_detail(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        slot_index: int,
    ) -> dict[str, Any] | None:
        frame = app.latest_frame
        if frame is None or frame.size == 0:
            return None

        raw_texts = self._extract_unique_texts(frame)
        detail_lines = self._extract_detail_lines(frame)
        detail_texts = [line.text for line in detail_lines]
        if not raw_texts:
            logger.warning(f"第 {slot_index} 张记忆卡详情页未识别到文本")
            return None

        skill_cards = self._collect_skill_card_pages(app)
        skill_texts = [
            text
            for page in skill_cards
            for text in (page.get("raw_texts") or [])
            if text
        ]
        skill_text_set = {text for text in skill_texts}
        non_skill_texts = [text for text in detail_texts if text not in skill_text_set]

        tag_matches = match_memory_tags(non_skill_texts)
        ability_matches = match_memory_abilities(
            non_skill_texts,
            produce_group_id=ctx.produce_group_id,
        )
        evaluation_candidates = sorted(
            {
                candidate["evaluation"]
                for match in ability_matches
                for candidate in match["metadata"].get("candidates", [])
            }
        )

        return {
            "slot_index": slot_index,
            "source": "memory-selection-detail",
            "raw_texts": raw_texts,
            "detail_texts": detail_texts,
            "stats": self._extract_memory_stats(detail_lines),
            "tags": tag_matches,
            "abilities": ability_matches,
            "evaluation_candidates": evaluation_candidates,
            "skill_cards": skill_cards,
            "skill_card_count": len(skill_cards),
            "produce_group_id": ctx.produce_group_id,
        }

    @staticmethod
    def _extract_detail_lines(frame) -> list[OCR_Result]:
        lines = CollectMemoryAttributesStep._extract_ocr_lines(frame)
        if frame is None or frame.size == 0:
            return lines

        height = frame.shape[0]
        inventory_header = CollectMemoryAttributesStep._find_line(lines, ProduceText.OWNED_MEMORY, fuzz_threshold=68)
        bottom_limit = int(height * 0.72)
        if inventory_header is not None:
            bottom_limit = min(bottom_limit, max(int(height * 0.42), inventory_header.y - 220))
        return [line for line in lines if line.y <= bottom_limit]

    def _collect_skill_card_pages(self, app: "AppProcessor") -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        seen_keys: set[tuple[int | None, str]] = set()

        for _ in range(4):
            page = self._extract_skill_card_page(app.latest_frame)
            if page is None:
                break

            page_key = (page.get("page_index"), page.get("title", ""))
            if page_key not in seen_keys:
                collected.append(page)
                seen_keys.add(page_key)

            page_index = page.get("page_index")
            total_pages = page.get("total_pages")
            if page_index is None or total_pages is None or page_index >= total_pages:
                break
            if not self._goto_next_skill_card_page(app, current_page=page_index):
                break

        return collected

    @staticmethod
    def _extract_skill_card_page(frame) -> dict[str, Any] | None:
        if frame is None or frame.size == 0:
            return None

        lines = CollectMemoryAttributesStep._extract_detail_lines(frame)
        skill_header = CollectMemoryAttributesStep._find_line(
            lines,
            ProduceText.AVAILABLE_SKILL_CARD,
            fuzz_threshold=60,
        )
        if skill_header is None:
            return None

        pager_line: OCR_Result | None = None
        page_index: int | None = None
        total_pages: int | None = None
        for line in lines:
            parsed = CollectMemoryAttributesStep._parse_page_indicator(line.text)
            if parsed is None:
                continue
            pager_line = line
            page_index, total_pages = parsed
            break

        section_lines = [line for line in lines if line.y > skill_header.y + 20]
        if pager_line is not None:
            section_lines = [line for line in section_lines if line.y < pager_line.y - 20]

        title_line: OCR_Result | None = None
        matched_entry: dict[str, Any] | None = None
        phase_texts: list[str] = []
        for line in section_lines:
            matches = [
                entry
                for entry in match_card_and_item_entries([line.text], threshold=78)
                if entry["kind"] == "produce_card"
            ]
            if matches:
                title_line = line
                matched_entry = matches[0]
                break
            phase_texts.append(line.text)

        effect_texts: list[str] = []
        if title_line is not None:
            effect_texts = [
                line.text
                for line in section_lines
                if line.y > title_line.y + 10
            ]

        db_description = ""
        description_match_score = 0.0
        if matched_entry is not None:
            db_description = CollectMemoryAttributesStep._get_produce_card_description(matched_entry["id"])
            if db_description and effect_texts:
                description_match_score = round(
                    fuzz.ratio(
                        CollectMemoryAttributesStep._normalize_text("".join(effect_texts)),
                        CollectMemoryAttributesStep._normalize_text(db_description),
                    ),
                    2,
                )

        return {
            "page_index": page_index,
            "total_pages": total_pages,
            "phase_texts": phase_texts,
            "title": title_line.text if title_line is not None else "",
            "raw_texts": [line.text for line in section_lines],
            "effect_texts": effect_texts,
            "matched_entry": matched_entry,
            "db_description": db_description,
            "description_match_score": description_match_score,
        }

    def _goto_next_skill_card_page(self, app: "AppProcessor", current_page: int) -> bool:
        frame = app.latest_frame
        if frame is None or frame.size == 0:
            return False

        width = frame.shape[1]
        lines = self._extract_ocr_lines(frame)
        pager_line = None
        for line in lines:
            parsed = self._parse_page_indicator(line.text)
            if parsed is None:
                continue
            if parsed[0] == current_page:
                pager_line = line
                break
        if pager_line is None:
            return False

        for offset in (150, 230, 310):
            tap_x = min(width - 80, pager_line.cx + offset)
            app.device.click(tap_x, pager_line.cy)
            sleep(0.5)
            wait_frame_stable(app, timeout=2.0)
            page = self._extract_skill_card_page(app.latest_frame)
            if page is not None and page.get("page_index") not in (None, current_page):
                return True

        logger.warning(f"技能卡分页停留在 {current_page}，未能翻到下一页")
        return False

    @staticmethod
    def _extract_memory_stats(lines: list[OCR_Result]) -> dict[str, int]:
        stats: dict[str, int] = {}
        for stat_key, queries in _STAT_LABELS.items():
            label_line = None
            for line in lines:
                if any(string_match(line.text, query, MatchConfig(fuzz_threshold=60)) for query in queries):
                    label_line = line
                    break
            if label_line is None:
                continue

            candidates = [
                line
                for line in lines
                if line.y > label_line.y
                and line.y <= label_line.y + 140
                and abs(line.cx - label_line.cx) <= 140
            ]
            candidates.sort(key=lambda line: (line.y, abs(line.cx - label_line.cx)))
            for candidate in candidates:
                value = CollectMemoryAttributesStep._extract_int(candidate.text)
                if value is not None:
                    stats[stat_key] = value
                    break
        return stats

    @staticmethod
    def _extract_int(text: str) -> int | None:
        digits = re.findall(r"\d+", text or "")
        if not digits:
            return None
        return int("".join(digits))

    @staticmethod
    def _extract_ocr_lines(frame) -> list[OCR_Result]:
        if frame is None or frame.size == 0:
            return []

        ocr_result = ocr_service.ocr(frame)
        lines: list[OCR_Result] = []
        seen: set[tuple[int, int, str]] = set()
        for item in sorted(ocr_result.results, key=lambda result: (result.y, result.x)):
            text = item.text.strip()
            if not text:
                continue
            if item.confidence is not None and item.confidence < 0.25:
                continue
            key = (item.x, item.y, text)
            if key in seen:
                continue
            seen.add(key)
            lines.append(item)
        return lines

    @staticmethod
    def _extract_unique_texts(frame) -> list[str]:
        texts: list[str] = []
        seen: set[str] = set()
        for item in CollectMemoryAttributesStep._extract_ocr_lines(frame):
            if item.text in seen:
                continue
            seen.add(item.text)
            texts.append(item.text)
        return texts

    @staticmethod
    def _find_line(
        lines: list[OCR_Result],
        query: str,
        *,
        fuzz_threshold: float = 70,
    ) -> OCR_Result | None:
        for line in lines:
            if string_match(line.text, query, MatchConfig(fuzz_threshold=fuzz_threshold)):
                return line
        return None

    @staticmethod
    def _parse_page_indicator(text: str) -> tuple[int, int] | None:
        match = _PAGE_RE.search(fullwidth_to_halfwidth(text or ""))
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return normalize_ocr_jp(fullwidth_to_halfwidth(text or "")).replace(" ", "").strip()

    @staticmethod
    def _get_produce_card_description(card_id: str) -> str:
        produce_card_db = GakumasDatabase_ProduceCardDataUtils()
        card = produce_card_db.get_by_id(f"{card_id}.0")
        if card is None:
            return ""
        source = getattr(card, "localization", None) or card
        return _concat_produce_descriptions(getattr(source, "produceDescriptions", []))

    def _close_memory_detail(self, app: "AppProcessor", slot_index: int) -> None:
        if self._ensure_memory_selection_page(app, timeout=1.0):
            return

        if self._get_memory_page_state(app) == "detail":
            exit_state = self._dismiss_memory_detail_overlay(app)
            if exit_state == "selection":
                return
            if exit_state == "candidate_list" and self._close_memory_candidate_list(app):
                return

        if not self._ensure_memory_selection_page(app, timeout=8.0):
            logger.warning(f"第 {slot_index} 张记忆卡关闭后未回到メモリー選択页")

    def _close_memory_candidate_list(self, app: "AppProcessor") -> bool:
        if self._dismiss_memory_candidate_list(app):
            return True
        if self._ensure_memory_selection_page(app, timeout=6.0):
            return True
        if self._get_memory_page_state(app) == "candidate_list":
            logger.warning("关闭メモリー編成一覧后未回到メモリー選択页")
        return False

    def _advance_to_final_confirm(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        if not self._ensure_memory_selection_page(app, timeout=10.0):
            raise TimeoutError("进入开始确认页前不在メモリー選択页")

        app.game_utils.click_button(
            ButtonText.NEXT,
            match_config=MatchConfig(fuzz_threshold=80),
        )
        app.game_utils.wait_loading()
        self._handle_rental_modal(app, ctx)

        if wait_for_final_confirm_page(app, timeout=15.0):
            logger.debug("记忆属性采集后成功进入最终确认页")
            return True
        raise TimeoutError("等待最终确认页超时")

    @staticmethod
    def _handle_rental_modal(app: "AppProcessor", ctx: "ProduceContext"):
        sleep(1)
        for _ in range(3):
            modal = app.game_utils.try_get_modal(no_body=True)
            if modal is None:
                logger.debug("未检测到レンタル弹窗")
                return

            if modal.modal_title and string_match(
                modal.modal_title,
                [ModalText.TITLE.RENTAL_AVAILABLE, ModalText.TITLE.RENTAL_CONFIRMATION],
                MatchConfig(fuzz_threshold=70),
            ):
                logger.info(f"检测到レンタル弹窗（{modal.modal_title!r}），确认")
                ctx.has_rental_memory = True
                if not click_modal_action_with_retry(app, modal, action_name="memory rental modal"):
                    raise TimeoutError(f"{modal.modal_title!r} 弹窗未能关闭")
                sleep(1)
            else:
                logger.debug(f"弹窗标题不匹配レンタル: {modal.modal_title!r}")
                return
