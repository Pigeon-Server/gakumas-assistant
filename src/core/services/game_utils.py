from copy import copy
from time import sleep, time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.Modal import Modal
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.Yolo import Yolo_Box
from src.utils.debug_tools import DebugTools
from src.utils.game_tools import get_current_location, get_modal
from src.utils.logger import logger
from src.utils.opencv_tools import compute_ssim_score
from src.utils.performance_tools import timeit
from src.utils.string_tools import string_match, MatchConfig

if TYPE_CHECKING:
    from src.main import AppProcessor

class GameUtils:
    _app_processor: "AppProcessor"
    debug_tools = DebugTools()

    def __init__(self, app_processor: "AppProcessor"):
        self._app_processor = app_processor

    def _get_current_frame(self) -> Optional[np.ndarray]:
        frame = getattr(self._app_processor, "latest_frame", None)
        if frame is not None and frame.size > 0:
            return frame
        results = getattr(self._app_processor, "latest_results", None)
        if results is None:
            return None
        frame = getattr(results, "frame", None)
        if frame is None or frame.size == 0:
            return None
        return frame

    def wait_for_label(self, label, timeout=15, interval=1, continuous=1):
        """
        等待指定标签的框出现
        :param label: 标签
        :param timeout: 超时时间
        :param interval: 轮询间隔
        :param continuous: 连续出现几次（新帧）再返回
        :return:
        """
        WAIT_TIME = 0
        COUNT = 0
        MISS_STREAK = 0        # 当前连续未检测到的新帧数
        MISS_TOLERANCE = 1     # 允许中间漏几帧，防止单帧漏检导致计数重置
        CENTER_MAX_SHIFT = 50  # 中心点最大允许偏移（像素），超过则视为不同元素
        last_seen_id = None
        last_cx, last_cy = None, None  # 上一次确认检测到的中心点
        logger.debug(f"waiting for label: {label}")
        while WAIT_TIME <= timeout:
            current = self._app_processor.latest_results
            # 帧未更新：等待下一帧，不纳入连续计数
            if current is None or id(current) == last_seen_id:
                sleep(0.1)
                WAIT_TIME += 0.1
                continue
            last_seen_id = id(current)
            if COUNT > continuous:
                logger.debug(f"Label '{label}' appeared {continuous} times. Returning True.")
                return True
            found = current.filter_by_label(label)
            if found:
                box = found.first()
                # 若存在上一帧中心点记录，检查偏移是否过大
                if last_cx is not None:
                    shift = ((box.cx - last_cx) ** 2 + (box.cy - last_cy) ** 2) ** 0.5
                    if shift > CENTER_MAX_SHIFT:
                        # 位置明显偏移，视为不同元素，重置计数
                        logger.debug(f"Label '{label}' center shifted {shift:.1f}px > {CENTER_MAX_SHIFT}px, resetting count.")
                        COUNT = 0
                        MISS_STREAK = 0
                last_cx, last_cy = box.cx, box.cy
                COUNT += 1
                MISS_STREAK = 0
                logger.debug(f"Found label '{label}' at ({box.cx},{box.cy}) (count={COUNT})")
                sleep(0.3)
                continue
            else:
                MISS_STREAK += 1
                if MISS_STREAK > MISS_TOLERANCE:
                    # 连续漏帧超过容忍度才重置，同时清除位置记录
                    COUNT = 0
                    MISS_STREAK = 0
                    last_cx, last_cy = None, None
                    logger.debug(f"Label '{label}' not found. Resetting count.")
                else:
                    logger.debug(f"Label '{label}' not found (miss_streak={MISS_STREAK}, count kept={COUNT}).")
            sleep(interval)
            WAIT_TIME += interval
            logger.debug(f"Waiting... {WAIT_TIME}/{timeout}s")
        logger.warning(f"Timeout reached ({timeout}s): Label '{label}' not found.")
        return False

    def wait_label_exist(self, label, timeout=15, interval=1, continuous=1):
        WAIT_TIME = 0
        COUNT = 0
        MISS_STREAK = 0        # 当前连续检测到的新帧数（对此函数而言，出现即为"miss"）
        MISS_TOLERANCE = 1     # 允许中间出现几帧，防止单帧误检导致计数重置
        CENTER_MAX_SHIFT = 50  # 中心点最大允许偏移（像素），超过则视为不同元素
        last_seen_id = None
        last_appeared_cx, last_appeared_cy = None, None  # 最近一次出现时的中心点（用于误检位置核验）
        logger.debug(f"Waiting label exist: {label}")
        while WAIT_TIME <= timeout:
            current = self._app_processor.latest_results
            # 帧未更新：等待下一帧，不纳入连续计数
            if current is None or id(current) == last_seen_id:
                sleep(0.1)
                WAIT_TIME += 0.1
                continue
            last_seen_id = id(current)
            if COUNT > continuous:
                logger.debug(f"Label '{label}' disappeared {continuous} times. Returning True.")
                return True
            found = current.filter_by_label(label)
            if not found:
                COUNT += 1
                MISS_STREAK = 0
                logger.debug(f"Not found label '{label}' (count={COUNT})")
                sleep(0.3)
                continue
            else:
                box = found.first()
                MISS_STREAK += 1
                if MISS_STREAK > MISS_TOLERANCE:
                    COUNT = 0
                    MISS_STREAK = 0
                    last_appeared_cx, last_appeared_cy = box.cx, box.cy
                    logger.debug(f"Label '{label}' found at ({box.cx},{box.cy}). Resetting count.")
                else:
                    # 在容忍范围内：检查出现位置是否与上次一致
                    if last_appeared_cx is not None:
                        shift = ((box.cx - last_appeared_cx) ** 2 + (box.cy - last_appeared_cy) ** 2) ** 0.5
                        if shift > CENTER_MAX_SHIFT:
                            # 位置偏移过大，不同元素出现，重置计数
                            COUNT = 0
                            MISS_STREAK = 0
                            logger.debug(f"Label '{label}' appeared at new position, shift={shift:.1f}px. Resetting count.")
                        else:
                            logger.debug(f"Label '{label}' found (miss_streak={MISS_STREAK}, shift={shift:.1f}px, count kept={COUNT}).")
                    else:
                        logger.debug(f"Label '{label}' found (miss_streak={MISS_STREAK}, count kept={COUNT}).")
                    last_appeared_cx, last_appeared_cy = box.cx, box.cy
            sleep(interval)
            WAIT_TIME += interval
            logger.debug(f"Waiting... {WAIT_TIME}/{timeout}s")
        logger.warning(f"Timeout reached ({timeout}s): Label '{label}' found.")
        return False

    def try_get_modal(self, no_body: bool = False, require_header: bool = True) -> Optional[Modal]:
        """
        尝试解析当前画面的模态框。

        默认要求存在 modal header，避免在普通页面上把按钮区误判为弹窗。
        只有在显式关闭 require_header 时，才允许进入更宽松的 modal 解析。

        这里只做轻量预筛选：当前帧存在按钮即可继续交给 ModalParser。
        是否真的是模态框，由 ModalParser 根据头部/面板/按钮布局综合判断。
        """
        results = self._app_processor.latest_results
        if results is None or results.frame is None or results.frame.size == 0:
            return None
        if require_header and not results.exists_label(BaseUILabels.MODAL_HEADER):
            return None

        buttons = results.filter_by_label(BaseUILabels.BUTTON)
        if not buttons:
            return None

        return get_modal(results, no_body=no_body, quiet=True)

    def wait_for_modal(
            self,
            modal_title,
            timeout=10,
            interval=1,
            no_body: bool = False,
            match_config: MatchConfig = None,
            require_header: bool = True,
    ) -> Optional[Modal]:
        """
        等待指定标题的模态框出现
        :param modal_title: 模态框标题
        :param timeout: 超时时间
        :param interval: 轮询间隔
        :param no_body: 不需要框体文本
        :param match_config: 匹配配置
        :param require_header: 是否要求存在 modal header
        :return:
        """
        logger.debug(f"Waiting for modal with title: {modal_title}")
        wait_time = 0
        match_config = match_config if match_config is not None else MatchConfig(fuzz_threshold=80)
        while wait_time < timeout:
            modal = self.try_get_modal(no_body=no_body, require_header=require_header)
            if modal:
                if modal_title is None or string_match(modal.modal_title, modal_title, match_config):
                    logger.debug(f"Modal found: {modal.modal_title}")
                    return modal
                logger.debug(f"Modal title '{modal.modal_title}' does not match '{modal_title}'")
            else:
                logger.debug(f"No visible modal found, waiting... ({wait_time}/{timeout})")

            sleep(interval)
            wait_time += interval
            logger.debug(f"Waiting... {wait_time}/{timeout}s")

        logger.warning(f"Timeout reached ({timeout}s): Modal with title '{modal_title}' not found.")
        return None

    def wait_modal_transition(self, previous_modal_title: str | None = None, timeout: float = 5.0, interval: float = 0.2) -> bool:
        """
        等待当前模态框关闭，或切换为另一个模态框。

        当 previous_modal_title 为 None 时，只要当前模态消失就返回 True；
        当 previous_modal_title 有值时，模态消失或标题变化都视为状态切换成功。
        """
        wait_time = 0.0
        while wait_time <= timeout:
            modal = self.try_get_modal(no_body=True)
            if modal is None:
                return True
            if previous_modal_title is not None and modal.modal_title != previous_modal_title:
                return True
            sleep(interval)
            wait_time += interval
        logger.warning(f"Timeout reached ({timeout}s): Modal '{previous_modal_title}' did not close or change.")
        return False

    def click_modal_button_and_wait_transition(
            self,
            button: Yolo_Box,
            previous_modal_title: str | None = None,
            retries: int = 2,
            timeout: float = 5.0,
            interval: float = 0.2,
    ) -> bool:
        """
        点击模态按钮，并等待模态关闭或切换到下一个模态。

        该方法用于“点击成功”不能只靠坐标触发来判断的模态流程。
        返回 True 表示模态状态确实发生了变化；
        返回 False 表示按钮点击后，原模态仍停留在当前画面。
        """
        baseline_title = previous_modal_title
        if baseline_title is None:
            modal = self.try_get_modal(no_body=True)
            baseline_title = None if modal is None else modal.modal_title

        if not self.click_element_and_wait_trigger(button, retries=retries, timeout=min(timeout, 1.5), interval=0.1):
            self._app_processor.device.click_element(button)

        return self.wait_modal_transition(
            previous_modal_title=baseline_title,
            timeout=timeout,
            interval=interval,
        )

    def click_on_label(self, label, timeout=10, interval=1):
        """
        等待指定标签并点击
        :param label: 标签
        :param timeout: 超时时间
        :param interval: 轮询间隔（同时决定早退阈值：连续 3×interval 秒的新帧未检测到则提前退出）
        :return:
        """
        WAIT_TIME = 0.0
        MISS_DURATION = 0.0    # 仅统计新帧未检测到的累计时间，跳过未更新的帧
        POLL = 0.3             # 内部轮询间隔
        CENTER_MAX_SHIFT = 50  # 中心点最大允许偏移（像素），超过则视为不同元素，不点击
        last_seen_id = None
        last_cx, last_cy = None, None  # 上一次检测到的中心点
        logger.debug(f"waiting to click label: {label}")
        while WAIT_TIME < timeout:
            current = self._app_processor.latest_results
            # 帧未更新（ADB 等慢速采集场景常见）：等待下一帧，不纳入检测统计
            if current is None or id(current) == last_seen_id:
                sleep(POLL)
                WAIT_TIME += POLL
                continue
            last_seen_id = id(current)
            boxs = current.filter_by_label(label)
            if boxs:
                box = boxs.first()
                # 检查中心点是否稳定（与上一帧偏移不超过阈值才点击）
                if last_cx is not None:
                    shift = ((box.cx - last_cx) ** 2 + (box.cy - last_cy) ** 2) ** 0.5
                    if shift > CENTER_MAX_SHIFT:
                        # 位置不稳定，更新位置记录，等待下一帧确认
                        logger.debug(f"Label '{label}' center shifted {shift:.1f}px, waiting for stable position.")
                        last_cx, last_cy = box.cx, box.cy
                        MISS_DURATION = 0.0
                        sleep(POLL)
                        WAIT_TIME += POLL
                        continue
                logger.debug(f"Found label '{label}' at ({box.cx},{box.cy}), clicking...")
                self._app_processor.device.click_element(box)
                return True
            last_cx, last_cy = None, None  # 未检测到时清除位置记录
            MISS_DURATION += POLL
            if MISS_DURATION >= 3 * interval:
                # 连续多个新帧均未检测到，提前退出
                logger.warning(f"Label '{label}' not found for {MISS_DURATION:.1f}s of new frames, breaking out of loop.")
                break
            sleep(POLL)
            WAIT_TIME += POLL
            logger.debug(f"Label '{label}' not found, retrying... ({WAIT_TIME:.1f}/{timeout}s)")
        logger.warning(f"Timeout reached ({timeout}s): Label '{label}' not found.")
        return False

    def wait_loading(self, timeout=-1):
        """
        等待加载
        :param timeout: 超时时间
        :return:
        """
        WAIT_TIME = 0
        COUNT = 0
        sleep(1)
        while timeout == -1 or WAIT_TIME < timeout:
            if self._app_processor.latest_results.filter_by_labels([BaseUILabels.GENERAL_LOADING1, BaseUILabels.GENERAL_LOADING2]):
                if WAIT_TIME == 0:
                   logger.debug("Waiting for loading")
                sleep(1)
                WAIT_TIME += 1
            else:
                if COUNT > 3:
                    logger.debug("Wait for the loading to finish")
                    return True
                else:
                    COUNT += 1
                    sleep(0.3)
        raise TimeoutError("Waiting for a load timeout")

    def check_label_exists_at_position(self, target_label, x: int, y: int, w: int, h: int, threshold: float = 0.8) -> bool:
        """
        检查目标标签是否存在于指定区域（支持部分重叠判断）
        :param y:
        :param x:
        :param w:
        :param h:
        :param target_label: 标签名
        :param threshold: IOU阈值
        """
        results = self._app_processor.latest_results
        if not results.exists_label(target_label):
            return False
        select_labels = results.filter_by_label(target_label)
        if not select_labels:
            return False
        # 当前检查区域
        x1, y1, x2, y2 = x, y, x + w, y + h
        for el in select_labels:
            ex1, ey1, ex2, ey2 = el.x, el.y, el.x + el.w, el.y + el.h
            # 计算交集
            inter_w = max(0, min(x2, ex2) - max(x1, ex1))
            inter_h = max(0, min(y2, ey2) - max(y1, ey1))
            inter_area = inter_w * inter_h
            # 计算并集
            union_area = (w * h) + (el.w * el.h) - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou >= threshold:
                return True
        return False


    def check_image_change_at_position(self, x, y, w, h, original: Optional[np.ndarray] = None, timeout=10, threshold: float = 0.8) -> bool:
        """
        检查指定位置图像是否变化
        :param x:
        :param y:
        :param w:
        :param h:
        :param original: 原图
        :param timeout: 超时时间
        :param threshold: 图像变化阈值
        :return:
        """
        current_source = self._get_current_frame()
        if current_source is None:
            return False

        reference_frame = original if original is not None and original.size > 0 else current_source[y:h, x:w]
        if reference_frame is None or reference_frame.size == 0:
            return False

        wait_time = 0.0
        stable_hits = 0
        while wait_time <= timeout:
            current_source = self._get_current_frame()
            if current_source is None:
                sleep(0.1)
                wait_time += 0.1
                continue

            current_frame = current_source[y:h, x:w]
            if current_frame is None or current_frame.size == 0:
                sleep(0.1)
                wait_time += 0.1
                continue

            if reference_frame.shape != current_frame.shape:
                return True

            score = compute_ssim_score(reference_frame, current_frame)
            if score < threshold:
                stable_hits += 1
                if stable_hits >= 2:
                    return True
            else:
                stable_hits = 0

            sleep(0.1)
            wait_time += 0.1
        return False

    def check_image_change_at_yolobox(self, target_yolobox: Yolo_Box, timeout=10, threshold: float = 0.8) -> bool:
        """
        检查目标YoloBox位置的图像是否变化
        :param target_yolobox:
        :param timeout:
        :param threshold:
        :return:
        """
        return self.check_image_change_at_position(
            target_yolobox.x,
            target_yolobox.y,
            target_yolobox.w,
            target_yolobox.h,
            target_yolobox.frame,
            timeout,
            threshold
        )

    def wait_for_action_trigger(
            self,
            element: Optional[Yolo_Box] = None,
            original_frame: Optional[np.ndarray] = None,
            timeout: float = 2.0,
            interval: float = 0.1,
            frame_threshold: float = 0.995,
            region_threshold: float = 0.8,
    ) -> bool:
        """
        等待一次操作真正触发。
        优先检查被点击元素区域是否发生变化，并辅以整帧变化兜底。
        """
        baseline_frame = original_frame if original_frame is not None and original_frame.size > 0 else self._get_current_frame()
        if baseline_frame is None:
            return False
        baseline_frame = baseline_frame.copy()

        region_reference = None
        if element is not None and getattr(element, "frame", None) is not None and element.frame.size > 0:
            region_reference = element.frame.copy()

        wait_time = 0.0
        while wait_time <= timeout:
            current_frame = self._get_current_frame()
            if current_frame is None:
                sleep(interval)
                wait_time += interval
                continue

            if region_reference is not None:
                x1, y1, x2, y2 = map(int, [element.x, element.y, element.w, element.h])
                current_region = current_frame[y1:y2, x1:x2]
                if current_region is not None and current_region.size > 0:
                    if current_region.shape != region_reference.shape:
                        return True
                    if compute_ssim_score(region_reference, current_region) < region_threshold:
                        return True

            if compute_ssim_score(baseline_frame, current_frame) < frame_threshold:
                return True

            sleep(interval)
            wait_time += interval

        return False

    def click_element_and_wait_trigger(
            self,
            element: Yolo_Box,
            retries: int = 3,
            timeout: float = 2.0,
            interval: float = 0.1,
            frame_threshold: float = 0.995,
            region_threshold: float = 0.8,
    ) -> bool:
        """
        点击元素并等待界面确认被触发。
        若点击未生效，可自动重试数次。
        """
        for attempt in range(1, retries + 1):
            baseline_frame = self._get_current_frame()
            baseline_frame = baseline_frame.copy() if baseline_frame is not None else None
            self._app_processor.device.click_element(element)
            if self.wait_for_action_trigger(
                    element=element,
                    original_frame=baseline_frame,
                    timeout=timeout,
                    interval=interval,
                    frame_threshold=frame_threshold,
                    region_threshold=region_threshold,
            ):
                return True
            logger.warning(f"Click did not trigger visible UI change ({attempt}/{retries}): {getattr(element, 'label', type(element).__name__)}")
        return False


    def click_button(self, text, timeout=10, match_config: MatchConfig = MatchConfig(use_fuzz=True, fuzz_threshold=80)):
        """
        点击指定文本按钮
        :param match_config:
        :param text: 按钮文本
        :param timeout: 超时时间
        :return:
        """
        logger.debug(f"waiting click button: {text}")
        self._app_processor.device.click_element(self.wait_button(text, timeout, match_config))

    def wait_button(self, text, timeout=10, match_config: MatchConfig = MatchConfig(use_fuzz=True, fuzz_threshold=80)):
        """
        等待指定文本按钮
        :param match_config:
        :param text: 按钮文本
        :param timeout: 超时时间
        :return:
        """
        COUNT = 0
        while COUNT < timeout:
            buttons = ButtonList(self._app_processor.latest_results)
            logger.debug(buttons)
            if button := buttons.get_button_by_text(text, match_config):
                return button
            sleep(1)
            COUNT += 1
        raise TimeoutError(f"Waiting for {text} button timeout")

    def _find_button_by_text(
            self,
            text: str,
            *,
            fuzz_threshold: float = 70,
            use_contains: bool = True,
    ):
        return ButtonList(self._app_processor.latest_results).get_button_by_text(
            text,
            MatchConfig(fuzz_threshold=fuzz_threshold, use_contains=use_contains, normalize=True),
        )

    def _get_top_right_action_button(self):
        buttons = ButtonList(self._app_processor.latest_results)
        candidates = [button for button in buttons if button.cx >= 720 and button.cy <= 280]
        candidates.sort(key=lambda button: (button.cy, -button.cx))
        return candidates[0] if candidates else None

    def _get_top_right_fallback_target(self) -> Optional[Yolo_Box]:
        frame = self._get_current_frame()
        if frame is None:
            return None
        height, width = frame.shape[:2]
        left = max(width - 220, 0)
        top = 0
        bottom = min(int(height * 0.18), height)
        if left >= width or top >= bottom:
            return None
        return Yolo_Box(left, top, width, bottom, BaseUILabels.CLOSE_BUTTON, frame[top:bottom, left:width].copy())

    def _wait_loading_safely(self, timeout: int = 8):
        try:
            self.wait_loading(timeout=timeout)
        except TimeoutError:
            logger.warning(f"Waiting for loading timed out after {timeout}s during navigation")

    def _try_exit_special_page(self, current_location: str | None) -> bool:
        results = self._app_processor.latest_results
        candidates: list = []
        if current_location == GamePageTypes.PRODUCER__MEMORY_DETAIL:
            if cancel_button := self._find_button_by_text("キャンセル", fuzz_threshold=60):
                candidates.append(cancel_button)
        elif current_location in {
            GamePageTypes.PRODUCER__MEMORY_CANDIDATE_LIST,
            GamePageTypes.PRODUCER__FORMATION_DETAILS,
        }:
            close_buttons = results.filter_by_label(BaseUILabels.CLOSE_BUTTON)
            if close_buttons:
                candidates.append(close_buttons.first())
            back_buttons = results.filter_by_label(BaseUILabels.BACK_BTN)
            if back_buttons:
                candidates.append(back_buttons.first())
            if current_location == GamePageTypes.PRODUCER__FORMATION_DETAILS:
                if close_button := self._find_button_by_text(ButtonText.CLOSE, fuzz_threshold=65):
                    candidates.append(close_button)
                if top_right_button := self._get_top_right_action_button():
                    candidates.append(top_right_button)
                if top_right_fallback := self._get_top_right_fallback_target():
                    candidates.append(top_right_fallback)
        else:
            return False

        seen: set[tuple[int | float | None, ...]] = set()
        for candidate in candidates:
            if candidate is None:
                continue
            key = (
                getattr(candidate, "x", None),
                getattr(candidate, "y", None),
                getattr(candidate, "w", None),
                getattr(candidate, "h", None),
                getattr(candidate, "text", None),
                getattr(candidate, "label", None),
            )
            if key in seen:
                continue
            seen.add(key)
            if not self.click_element_and_wait_trigger(candidate, retries=2, timeout=2.5, interval=0.1):
                continue
            self._wait_loading_safely(timeout=8)
            return True
        return False

    def go_home(self, max_try: int = 5):
        """
        返回主页
        :return:
        """
        self.update_current_location()
        if self._app_processor.game_status_manager.current_location == GamePageTypes.MAIN_MENU__HOME:
            return
        for _ in range(max_try):
            logger.debug(f"[{max_try}/{_}]Try going home")
            main_menu_items = [
                value for name, value in vars(GamePageTypes).items()
                if name.startswith("MAIN_MENU__")
            ]
            current_location = self.update_current_location()
            if current_location == GamePageTypes.MAIN_MENU__HOME:
                return
            if self._try_exit_special_page(current_location):
                current_location = self.update_current_location()
                if current_location == GamePageTypes.MAIN_MENU__HOME:
                    return
                sleep(1)
                continue
            navigation_candidates = []
            if current_location in main_menu_items:
                if home_tab := self._app_processor.latest_results.filter_by_label(BaseUILabels.TAB_HOME):
                    navigation_candidates.append(home_tab.first())
            else:
                if go_home_btn := self._app_processor.latest_results.filter_by_label(BaseUILabels.GO_HOME_BTN):
                    navigation_candidates.append(go_home_btn.first())
                if back_btn := self._app_processor.latest_results.filter_by_label(BaseUILabels.BACK_BTN):
                    navigation_candidates.append(back_btn.first())

            if navigation_candidates:
                for candidate in navigation_candidates:
                    if not self.click_element_and_wait_trigger(candidate, retries=3, timeout=2.5, interval=0.1):
                        logger.warning(
                            "Navigation click did not trigger visible UI change during go_home: {}",
                            getattr(candidate, "label", type(candidate).__name__),
                        )
                        continue
                    self._wait_loading_safely(timeout=8)
                    current_location = self.update_current_location()
                    if current_location == GamePageTypes.MAIN_MENU__HOME:
                        return
                    logger.warning(
                        "Navigation click changed the UI but did not reach HOME. Current location: {}",
                        current_location,
                    )
                sleep(1)
                continue

            if modal_header := self._app_processor.latest_results.filter_by_label(BaseUILabels.MODAL_HEADER):
                modal_header = modal_header.first()
                self._app_processor.device.click(modal_header.cx, max(modal_header.y - 50, 0))
            sleep(1)
        raise RuntimeError("Going home failed")


    def back_next_page(self):
        """
        返回上一页。

        只有在点击返回按钮后检测到可见 UI 变化，才认为返回动作成功。
        """
        logger.debug("Going back next page")
        current_location = self.update_current_location()
        if self._try_exit_special_page(current_location):
            return True
        if not self.wait_for_label(BaseUILabels.BACK_BTN, 3):
            raise TimeoutError("Waiting for a back button timeout")
        back_button = self._app_processor.latest_results.filter_by_label(BaseUILabels.BACK_BTN).first()
        if back_button is None:
            raise TimeoutError("Back button disappeared before click.")
        if not self.click_element_and_wait_trigger(back_button, retries=3, timeout=2.5, interval=0.1):
            raise TimeoutError("Back button click did not trigger page transition.")
        return True

    def update_current_location(self, new_location: str = None):
        """
        更细游戏管理器中的当前位置
        :param new_location: 可选，直接按输入的位置
        :return:
        """
        update = False
        if new_location and new_location != self._app_processor.game_status_manager.current_location:
            update = True
            self._app_processor.game_status_manager.current_location = new_location
        else:
            current_location = get_current_location(self._app_processor.latest_results)
            if current_location and current_location != self._app_processor.game_status_manager.current_location:
                update = True
                self._app_processor.game_status_manager.current_location = current_location
        if update:
            logger.debug(f"Current location: {self._app_processor.game_status_manager.current_location}")
            self._app_processor.broadcast_app_status()
        return self._app_processor.game_status_manager.current_location

    def wait_location_update(self, target_location: str, timeout=15, ignore_loading=True):
        """
        等待当前位置刷新
        :param target_location: 目标位置
        :param timeout: 超时时间 (increased default to 15s for reliability)
        :param ignore_loading: 是否忽略LOADING状态 (允许在加载过程中继续等待)
        :return:
        """
        logger.debug(f"Wait for the location to be updated to {target_location}......")
        COUNT = 0
        while True:
            if COUNT > timeout:
                current = self.update_current_location()
                logger.error(f"Timeout waiting for location update. Target: {target_location}, Current: {current}")
                raise TimeoutError("Timeout for waiting for location update")
            current_loc = self.update_current_location()
            if current_loc == target_location:
                logger.debug(f"Location successfully updated to {target_location}")
                return True
            # If ignore_loading is True, don't count LOADING states towards timeout
            elif ignore_loading and current_loc == GamePageTypes.LOADING:
                logger.debug(f"Detected LOADING state, continuing to wait (count={COUNT})")
            else:
                COUNT += 1
            sleep(1)

    def wait_frame_stable(self, threshold=0.98, stable_count=3, timeout=5,
                          exclude_region=None, min_stable_duration=0):
        """
        等待画面稳定（SSIM）

        通过比较来自不同 YOLO 推理周期的帧来判断画面是否稳定。
        使用 latest_results 对象身份来确保每次比较的是不同的推理帧，
        避免因高帧率下多次读取同一缓存帧而误判为稳定。

        Args:
            threshold: 画面相似度阈值
            stable_count: 连续多少帧满足才算稳定
            timeout: 超时时间（秒）
            exclude_region: 可选 (x, y, w, h) 比例元组 (0~1)，
                           将该区域遮黑后再算 SSIM，用于排除动画区域（如 Live2D 卡面）。
                           例如 (0.1, 0.2, 0.8, 0.7) 表示排除 10%~90% 宽、20%~90% 高的中心区域。
            min_stable_duration: 最小稳定持续时间（秒），要求画面稳定持续至少这么长时间才返回。
                                用于滚动列表等场景，防止动画中的短暂停顿被误判为稳定。
        """
        start = time()
        prev_frame = None
        prev_results = None
        stable_times = 0
        stable_since = None

        while True:
            curr_results = self._app_processor.latest_results
            if curr_results is None:
                sleep(0.05)
                continue

            # 确保是不同的推理周期产生的帧，避免比较同一缓存帧
            if curr_results is prev_results:
                sleep(0.05)
                continue

            curr_frame = curr_results.frame
            if curr_frame is None:
                prev_results = curr_results
                sleep(0.05)
                continue

            # 第一帧，无对比，跳过
            if prev_frame is None:
                prev_frame = curr_frame.copy()
                prev_results = curr_results
                continue

            a = prev_frame
            b = curr_frame
            if exclude_region is not None:
                a = a.copy()
                b = b.copy()
                h, w = a.shape[:2]
                rx, ry, rw, rh = exclude_region
                x1, y1 = int(rx * w), int(ry * h)
                x2, y2 = int((rx + rw) * w), int((ry + rh) * h)
                a[y1:y2, x1:x2] = 0
                b[y1:y2, x1:x2] = 0

            score = compute_ssim_score(a, b)
            logger.debug(f"SSIM: {score}")
            if score >= threshold:
                stable_times += 1
                if stable_since is None:
                    stable_since = time()
            else:
                stable_times = 0
                stable_since = None
            # 判断是否连续稳定且持续时间足够
            if stable_times >= stable_count:
                if min_stable_duration <= 0 or (time() - stable_since) >= min_stable_duration:
                    return True

            # timeout < 0 表示不限制时间
            if timeout >= 0 and time() - start > timeout:
                return False

            prev_frame = curr_frame.copy()
            prev_results = curr_results
