"""可扩展的 gameplay handler 基础设施。

架构:
  - GameplayHandler: 所有阶段 handler 的抽象基类。
  - HandlerResult: handler 执行后的返回类型。
  - GameplayDispatcher: 将 phase 路由到对应 handler 的注册中心。

新增 gameplay 阶段的步骤（如 NIA オーディション）:
  1. （可选）在 `src/constants/game/producer_gameplay.py` 中添加阶段值
  2. （可选）在 ui.py 的 classify_gameplay_phase() 中添加检测规则
  3. 在 gameplay/ 下新建模块，继承 GameplayHandler
  4. 在 __init__.py 的 build_default_dispatcher() 中注册

每个 handler 必须实现:
  - can_handle(app, ctx, phase, position) -> bool
  - handle(app, ctx, phase, position) -> HandlerResult

调度器按 priority 从高到低尝试所有 handler，
委托给第一个 can_handle() 返回 True 的 handler。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import cv2

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.shared.common import click_relative_point
from src.utils.debug_tools import DebugTools
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig, fullwidth_to_halfwidth, string_match

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_RESULT_SCREEN_OCR = OCRService()
_RESULT_RETRY_RE = re.compile(ProduceText.REMAINING_COUNT_PATTERN)
_RESULT_RETRY_STUCK_LIMIT = 3
_RESULT_EXAM_FAILURE_RETRY_KEY = "result_exam_failure_retry_clicks"
_RESULT_EXAM_FAILURE_CENTER_TAP_KEY = "result_exam_failure_center_taps"
_RESULT_EXAM_FAILURE_CENTER_TAP_LIMIT_KEY = "result_exam_failure_center_tap_limit"
_RESULT_EXAM_FAILURE_CENTER_TAP_LIMIT = 2
_RESULT_PANEL_WHITE_HSV_LOWER = (0, 0, 145)
_RESULT_PANEL_WHITE_HSV_UPPER = (180, 60, 255)


# ────────────────────────────────────────────────────────────
# Handler 返回值
# ────────────────────────────────────────────────────────────

@dataclass
class HandlerResult:
    """handler 执行后的返回值。

    状态:
      ok        — 操作成功执行
      waiting   — 暂无操作，下次循环重试
      exit      — gameplay 循环应当终止（如结果画面）
      no_action — handler 匹配但无法执行（无候选元素等）
      unhandled — 没有 handler 匹配
    """
    status: str
    detail: str = ""
    sleep_after: float = 0.5

    @staticmethod
    def ok(detail: str = "", sleep_after: float = 0.5) -> "HandlerResult":
        """返回状态为 ok 的 handler 结果，表示当前帧的操作已成功执行。

        Args:
            detail: 本次操作的简要描述，用于日志和调试输出。
            sleep_after: 操作完成后需要等待的秒数，避免画面未切换完成即进入下一帧。

        Returns:
            HandlerResult: status 为 "ok" 的结果对象。
        """
        return HandlerResult("ok", detail, sleep_after)

    @staticmethod
    def waiting(detail: str = "", sleep_after: float = 1.0) -> "HandlerResult":
        """返回状态为 waiting 的 handler 结果，表示当前画面暂无可操作元素，需等待下次循环。

        Args:
            detail: 等待原因的说明文本。
            sleep_after: 等待期间的间隔秒数，默认比 ok 略长以减少无效轮询。

        Returns:
            HandlerResult: status 为 "waiting" 的结果对象。
        """
        return HandlerResult("waiting", detail, sleep_after)

    @staticmethod
    def exit(detail: str = "") -> "HandlerResult":
        """返回状态为 exit 的 handler 结果，表示 gameplay 主循环应当终止。

        通常在到达培育结果画面、记忆卡面选择等收尾阶段时触发，
        使流水线退出 PRODUCER 模型并切换到 BASE_UI 或结束整个流程。

        Args:
            detail: 退出原因的说明文本。

        Returns:
            HandlerResult: status 为 "exit" 的结果对象，sleep_after 固定为 0。
        """
        return HandlerResult("exit", detail, 0.0)

    @staticmethod
    def no_action(detail: str = "", sleep_after: float = 0.8) -> "HandlerResult":
        """返回状态为 no_action 的 handler 结果，表示 handler 匹配当前阶段但无法执行有效操作。

        常见场景：页面上没有候选元素、所有按钮均不可点击、识别到的元素数量为零等。
        与 waiting 不同，no_action 表示已经检查过但确实无可用操作。

        Args:
            detail: 无法执行操作的原因说明。
            sleep_after: 等待后再次尝试的间隔秒数。

        Returns:
            HandlerResult: status 为 "no_action" 的结果对象。
        """
        return HandlerResult("no_action", detail, sleep_after)

    @staticmethod
    def unhandled() -> "HandlerResult":
        """返回状态为 unhandled 的 handler 结果，表示没有任何已注册的 handler 匹配当前画面。

        该结果由 GameplayDispatcher.dispatch() 在所有 handler 的 can_handle() 均返回 False 时返回，
        通常不应由具体 handler 主动构造。

        Returns:
            HandlerResult: status 为 "unhandled" 的结果对象。
        """
        return HandlerResult("unhandled", "", 0.0)


# ────────────────────────────────────────────────────────────
# Handler 抽象基类
# ────────────────────────────────────────────────────────────

class GameplayHandler(ABC):
    """所有 gameplay 阶段 handler 的抽象基类。

    子类需覆盖 ``phase_tag`` 和 ``priority``。

    推荐优先级范围:
      95   结果画面 / 退出检测
      90   弹窗覆盖层
      50   常规 gameplay 阶段
      10   过场效果链
      -100 兜底（点击推进）
    """

    phase_tag: str = ""
    priority: int = 50

    @abstractmethod
    def can_handle(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        phase: str,
        position: str,
    ) -> bool:
        """判断此 handler 是否应处理当前画面，返回 True 表示匹配。"""
        ...

    @abstractmethod
    def handle(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        phase: str,
        position: str,
    ) -> HandlerResult:
        """执行操作。仅在 can_handle() 返回 True 时调用。"""
        ...

    def __repr__(self) -> str:
        """返回 handler 的字符串表示，包含类名、阶段标签和优先级，便于日志输出和调试。

        Returns:
            str: 格式为 `<ClassName phase='xxx' priority=N>` 的字符串。
        """
        return f"<{self.__class__.__name__} phase={self.phase_tag!r} priority={self.priority}>"


# ────────────────────────────────────────────────────────────
# 调度器
# ────────────────────────────────────────────────────────────

class GameplayDispatcher:
    """将 gameplay 画面帧路由到已注册 handler 的调度器。

    按 priority 从高到低依次尝试 handler，
    第一个 can_handle() 返回 True 的 handler 处理该帧。

    用法::

        dispatcher = GameplayDispatcher()
        dispatcher.register(ScheduleHandler())
        dispatcher.register(DialogueHandler())
        result = dispatcher.dispatch(app, ctx, phase, position)
    """

    def __init__(self) -> None:
        """初始化调度器，创建空的 handler 注册列表。

        Returns:
            None: 仅产生副作用，不返回业务值。
        """
        self._handlers: List[GameplayHandler] = []

    def register(self, handler: GameplayHandler) -> "GameplayDispatcher":
        """注册 handler，按 priority 降序重新排序。"""
        self._handlers.append(handler)
        self._handlers.sort(key=lambda h: -h.priority)
        return self

    def unregister(self, handler_type: type) -> "GameplayDispatcher":
        """从调度器中移除指定类型的所有 handler，并返回自身以支持链式调用。

        Args:
            handler_type: 需要注销的 handler 类类型，所有该类型的实例都会被移除。

        Returns:
            GameplayDispatcher: 当前调度器对象，便于连续调用。
        """
        self._handlers = [h for h in self._handlers if not isinstance(h, handler_type)]
        return self

    def dispatch(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        phase: str,
        position: str,
    ) -> HandlerResult:
        """找到第一个匹配的 handler 并执行。"""
        for handler in self._handlers:
            if handler.can_handle(app, ctx, phase, position):
                # 非弹窗阶段成功处理时，重置弹窗卡住计数器
                if handler.phase_tag != "modal":
                    ctx.handler_state.pop("modal_stuck_count", None)
                result = handler.handle(app, ctx, phase, position)
                return result
        return HandlerResult.unhandled()

    @property
    def handlers(self) -> List[GameplayHandler]:
        """当前已注册的 handler（按 priority 排序）。"""
        return list(self._handlers)


# ────────────────────────────────────────────────────────────
# 内置通用处理器
# ────────────────────────────────────────────────────────────

class ResultHandler(GameplayHandler):
    """检测到结果画面时标记培育进入收尾阶段。

    收尾分两阶段：
      1. produce_finishing_pending: 已决定结束培育，但仍在 PRODUCER 模型下处理
         （考试结果、LIVE演出、记忆生成等）
      2. produce_finishing: 到达记忆卡面选择等后期通用 UI 页面，切 BASE_UI 推进回主页
    """

    phase_tag = GameplayPhase.RESULT
    priority = 95

    # 到达这些位置时切换到 BASE_UI 收尾（记忆卡面选择及之后的通用 UI）
    _BASEUI_POSITIONS = {
        GameplayPosition.RESULT_MEMORY_PAGE,
        GameplayPosition.RESULT_REWARD_SUMMARY,
        GameplayPosition.RESULT_ACHIEVEMENT_PROGRESS,
        GameplayPosition.RESULT_EVENT_REWARD_PROGRESS,
    }

    # 这些位置说明培育已结束，但还在 PRODUCER 可处理的阶段（点击推进即可）
    _PENDING_POSITIONS = {
        GameplayPosition.RESULT_FINAL_EVALUATION,
        GameplayPosition.RESULT_MEMORY_GENERATION,
    }

    def can_handle(self, app, ctx, phase, position):
        """当当前画面阶段为 RESULT 时返回 True，表示由该处理器接管所有结果页面。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            bool: phase 等于 GameplayPhase.RESULT 时返回 True。
        """
        return phase == GameplayPhase.RESULT

    @staticmethod
    def _find_result_gray_white_panel(frame) -> tuple[int, int, int, int] | None:
        """在结果画面中查找灰白色面板区域，用于考试失败页的框外点击推进。

        通过 HSV 颜色阈值分割 + 形态学操作定位画面中央的大面积灰白矩形，
        该矩形通常是考试结果面板。找到后可以在其外部安全区域点击以推进流程。

        Args:
            frame: 当前游戏截图的 OpenCV 图像帧（BGR 格式）。

        Returns:
            tuple[int, int, int, int] | None: 面板的边界框坐标 (x1, y1, x2, y2)，
            未找到符合条件的面板时返回 None。筛选条件包括面积占比 >= 18%、
            宽高下限、水平居中、垂直位置在 8%-65% 之间且底部延伸到 62% 以下。
        """
        if frame is None or getattr(frame, "size", 0) <= 0:
            return None
        frame_h, frame_w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _RESULT_PANEL_WHITE_HSV_LOWER, _RESULT_PANEL_WHITE_HSV_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_rect: tuple[int, int, int, int] | None = None
        best_area = 0.0
        for contour in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            if w_rect <= 0 or h_rect <= 0:
                continue
            area = float(w_rect * h_rect)
            area_ratio = area / max(float(frame_w * frame_h), 1.0)
            if area_ratio < 0.18:
                continue
            if w_rect < int(frame_w * 0.45) or h_rect < int(frame_h * 0.35):
                continue
            cx = x + w_rect / 2.0
            if abs(cx - frame_w / 2.0) > frame_w * 0.30:
                continue
            if y < int(frame_h * 0.08) or y > int(frame_h * 0.65):
                continue
            if y + h_rect < int(frame_h * 0.62):
                continue
            if area > best_area:
                best_area = area
                best_rect = (x, y, x + w_rect, y + h_rect)
        return best_rect

    @staticmethod
    def _click_outside_result_panel(
        app: "AppProcessor",
        *,
        label: str,
    ) -> bool:
        """在结果画面灰白色面板的外部安全区域点击，用于推进考试失败等结果页面。

        先通过 `_find_result_gray_white_panel` 定位面板位置，然后在面板上方或下方
        的安全边距区域点击，避免误触面板内的按钮。如果无法检测到面板则回退失败。

        Args:
            app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
            label: 本次点击操作的标识，用于日志记录和 DebugTools 标注。

        Returns:
            bool: 是否成功执行了点击操作。返回 False 表示无法获取画面或未检测到面板。
        """
        frame = getattr(getattr(app, "latest_results", None), "frame", None)
        if frame is None:
            frame = getattr(app, "latest_frame", None)
        if frame is None:
            return False
        frame_h, frame_w = frame.shape[:2]
        panel_rect = ResultHandler._find_result_gray_white_panel(frame)
        if panel_rect is None:
            return False
        x1, y1, x2, y2 = panel_rect
        safe_margin = max(int(frame_h * 0.035), 24)
        tap_x = int(max(int(frame_w * 0.12), min(int(frame_w * 0.88), (x1 + x2) / 2.0)))
        if y1 > int(frame_h * 0.18) + safe_margin:
            tap_y = max(int(frame_h * 0.12), y1 - safe_margin)
        else:
            tap_y = min(int(frame_h * 0.92), y2 + safe_margin)
        DebugTools().add_box(
            x1,
            y1,
            x2,
            y2,
            label="ResultGrayWhitePanel",
            color=(200, 200, 80),
            alpha=0.10,
            duration=2.5,
            font_size=14,
        )
        DebugTools().add_point(
            tap_x,
            tap_y,
            radius=8,
            color=(80, 220, 120),
            alpha=0.35,
            duration=2.5,
        )
        app.device.click(tap_x, tap_y, label)
        return True

    @staticmethod
    def _extract_retry_remaining(frame_text: str) -> int | None:
        """从结果画面 OCR 文本中提取剩余重试次数。

        使用正则表达式匹配画面中的剩余次数文案（如「残り○回」），
        用于判断考试失败页是否还有重试机会。

        Args:
            frame_text: 当前画面 OCR 识别到的全部文本内容，已做全角转半角处理。

        Returns:
            int | None: 剩余重试次数，匹配失败或文本为空时返回 None。
        """
        normalized = fullwidth_to_halfwidth(str(frame_text or ""))
        match = _RESULT_RETRY_RE.search(normalized)
        if not match:
            return None
        try:
            return int(match.group(1) or 0)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _should_choose_next_on_exam_failure(ctx: "ProduceContext", frame_text: str) -> bool:
        """判断考试失败页是否应该放弃重试并转向下一步。"""
        remaining = ResultHandler._extract_retry_remaining(frame_text)
        if remaining is not None and remaining <= 0:
            return True
        retry_click_count = int(ctx.handler_state.get(_RESULT_EXAM_FAILURE_RETRY_KEY, 0) or 0)
        return retry_click_count >= _RESULT_RETRY_STUCK_LIMIT

    @staticmethod
    def _click_result_button_by_ocr(
        app: "AppProcessor",
        *,
        tokens: tuple[str, ...],
        label: str,
    ) -> bool:
        """通过 OCR 在结果页中点击指定按钮。"""
        frame = getattr(getattr(app, "latest_results", None), "frame", None)
        if frame is None:
            frame = getattr(app, "latest_frame", None)
        if frame is None:
            return False
        ocr_results = _RESULT_SCREEN_OCR.ocr(frame)
        if not ocr_results or not ocr_results.results:
            return False
        merged = ocr_results.auto_merge_lines(cy_range=8, width_gap=24)
        match_config = MatchConfig(fuzz_threshold=55, use_contains=True, normalize=True)
        best = None
        best_score = -1.0
        for line in merged.results:
            text = str(getattr(line, "text", "") or "").strip()
            if not text:
                continue
            matched = string_match(text, list(tokens), config=match_config)
            if not matched:
                continue
            score = float(getattr(matched, "threshold", 0.0) or 0.0)
            # 同分时优先底部按钮，避免误点标题文案里的同词片段。
            if best is None or score > best_score or (score == best_score and int(line.cy) > int(best.cy)):
                best = line
                best_score = score
        if best is None:
            return False
        x1 = int(best.x)
        y1 = int(best.y)
        x2 = int(best.x + best.w)
        y2 = int(best.y + best.h)
        DebugTools().add_box(
            x1,
            y1,
            x2,
            y2,
            label=f"ResultButtonOCR:{label}",
            color=(80, 220, 120),
            alpha=0.15,
            duration=2.5,
            font_size=16,
        )
        app.device.click(int(best.cx), int(best.cy), label)
        return True

    @staticmethod
    def _handle_result_exam_failure(app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """处理考试失败结果页面的点击决策逻辑。

        策略：
        1. 优先采用用户习惯 — 在面板外部点击推进，连续若干次无效后切换策略。
        2. 收集画面 OCR 文本，判断是否应该放弃重试、点击「次へ」进入下一结果。
        3. 如果还有重试机会，则点击「再挑戦」按钮重新考试。
        4. 按钮识别失败时回退到 OCR 文本匹配方式查找按钮。

        Args:
            app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
            ctx: 培育上下文对象，保存跨步骤状态与策略配置，包括重试计数器等。

        Returns:
            bool: 是否成功执行了点击操作。返回 False 表示画面中未找到任何可点击元素。
        """
        from src.core.tasks.producer_challenge.ui import collect_frame_text, find_button

        center_tap_limit = int(
            ctx.handler_state.get(
                _RESULT_EXAM_FAILURE_CENTER_TAP_LIMIT_KEY,
                _RESULT_EXAM_FAILURE_CENTER_TAP_LIMIT,
            )
            or 0
        )
        center_tap_count = int(ctx.handler_state.get(_RESULT_EXAM_FAILURE_CENTER_TAP_KEY, 0) or 0)
        # 先按用户习惯走“点中间推进”，连续几次无效后再切换到按钮识别兜底。
        if center_tap_limit > 0 and center_tap_count < center_tap_limit:
            clicked = ResultHandler._click_outside_result_panel(
                app,
                label="result-exam-failure-outside",
            )
            if not clicked:
                # 未检测到灰白框时退回顶部安全区，避免误点灰白框内部。
                click_relative_point(
                    app,
                    x_ratio=0.5,
                    y_ratio=0.16,
                    label="result-exam-failure-outside-fallback",
                )
            ctx.handler_state[_RESULT_EXAM_FAILURE_CENTER_TAP_KEY] = center_tap_count + 1
            logger.info(
                "result: 考试失败页先尝试框外推进({}/{})",
                ctx.handler_state[_RESULT_EXAM_FAILURE_CENTER_TAP_KEY],
                center_tap_limit,
            )
            return True

        frame_text = collect_frame_text(getattr(app, "latest_results", None))
        choose_next = ResultHandler._should_choose_next_on_exam_failure(ctx, frame_text)
        retry_button = find_button(app, ButtonText.RETRY, fuzz_threshold=50)
        next_button = find_button(app, ButtonText.NEXT, fuzz_threshold=50)

        if choose_next:
            if next_button is not None:
                app.device.click_element(next_button)
                ctx.handler_state[_RESULT_EXAM_FAILURE_RETRY_KEY] = 0
                logger.info("result: 考试失败页命中「次へ」，放弃重试继续结果链")
                return True
            if ResultHandler._click_result_button_by_ocr(
                app,
                tokens=(ButtonText.NEXT,),
                label="result-exam-failure-next",
            ):
                ctx.handler_state[_RESULT_EXAM_FAILURE_RETRY_KEY] = 0
                logger.info("result: 考试失败页 OCR 命中「次へ」，放弃重试继续结果链")
                return True
            return False

        if retry_button is not None:
            app.device.click_element(retry_button)
            ctx.handler_state[_RESULT_EXAM_FAILURE_RETRY_KEY] = int(
                ctx.handler_state.get(_RESULT_EXAM_FAILURE_RETRY_KEY, 0) or 0
            ) + 1
            logger.info(
                "result: 考试失败页点击「再挑戦」(连续尝试={})",
                ctx.handler_state[_RESULT_EXAM_FAILURE_RETRY_KEY],
            )
            return True
        if ResultHandler._click_result_button_by_ocr(
            app,
            tokens=(ButtonText.RETRY,),
            label="result-exam-failure-retry",
        ):
            ctx.handler_state[_RESULT_EXAM_FAILURE_RETRY_KEY] = int(
                ctx.handler_state.get(_RESULT_EXAM_FAILURE_RETRY_KEY, 0) or 0
            ) + 1
            logger.info(
                "result: 考试失败页 OCR 命中「再挑戦」(连续尝试={})",
                ctx.handler_state[_RESULT_EXAM_FAILURE_RETRY_KEY],
            )
            return True
        return False

    def handle(self, app, ctx, phase, position):
        """处理结果画面的点击推进与阶段切换逻辑。

        根据 position 细分位置决定行为：
        - BASEUI_POSITIONS：标记 produce_finishing，切换到 BASE_UI 收尾流程。
        - PENDING_POSITIONS：标记 produce_finishing_pending，继续 PRODUCER 推进。
        - RESULT_EXAM_FAILURE：调用专用方法处理考试失败的重试/放弃决策。
        - 其他结果页面：优先点击确认按钮，找不到则点击屏幕中心推进。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            HandlerResult: 包含操作状态、描述和等待时间的结果对象。
        """
        if position != GameplayPosition.RESULT_EXAM_FAILURE:
            ctx.handler_state.pop(_RESULT_EXAM_FAILURE_RETRY_KEY, None)
            ctx.handler_state.pop(_RESULT_EXAM_FAILURE_CENTER_TAP_KEY, None)

        # 到达记忆卡面选择等后期页面后，切换到 BASE_UI 收尾
        if position in self._BASEUI_POSITIONS:
            ctx.handler_state["produce_finishing"] = True
            logger.info("result: 到达 {} → 标记 produce_finishing，将切换 BASE_UI", position)
            return HandlerResult.ok(f"result ({position}) → produce_finishing", sleep_after=0.5)

        # 培育已结束的中间页面（最终评价、记忆生成等），标记 pending 后继续 PRODUCER 推进
        if position in self._PENDING_POSITIONS:
            ctx.handler_state["produce_finishing_pending"] = True
            logger.info("result: 到达 {} → 标记 produce_finishing_pending，继续 PRODUCER 推进", position)

        if position == GameplayPosition.RESULT_EXAM_FAILURE:
            if ResultHandler._handle_result_exam_failure(app, ctx):
                ctx.handler_state["unknown_retry_override"] = {
                    "reason": "result_exam_failure_transition",
                    "retry_limit": 15,
                    "retry_sleep": 1.0,
                }
                return HandlerResult.ok("result (result_exam_failure) → handled", sleep_after=1.0)
            logger.warning("result: 考试失败页未命中可点击按钮，回退中心点击推进")

        # 所有非 BASE_UI 的结果页面：优先点击确认按钮，否则点中心推进
        from src.constants.yolo.labels.producer_Labels import ProducerLabels
        confirm_boxes = list(app.latest_results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
        if confirm_boxes:
            box = confirm_boxes[0]
            app.device.click_element(box)
        else:
            click_relative_point(app, x_ratio=0.5, y_ratio=0.5, label="result-advance")
        # 结果页后常有切页动画或对话过渡，因此给更长的 unknown 重试
        ctx.handler_state["unknown_retry_override"] = {
            "reason": "result_midgame_transition",
            "retry_limit": 10,
            "retry_sleep": 1.0,
        }
        return HandlerResult.ok(f"result ({position}) → advance", sleep_after=0.8)


class AdvanceHandler(GameplayHandler):
    """兜底 handler：点击屏幕中央推进未知/加载画面。

    最低优先级 — 仅在无其他 handler 匹配时激活。
    """

    phase_tag = ""
    priority = -100

    def can_handle(self, app, ctx, phase, position):
        """始终返回 True，作为兜底 handler 匹配所有未被其他处理器接管的画面。

        由于优先级设为 -100（最低），只有在所有其他 handler 的 can_handle() 均返回 False 时
        才会被调度器选中。适用于加载画面、过场动画、未知页面等需要点击推进的场景。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            bool: 始终返回 True。
        """
        return True  # 兜底匹配所有未被其他 handler 处理的画面

    def handle(self, app, ctx, phase, position):
        """点击屏幕下方偏中心位置以推进当前画面。

        适用于加载画面、过场动画、未知页面等没有明确可点击元素的场景。
        点击位置设为 (0.5, 0.82)，即屏幕水平居中、垂直方向偏下方，避免误触顶部 HUD。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            HandlerResult: status 为 "ok" 的结果对象，附带 1 秒等待时间。
        """
        from src.utils.logger import logger
        logger.debug(f"advance: tap to progress (phase={phase}, position={position})")
        click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="advance")
        return HandlerResult.ok("advance tap", sleep_after=1.0)
