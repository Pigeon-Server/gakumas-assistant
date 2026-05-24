from dataclasses import dataclass, field

import cv2
import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ocr_engine import OCRService
from src.entity.Game.Components.Button import Button
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.utils.debug_tools import DebugTools
from src.utils.logger import logger

ocr_service = OCRService()
debug_tools = DebugTools()


@dataclass
class Modal:
    modal_title: str
    modal_body: np.ndarray
    modal_body_text: str | None = None
    confirm_button: Button = None
    cancel_button: Button = None
    panel_box: Yolo_Box | None = None
    header_box: Yolo_Box | None = None
    body_box: Yolo_Box | None = None
    action_buttons: list[Button] = field(default_factory=list)
    source_frame: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_yolo_results(
            cls,
            yolo_result: Yolo_Results,
            no_body: bool = False,
            quiet: bool = False,
    ) -> "Modal | None":
        return ModalParser(yolo_result, no_body=no_body, quiet=quiet).parse()

    def draw_debug(self, image: np.ndarray | None = None) -> np.ndarray:
        frame = (image if image is not None else self.source_frame)
        if frame is None or frame.size == 0:
            raise ValueError("Modal debug image requires a source frame")

        debug_tools.clear_all()
        try:
            if self.panel_box:
                debug_tools.add_box(
                    int(self.panel_box.x),
                    int(self.panel_box.y),
                    int(self.panel_box.w),
                    int(self.panel_box.h),
                    color=(0, 255, 255),
                    label="panel",
                    duration=300,
                )
            if self.header_box:
                debug_tools.add_box(
                    int(self.header_box.x),
                    int(self.header_box.y),
                    int(self.header_box.w),
                    int(self.header_box.h),
                    color=(0, 165, 255),
                    label=f"header:{self.modal_title}",
                    duration=300,
                )
            if self.body_box:
                debug_tools.add_box(
                    int(self.body_box.x),
                    int(self.body_box.y),
                    int(self.body_box.w),
                    int(self.body_box.h),
                    color=(255, 215, 0),
                    label="body",
                    duration=300,
                )
            for button in self.action_buttons:
                role = "action"
                color = (255, 0, 255)
                if self.confirm_button and button == self.confirm_button:
                    role = "confirm"
                    color = (0, 255, 0)
                elif self.cancel_button and button == self.cancel_button:
                    role = "cancel"
                    color = (255, 0, 0)
                debug_tools.add_box(
                    int(button.x),
                    int(button.y),
                    int(button.w),
                    int(button.h),
                    color=color,
                    label=role,
                    duration=300,
                )
            return debug_tools.draw_boxes(frame.copy())
        finally:
            debug_tools.clear_all()


class ModalParser:
    def __init__(self, yolo_result: Yolo_Results, no_body: bool = False, quiet: bool = False):
        self.yolo_result = yolo_result
        self.frame = yolo_result.frame
        self.no_body = no_body
        self.quiet = quiet
        self.buttons = yolo_result.filter_by_label(BaseUILabels.BUTTON)
        self.action_buttons: Yolo_Results | None = None
        self.panel_box: Yolo_Box | None = None
        self.header_box: Yolo_Box | None = None
        self.body_box: Yolo_Box | None = None

    def _warn(self, message: str):
        if not self.quiet:
            logger.warning(message)

    def parse(self) -> Modal | None:
        if not self.buttons:
            self._warn("Modal buttons not found")
            return None

        self.action_buttons = self._get_action_buttons()
        if not self.action_buttons:
            self._warn("Modal action buttons not found")
            return None

        self.header_box = self._get_header_box()
        if self.header_box is None:
            self._warn("Modal header not found")
            return None

        self.panel_box = self._get_panel_box()

        modal_title = self._extract_title()
        if not modal_title:
            self._warn("Modal title OCR failed")
            return None

        confirm_button, cancel_button = self._get_confirm_cancel_buttons()
        if not confirm_button and not cancel_button:
            self._warn("Cancel or Confirm buttons not found")
            return None

        body_frame = self._build_body_box(confirm_button, cancel_button)
        if body_frame is None:
            self._warn("Modal body region is invalid")
            return None

        modal_body_text = None
        if not self.no_body:
            ocr_result = ocr_service.ocr(body_frame)
            modal_body_text = " ".join(item.text for item in ocr_result) if ocr_result else ""

        action_buttons = [Button(button, no_text=True) for button in self.action_buttons.boxes]
        return Modal(
            modal_title,
            body_frame,
            modal_body_text,
            confirm_button,
            cancel_button,
            self.panel_box,
            self.header_box,
            self.body_box,
            action_buttons,
            self.frame.copy(),
        )

    def draw_debug(self, modal: Modal | None = None) -> np.ndarray:
        if modal is not None:
            return modal.draw_debug(self.frame)

        if self.frame is None or self.frame.size == 0:
            raise ValueError("Modal debug image requires a source frame")

        debug_tools.clear_all()
        try:
            if self.panel_box:
                debug_tools.add_box(
                    int(self.panel_box.x),
                    int(self.panel_box.y),
                    int(self.panel_box.w),
                    int(self.panel_box.h),
                    color=(0, 255, 255),
                    label="panel",
                    duration=300,
                )
            if self.header_box:
                debug_tools.add_box(
                    int(self.header_box.x),
                    int(self.header_box.y),
                    int(self.header_box.w),
                    int(self.header_box.h),
                    color=(0, 165, 255),
                    label="header",
                    duration=300,
                )
            if self.body_box:
                debug_tools.add_box(
                    int(self.body_box.x),
                    int(self.body_box.y),
                    int(self.body_box.w),
                    int(self.body_box.h),
                    color=(255, 215, 0),
                    label="body",
                    duration=300,
                )
            if self.action_buttons:
                for button in self.action_buttons:
                    debug_tools.add_box(
                        int(button.x),
                        int(button.y),
                        int(button.w),
                        int(button.h),
                        color=(255, 0, 255),
                        label="action",
                        duration=300,
                    )
            return debug_tools.draw_boxes(self.frame.copy())
        finally:
            debug_tools.clear_all()

    def _get_action_buttons(self) -> Yolo_Results | None:
        button_groups = self.buttons.group_yolo_boxes_by_position(30, None)
        if not button_groups:
            return None
        frame_height = self.frame.shape[0]
        lower_half_groups = [
            group for group in button_groups
            if max(box.cy for box in group.boxes) >= frame_height * 0.4
        ]
        candidate_groups = lower_half_groups or button_groups
        return max(
            candidate_groups,
            key=lambda group: (
                max(box.cy for box in group.boxes),
                max(box.w for box in group.boxes) - min(box.x for box in group.boxes),
                len(group.boxes),
            ),
        )

    def _get_header_box(self) -> Yolo_Box | None:
        headers = self.yolo_result.filter_by_label(BaseUILabels.MODAL_HEADER)
        if headers:
            action_top = min(box.y for box in self.action_buttons.boxes)
            action_left = min(box.x for box in self.action_buttons.boxes)
            action_right = max(box.w for box in self.action_buttons.boxes)
            candidate_headers = [
                header for header in headers
                if header.h <= action_top and max(0, min(header.w, action_right) - max(header.x, action_left)) > 0
            ]
            if candidate_headers:
                return max(candidate_headers, key=lambda header: (header.w - header.x, -header.y))
            return headers.get_x_min_element().first()
        return self._infer_header_from_frame()

    def _get_panel_box(self) -> Yolo_Box | None:
        frame_height, frame_width = self.frame.shape[:2]
        action_left = min(box.x for box in self.action_buttons.boxes)
        action_right = max(box.w for box in self.action_buttons.boxes)
        action_bottom = max(box.h for box in self.action_buttons.boxes)
        search_left = max(0, int(min(self.header_box.x, action_left) - 24))
        search_right = min(frame_width, int(max(self.header_box.w, action_right) + 24))
        search_top = max(0, int(self.header_box.y - 24))
        search_bottom = min(frame_height, int(action_bottom + 24))
        search_frame = self.frame[search_top:search_bottom, search_left:search_right]
        if search_frame.size == 0:
            return None

        lab = cv2.cvtColor(search_frame, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]
        a_ch = lab[:, :, 1]
        b_ch = lab[:, :, 2]
        # White in LAB: high lightness, neutral chroma (a* and b* near 128)
        white_mask = (
            (l_ch > 200).astype(np.uint8)
            & (a_ch > 118).astype(np.uint8)
            & (a_ch < 138).astype(np.uint8)
            & (b_ch > 115).astype(np.uint8)
            & (b_ch < 140).astype(np.uint8)
        ) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        header_width = self.header_box.w - self.header_box.x
        action_top = min(box.y for box in self.action_buttons.boxes)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x1 = search_left + x
            y1 = search_top + y
            x2 = x1 + w
            y2 = y1 + h
            if w < header_width * 0.85:
                continue
            if h < max(120, (action_bottom - self.header_box.y) * 0.55):
                continue
            if x1 > self.header_box.x + 12 or x2 < self.header_box.w - 12:
                continue
            if y1 > self.header_box.y + 12 or y2 < action_top:
                continue
            candidates.append((x1, y1, x2, y2, w * h))

        if candidates:
            x1, y1, x2, y2, _ = max(
                candidates,
                key=lambda box: (
                    box[4],
                    -abs(((box[0] + box[2]) // 2) - frame_width // 2),
                ),
            )
            return Yolo_Box(x1, y1, x2, y2, "ModalPanel", self.frame[y1:y2, x1:x2])

        # Fall back to a layout box derived from the stable header/action anchors.
        x1 = max(0, int(min(self.header_box.x, action_left) - 12))
        y1 = max(0, int(self.header_box.y - 12))
        x2 = min(frame_width, int(max(self.header_box.w, action_right) + 12))
        y2 = min(frame_height, int(action_bottom + 12))
        return Yolo_Box(x1, y1, x2, y2, "ModalPanel", self.frame[y1:y2, x1:x2])

    def _infer_header_from_frame(self) -> Yolo_Box | None:
        """Infer the modal header location using LAB color space detection.

        The modal header is an orange/yellow gradient bar above the action
        buttons.  We detect it by converting to CIELAB and looking for warm
        colours (high *a** and *b** channels) that form a wide, thin
        horizontal band aligned with the action buttons.

        LAB is more robust to JPEG compression artifacts than HSV because
        luminance (L*) is decorrelated from chrominance (a*, b*), so
        quantization noise in brightness does not shift the perceived hue.
        """
        frame_height, frame_width = self.frame.shape[:2]
        action_top = min(box.y for box in self.action_buttons.boxes)
        action_left = min(box.x for box in self.action_buttons.boxes)
        action_right = max(box.w for box in self.action_buttons.boxes)
        action_width = action_right - action_left

        search_top = max(0, int(frame_height * 0.25))
        search_bottom = max(search_top + 1, action_top - 20)
        if search_bottom <= search_top:
            return None

        search_frame = self.frame[search_top:search_bottom, :]
        blurred = cv2.GaussianBlur(search_frame, (5, 5), 0)

        # --- LAB-based warm colour detection ---
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # Orange / warm-yellow in LAB:
        #   L* > 160  (bright),  a* > 135  (toward red),  b* > 150  (toward yellow)
        # A secondary, slightly looser mask catches paler header variants.
        warm_mask = (
            (l_ch > 160).astype(np.uint8)
            & (a_ch > 135).astype(np.uint8)
            & (b_ch > 150).astype(np.uint8)
        ) * 255
        pale_mask = (
            (l_ch > 180).astype(np.uint8)
            & (a_ch > 128).astype(np.uint8)
            & (b_ch > 140).astype(np.uint8)
        ) * 255
        orange_mask = cv2.bitwise_or(warm_mask, pale_mask)

        orange_mask = cv2.morphologyEx(
            orange_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7)),
        )
        orange_mask = cv2.morphologyEx(
            orange_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        )

        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        min_width = max(int(frame_width * 0.6), int(action_width * 0.9))
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_width or h < 25:
                continue
            x1 = x
            y1 = y + search_top
            x2 = x + w
            y2 = y1 + h
            overlap = max(0, min(x2, action_right) - max(x1, action_left))
            if overlap < action_width * 0.5:
                continue
            candidates.append((x1, y1, x2, y2))

        if not candidates:
            return None

        x1, y1, x2, y2 = min(candidates, key=lambda box: (box[1], -(box[2] - box[0])))
        return Yolo_Box(x1, y1, x2, y2, BaseUILabels.MODAL_HEADER, self.frame[y1:y2, x1:x2])

    def _extract_title(self) -> str:
        header_buttons = [
            button for button in self.buttons.boxes
            if self.header_box.x < button.cx < self.header_box.w and self.header_box.y < button.cy < self.header_box.h
        ]
        header_button_min_x = min(
            (button.x - self.header_box.x for button in header_buttons),
            default=self.header_box.frame.shape[1],
        )
        header_title_frame = self.header_box.frame[:, :max(1, header_button_min_x)]
        ocr_result = ocr_service.ocr(header_title_frame)
        if not ocr_result:
            ocr_result = ocr_service.ocr(self.header_box.frame)
            if not ocr_result:
                return ""
        ocr_result = ocr_result.auto_merge_lines(width_gap=30)
        return "".join(item.text for item in ocr_result).replace(" ", "")

    def _get_confirm_cancel_buttons(self) -> tuple[Button | None, Button | None]:
        if len(self.action_buttons) >= 2:
            confirm_result = self.action_buttons.get_x_max_element()
            cancel_result = self.action_buttons.get_x_min_element()
            return Button(confirm_result.first()), Button(cancel_result.first())
        single_result = self.action_buttons.get_y_max_element()
        return None, Button(single_result.first())

    def _build_body_box(self, confirm_button: Button | None, cancel_button: Button | None) -> np.ndarray | None:
        if confirm_button and cancel_button:
            modal_body_y = max(cancel_button.y, confirm_button.y)
        else:
            modal_body_y = confirm_button.y if confirm_button else cancel_button.y

        panel_left = int(self.panel_box.x) if self.panel_box else int(self.header_box.x)
        panel_right = int(self.panel_box.w) if self.panel_box else int(self.header_box.w)
        panel_top = int(self.panel_box.y) if self.panel_box else int(self.header_box.y)
        if modal_body_y <= self.header_box.h or modal_body_y <= panel_top:
            return None

        x1 = panel_left
        y1 = int(self.header_box.h)
        x2 = panel_right
        y2 = int(min(modal_body_y, self.panel_box.h if self.panel_box else modal_body_y))
        self.body_box = Yolo_Box(x1, y1, x2, y2, "ModalBody", self.frame[y1:y2, x1:x2])
        return self.body_box.frame
