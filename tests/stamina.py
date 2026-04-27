#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.constants.yolo.model_type import YoloModelType
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.tasks.producer_challenge.gameplay.decision_support.hud import (
    _extract_noisy_hud_value,
    _parse_stamina_text,
)
from src.core.tasks.producer_challenge.shared.common import ocr_text
from src.utils.string_tools import fullwidth_to_halfwidth

DEFAULT_IMAGE = PROJECT_ROOT / "tests" / "stamina.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "_artifacts"


@dataclass
class AnchorSpec:
    name: str
    lower_hsv: tuple[int, int, int]
    upper_hsv: tuple[int, int, int]
    search_y1_ratio: float
    search_y2_ratio: float
    min_area_ratio: float
    min_aspect: float
    max_aspect: float
    prefer_rightmost: bool


_SHIELD_SPEC = AnchorSpec(
    name="shield",
    lower_hsv=(62, 0, 175),
    upper_hsv=(93, 190, 255),
    search_y1_ratio=0.00,
    search_y2_ratio=0.62,
    min_area_ratio=0.004,
    min_aspect=0.50,
    max_aspect=2.60,
    prefer_rightmost=True,
)

_HEART_SPEC = AnchorSpec(
    name="heart",
    lower_hsv=(0, 39, 97),
    upper_hsv=(88, 169, 255),
    search_y1_ratio=0.34,
    search_y2_ratio=1.00,
    min_area_ratio=0.003,
    min_aspect=0.45,
    max_aspect=2.80,
    prefer_rightmost=False,
)

_BAR_SPEC = AnchorSpec(
    name="stamina_bar",
    lower_hsv=(19, 143, 0),
    upper_hsv=(100, 186, 255),
    search_y1_ratio=0.00,
    search_y2_ratio=0.56,
    min_area_ratio=0.010,
    min_aspect=2.0,
    max_aspect=20.0,
    prefer_rightmost=False,
)


def _capture_adb(serial: str) -> np.ndarray:
    raw = subprocess.check_output(
        ["adb", "-s", serial, "exec-out", "screencap", "-p"],
        timeout=10,
    )
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("ADB 截图解码失败")
    return image


def _locate_stamina_panel(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[str, float, tuple[int, int, int, int]]]]:
    model = YoloModelFromONNX(config.model_config[YoloModelType.PRODUCER])
    result = model(frame, conf_threshold=0.35)
    detections: list[tuple[str, float, tuple[int, int, int, int]]] = []
    panel_box: tuple[int, int, int, int] | None = None
    panel_score = -1.0
    h, w = frame.shape[:2]

    for box, score, cls_id in zip(result.boxes, result.scores, result.class_ids):
        x, y, bw, bh = map(int, box)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        bw = max(1, bw)
        bh = max(1, bh)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        if x2 <= x or y2 <= y:
            continue
        label = result.model_mata.names.get(int(cls_id), str(cls_id))
        detections.append((label, float(score), (x, y, x2, y2)))
        if label == ProducerLabels.PC_STAMINA and float(score) > panel_score:
            panel_score = float(score)
            panel_box = (x, y, x2, y2)

    if panel_box is None:
        return frame, (0, 0, frame.shape[1], frame.shape[0]), detections
    x1, y1, x2, y2 = panel_box
    return frame[y1:y2, x1:x2], panel_box, detections


def _find_anchor(panel: np.ndarray, spec: AnchorSpec) -> tuple[tuple[int, int, int, int] | None, np.ndarray]:
    h, w = panel.shape[:2]
    search_y1 = int(h * spec.search_y1_ratio)
    search_y2 = int(h * spec.search_y2_ratio)
    search = panel[search_y1:search_y2, :]
    if search.size == 0:
        return None, np.zeros((h, w), dtype=np.uint8)

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(spec.lower_hsv, dtype=np.uint8),
        np.array(spec.upper_hsv, dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[search_y1:search_y2, :] = mask

    min_area = float(mask.shape[0] * mask.shape[1]) * spec.min_area_ratio
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        ww = int(stats[idx, cv2.CC_STAT_WIDTH])
        hh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if ww <= 0 or hh <= 0 or area < min_area:
            continue
        aspect = ww / max(1.0, float(hh))
        if aspect < spec.min_aspect or aspect > spec.max_aspect:
            continue
        right_bias = float(x + ww) / max(1.0, float(mask.shape[1]))
        score = area * 1.0 + (right_bias * 120.0 if spec.prefer_rightmost else 0.0)
        candidates.append((score, (x, y + search_y1, x + ww, y + hh + search_y1)))

    if not candidates:
        return None, full_mask
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], full_mask


def _ocr_right_of_anchor(panel: np.ndarray, anchor: tuple[int, int, int, int], *, min_x1_ratio: float, x_padding: int = 4, y_padding: int = 6) -> tuple[str, tuple[int, int, int, int]]:
    h, w = panel.shape[:2]
    ax1, ay1, ax2, ay2 = anchor
    x1 = min(w - 1, max(int(w * min_x1_ratio), ax2 + x_padding))
    y1 = max(0, ay1 - y_padding)
    y2 = min(h, ay2 + y_padding)
    crop = panel[y1:y2, x1:w]
    if crop.size == 0 or crop.shape[1] < max(8, int(w * 0.04)):
        fx1 = max(0, ax1 + int((ax2 - ax1) * 0.45))
        fx2 = min(w, ax2 + max(2, x_padding))
        fy1 = max(0, ay1 - y_padding)
        fy2 = min(h, ay2 + y_padding)
        fallback = panel[fy1:fy2, fx1:fx2]
        return ocr_text(fallback), (fx1, fy1, fx2, fy2)
    return ocr_text(crop), (x1, y1, w, y2)


def _extract_digits(text: str) -> str:
    return "".join(re.findall(r"\d+", fullwidth_to_halfwidth(str(text or ""))))


def _draw_box(canvas: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        canvas,
        label,
        (x1, max(14, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def run(image: np.ndarray) -> tuple[np.ndarray, dict[str, str | int]]:
    panel, panel_box, detections = _locate_stamina_panel(image)
    shield_anchor, shield_mask = _find_anchor(panel, _SHIELD_SPEC)
    heart_anchor, heart_mask = _find_anchor(panel, _HEART_SPEC)
    bar_anchor, bar_mask = _find_anchor(panel, _BAR_SPEC)

    genki_sources: list[str] = []
    stamina_sources: list[str] = []
    ocr_regions: dict[str, tuple[int, int, int, int]] = {}

    if shield_anchor is not None:
        text, rect = _ocr_right_of_anchor(panel, shield_anchor, min_x1_ratio=0.56)
        genki_sources.append(text)
        ocr_regions["genki"] = rect
    if bar_anchor is not None:
        text, rect = _ocr_right_of_anchor(panel, bar_anchor, min_x1_ratio=0.46)
        genki_sources.append(text)
        ocr_regions["genki_bar"] = rect
    # 额外固定区域补样，增强小图与压缩图的鲁棒性。
    h, w = panel.shape[:2]
    strict_genki = panel[int(h * 0.02):int(h * 0.48), int(w * 0.66):int(w * 0.99)]
    strict_stamina = panel[int(h * 0.44):int(h * 0.99), int(w * 0.34):int(w * 0.99)]
    genki_sources.append(ocr_text(strict_genki))
    stamina_sources.append(ocr_text(strict_stamina))
    ocr_regions["genki_strict"] = (int(w * 0.66), int(h * 0.02), int(w * 0.99), int(h * 0.48))
    ocr_regions["stamina_strict"] = (int(w * 0.34), int(h * 0.44), int(w * 0.99), int(h * 0.99))
    if heart_anchor is not None:
        text, rect = _ocr_right_of_anchor(panel, heart_anchor, min_x1_ratio=0.30)
        stamina_sources.append(text)
        ocr_regions["stamina"] = rect

    genki_value, _ = _extract_noisy_hud_value(*genki_sources, upper_bound=999)
    stamina_value, _ = _extract_noisy_hud_value(*stamina_sources, upper_bound=999)
    stamina_joined = " ".join(stamina_sources)
    parsed_stamina, parsed_max = _parse_stamina_text(
        stamina_joined,
        previous_stamina=stamina_value,
        previous_max_stamina=0,
    )
    has_slash = "/" in fullwidth_to_halfwidth(stamina_joined)
    if has_slash and parsed_stamina > 0:
        stamina_value = parsed_stamina
    max_stamina = parsed_max if has_slash and parsed_max > 0 else stamina_value

    canvas = image.copy()
    px1, py1, px2, py2 = panel_box
    _draw_box(canvas, panel_box, (255, 200, 0), "pc_stamina_panel")

    def _to_abs(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = rect
        return (px1 + x1, py1 + y1, px1 + x2, py1 + y2)

    if shield_anchor is not None:
        _draw_box(canvas, _to_abs(shield_anchor), (255, 100, 80), "shield_anchor")
    if heart_anchor is not None:
        _draw_box(canvas, _to_abs(heart_anchor), (80, 220, 120), "heart_anchor")
    if bar_anchor is not None:
        _draw_box(canvas, _to_abs(bar_anchor), (80, 180, 255), "bar_anchor")

    for key, rect in ocr_regions.items():
        _draw_box(canvas, _to_abs(rect), (220, 220, 220), f"ocr_{key}")

    mask_board = np.zeros((panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
    mask_board[:, :, 2] = shield_mask
    mask_board[:, :, 1] = heart_mask
    mask_board[:, :, 0] = bar_mask
    mask_board = cv2.resize(mask_board, (max(1, panel.shape[1] * 2), max(1, panel.shape[0] * 2)))

    hud_text = (
        f"GENKI={genki_value} | STAMINA={stamina_value}/{max_stamina} | "
        f"genki_raw={[_extract_digits(s) for s in genki_sources]} | "
        f"stamina_raw={[_extract_digits(s) for s in stamina_sources]}"
    )
    cv2.putText(canvas, hud_text[:160], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

    panel_large = cv2.resize(panel, (max(1, panel.shape[1] * 2), max(1, panel.shape[0] * 2)))
    board_h = max(canvas.shape[0], panel_large.shape[0], mask_board.shape[0])
    total_w = canvas.shape[1] + panel_large.shape[1] + mask_board.shape[1]
    board = np.zeros((board_h, total_w, 3), dtype=np.uint8)
    board[:canvas.shape[0], :canvas.shape[1]] = canvas
    board[:panel_large.shape[0], canvas.shape[1]:canvas.shape[1] + panel_large.shape[1]] = panel_large
    board[:mask_board.shape[0], canvas.shape[1] + panel_large.shape[1]:] = mask_board

    summary = {
        "genki": int(genki_value),
        "stamina": int(stamina_value),
        "max_stamina": int(max_stamina),
        "genki_sources": " | ".join(genki_sources),
        "stamina_sources": " | ".join(stamina_sources),
        "detections": len(detections),
    }
    return board, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="体力/元气 HUD 可视化测试")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE), help="输入截图路径")
    parser.add_argument("--adb", action="store_true", help="从 ADB 实时截图")
    parser.add_argument("--serial", type=str, default="127.0.0.1:16384", help="ADB 设备序列号")
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    if args.adb:
        image = _capture_adb(args.serial)
        source = f"adb:{args.serial}"
    else:
        image_path = Path(args.image).expanduser().resolve()
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        source = str(image_path)

    board, summary = run(image)
    output_dir = Path(args.save_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = output_dir / f"stamina_debug_{ts}.png"
    cv2.imwrite(str(out_path), board)

    print(f"[stamina] source={source}")
    print(f"[stamina] output={out_path}")
    print(
        f"[stamina] genki={summary['genki']} "
        f"stamina={summary['stamina']}/{summary['max_stamina']} "
        f"detections={summary['detections']}"
    )
    print(f"[stamina] genki_sources={summary['genki_sources']}")
    print(f"[stamina] stamina_sources={summary['stamina_sources']}")


if __name__ == "__main__":
    main()
