"""
竞技场(Contest)页面 OCR 诊断 + 图像采集脚本

连接ADB设备，截图后使用YOLO检测和OCR识别，
逐步分析ContestItem解析失败的根因（"総合力合計"锚点找不到）。

用法：
  1. 手动将游戏导航到竞技场主页（能看到3个对手）
  2. 运行:
     python -m tests.test_contest_ocr_diagnostic
"""
import os
import sys
import json
import time

import cv2
import numpy as np
import adbutils

# 保证项目根在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.inference.ocr_engine import OCRService
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Contest import ContestList, ContestItem
from src.utils.opencv_tools import gen_color_mask
from src.utils.string_tools import string_match, MatchConfig

OUT_DIR = os.path.join("logs", "debug", "test_captures", "contest")
os.makedirs(OUT_DIR, exist_ok=True)

model = YoloModelFromONNX(config.model_config["BASE_UI"])
ocr_service = OCRService()


def adb_screenshot(dev) -> np.ndarray:
    pil_img = dev.screenshot()
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def detect(frame: np.ndarray) -> Yolo_Results:
    raw = model(frame, conf_threshold=0.5, iou_threshold=0.5)
    return Yolo_Results(raw, frame)


def save_image(frame, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    cv2.imwrite(path, frame)
    print(f"  [SAVED] {name}.png  ({frame.shape[1]}x{frame.shape[0]})")
    return path


def diagnose_contest_ocr(frame: np.ndarray, results: Yolo_Results, capture_index: int):
    """
    从完整帧中提取竞技场区域，用颜色分割找到对手卡片，
    逐张卡片跑OCR，输出每个OCR结果的 text/confidence/位置。
    """
    height, width = frame.shape[:2]
    report = {"capture_index": capture_index, "items": []}

    contest_list = ContestList(results, frame)
    start_y = int(contest_list._start_y)
    end_y = int(contest_list._end_y)
    contest_area = contest_list.contest_area

    if contest_area is None or contest_area.size == 0:
        print("  [WARN] Contest area is empty after bounds fallback")
        return report

    print(f"  Contest area: y=[{start_y}, {end_y}], width={width}")
    save_image(contest_area, f"cap{capture_index:02d}_contest_area")

    # 颜色分割找对手卡片（复刻 _get_contest_items 逻辑）
    ca_h, ca_w = contest_area.shape[:2]

    lower1, upper1 = (0, 0, 75), (179, 75, 140)
    lower2, upper2 = (0, 0, 235), (179, 15, 255)
    mask1 = gen_color_mask(contest_area, lower1, upper1)
    mask2 = gen_color_mask(contest_area, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > ca_w * 0.5:
            roi = contest_area[y:y + h, x:x + w]
            rois.append((x, y, w, h, roi))

    print(f"  Found {len(rois)} contest item ROIs (need 3)")

    for idx, (rx, ry, rw, rh, roi) in enumerate(rois):
        save_image(roi, f"cap{capture_index:02d}_item{idx}")
        ocr_results = ocr_service.ocr(roi)
        item_report = {
            "roi_index": idx,
            "roi_xywh": [rx, ry, rw, rh],
            "ocr_results": [],
        }

        print(f"\n  --- Item {idx} (roi {rw}x{rh}) ---")
        for r in ocr_results:
            record = {
                "text": r.text,
                "confidence": float(r.confidence),
                "x": int(r.x), "y": int(r.y),
                "w": int(r.w), "h": int(r.h),
            }
            item_report["ocr_results"].append(record)
            # 检查是否 fuzzy 匹配 "総合力合計"
            match_result = string_match(r.text, "総合力合計", MatchConfig(use_fuzz=True, fuzz_threshold=60))
            marker = " <<<ANCHOR>>>" if match_result else ""
            contains = " [CONTAINS]" if "総合力合計" in r.text else ""
            print(f"    OCR: text={r.text!r:30s}  conf={r.confidence:.2f}  "
                  f"pos=({r.x},{r.y},{r.w},{r.h}){contains}{marker}")

        # 尝试直接用 ContestItem 解析
        try:
            ci = ContestItem(rx, start_y + ry, rx + rw, start_y + ry + rh, f"test_{idx}", roi)
            item_report["parse_ok"] = True
            item_report["combat_power"] = ci.combat_power
            item_report["pt"] = ci.pt
            item_report["username"] = ci.username
            print(f"    ✓ ContestItem parsed: power={ci.combat_power}, pt={ci.pt}, user={ci.username}")
        except Exception as e:
            item_report["parse_ok"] = False
            item_report["error"] = str(e)
            print(f"    ✗ ContestItem FAILED: {e}")

        report["items"].append(item_report)

    return report


def main():
    print("=== 竞技场 OCR 诊断脚本 ===")
    print("请确保游戏已导航到竞技场页面（能看到3个对手）\n")

    dev = adbutils.adb.device_list()
    if not dev:
        print("ERROR: No ADB device found!")
        return
    dev = dev[0]
    print(f"设备: {dev.serial}")

    all_reports = []
    NUM_CAPTURES = 5

    for cap_idx in range(NUM_CAPTURES):
        print(f"\n{'='*60}")
        print(f"Capture {cap_idx + 1}/{NUM_CAPTURES}")
        print(f"{'='*60}")

        frame = adb_screenshot(dev)
        save_image(frame, f"cap{cap_idx:02d}_full")

        results = detect(frame)
        print(f"  YOLO boxes: {len(results.boxes)}")
        for b in results.boxes:
            print(f"    {b.label:25s} ({b.x},{b.y},{b.w},{b.h})")

        report = diagnose_contest_ocr(frame, results, cap_idx)
        all_reports.append(report)

        if cap_idx < NUM_CAPTURES - 1:
            # 点击刷新按钮或等待一小会儿让帧略有差异
            print("\n  等待2秒后再截图（可手动操作切换对手）...")
            time.sleep(2)

    # 保存诊断报告
    report_path = os.path.join(OUT_DIR, "ocr_diagnostic_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f"\n全部诊断报告已保存到: {report_path}")

    # 统计
    total_items = sum(len(r["items"]) for r in all_reports)
    parsed_ok = sum(1 for r in all_reports for item in r["items"] if item.get("parse_ok"))
    print(f"\n总计: {total_items} 个对手项, {parsed_ok} 个解析成功, {total_items - parsed_ok} 个失败")
    if parsed_ok < total_items:
        print(">>> 需要修复 ContestItem 的锚点匹配逻辑以提高鲁棒性 <<<")


if __name__ == "__main__":
    main()
