"""
支援卡详情页按钮检测诊断脚本（上限解放 / サポート変換 / Lv強化）

在设备上采集多张支援卡的详情页截图，
验证 YOLO + OCR 能否正确识别所有功能按钮。

用法：
  1. 手动将游戏导航到支援卡列表页面
  2. 运行:
     python -m tests.test_support_card_detail_diagnostic
"""
import os
import sys
import json
import time

import cv2
import numpy as np
import adbutils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.inference.ocr_engine import OCRService
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.SupportCard import SupportCardListParser
from src.constants.game.text.support_card_text import SupportCardText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.utils.string_tools import MatchConfig

OUT_DIR = os.path.join("logs", "debug", "test_captures", "support_card_detail")
os.makedirs(OUT_DIR, exist_ok=True)

model = YoloModelFromONNX(config.model_config["BASE_UI"])
ocr_service = OCRService()
_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)

# 需要检测的按钮文本列表
BUTTON_TEXTS = {
    "Lv強化": SupportCardText.LV_ENHANCE,
    "上限解放": SupportCardText.LIMIT_BREAK,
    "サポート変換": SupportCardText.SUPPORT_CONVERT,
}


def adb_screenshot(dev) -> np.ndarray:
    pil_img = dev.screenshot()
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def detect(frame: np.ndarray) -> Yolo_Results:
    raw = model(frame, conf_threshold=0.5, iou_threshold=0.5)
    return Yolo_Results(raw, frame)


def save_image(frame, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    cv2.imwrite(path, frame)
    print(f"  [SAVED] {name}.png")
    return path


def diagnose_detail_page(frame: np.ndarray, results: Yolo_Results, card_idx: int):
    """分析详情页上的按钮检测情况"""
    report = {"card_index": card_idx, "buttons": {}}

    buttons = ButtonList(results)
    print(f"\n  全部按钮 ({len(buttons.buttons)}):")
    for btn in buttons.buttons:
        print(f"    text={btn.text!r:30s}  disabled={btn.is_disabled()}  "
              f"pos=({btn.x},{btn.y},{btn.w},{btn.h})")

    for label, text_const in BUTTON_TEXTS.items():
        btn = buttons.get_button_by_text(text_const, _FUZZ)
        if btn:
            report["buttons"][label] = {
                "found": True,
                "disabled": btn.is_disabled(),
                "text": btn.text,
                "pos": [int(btn.x), int(btn.y), int(btn.w), int(btn.h)],
            }
            status = "DISABLED" if btn.is_disabled() else "ENABLED"
            print(f"  [{status}] {label}: text={btn.text!r}")
        else:
            report["buttons"][label] = {"found": False}
            print(f"  [NOT FOUND] {label}")

    return report


def navigate_to_card_detail(dev, card_cx, card_cy):
    """Click a card, then click 詳細を見る to enter detail page."""
    dev.click(card_cx, card_cy)
    time.sleep(1)

    # Check if we need to click 詳細を見る
    frame = adb_screenshot(dev)
    results = detect(frame)
    buttons = ButtonList(results)
    detail_btn = buttons.get_button_by_text(SupportCardText.VIEW_DETAIL, _FUZZ)
    if detail_btn:
        dev.click(detail_btn.cx, detail_btn.cy)
        time.sleep(1.5)
    else:
        # Maybe clicking the card directly went to detail (already selected)
        time.sleep(0.5)

    return adb_screenshot(dev), detect(adb_screenshot(dev))


def navigate_back_to_list(dev):
    """From detail page, press back to return to card list."""
    frame = adb_screenshot(dev)
    results = detect(frame)
    back_btns = results.filter_by_label(BaseUILabels.BACK_BTN)
    if back_btns:
        back = list(back_btns)[0]
        dev.click(back.cx, back.cy)
    else:
        dev.keyevent(4)  # KEYCODE_BACK
    time.sleep(1.5)


def main():
    print("=== 支援卡详情页按钮检测诊断 ===")
    print(f"保存目录: {OUT_DIR}")
    print("请确保游戏已导航到支援卡列表页面\n")

    dev = adbutils.adb.device_list()
    if not dev:
        print("ERROR: No ADB device found!")
        return
    dev = dev[0]
    print(f"设备: {dev.serial}")

    all_reports = []
    NUM_CARDS = 5

    # 首先截取列表页面
    frame = adb_screenshot(dev)
    results = detect(frame)
    save_image(frame, "00_list_page")

    card_list = SupportCardListParser(results).parse()
    cards = card_list.cards
    print(f"列表页检测到 {len(cards)} 张卡")

    for card_idx in range(min(NUM_CARDS, len(cards))):
        print(f"\n{'='*60}")
        print(f"Card {card_idx + 1}/{NUM_CARDS}")
        print(f"{'='*60}")

        card = cards[card_idx]
        print(f"  卡片: Lv={card.level} Rarity={card.rarity} Stars={card.stars}")
        print(f"  位置: cx={card.box.cx} cy={card.box.cy}")

        # 进入详情页
        detail_frame, detail_results = navigate_to_card_detail(dev, card.box.cx, card.box.cy)
        save_image(detail_frame, f"card{card_idx:02d}_detail")

        # 重新截图确保稳定
        time.sleep(0.5)
        detail_frame = adb_screenshot(dev)
        detail_results = detect(detail_frame)
        save_image(detail_frame, f"card{card_idx:02d}_detail_stable")

        report = diagnose_detail_page(detail_frame, detail_results, card_idx)
        report["card_info"] = {
            "level": card.level,
            "rarity": card.rarity,
            "stars": card.stars,
        }
        all_reports.append(report)

        # 返回列表
        navigate_back_to_list(dev)

        # 重新截图获取列表
        time.sleep(0.5)
        frame = adb_screenshot(dev)
        results = detect(frame)
        card_list = SupportCardListParser(results).parse()
        cards = card_list.cards
        print(f"  返回列表，检测到 {len(cards)} 张卡")

    # 保存报告
    report_path = os.path.join(OUT_DIR, "detail_diagnostic_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f"\n诊断报告已保存到: {report_path}")

    # 统计
    total = len(all_reports)
    for label in BUTTON_TEXTS:
        found = sum(1 for r in all_reports if r["buttons"].get(label, {}).get("found"))
        print(f"  {label}: {found}/{total} 找到")


if __name__ == "__main__":
    main()
