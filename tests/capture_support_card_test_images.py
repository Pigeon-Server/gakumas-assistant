"""
采集支援卡强化流程各步骤的测试图像。

使用方式：
  1. 确保 ADB 已连接设备，设备处于支援卡列表页面
  2. python tests/capture_support_card_test_images.py
  3. 脚本会自动操作设备，采集不同卡片在各个步骤的截图
  4. 截图保存到 logs/debug/test_captures/support_card/

采集步骤：
  Step 1: 卡片列表页面 (card_list)
  Step 2: 点击卡片后（选中状态，詳細を見る 按钮出现）(card_selected)
  Step 3: 详情页面 (detail_page)
  Step 4: 强化页面 (enhance_page)
  Step 5: 返回列表 (back_to_list)
"""

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.device.Android.app import Android_App
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.SupportCard import SupportCardListParser
from src.constants.game.text.support_card_text import SupportCardText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.utils.string_tools import MatchConfig

_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)
CAPTURE_DIR = os.path.join("logs", "debug", "test_captures", "support_card")
NUM_CARDS_TO_TEST = 3  # 测试几张不同的卡片


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save(frame, step_name, card_idx):
    ensure_dir(CAPTURE_DIR)
    fname = f"{step_name}_card{card_idx}.png"
    path = os.path.join(CAPTURE_DIR, fname)
    cv2.imwrite(path, frame)
    print(f"  [SAVED] {fname}")
    return path


def detect(model, frame):
    results = model(frame)
    return Yolo_Results(results, frame)


def main():
    print("=== 支援卡强化流程测试图像采集 ===")
    print(f"目标采集卡片数: {NUM_CARDS_TO_TEST}")
    print(f"保存目录: {CAPTURE_DIR}")
    ensure_dir(CAPTURE_DIR)

    device = Android_App()
    model = YoloModelFromONNX("model/base_ui.onnx")
    w, h = device.get_window_size()
    print(f"设备: {w}x{h}")

    # Wait for frame stable
    time.sleep(2)

    for card_idx in range(NUM_CARDS_TO_TEST):
        print(f"\n--- Card {card_idx + 1}/{NUM_CARDS_TO_TEST} ---")

        # ── Step 1: Card list page ──
        print("Step 1: 卡片列表页面")
        time.sleep(1)
        frame = device.capture()
        yr = detect(model, frame)
        save(frame, "step1_card_list", card_idx)

        # Parse support cards
        card_list = SupportCardListParser(yr).parse()
        print(f"  检测到 {len(card_list)} 张卡片")
        for c in card_list:
            if not c.occluded:
                print(f"    {c.rarity} Lv{c.level} ★{c.stars} lb={c.limit_break}")

        # Select a visible non-occluded card
        visible = [c for c in card_list if not c.occluded]
        if not visible:
            print("  没有可见卡片，跳过")
            continue

        # Pick different cards for each iteration
        target_card = visible[min(card_idx, len(visible) - 1)]
        print(f"  选择: {target_card.rarity} Lv{target_card.level}")

        # ── Step 2: Click card thumbnail → selected state ──
        print("Step 2: 点击卡片")
        device.click_element(target_card.box)
        time.sleep(1.5)
        frame = device.capture()
        yr = detect(model, frame)
        save(frame, "step2_card_selected", card_idx)

        btns = ButtonList(yr)
        detail_btn = btns.get_button_by_text(SupportCardText.VIEW_DETAIL, _FUZZ)
        print(f"  詳細を見る: {'FOUND' if detail_btn else 'NOT FOUND'}")

        convert_btn = btns.get_button_by_text(SupportCardText.SUPPORT_CONVERT, _FUZZ)
        print(f"  サポート変換: {'FOUND' if convert_btn else 'NOT FOUND'}")

        if not detail_btn:
            print("  [SKIP] 詳細を見る 未找到，跳过此卡")
            continue

        # ── Step 3: Enter detail page ──
        print("Step 3: 进入详情页")
        device.click_element(detail_btn)
        time.sleep(2)
        frame = device.capture()
        yr = detect(model, frame)
        save(frame, "step3_detail_page", card_idx)

        btns = ButtonList(yr)
        lv_btn = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
        lb_btn = btns.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
        print(f"  Lv強化: {'FOUND' if lv_btn else 'NOT FOUND'}{' disabled' if lv_btn and lv_btn.is_disabled() else ''}")
        print(f"  上限解放: {'FOUND' if lb_btn else 'NOT FOUND'}{' disabled' if lb_btn and lb_btn.is_disabled() else ''}")

        # Check for back button
        has_back = yr.exists_label(BaseUILabels.BACK_BTN)
        print(f"  Back Button: {'FOUND' if has_back else 'NOT FOUND'}")

        # ── Step 4: Enter enhancement page (if Lv強化 available) ──
        if lv_btn and not lv_btn.is_disabled():
            print("Step 4: 进入强化页面")
            device.click_element(lv_btn)
            time.sleep(2)
            frame = device.capture()
            yr = detect(model, frame)
            save(frame, "step4_enhance_page", card_idx)

            btns = ButtonList(yr)
            max_btn = btns.get_button_by_text(SupportCardText.MAX_LEVEL_BUTTON, _FUZZ)
            if max_btn is None:
                max_btn = btns.get_button_by_text("»", _FUZZ)
            confirm_btn = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
            cancel_btn = btns.get_button_by_text(SupportCardText.ENHANCE_CANCEL, _FUZZ)

            print(f"  >> 按钮: {'FOUND' if max_btn else 'NOT FOUND'}{' disabled' if max_btn and max_btn.is_disabled() else ''}")
            print(f"  Lv強化(确认): {'FOUND' if confirm_btn else 'NOT FOUND'}{' disabled' if confirm_btn and confirm_btn.is_disabled() else ''}")
            print(f"  キャンセル: {'FOUND' if cancel_btn else 'NOT FOUND'}")

            # List ALL buttons for debugging
            print("  [DEBUG] All buttons on enhance page:")
            for b in btns.buttons:
                print(f"    \"{b.text}\" disabled={b.is_disabled()}")

            # Go back via cancel
            if cancel_btn:
                print("Step 4b: 取消返回详情页")
                device.click_element(cancel_btn)
                time.sleep(1.5)
                frame = device.capture()
                save(frame, "step4b_cancel_back", card_idx)
        else:
            print("Step 4: [SKIP] Lv強化 不可用")
            save(frame, "step4_enhance_skipped", card_idx)

        # ── Step 5: Back to card list ──
        print("Step 5: 返回列表")
        frame = device.capture()
        yr = detect(model, frame)
        if yr.exists_label(BaseUILabels.BACK_BTN):
            back_boxes = yr.filter_by_label(BaseUILabels.BACK_BTN)
            device.click_element(back_boxes.first())
        else:
            device.back()
        time.sleep(2)
        frame = device.capture()
        yr = detect(model, frame)
        save(frame, "step5_back_to_list", card_idx)

        has_support_cards = yr.exists_label(BaseUILabels.SUPPORT_CARD)
        print(f"  回到列表: {'OK' if has_support_cards else 'FAIL'}")

        # Scroll down for next card to test a different one
        if card_idx < NUM_CARDS_TO_TEST - 1:
            print("  向下滚动...")
            device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
            time.sleep(1.5)

    print(f"\n=== 采集完毕 ===")
    print(f"图像保存在: {CAPTURE_DIR}")
    print(f"共采集 {NUM_CARDS_TO_TEST} 张卡片的流程截图")


if __name__ == "__main__":
    main()
