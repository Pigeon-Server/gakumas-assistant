"""
Quick capture of 2 more non-max card enhance pages.
"""
import os, sys, time
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.device.Android.app import Android_App
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.SupportCard import SupportCardListParser
from src.constants.game.text.support_card_text import SupportCardText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.base_ui.auto_enhancement_support_card import _get_level_cap
from src.utils.string_tools import MatchConfig

_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)
CAPTURE_DIR = os.path.join("logs", "debug", "test_captures", "support_card")

def detect(model, frame):
    return Yolo_Results(model(frame), frame)

def save(frame, name):
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, f"{name}.png")
    cv2.imwrite(path, frame)
    print(f"  [SAVED] {name}.png")

def main():
    device = Android_App()
    model = YoloModelFromONNX("model/base_ui.onnx")
    w, h = device.get_window_size()

    captured_count = 0
    start_index = 1  # already have enhance_nonmax_0

    for attempt in range(12):
        time.sleep(1)
        frame = device.capture()
        yr = detect(model, frame)
        card_list = SupportCardListParser(yr).parse()

        enhanceable = []
        for c in card_list:
            if c.occluded or c.level is None or c.rarity is None:
                continue
            cap = _get_level_cap(c.rarity, c.stars)
            if c.level < cap:
                enhanceable.append((c, cap))

        if not enhanceable:
            device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
            time.sleep(1.5)
            continue

        card, cap = enhanceable[0]
        tag = f"enhance_nonmax_{start_index + captured_count}"
        print(f"\n>>> Card: {card.rarity} Lv{card.level}/{cap} ★{card.stars}")

        device.click_element(card.box)
        time.sleep(1.5)
        frame = device.capture()
        yr = detect(model, frame)

        btns = ButtonList(yr)
        detail_btn = btns.get_button_by_text(SupportCardText.VIEW_DETAIL, _FUZZ)
        if not detail_btn:
            print("  No 詳細を見る, skip")
            device.back()
            time.sleep(2)
            device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
            time.sleep(1.5)
            continue

        device.click_element(detail_btn)
        time.sleep(2)
        frame = device.capture()
        yr = detect(model, frame)
        save(frame, f"{tag}_detail")

        btns = ButtonList(yr)
        lv_btn = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
        if not lv_btn or lv_btn.is_disabled():
            print("  Lv強化 not found or disabled, back")
            if yr.exists_label(BaseUILabels.BACK_BTN):
                device.click_element(yr.filter_by_label(BaseUILabels.BACK_BTN).first())
            time.sleep(2)
            device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
            time.sleep(1.5)
            continue

        device.click_element(lv_btn)
        time.sleep(2)
        frame = device.capture()
        yr = detect(model, frame)
        save(frame, f"{tag}_enhance_page")

        btns = ButtonList(yr)
        print(f"  Buttons ({len(btns.buttons)}):")
        for b in btns.buttons:
            print(f"    \"{b.text}\" disabled={b.is_disabled()} x={int(b.x)} w={int(b.w - b.x)}")

        # Position fallback for >> button
        frame_w = frame.shape[1]
        empty_btns = [
            b for b in btns.buttons
            if (b.text is not None and b.text.strip() == "")
            and (b.w - b.x) < frame_w * 0.15
        ]
        if empty_btns:
            right_btn = max(empty_btns, key=lambda b: b.x + (b.w - b.x) / 2)
            print(f"  >> (position fallback): x={int(right_btn.x)}, disabled={right_btn.is_disabled()}")

        # Go back
        cancel_btn = btns.get_button_by_text(SupportCardText.ENHANCE_CANCEL, _FUZZ)
        if cancel_btn:
            device.click_element(cancel_btn)
            time.sleep(1.5)
        frame = device.capture()
        yr = detect(model, frame)
        if yr.exists_label(BaseUILabels.BACK_BTN):
            device.click_element(yr.filter_by_label(BaseUILabels.BACK_BTN).first())
        else:
            device.back()
        time.sleep(2)

        captured_count += 1
        if captured_count >= 2:
            break

        # Scroll to next set of cards
        device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
        time.sleep(1.5)

    print(f"\n=== Done: captured {captured_count} more non-max enhance pages ===")

if __name__ == "__main__":
    main()
