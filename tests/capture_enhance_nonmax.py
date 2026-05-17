"""
采集非满级卡片的强化页面截图，用于测试 >> 按钮和 Lv強化 确认按钮。
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
    print("=== 采集非满级卡片的强化页面截图 ===")
    device = Android_App()
    model = YoloModelFromONNX("model/base_ui.onnx")
    w, h = device.get_window_size()
    print(f"设备: {w}x{h}")

    # Scroll down to find low-level cards first
    for scroll in range(5):
        device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
        time.sleep(1.5)
    print("已滚动到低级卡片区域")

    captured_count = 0
    target_count = 3

    for attempt in range(8):
        time.sleep(1)
        frame = device.capture()
        yr = detect(model, frame)
        card_list = SupportCardListParser(yr).parse()

        # Find cards that need enhancement (not at max for their star level)
        enhanceable = []
        for c in card_list:
            if c.occluded or c.level is None or c.rarity is None:
                continue
            cap = _get_level_cap(c.rarity, c.stars)
            if c.level < cap:
                enhanceable.append((c, cap))
                print(f"  可强化: {c.rarity} Lv{c.level}/{cap} ★{c.stars}")

        if not enhanceable:
            print(f"  当前页面无可强化卡片，滚动...")
            device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
            time.sleep(1.5)
            continue

        for card, cap in enhanceable[:min(target_count - captured_count, 2)]:
            tag = f"enhance_nonmax_{captured_count}"
            card_label = f"{card.rarity} Lv{card.level}/{cap} ★{card.stars}"
            print(f"\n>>> 处理: {card_label}")

            # Click card
            device.click_element(card.box)
            time.sleep(1.5)
            frame = device.capture()
            yr = detect(model, frame)

            # Click 詳細を見る
            btns = ButtonList(yr)
            detail_btn = btns.get_button_by_text(SupportCardText.VIEW_DETAIL, _FUZZ)
            if not detail_btn:
                print("  詳細を見る not found, skip")
                continue
            device.click_element(detail_btn)
            time.sleep(2)
            frame = device.capture()
            yr = detect(model, frame)
            save(frame, f"{tag}_detail")

            # Click Lv強化
            btns = ButtonList(yr)
            lv_btn = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
            if not lv_btn:
                print("  Lv強化 not found, go back")
                if yr.exists_label(BaseUILabels.BACK_BTN):
                    device.click_element(yr.filter_by_label(BaseUILabels.BACK_BTN).first())
                time.sleep(2)
                continue
            if lv_btn.is_disabled():
                print(f"  Lv強化 disabled for {card_label}")
                if yr.exists_label(BaseUILabels.BACK_BTN):
                    device.click_element(yr.filter_by_label(BaseUILabels.BACK_BTN).first())
                time.sleep(2)
                continue

            device.click_element(lv_btn)
            time.sleep(2)
            frame = device.capture()
            yr = detect(model, frame)
            save(frame, f"{tag}_enhance_page")

            # Check ALL buttons
            btns = ButtonList(yr)
            print(f"  所有按钮 ({len(btns.buttons)}):")
            for b in btns.buttons:
                print(f"    \"{b.text}\" disabled={b.is_disabled()}")

            # Check specific buttons
            max_btn = btns.get_button_by_text(SupportCardText.MAX_LEVEL_BUTTON, _FUZZ)
            if max_btn is None:
                max_btn = btns.get_button_by_text("»", _FUZZ)
            confirm_btn = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
            cancel_btn = btns.get_button_by_text(SupportCardText.ENHANCE_CANCEL, _FUZZ)

            print(f"  >> 按钮: {'FOUND' if max_btn else 'NOT FOUND'}{' disabled' if max_btn and max_btn.is_disabled() else ''}")
            print(f"  Lv強化(确认): {'FOUND' if confirm_btn else 'NOT FOUND'}{' disabled' if confirm_btn and confirm_btn.is_disabled() else ''}")
            print(f"  キャンセル: {'FOUND' if cancel_btn else 'NOT FOUND'}")

            # 如果 >> 找不到, 尝试通过 OCR 搜索整个画面
            if max_btn is None:
                from src.core.inference.ocr_engine import OCRService
                ocr = OCRService()
                # OCR the bottom portion of the enhance page where level selection is
                bottom = frame[int(frame.shape[0] * 0.7):, :]
                ocr_result = ocr.ocr(bottom)
                if ocr_result and ocr_result.results:
                    print("  OCR (bottom):")
                    for r in ocr_result.results:
                        print(f"    \"{r.text}\" at {r.box}")

            # Go back: cancel → back
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
            if captured_count >= target_count:
                break

        if captured_count >= target_count:
            break
        # Scroll for more
        device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
        time.sleep(1.5)

    print(f"\n=== 完成: 采集了 {captured_count} 张非满级卡片的强化页面 ===")

if __name__ == "__main__":
    main()
