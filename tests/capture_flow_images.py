"""
Comprehensive test image capture for support card enhancement flow.
Captures screenshots at every step for multiple cards.

Adaptive flow: after clicking a card thumbnail, check whether we landed on
the detail page directly (Lv強化 visible) OR just selected it on list page
(詳細を見る visible). Handles both cases.
"""
import os
import sys
import time
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.device.Android.app import Android_App
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.SupportCard import SupportCardListParser
from src.constants.game.text.support_card_text import SupportCardText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.base_ui.auto_enhancement_support_card import (
    _get_level_cap, _detect_chevron_count,
)
from src.utils.string_tools import MatchConfig

_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)
CAPTURE_DIR = os.path.join("logs", "debug", "test_captures", "support_card")
METADATA_PATH = os.path.join(CAPTURE_DIR, "metadata.json")


def detect(model, frame):
    return Yolo_Results(model(frame), frame)


def save(frame, name):
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, f"{name}.png")
    cv2.imwrite(path, frame)
    return path


def btn_info(buttons: ButtonList):
    return [
        {"text": b.text or "", "disabled": b.is_disabled(),
         "chevrons": _detect_chevron_count(b.frame),
         "x": int(b.x), "y": int(b.y), "w": int(b.w - b.x), "h": int(b.h - b.y)}
        for b in buttons.buttons
    ]


def card_info(card):
    return {
        "rarity": card.rarity, "level": card.level, "stars": card.stars,
        "limit_break": card.limit_break, "occluded": card.occluded,
        "cap": _get_level_cap(card.rarity, card.stars) if card.rarity else None,
    }


def _on_detail_page(btns):
    """Check if we're on the detail page by looking for Lv強化 or 上限解放."""
    return (btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ) is not None or
            btns.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ) is not None)


def _on_list_page(yr):
    """Check if we're on the card list page by SUPPORT_CARD label count."""
    sc_boxes = yr.filter_by_label(BaseUILabels.SUPPORT_CARD).boxes
    return len(list(sc_boxes)) >= 3


def _navigate_back_to_list(device, model, max_attempts=8):
    """Navigate back to card list page from anywhere."""
    for i in range(max_attempts):
        frame = device.capture()
        yr = detect(model, frame)
        if _on_list_page(yr):
            return True
        btns = ButtonList(yr)
        cancel = btns.get_button_by_text(SupportCardText.ENHANCE_CANCEL, _FUZZ)
        if cancel:
            device.click_element(cancel)
            time.sleep(1.5)
            continue
        if yr.exists_label(BaseUILabels.BACK_BTN):
            device.click_element(yr.filter_by_label(BaseUILabels.BACK_BTN).first())
            time.sleep(1.5)
            continue
        device.back()
        time.sleep(1.5)
    return False


def main():
    print("=== Comprehensive support card flow image capture ===")
    device = Android_App()
    model = YoloModelFromONNX("model/base_ui.onnx")
    w, h = device.get_window_size()
    print(f"Device: {w}x{h}")

    metadata = {"device": {"width": w, "height": h}, "captures": []}
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    # Navigate to card list page
    print("Navigating to support card list page...")
    if not _navigate_back_to_list(device, model):
        print("FATAL: Could not reach card list page")
        return

    captured = []
    cards_captured = 0
    cards_target = 5
    scroll_count = 0
    processed_yolo_centers = set()  # avoid re-processing same card after scroll back

    while cards_captured < cards_target and scroll_count < 10:
        time.sleep(1)
        frame = device.capture()
        yr = detect(model, frame)
        card_list = SupportCardListParser(yr).parse()
        has_sc = yr.exists_label(BaseUILabels.SUPPORT_CARD)
        print(f"\nPage {scroll_count}: {len(card_list)} cards, SUPPORT_CARD={has_sc}")

        if not card_list:
            scroll_count += 1
            device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
            time.sleep(1.5)
            continue

        page_had_capture = False
        for card in list(card_list):
            if cards_captured >= cards_target:
                break
            if card.occluded or card.level is None or card.rarity is None:
                continue
            # Dedup by approximate center
            center_key = (round(card.box.cx / 20) * 20, round(card.box.cy / 20) * 20)
            if center_key in processed_yolo_centers:
                continue

            cap = _get_level_cap(card.rarity, card.stars)
            is_max = card.level >= cap
            tag = f"card_{cards_captured}"
            meta = card_info(card)
            entry = {"tag": tag, "card": meta, "steps": {}}
            print(f"\n>>> [{tag}] {card.rarity} Lv{card.level}/{cap} ★{card.stars} max={is_max}")

            # ─── Step 1: Card list page ───
            save(frame, f"{tag}_step1_list")
            entry["steps"]["step1_list"] = {
                "file": f"{tag}_step1_list.png",
                "has_support_card_label": has_sc,
                "total_cards": len(card_list),
            }

            # ─── Step 2: Click card thumbnail ───
            device.click_element(card.box)
            time.sleep(2)
            frame = device.capture()
            yr = detect(model, frame)
            btns = ButtonList(yr)
            save(frame, f"{tag}_step2_after_click")

            on_detail = _on_detail_page(btns)
            on_list = _on_list_page(yr)
            view_detail_btn = btns.get_button_by_text(SupportCardText.VIEW_DETAIL, _FUZZ)

            entry["steps"]["step2_after_click"] = {
                "file": f"{tag}_step2_after_click.png",
                "buttons": btn_info(btns),
                "on_detail_page": on_detail,
                "on_list_page": on_list,
                "has_view_detail": view_detail_btn is not None,
            }
            print(f"  step2: on_detail={on_detail} on_list={on_list} view_detail={view_detail_btn is not None}")

            # ─── Step 3: Get to detail page ───
            if on_detail:
                # Already on detail page (card was pre-selected)
                save(frame, f"{tag}_step3_detail")
                detail_yr = yr
                detail_btns = btns
            elif view_detail_btn:
                # On list with card selected, click 詳細を見る
                device.click_element(view_detail_btn)
                time.sleep(2)
                frame = device.capture()
                detail_yr = detect(model, frame)
                detail_btns = ButtonList(detail_yr)
                save(frame, f"{tag}_step3_detail")
            else:
                print(f"  SKIP: neither detail page nor 詳細を見る found")
                _navigate_back_to_list(device, model)
                processed_yolo_centers.add(center_key)
                continue

            lv_btn = detail_btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
            lb_btn = detail_btns.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
            has_back = detail_yr.exists_label(BaseUILabels.BACK_BTN)
            entry["steps"]["step3_detail"] = {
                "file": f"{tag}_step3_detail.png",
                "buttons": btn_info(detail_btns),
                "has_back_button": has_back,
                "has_lv_enhance": lv_btn is not None,
                "lv_enhance_disabled": lv_btn.is_disabled() if lv_btn else None,
                "has_limit_break": lb_btn is not None,
                "limit_break_disabled": lb_btn.is_disabled() if lb_btn else None,
            }
            print(f"  step3: Lv強化={lv_btn is not None}(disabled={lv_btn.is_disabled() if lv_btn else None}) "
                  f"上限解放={lb_btn is not None}(disabled={lb_btn.is_disabled() if lb_btn else None}) back={has_back}")

            # ─── Step 4: Enter enhancement page ───
            if lv_btn and not lv_btn.is_disabled():
                device.click_element(lv_btn)
                time.sleep(2)
                frame = device.capture()
                yr = detect(model, frame)
                save(frame, f"{tag}_step4_enhance")
                btns = ButtonList(yr)

                enhance_btns = btn_info(btns)
                confirm_btn = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
                cancel_btn = btns.get_button_by_text(SupportCardText.ENHANCE_CANCEL, _FUZZ)
                chevron_btns = [b for b in btns.buttons if _detect_chevron_count(b.frame) > 0]
                double_chevron = [b for b in btns.buttons if _detect_chevron_count(b.frame) == 2]

                entry["steps"]["step4_enhance"] = {
                    "file": f"{tag}_step4_enhance.png",
                    "buttons": enhance_btns,
                    "has_confirm": confirm_btn is not None,
                    "confirm_disabled": confirm_btn.is_disabled() if confirm_btn else None,
                    "has_cancel": cancel_btn is not None,
                    "chevron_buttons_count": len(chevron_btns),
                    "double_chevron_found": len(double_chevron) > 0,
                }
                print(f"  step4: confirm={confirm_btn is not None} cancel={cancel_btn is not None} "
                      f"chevrons={len(chevron_btns)} >>={len(double_chevron)>0}")

                for bi, b in enumerate(btns.buttons):
                    cv2.imwrite(os.path.join(CAPTURE_DIR, f"{tag}_btn{bi}.png"), b.frame)

                # Go back via cancel
                if cancel_btn:
                    device.click_element(cancel_btn)
                    time.sleep(1.5)
                else:
                    device.back()
                    time.sleep(1.5)
            else:
                entry["steps"]["step4_enhance"] = {
                    "file": None, "skipped": True,
                    "reason": "Lv強化 disabled or not found",
                }
                print(f"  step4: SKIPPED")

            # ─── Step 5: Go back to list ───
            if not _navigate_back_to_list(device, model):
                print("  WARNING: Could not navigate back to list")
                break

            frame = device.capture()
            yr = detect(model, frame)
            save(frame, f"{tag}_step5_back_to_list")
            has_sc_after = yr.exists_label(BaseUILabels.SUPPORT_CARD)
            card_count_after = len(SupportCardListParser(yr).parse())
            entry["steps"]["step5_back_to_list"] = {
                "file": f"{tag}_step5_back_to_list.png",
                "has_support_card_label": has_sc_after,
                "card_count": card_count_after,
            }
            print(f"  step5: SUPPORT_CARD={has_sc_after}, cards={card_count_after}")

            cards_captured += 1
            processed_yolo_centers.add(center_key)
            captured.append(entry)
            metadata["captures"].append(entry)
            page_had_capture = True
            time.sleep(0.5)

        if cards_captured >= cards_target:
            break
        if not page_had_capture:
            print(f"\nScrolling... (captured: {cards_captured}/{cards_target})")
            device.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.35), offset_y=0)
            time.sleep(1.5)
            scroll_count += 1

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\n=== Done: captured {cards_captured} cards ===")
    for entry in captured:
        tag = entry["tag"]
        c = entry["card"]
        steps = list(entry["steps"].keys())
        print(f"  {tag}: {c['rarity']} Lv{c['level']}/{c['cap']} ★{c['stars']} steps={steps}")


if __name__ == "__main__":
    main()
