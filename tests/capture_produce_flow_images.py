"""
Produce flow capture tool — captures screenshots at every step of the
プロデュース setup flow (Home → Scenario → Difficulty → Idol → Support → Memory → Start).

This script connects to an ADB device, loads the YOLO model, and interactively
guides through the produce setup flow while capturing multiple images at each step.

Usage:
    python tests/capture_produce_flow_images.py
"""
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.device.Android.app import Android_App
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Button import ButtonList
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.utils.string_tools import MatchConfig

CAPTURE_DIR = os.path.join("tests", "_artifacts", "produce_flow")
METADATA_PATH = os.path.join(CAPTURE_DIR, "metadata.json")
_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)

# Number of additional captures per step (for robustness testing)
EXTRA_CAPTURES = 2


def detect(model, frame):
    return Yolo_Results(model(frame), frame)


def save(frame, name):
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, f"{name}.png")
    cv2.imwrite(path, frame)
    # Also save a JPG version with moderate compression to test anti-noise
    jpg_path = os.path.join(CAPTURE_DIR, f"{name}.jpg")
    cv2.imwrite(jpg_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    print(f"  saved: {name}.png + {name}.jpg")
    return path


def detect_and_save(model, device, step_name, extra_captures=EXTRA_CAPTURES):
    """Capture frame, run YOLO, save image, and return (frame, yr, buttons)."""
    captures = []
    for i in range(1 + extra_captures):
        time.sleep(0.5)
        frame = device.capture()
        yr = detect(model, frame)
        btns = ButtonList(yr)
        suffix = f"_{i}" if i > 0 else ""
        save(frame, f"{step_name}{suffix}")
        captures.append({"frame": frame, "yr": yr, "btns": btns})

        labels = [b.label for b in yr.boxes]
        btn_texts = [b.text or "" for b in btns.buttons]
        print(f"  [{step_name}{suffix}] labels={len(labels)}, buttons={btn_texts}")
    return captures[0]["frame"], captures[0]["yr"], captures[0]["btns"]


def wait_and_detect(model, device, wait=1.5):
    """Wait a bit, then detect."""
    time.sleep(wait)
    frame = device.capture()
    yr = detect(model, frame)
    return frame, yr, ButtonList(yr)


def print_step(step_num, title):
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*60}")


def main():
    print("=== Gakumas Produce Flow Capture Tool ===\n")

    device = Android_App()
    model = YoloModelFromONNX("model/base_ui.onnx")
    w, h = device.get_window_size()
    print(f"Device: {w}x{h}\n")

    metadata = {
        "device": {"width": w, "height": h},
        "steps": {},
        "captures_per_step": 1 + EXTRA_CAPTURES,
    }
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    # ── STEP 0: Verify Home Page ──
    print_step(0, "Verify Home Page (ホーム)")
    frame, yr, btns = detect_and_save(model, device, "step0_home")

    produce_btn = yr.filter_by_label(BaseUILabels.HOME_PRODUCE_BTN)
    if not produce_btn:
        print("ERROR: Not on home page (Produce button not found)")
        return
    print(f"  ✓ Produce button found at {produce_btn.first().cx},{produce_btn.first().cy}")
    metadata["steps"]["step0_home"] = {
        "produce_btn": True,
        "labels": [b.label for b in yr.boxes],
    }

    # ── STEP 1: Click Produce → Enter scenario selection ──
    print_step(1, "Navigate to Produce (プロデュース)")
    device.click_element(produce_btn.first())
    time.sleep(3)  # Wait for transition + loading
    frame, yr, btns = detect_and_save(model, device, "step1_produce_page")

    # Check for difficulty labels (Regular/Pro/Master/NIA)
    difficulty_labels = [
        BaseUILabels.PRODUCER_REGULAR,
        BaseUILabels.PRODUCER_PRO,
        BaseUILabels.PRODUCER_MASTER,
        BaseUILabels.PRODUCER_NIA,
    ]
    found_difficulties = {}
    for dl in difficulty_labels:
        boxes = yr.filter_by_label(dl)
        found_difficulties[dl] = bool(boxes)
        if boxes:
            print(f"  ✓ Found: {dl}")

    # Also check for any modal about destroying existing produce data
    btn_texts = [b.text for b in btns.buttons if b.text]
    print(f"  Button texts: {btn_texts}")
    metadata["steps"]["step1_produce_page"] = {
        "difficulties": found_difficulties,
        "labels": [b.label for b in yr.boxes],
        "buttons": btn_texts,
    }

    # Check if there's a "データ破棄" modal (existing produce data warning)
    if yr.exists_label(BaseUILabels.MODAL_HEADER):
        print("  ⚠ Modal detected, possibly 'プロデュースデータの破棄'")
        frame, yr, btns = detect_and_save(model, device, "step1_modal_data_destroy")
        for b in btns.buttons:
            print(f"    Modal button: [{b.text}]")
        metadata["steps"]["step1_modal_data_destroy"] = {
            "labels": [b.label for b in yr.boxes],
            "buttons": [b.text for b in btns.buttons],
        }
        # Click confirm to proceed
        if btns.buttons:
            # The confirm/positive button is usually the rightmost
            confirm = btns.buttons[-1] if len(btns.buttons) > 1 else btns.buttons[0]
            print(f"  Clicking modal confirm: [{confirm.text}]")
            device.click_element(confirm)
            time.sleep(2)
            frame, yr, btns = detect_and_save(model, device, "step1_after_modal")

    # ── STEP 2: Scenario / Difficulty Selection ──
    print_step(2, "Scenario & Difficulty Selection")

    # Look for scenario tabs or difficulty labels
    # The main page shows difficulty options: Regular, Pro, Master (for 初)
    # and NIA might be a separate tab
    frame, yr, btns = detect_and_save(model, device, "step2_scenario_selection")

    # Find and click Regular difficulty as default test
    regular = yr.filter_by_label(BaseUILabels.PRODUCER_REGULAR)
    if regular:
        print(f"  ✓ Regular difficulty found, clicking...")
        device.click_element(regular.first())
        time.sleep(2)
        frame, yr, btns = detect_and_save(model, device, "step2_after_regular_click")
    else:
        print("  ✗ Regular difficulty not found, trying buttons...")
        # Try clicking on visible text buttons
        for b in btns.buttons:
            if b.text:
                print(f"    Button: [{b.text}]")

    metadata["steps"]["step2_scenario_selection"] = {
        "labels": [b.label for b in yr.boxes],
        "buttons": [b.text for b in btns.buttons if b.text],
    }

    # ── STEP 3: Idol Card Selection ──
    print_step(3, "Idol Card Selection (プロデュースアイドル選択)")

    # After clicking difficulty, we should be on the idol card selection page
    frame, yr, btns = detect_and_save(model, device, "step3_idol_selection")

    # Look for Produce Cards (Vocal/Dance/Visual) which appear on idol cards
    produce_card_labels = [
        BaseUILabels.PRODUCE_CARD_VOCAL,
        BaseUILabels.PRODUCE_CARD_DANCE,
        BaseUILabels.PRODUCE_CARD_VISUAL,
    ]
    for pcl in produce_card_labels:
        boxes = yr.filter_by_label(pcl)
        if boxes:
            print(f"  ✓ Found: {pcl} ({len(boxes)} instances)")

    # Try swiping left to see more idol cards
    print("  Swiping left to see more cards...")
    for swipe_idx in range(3):
        device.swipe(int(w * 0.75), h // 2, int(w * 0.25), h // 2, duration=0.6)
        time.sleep(1.5)
        frame, yr, btns = detect_and_save(model, device, f"step3_idol_swipe_{swipe_idx}")

    # Swipe back to first card
    print("  Swiping right to return...")
    for swipe_idx in range(3):
        device.swipe(int(w * 0.25), h // 2, int(w * 0.75), h // 2, duration=0.6)
        time.sleep(1.5)

    frame, yr, btns = detect_and_save(model, device, "step3_idol_back_to_first")

    metadata["steps"]["step3_idol_selection"] = {
        "labels": [b.label for b in yr.boxes],
        "buttons": [b.text for b in btns.buttons if b.text],
    }

    # Click "次へ" (Next) button to proceed
    next_btn = btns.get_button_by_text("次へ", _FUZZ)
    if next_btn:
        print(f"  ✓ Found Next button, clicking...")
        device.click_element(next_btn)
        time.sleep(3)
    else:
        print("  ✗ Next button not found, looking for alternative...")
        for b in btns.buttons:
            print(f"    [{b.text}]")
        # Try the last button which is often "Next"
        if btns.buttons:
            print(f"  Trying last button: [{btns.buttons[-1].text}]")
            device.click_element(btns.buttons[-1])
            time.sleep(3)

    # ── STEP 4: Support Card Selection ──
    print_step(4, "Support Card Selection (サポートカード編成)")

    frame, yr, btns = detect_and_save(model, device, "step4_support_selection")

    # Check for support card and blank slot labels
    support_cards = yr.filter_by_label(BaseUILabels.SUPPORT_CARD)
    blank_slots = yr.filter_by_label(BaseUILabels.BLANK_SLOT)
    print(f"  Support cards: {len(support_cards)}, Blank slots: {len(blank_slots)}")

    # Look for "おまかせ" button
    omakase_btn = btns.get_button_by_text("おまかせ", _FUZZ)
    if omakase_btn:
        print(f"  ✓ Found おまかせ button")
    else:
        print("  ✗ おまかせ not found directly")
        for b in btns.buttons:
            print(f"    [{b.text}]")

    metadata["steps"]["step4_support_selection"] = {
        "labels": [b.label for b in yr.boxes],
        "buttons": [b.text for b in btns.buttons if b.text],
        "support_cards": len(support_cards),
        "blank_slots": len(blank_slots),
    }

    # Click おまかせ to auto-fill support cards
    if omakase_btn:
        print("  Clicking おまかせ to auto-fill...")
        device.click_element(omakase_btn)
        time.sleep(2)
        frame, yr, btns = detect_and_save(model, device, "step4_after_omakase")

        # Look for "決定" button
        confirm_btn = btns.get_button_by_text("決定", _FUZZ)
        if confirm_btn:
            print(f"  ✓ Found 決定 button, clicking...")
            device.click_element(confirm_btn)
            time.sleep(2)
            frame, yr, btns = detect_and_save(model, device, "step4_after_confirm")
        else:
            print("  ✗ 決定 not found")
            for b in btns.buttons:
                print(f"    [{b.text}]")

    # Click "次へ" to proceed to memory selection
    frame, yr, btns = wait_and_detect(model, device, 1)
    next_btn = btns.get_button_by_text("次へ", _FUZZ)
    if next_btn:
        print(f"  ✓ Found Next button for support, clicking...")
        device.click_element(next_btn)
        time.sleep(3)
    else:
        # Try "編成する" or similar
        for b in btns.buttons:
            if b.text:
                print(f"    [{b.text}]")

    # ── STEP 5: Memory Selection ──
    print_step(5, "Memory Selection (メモリー編成)")

    frame, yr, btns = detect_and_save(model, device, "step5_memory_selection")

    memory_cards = yr.filter_by_label(BaseUILabels.MEMORY_CARD)
    blank_slots = yr.filter_by_label(BaseUILabels.BLANK_SLOT)
    print(f"  Memory cards: {len(memory_cards)}, Blank slots: {len(blank_slots)}")

    btn_texts = [b.text for b in btns.buttons if b.text]
    print(f"  Buttons: {btn_texts}")

    metadata["steps"]["step5_memory_selection"] = {
        "labels": [b.label for b in yr.boxes],
        "buttons": btn_texts,
        "memory_cards": len(memory_cards),
        "blank_slots": len(blank_slots),
    }

    # Click おまかせ for memory
    omakase_btn = btns.get_button_by_text("おまかせ", _FUZZ)
    if omakase_btn:
        print("  Clicking おまかせ for memory...")
        device.click_element(omakase_btn)
        time.sleep(2)
        frame, yr, btns = detect_and_save(model, device, "step5_after_omakase")

        confirm_btn = btns.get_button_by_text("決定", _FUZZ)
        if confirm_btn:
            print(f"  ✓ Found 決定, clicking...")
            device.click_element(confirm_btn)
            time.sleep(2)
            frame, yr, btns = detect_and_save(model, device, "step5_after_confirm")

    # Click 次へ to proceed
    frame, yr, btns = wait_and_detect(model, device, 1)
    next_btn = btns.get_button_by_text("次へ", _FUZZ)
    if next_btn:
        print(f"  ✓ Found Next button for memory, clicking...")
        device.click_element(next_btn)
        time.sleep(3)
    else:
        for b in btns.buttons:
            if b.text:
                print(f"    [{b.text}]")

    # ── STEP 5.5: Handle potential rental popup ──
    print_step("5.5", "Check for Rental Popup (レンタル可能)")

    frame, yr, btns = detect_and_save(model, device, "step5_5_rental_check")

    if yr.exists_label(BaseUILabels.MODAL_HEADER):
        print("  ⚠ Modal detected! Possibly rental popup")
        from src.utils.game_tools import get_modal
        modal = get_modal(yr, no_body=True, quiet=True)
        if modal:
            print(f"  Modal title: {modal.modal_title}")
            metadata["steps"]["step5_5_rental_popup"] = {
                "modal_title": modal.modal_title,
                "labels": [b.label for b in yr.boxes],
            }
            # Click confirm to dismiss
            if modal.confirm_button:
                device.click_element(modal.confirm_button)
                time.sleep(2)
                frame, yr, btns = detect_and_save(model, device, "step5_5_after_rental")

    # ── STEP 6: Final Confirmation Screen ──
    print_step(6, "Final Confirmation & Start (プロデュース開始)")

    frame, yr, btns = detect_and_save(model, device, "step6_final_confirm")

    btn_texts = [b.text for b in btns.buttons if b.text]
    print(f"  Buttons: {btn_texts}")

    # Look for プロデュース開始 button
    start_btn = btns.get_button_by_text("プロデュース開始", _FUZZ)
    if start_btn:
        print("  ✓✓✓ Found プロデュース開始 button! Flow capture complete!")
    else:
        print("  ✗ プロデュース開始 not found directly, checking all buttons:")
        for b in btns.buttons:
            print(f"    [{b.text}] at [{b.x},{b.y}]")

    metadata["steps"]["step6_final_confirm"] = {
        "labels": [b.label for b in yr.boxes],
        "buttons": btn_texts,
        "start_btn_found": start_btn is not None,
    }

    # DO NOT CLICK START — we only capture up to this point

    # ── Save Metadata ──
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print(f"  Capture complete! Files saved to: {CAPTURE_DIR}")
    print(f"  Metadata: {METADATA_PATH}")
    print(f"{'='*60}")

    # Summary
    total_files = len([f for f in os.listdir(CAPTURE_DIR) if f.endswith(('.png', '.jpg'))])
    print(f"\n  Total image files: {total_files}")
    for step_name, step_data in metadata["steps"].items():
        print(f"  {step_name}: labels={len(step_data.get('labels', []))}, "
              f"buttons={step_data.get('buttons', [])}")


if __name__ == "__main__":
    main()
