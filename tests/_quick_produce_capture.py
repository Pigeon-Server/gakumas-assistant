#!/usr/bin/env python3
"""Quick ADB capture + YOLO inference for produce flow steps.
Start from wherever the device currently is and walk through the flow."""
import cv2, numpy as np, subprocess, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.constants.yolo.model_type import YoloModelType
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Button import ButtonList
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.utils.string_tools import MatchConfig

ARTIFACTS = os.path.join('tests', '_artifacts', 'produce_flow')
os.makedirs(ARTIFACTS, exist_ok=True)

model = YoloModelFromONNX(config.model_config[YoloModelType.BASE_UI])

def screenshot():
    proc = subprocess.run(['adb', 'exec-out', 'screencap', '-p'], capture_output=True, timeout=10)
    return cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)

def detect(frame):
    return Yolo_Results(model(frame, conf_threshold=0.5, iou_threshold=0.5), frame)

def save(frame, name):
    cv2.imwrite(os.path.join(ARTIFACTS, f'{name}.png'), frame)
    cv2.imwrite(os.path.join(ARTIFACTS, f'{name}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'  Saved: {name}')

def tap(x, y):
    subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)], timeout=5)

def swipe(x1, y1, x2, y2, ms=500):
    subprocess.run(['adb', 'shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2), str(ms)], timeout=5)

def click_box(box):
    cx, cy = int((box.x + box.w) / 2), int((box.y + box.h) / 2)
    tap(cx, cy)
    return cx, cy

def find_button(yr, text):
    buttons = ButtonList(yr)
    for b in buttons:
        if b.text and text in b.text:
            return b
    return None

def summary(yr):
    labels = {}
    for box in yr:
        labels[box.label] = labels.get(box.label, 0) + 1
    for lbl, cnt in sorted(labels.items()):
        print(f'    {lbl}: {cnt}')
    buttons = ButtonList(yr)
    if buttons:
        print(f'    Buttons: {[b.text for b in buttons]}')

# ═══════ Current state: NIA scenario page ═══════
# We're already on the NIA scenario page with Pro/Master buttons visible.
# Let's capture it and then walk through the NIA Pro flow.

print('=== Step A: NIA Scenario Page ===')
frame = screenshot()
yr = detect(frame)
save(frame, 'stepA_scenario_nia')
summary(yr)
time.sleep(0.5)
save(screenshot(), 'stepA_scenario_nia_1')

# Now swipe right to go to HAJIME page
print('\n=== Swipe right to HAJIME ===')
h, w = frame.shape[:2]
swipe(w // 4, h // 2, w * 3 // 4, h // 2, 500)
time.sleep(2)

print('=== Step A: HAJIME Scenario Page ===')
frame = screenshot()
yr = detect(frame)
save(frame, 'stepA_scenario_hajime')
summary(yr)
time.sleep(0.5)
save(screenshot(), 'stepA_scenario_hajime_1')

# Try to capture HAJIME page with different difficulties visible
time.sleep(1)
save(screenshot(), 'stepA_scenario_hajime_2')

# Now swipe left to try Legend page
print('\n=== Swipe left for Legend ===')
swipe(w * 3 // 4, h // 2, w // 4, h // 2, 500)
time.sleep(2)
frame = screenshot()
yr = detect(frame)
summary(yr)

# If we're on NIA, swipe left once more
if yr.exists_label(BaseUILabels.PRODUCER_NIA):
    print('  On NIA, swiping left again...')
    swipe(w * 3 // 4, h // 2, w // 4, h // 2, 500)
    time.sleep(2)
    frame = screenshot()
    yr = detect(frame)
    summary(yr)

# Check if we found Legend or not
buttons = ButtonList(yr)
has_legend = any('レジェンド' in (b.text or '') for b in buttons)
if has_legend:
    print('  [OK] Found Legend page!')
    save(frame, 'stepB_legend_page')
    time.sleep(0.5)
    save(screenshot(), 'stepB_legend_page_1')
else:
    print('  Legend page not found, continuing...')
    save(frame, 'stepB_legend_page')  # save whatever we have

# Navigate back to HAJIME and click Regular
print('\n=== Navigate back to HAJIME Regular ===')
# Swipe right to get back
swipe(w // 4, h // 2, w * 3 // 4, h // 2, 500)
time.sleep(2)
frame = screenshot()
yr = detect(frame)

if not yr.exists_label(BaseUILabels.PRODUCER_REGULAR):
    swipe(w // 4, h // 2, w * 3 // 4, h // 2, 500)
    time.sleep(2)
    frame = screenshot()
    yr = detect(frame)

regular_boxes = yr.filter_by_label(BaseUILabels.PRODUCER_REGULAR)
if regular_boxes:
    box = list(regular_boxes)[0]
    cx, cy = click_box(box)
    print(f'  Clicked Regular at ({cx}, {cy})')
    time.sleep(3)
else:
    print('  ERROR: Regular not found')
    summary(yr)
    sys.exit(1)

# --- Step C: Idol Selection ---
print('\n=== Step C: Idol Selection ===')
frame = screenshot()
yr = detect(frame)
save(frame, 'stepC_idol_selection')
summary(yr)
save(screenshot(), 'stepC_idol_selection_1')

btn = find_button(yr, '次')
if btn:
    print(f'  Clicking [{btn.text}]')
    click_box(btn)
    time.sleep(3)

# --- Step D: Support Card Selection ---
print('\n=== Step D: Support Card Selection ===')
frame = screenshot()
yr = detect(frame)
save(frame, 'stepD_support_selection')
summary(yr)

btn = find_button(yr, 'おまかせ')
if btn:
    print(f'  Clicking [{btn.text}]')
    click_box(btn)
    time.sleep(2)

frame = screenshot()
yr = detect(frame)
save(frame, 'stepD_after_omakase')
summary(yr)

btn = find_button(yr, '決定')
if btn:
    print(f'  Clicking [{btn.text}]')
    click_box(btn)
    time.sleep(2)

frame = screenshot()
yr = detect(frame)
save(frame, 'stepD_before_next')

btn = find_button(yr, '次')
if btn:
    print(f'  Clicking [{btn.text}]')
    click_box(btn)
    time.sleep(3)

# --- Step E: Memory Selection ---
print('\n=== Step E: Memory Selection ===')
frame = screenshot()
yr = detect(frame)
save(frame, 'stepE_memory_selection')
summary(yr)

if yr.exists_label(BaseUILabels.CHECKBOX):
    print('  [OK] Checkbox detected!')
    save(frame, 'stepE_rental_checkbox')

btn = find_button(yr, 'おまかせ')
if btn:
    print(f'  Clicking [{btn.text}]')
    click_box(btn)
    time.sleep(2)

frame = screenshot()
yr = detect(frame)
save(frame, 'stepE_after_omakase')
summary(yr)

btn = find_button(yr, '決定')
if btn:
    print(f'  Clicking [{btn.text}]')
    click_box(btn)
    time.sleep(2)

frame = screenshot()
yr = detect(frame)
btn = find_button(yr, '次')
if btn:
    print(f'  Clicking [{btn.text}]')
    click_box(btn)
    time.sleep(3)

# Handle rental modal
print('\n=== Step F: Check Rental Modal ===')
frame = screenshot()
yr = detect(frame)
if yr.exists_label(BaseUILabels.MODAL_HEADER):
    save(frame, 'stepF_rental_modal')
    print('  Rental modal detected')
    buttons = ButtonList(yr)
    for b in buttons:
        if b.text:
            click_box(b)
            time.sleep(2)
            break
    frame = screenshot()
    yr = detect(frame)

# --- Step F: Final Confirm Page ---
print('\n=== Step F: Final Confirm ===')
save(frame, 'stepF_final')
summary(yr)
time.sleep(0.5)
save(screenshot(), 'stepF_final_1')

if yr.exists_label(BaseUILabels.SPECIAL_ITEMS):
    print('  [OK] SPECIAL_ITEMS detected!')
    save(frame, 'stepG_boost_items')
    save(screenshot(), 'stepG_boost_items_1')

# Check formation details button
btn = find_button(yr, '編成詳細')
if btn:
    print('\n=== Step H: Formation Details ===')
    click_box(btn)
    time.sleep(2)
    frame = screenshot()
    yr = detect(frame)
    save(frame, 'stepH_formation_idol')
    summary(yr)

    # Try switching to Support tab
    btn = find_button(yr, 'サポートカード')
    if btn:
        click_box(btn)
        time.sleep(1)
        frame = screenshot()
        save(frame, 'stepH_formation_support')

    # Try switching to Memory tab
    yr = detect(screenshot())
    btn = find_button(yr, 'メモリー')
    if btn:
        click_box(btn)
        time.sleep(1)
        frame = screenshot()
        save(frame, 'stepH_formation_memory')
        save(screenshot(), 'stepH_formation_memory_1')

    # Close overlay
    close_boxes = yr.filter_by_label(BaseUILabels.CLOSE_BUTTON)
    if close_boxes:
        click_box(list(close_boxes)[0])
        time.sleep(1)
    else:
        back_boxes = yr.filter_by_label(BaseUILabels.BACK_BTN)
        if back_boxes:
            click_box(list(back_boxes)[0])
            time.sleep(1)

# Don't click produce start - just capture
print('\n=== ALL DONE ===')
import glob
files = glob.glob(os.path.join(ARTIFACTS, '*.png'))
print(f'Total PNG files: {len(files)}')
for f in sorted(files):
    print(f'  {os.path.basename(f)}')
