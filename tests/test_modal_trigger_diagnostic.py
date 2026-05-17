"""
触发并分析模态框的 MODAL_HEADER YOLO 检测

流程:
  支援卡列表 → 选一张非满级卡 → 点Lv強化 → 进入强化页 →
  点 ">>" 设最高 → 点「強化する」→ 弹出确认模态 → 分析 → 点取消 → 返回

或者:
  支援卡列表 → 选卡 → 详情 → 点击右下角 ≡ →（可能弹出菜单模态）→ 分析 → 返回

也可以:
  直接从当前页面找任何可触发模态的操作

用法:
  确保在支援卡片列表页面
  python -m tests.test_modal_trigger_diagnostic
"""
import os
import sys
import time
import json

import cv2
import numpy as np
import adbutils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.inference.ocr_engine import OCRService
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Modal import ModalParser
from src.entity.Game.Components.Button import ButtonList
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.utils.string_tools import string_match, MatchConfig

OUT_DIR = os.path.join("logs", "debug", "test_captures", "modal_trigger")
os.makedirs(OUT_DIR, exist_ok=True)

model = YoloModelFromONNX(config.model_config["BASE_UI"])
ocr_service = OCRService()
_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)


def screenshot(dev) -> np.ndarray:
    pil_img = dev.screenshot()
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def detect(frame: np.ndarray, conf: float = 0.5) -> Yolo_Results:
    raw = model(frame, conf_threshold=conf, iou_threshold=0.5)
    return Yolo_Results(raw, frame)


def analyze_mh(frame: np.ndarray, tag: str):
    """分析 MODAL_HEADER 在三个置信度下的检测"""
    for conf in [0.5, 0.25, 0.1]:
        yolo = detect(frame, conf=conf)
        mh = yolo.filter_by_label(BaseUILabels.MODAL_HEADER)
        found = mh and len(mh) > 0
        confs = [round(b.confidence, 4) for b in mh.boxes] if mh else []
        mark = "✓" if found else "✗"
        print(f"    {tag} @{conf}: {mark} confs={confs}")

    # HSV fallback
    yolo_50 = detect(frame, conf=0.5)
    btns = yolo_50.filter_by_label(BaseUILabels.BUTTON)
    if btns and len(btns) > 0:
        parser = ModalParser(yolo_50, quiet=True)
        parser.action_buttons = parser._get_action_buttons()
        if parser.action_buttons:
            hsv = parser._infer_header_from_frame()
            print(f"    {tag} HSV: {'✓' if hsv else '✗'} box={f'{hsv.x},{hsv.y},{hsv.w},{hsv.h}' if hsv else 'N/A'}")

    # Full modal parse
    modal = ModalParser(yolo_50, quiet=True).parse()
    if modal:
        print(f"    {tag} Modal parsed! title={modal.modal_title}")
        return modal
    else:
        # Try lower conf
        yolo_25 = detect(frame, conf=0.25)
        modal25 = ModalParser(yolo_25, quiet=True).parse()
        if modal25:
            print(f"    {tag} Modal parsed @0.25! title={modal25.modal_title}")
            return modal25
    return None


def click(dev, x, y, wait=1.5):
    dev.click(int(x), int(y))
    time.sleep(wait)


def back(dev, wait=1.5):
    dev.keyevent(4)  # KEYCODE_BACK
    time.sleep(wait)


def find_label(frame, label, conf=0.5):
    yolo = detect(frame, conf=conf)
    items = yolo.filter_by_label(label)
    return items.boxes if items and len(items) > 0 else []


def find_button_text(frame, text, conf=0.5):
    yolo = detect(frame, conf=conf)
    buttons = ButtonList(yolo)
    return buttons.get_button_by_text(text, _FUZZ)


def save(frame, name):
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}.png"), frame)


def main():
    print("=" * 60)
    print("  MODAL_HEADER 触发诊断")
    print("=" * 60)

    adb = adbutils.AdbClient()
    devices = adb.device_list()
    if not devices:
        print("ERROR: 没有 ADB 设备")
        return
    dev = devices[0]
    print(f"设备: {dev.serial}")

    # === Step 1: 当前页面 ===
    frame = screenshot(dev)
    save(frame, "step0_current")
    yolo = detect(frame)
    labels = {}
    for b in yolo.boxes:
        labels[b.label] = labels.get(b.label, 0) + 1
    print(f"\n当前页面 labels: {labels}")

    # === Step 2: 找到支援卡并进入详情 ===
    cards = find_label(frame, BaseUILabels.SUPPORT_CARD)
    if not cards:
        print("没有检测到 Support Card，尝试找 ITEM")
        cards = find_label(frame, BaseUILabels.ITEM)
    if not cards:
        print("没有可点击的卡片元素，退出")
        return

    # 选一张靠后的卡片（可能是低级/有上限解放余地的）
    card = cards[min(4, len(cards) - 1)]
    print(f"\n点击卡片 ({card.cx}, {card.cy})")
    click(dev, card.cx, card.cy)

    frame = screenshot(dev)
    save(frame, "step1_detail")
    print("\n--- 详情页 ---")
    analyze_mh(frame, "detail")

    # === Step 3: 点击 Lv強化 ===
    lv_btn = find_button_text(frame, "Lv強化")
    if lv_btn:
        print(f"\n点击 Lv強化 ({lv_btn.cx}, {lv_btn.cy})")
        click(dev, lv_btn.cx, lv_btn.cy, wait=2.0)

        frame = screenshot(dev)
        save(frame, "step2_enhance_page")
        print("\n--- 强化页 ---")
        yolo_e = detect(frame)
        labels_e = {}
        for b in yolo_e.boxes:
            labels_e[b.label] = labels_e.get(b.label, 0) + 1
        print(f"  labels: {labels_e}")
        analyze_mh(frame, "enhance_page")

        # 找 >> button (max level)
        btns = ButtonList(yolo_e)
        # 看看所有按钮的文本
        print(f"  所有按钮:")
        for btn_box in yolo_e.filter_by_label(BaseUILabels.BUTTON).boxes if yolo_e.filter_by_label(BaseUILabels.BUTTON) else []:
            btn = btns._get_button_obj(btn_box)
            if btn:
                print(f"    [{btn.text}] at ({btn.cx}, {btn.cy}) disabled={btn.is_disabled()}")

        # 点击强化按钮（「強化する」或类似）
        confirm_btn = find_button_text(frame, "強化する")
        if not confirm_btn:
            confirm_btn = find_button_text(frame, "強化")
        if confirm_btn:
            print(f"\n点击強化する ({confirm_btn.cx}, {confirm_btn.cy})")
            click(dev, confirm_btn.cx, confirm_btn.cy, wait=2.0)

            # ===== 这里应该弹出确认模态框 =====
            for cap_i in range(6):
                time.sleep(0.5)
                frame = screenshot(dev)
                save(frame, f"step3_modal_cap{cap_i}")
                print(f"\n--- 模态框检测 cap{cap_i} ---")
                modal = analyze_mh(frame, f"modal_cap{cap_i}")

                if modal:
                    # 画 debug 图
                    try:
                        debug_img = modal.draw_debug()
                        save(debug_img, f"step3_modal_cap{cap_i}_debug")
                    except Exception:
                        pass
                    break

            # 点击取消/返回
            print("\n返回...")
            yolo_m = detect(frame)
            close_btns = find_label(frame, BaseUILabels.CLOSE_BUTTON)
            if close_btns:
                click(dev, close_btns[0].cx, close_btns[0].cy)
            else:
                back(dev)
        else:
            print("  没找到強化する按钮")

        # 返回详情页
        back(dev)
    else:
        print("  Lv強化 按钮未找到")

    # === Step 4: 尝试通过右下角菜单按钮 (≡) 触发模态 ===
    frame = screenshot(dev)
    save(frame, "step4_back_to_detail")
    yolo4 = detect(frame)

    # 寻找右下角的菜单按钮 (hamburger icon)
    # 通常在屏幕右下角，是一个独立的按钮
    all_buttons = yolo4.filter_by_label(BaseUILabels.BUTTON)
    if all_buttons:
        frame_h, frame_w = frame.shape[:2]
        # 找在右下角的独立按钮
        bottom_right_btns = [
            b for b in all_buttons.boxes
            if b.cx > frame_w * 0.8 and b.cy > frame_h * 0.85
        ]
        if bottom_right_btns:
            menu_btn = bottom_right_btns[0]
            print(f"\n点击右下角菜单按钮 ({menu_btn.cx}, {menu_btn.cy})")
            click(dev, menu_btn.cx, menu_btn.cy, wait=1.5)

            frame = screenshot(dev)
            save(frame, "step5_menu_modal")
            print("\n--- 菜单模态检测 ---")
            modal = analyze_mh(frame, "menu_modal")
            if modal:
                try:
                    save(modal.draw_debug(), "step5_menu_modal_debug")
                except Exception:
                    pass

            # 返回
            back(dev)
        else:
            print("\n没有找到右下角菜单按钮")

    # 最终返回列表
    back(dev)
    time.sleep(1)

    print(f"\n截图保存: {OUT_DIR}")
    print("完成!")


if __name__ == "__main__":
    main()
