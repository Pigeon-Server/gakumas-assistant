"""
MODAL_HEADER YOLO 检测诊断

在设备上触发模态框，然后多次截图分析 YOLO 对 MODAL_HEADER 的检测情况。
同时运行 HSV fallback 作为对照。

用法:
  1. 确保设备在支援卡片列表页面（或任何可以触发模态框的页面）
  2. python -m tests.test_modal_header_diagnostic

脚本行为（只读，不会实际执行上限解放/变换操作）:
  - 进入卡片详情页 → 点击上限解放 → 截图分析 → 返回
"""
import os
import sys
import time
import json
import traceback

import cv2
import numpy as np
import adbutils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.inference.ocr_engine import OCRService
from src.entity.Yolo import Yolo_Results, Yolo_Box
from src.entity.Game.Components.Modal import ModalParser
from src.entity.Game.Components.Button import ButtonList
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.game.text.support_card_text import SupportCardText
from src.utils.string_tools import string_match, MatchConfig

OUT_DIR = os.path.join("logs", "debug", "test_captures", "modal_header")
os.makedirs(OUT_DIR, exist_ok=True)

model = YoloModelFromONNX(config.model_config["BASE_UI"])
ocr_service = OCRService()
_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)


def adb_screenshot(dev) -> np.ndarray:
    pil_img = dev.screenshot()
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def detect(frame: np.ndarray, conf: float = 0.5) -> Yolo_Results:
    raw = model(frame, conf_threshold=conf, iou_threshold=0.5)
    return Yolo_Results(raw, frame)


def analyze_modal_header(frame: np.ndarray, tag: str) -> dict:
    """对一帧图像全面分析 MODAL_HEADER 检测"""
    result = {"tag": tag}

    # 1. 标准阈值(0.5) YOLO
    yolo_50 = detect(frame, conf=0.5)
    mh_50 = yolo_50.filter_by_label(BaseUILabels.MODAL_HEADER)
    result["yolo_conf50"] = {
        "found": len(mh_50) > 0 if mh_50 else False,
        "count": len(mh_50) if mh_50 else 0,
        "confs": [round(b.confidence, 4) for b in mh_50.boxes] if mh_50 else [],
        "boxes": [{"x": int(b.x), "y": int(b.y), "w": int(b.w), "h": int(b.h)} for b in mh_50.boxes] if mh_50 else [],
    }

    # 2. 低阈值(0.25) YOLO — 看是否被 conf 卡掉
    yolo_25 = detect(frame, conf=0.25)
    mh_25 = yolo_25.filter_by_label(BaseUILabels.MODAL_HEADER)
    result["yolo_conf25"] = {
        "found": len(mh_25) > 0 if mh_25 else False,
        "count": len(mh_25) if mh_25 else 0,
        "confs": [round(b.confidence, 4) for b in mh_25.boxes] if mh_25 else [],
        "boxes": [{"x": int(b.x), "y": int(b.y), "w": int(b.w), "h": int(b.h)} for b in mh_25.boxes] if mh_25 else [],
    }

    # 3. 极低阈值(0.1) YOLO
    yolo_10 = detect(frame, conf=0.1)
    mh_10 = yolo_10.filter_by_label(BaseUILabels.MODAL_HEADER)
    result["yolo_conf10"] = {
        "found": len(mh_10) > 0 if mh_10 else False,
        "count": len(mh_10) if mh_10 else 0,
        "confs": [round(b.confidence, 4) for b in mh_10.boxes] if mh_10 else [],
        "boxes": [{"x": int(b.x), "y": int(b.y), "w": int(b.w), "h": int(b.h)} for b in mh_10.boxes] if mh_10 else [],
    }

    # 4. HSV fallback (ModalParser._infer_header_from_frame)
    # 只有在有 buttons 时才能运行 ModalParser
    buttons_50 = yolo_50.filter_by_label(BaseUILabels.BUTTON)
    if buttons_50 and len(buttons_50) > 0:
        parser = ModalParser(yolo_50, quiet=True)
        parser.action_buttons = parser._get_action_buttons()
        if parser.action_buttons:
            hsv_header = parser._infer_header_from_frame()
            result["hsv_fallback"] = {
                "found": hsv_header is not None,
                "box": {"x": int(hsv_header.x), "y": int(hsv_header.y), "w": int(hsv_header.w), "h": int(hsv_header.h)} if hsv_header else None,
            }
        else:
            result["hsv_fallback"] = {"found": False, "reason": "no action buttons"}
    else:
        result["hsv_fallback"] = {"found": False, "reason": "no buttons detected"}

    # 5. 所有检测到的 label 汇总
    all_labels = {}
    for box in yolo_50.boxes:
        lbl = box.label
        if lbl not in all_labels:
            all_labels[lbl] = 0
        all_labels[lbl] += 1
    result["all_labels_conf50"] = all_labels

    return result


def draw_debug_overlay(frame: np.ndarray, analysis: dict) -> np.ndarray:
    """在图片上绘制检测结果"""
    vis = frame.copy()

    # 绘制 conf=0.5 YOLO MODAL_HEADER (绿色)
    for box in analysis["yolo_conf50"].get("boxes", []):
        cv2.rectangle(vis, (box["x"], box["y"]), (box["w"], box["h"]), (0, 255, 0), 3)
        cv2.putText(vis, "MH@0.5", (box["x"], box["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 绘制 conf=0.25 独有 MODAL_HEADER (黄色)
    for box in analysis["yolo_conf25"].get("boxes", []):
        if not analysis["yolo_conf50"]["found"]:
            cv2.rectangle(vis, (box["x"], box["y"]), (box["w"], box["h"]), (0, 255, 255), 2)
            cv2.putText(vis, "MH@0.25", (box["x"], box["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 绘制 conf=0.1 独有 MODAL_HEADER (红色)
    for box in analysis["yolo_conf10"].get("boxes", []):
        if not analysis["yolo_conf25"]["found"]:
            cv2.rectangle(vis, (box["x"], box["y"]), (box["w"], box["h"]), (0, 0, 255), 2)
            cv2.putText(vis, "MH@0.1", (box["x"], box["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 绘制 HSV fallback (蓝色)
    hsv_box = analysis.get("hsv_fallback", {}).get("box")
    if hsv_box:
        cv2.rectangle(vis, (hsv_box["x"], hsv_box["y"]), (hsv_box["w"], hsv_box["h"]), (255, 0, 0), 2)
        cv2.putText(vis, "HSV", (hsv_box["x"], hsv_box["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return vis


def click_adb(dev, x, y):
    dev.click(x, y)
    time.sleep(1.5)


def main():
    print("=" * 60)
    print("  MODAL_HEADER YOLO 检测诊断")
    print("=" * 60)

    adb = adbutils.AdbClient()
    devices = adb.device_list()
    if not devices:
        print("ERROR: 没有检测到 ADB 设备")
        return
    dev = devices[0]
    print(f"设备: {dev.serial}")

    all_results = []

    # === Phase 1: 当前页面分析（不管是什么页面） ===
    print("\n=== Phase 1: 当前页面分析 ===")
    frame = adb_screenshot(dev)
    analysis = analyze_modal_header(frame, "current_page")
    all_results.append(analysis)
    cv2.imwrite(os.path.join(OUT_DIR, "phase1_current.png"), frame)
    cv2.imwrite(os.path.join(OUT_DIR, "phase1_current_debug.png"), draw_debug_overlay(frame, analysis))
    print(f"  YOLO@0.50: MH={analysis['yolo_conf50']['found']}, confs={analysis['yolo_conf50']['confs']}")
    print(f"  YOLO@0.25: MH={analysis['yolo_conf25']['found']}, confs={analysis['yolo_conf25']['confs']}")
    print(f"  YOLO@0.10: MH={analysis['yolo_conf10']['found']}, confs={analysis['yolo_conf10']['confs']}")
    print(f"  HSV fallback: {analysis['hsv_fallback']}")
    print(f"  All labels: {analysis['all_labels_conf50']}")

    # === Phase 2: 尝试在当前页面找到可以触发模态框的操作 ===
    print("\n=== Phase 2: 尝试进入卡片详情并触发模态框 ===")

    # 检查当前页面是否是支援卡片列表
    yolo_res = detect(frame)
    # 尝试 ITEM 或 Support Card
    items = yolo_res.filter_by_label(BaseUILabels.ITEM)
    if not items or len(items) == 0:
        items = yolo_res.filter_by_label(BaseUILabels.SUPPORT_CARD)

    if items and len(items) >= 1:
        print(f"  检测到 {len(items)} 个可点击元素，尝试点击第一个进入详情")
        first_item = items.boxes[0]
        click_adb(dev, first_item.cx, first_item.cy)
        time.sleep(1.0)

        # 详情页分析
        frame = adb_screenshot(dev)
        cv2.imwrite(os.path.join(OUT_DIR, "phase2_detail.png"), frame)
        analysis = analyze_modal_header(frame, "detail_page")
        all_results.append(analysis)
        cv2.imwrite(os.path.join(OUT_DIR, "phase2_detail_debug.png"), draw_debug_overlay(frame, analysis))
        print(f"  详情页 YOLO@0.50: MH={analysis['yolo_conf50']['found']}")
        print(f"  详情页 All labels: {analysis['all_labels_conf50']}")

        # 找 上限解放 按钮
        yolo_res = detect(frame)
        buttons = ButtonList(yolo_res)
        lb_btn = buttons.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
        convert_btn = buttons.get_button_by_text(SupportCardText.SUPPORT_CONVERT, _FUZZ)

        print(f"  上限解放: {'找到' if lb_btn else '未找到'}{' (disabled)' if lb_btn and lb_btn.is_disabled() else ''}")
        print(f"  サポート変換: {'找到' if convert_btn else '未找到'}{' (disabled)' if convert_btn and convert_btn.is_disabled() else ''}")

        # 尝试点击 上限解放 触发模态/页面
        target_btn = None
        target_name = ""
        if lb_btn and not lb_btn.is_disabled():
            target_btn = lb_btn
            target_name = "上限解放"
        elif convert_btn and not convert_btn.is_disabled():
            target_btn = convert_btn
            target_name = "サポート変換"
        elif lb_btn:
            # 即使 disabled 也点，看看是否会弹 toast/modal
            target_btn = lb_btn
            target_name = "上限解放(disabled)"

        if target_btn:
            print(f"\n  === 点击 {target_name} ===")
            click_adb(dev, target_btn.cx, target_btn.cy)
            time.sleep(1.5)

            # 多次截图分析（模态框可能有动画）
            for cap_idx in range(5):
                time.sleep(0.5)
                frame = adb_screenshot(dev)
                tag = f"after_{target_name}_cap{cap_idx}"
                analysis = analyze_modal_header(frame, tag)
                all_results.append(analysis)

                fname = f"phase2_{target_name}_cap{cap_idx}"
                cv2.imwrite(os.path.join(OUT_DIR, f"{fname}.png"), frame)
                cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_debug.png"), draw_debug_overlay(frame, analysis))

                found_50 = analysis["yolo_conf50"]["found"]
                found_25 = analysis["yolo_conf25"]["found"]
                found_10 = analysis["yolo_conf10"]["found"]
                hsv = analysis["hsv_fallback"]["found"]
                confs_10 = analysis["yolo_conf10"]["confs"]
                print(f"    cap{cap_idx}: YOLO@0.50={found_50}, @0.25={found_25}, @0.10={found_10} (confs={confs_10}), HSV={hsv}")
                print(f"    labels: {analysis['all_labels_conf50']}")

            # 尝试 ModalParser 完整解析
            print(f"\n  === ModalParser 完整解析（最后一帧） ===")
            yolo_res = detect(frame, conf=0.5)
            try:
                modal = ModalParser(yolo_res, quiet=False).parse()
                if modal:
                    print(f"    标题: {modal.modal_title}")
                    print(f"    确认按钮: {modal.confirm_button}")
                    print(f"    取消按钮: {modal.cancel_button}")
                    cv2.imwrite(os.path.join(OUT_DIR, "phase2_modal_parsed.png"), modal.draw_debug())
                else:
                    print("    ModalParser 解析失败（返回 None）")
                    # 试 conf=0.25
                    yolo_res_25 = detect(frame, conf=0.25)
                    modal_25 = ModalParser(yolo_res_25, quiet=False).parse()
                    if modal_25:
                        print(f"    用 conf=0.25 可以解析! 标题: {modal_25.modal_title}")
                    else:
                        print("    conf=0.25 也无法解析")
            except Exception as e:
                print(f"    ModalParser 异常: {e}")
                traceback.print_exc()

            # 返回详情页
            print("\n  === 返回操作 ===")
            # 找取消/返回按钮
            yolo_res = detect(frame)
            back_btns = yolo_res.filter_by_label(BaseUILabels.BACK_BTN)
            cancel_btns = yolo_res.filter_by_label(BaseUILabels.CLOSE_BUTTON)
            buttons = ButtonList(yolo_res)

            if cancel_btns and len(cancel_btns) > 0:
                btn = cancel_btns.boxes[0]
                print(f"    点击 CLOSE_BUTTON ({btn.cx}, {btn.cy})")
                click_adb(dev, btn.cx, btn.cy)
            elif back_btns and len(back_btns) > 0:
                btn = back_btns.boxes[0]
                print(f"    点击 BACK_BTN ({btn.cx}, {btn.cy})")
                click_adb(dev, btn.cx, btn.cy)
            else:
                # 按物理返回键
                print("    按物理返回键")
                dev.keyevent(4)
                time.sleep(1)
        else:
            print("  没有找到可以触发模态框的按钮")

        # 返回列表页
        time.sleep(1)
        frame_after = adb_screenshot(dev)
        cv2.imwrite(os.path.join(OUT_DIR, "phase2_after_back.png"), frame_after)

        # 如果还在详情页，再返回一次
        yolo_after = detect(frame_after)
        back_btns = yolo_after.filter_by_label(BaseUILabels.BACK_BTN)
        if back_btns and len(back_btns) > 0:
            print("  还在子页面，再返回一次")
            dev.keyevent(4)
            time.sleep(1.5)
    else:
        print("  当前页面没有检测到 ITEM，跳过卡片交互")

    # === Phase 3: 汇总 ===
    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)

    has_modal_pages = [r for r in all_results if r["yolo_conf10"]["found"]]
    no_modal_pages = [r for r in all_results if not r["yolo_conf10"]["found"]]

    print(f"  总分析帧数: {len(all_results)}")
    print(f"  YOLO@0.50 检到 MH 的帧: {sum(1 for r in all_results if r['yolo_conf50']['found'])}")
    print(f"  YOLO@0.25 检到 MH 的帧: {sum(1 for r in all_results if r['yolo_conf25']['found'])}")
    print(f"  YOLO@0.10 检到 MH 的帧: {sum(1 for r in all_results if r['yolo_conf10']['found'])}")
    print(f"  HSV fallback 检到的帧: {sum(1 for r in all_results if r['hsv_fallback'].get('found', False))}")

    if has_modal_pages:
        conf10_vals = []
        for r in has_modal_pages:
            conf10_vals.extend(r["yolo_conf10"]["confs"])
        if conf10_vals:
            print(f"  MH 置信度范围(@0.10): [{min(conf10_vals):.4f}, {max(conf10_vals):.4f}]")
            below_50 = [c for c in conf10_vals if c < 0.5]
            if below_50:
                print(f"  ⚠️  有 {len(below_50)} 次 MH 置信度 < 0.5，会被默认阈值过滤!")
                print(f"     这些置信度: {sorted(below_50)}")

    # 保存完整报告
    report_path = os.path.join(OUT_DIR, "modal_header_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告: {report_path}")
    print(f"截图保存: {OUT_DIR}")


if __name__ == "__main__":
    main()
