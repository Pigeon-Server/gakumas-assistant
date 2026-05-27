"""
快速 MODAL_HEADER 诊断: 点击 ≡ 菜单按钮触发模态
"""
import os, sys, time, json
import cv2, numpy as np, adbutils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.inference.ocr_engine import OCRService
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Modal import ModalParser
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels

OUT = os.path.join("logs", "debug", "test_captures", "modal_trigger")
os.makedirs(OUT, exist_ok=True)
model = YoloModelFromONNX(config.model_config["BASE_UI"])
ocr = OCRService()


def ss(dev):
    return cv2.cvtColor(np.asarray(dev.screenshot()), cv2.COLOR_RGB2BGR)


def det(frame, conf=0.5):
    return Yolo_Results(model(frame, conf_threshold=conf, iou_threshold=0.5), frame)


def do_analysis(frame, tag):
    """在多置信度下分析 MODAL_HEADER"""
    results = {}
    for c in [0.5, 0.25, 0.15, 0.1]:
        y = det(frame, conf=c)
        mh = y.filter_by_label(BaseUILabels.MODAL_HEADER)
        found = mh and len(mh) > 0
        confs = [round(b.confidence, 4) for b in mh.boxes] if mh else []
        results[f"conf_{c}"] = {"found": found, "confs": confs}
        mk = "✓" if found else "✗"
        print(f"  {tag}@{c}: {mk} {confs}")

    # All labels at 0.5
    y50 = det(frame)
    labels = {}
    for b in y50.boxes:
        labels[b.label] = labels.get(b.label, 0) + 1
    print(f"  labels@0.5: {labels}")

    # HSV fallback
    btns = y50.filter_by_label(BaseUILabels.BUTTON)
    hsv_found = False
    if btns and len(btns) > 0:
        parser = ModalParser(y50, quiet=True)
        parser.action_buttons = parser._get_action_buttons()
        if parser.action_buttons:
            hsv = parser._infer_header_from_frame()
            hsv_found = hsv is not None
            if hsv:
                print(f"  HSV fallback: ✓ box=({hsv.x},{hsv.y},{hsv.w},{hsv.h})")
            else:
                print(f"  HSV fallback: ✗")
    results["hsv_found"] = hsv_found

    # Full ModalParser
    for c in [0.5, 0.25]:
        yc = det(frame, conf=c)
        modal = ModalParser(yc, quiet=True).parse()
        if modal:
            print(f"  ModalParser@{c}: ✓ title={modal.modal_title}")
            try:
                cv2.imwrite(os.path.join(OUT, f"{tag}_modal_debug_c{c}.png"), modal.draw_debug())
            except Exception:
                pass  # 调试绘图失败不影响主流程
            results["modal_title"] = modal.modal_title
            results["parsed_conf"] = c
            break
    else:
        print(f"  ModalParser: ✗ (未解析出模态框)")

    return results


def main():
    print("=" * 50)
    print("  MODAL_HEADER 快速诊断")
    print("=" * 50)

    adb = adbutils.AdbClient()
    dev = adb.device_list()[0]
    print(f"设备: {dev.serial}")

    all_results = {}

    # 1. 点击 ≡ 菜单按钮 (屏幕右下角)
    frame0 = ss(dev)
    h, w = frame0.shape[:2]
    # ≡ 按钮位置大约在右下角
    menu_x, menu_y = int(w * 0.9), int(h * 0.95)
    print(f"\n--- 点击 ≡ 菜单 ({menu_x}, {menu_y}) ---")
    dev.click(menu_x, menu_y)
    time.sleep(2.0)

    for i in range(5):
        time.sleep(0.5)
        frame = ss(dev)
        cv2.imwrite(os.path.join(OUT, f"menu_cap{i}.png"), frame)
        print(f"\ncap{i}:")
        r = do_analysis(frame, f"menu_{i}")
        all_results[f"menu_cap{i}"] = r

    # 返回
    dev.keyevent(4)
    time.sleep(1)

    # 2. 试试サポートカード → 详情 → Lv強化流程
    print(f"\n\n--- 导航到サポートカード ---")
    frame = ss(dev)
    # 点击サポートカード按钮 (从截图看大约在 x=330, y=1730)
    dev.click(330, 1730)
    time.sleep(2.0)

    frame = ss(dev)
    cv2.imwrite(os.path.join(OUT, "support_list.png"), frame)

    # 找卡片
    y = det(frame)
    cards = y.filter_by_label(BaseUILabels.SUPPORT_CARD)
    if cards and len(cards) > 0:
        # 滚动到底部找低级卡
        print(f"  找到 {len(cards)} 张支援卡")

        # 点击靠后的卡片（可能有 enabled 上限解放）
        target = cards.boxes[min(len(cards.boxes) - 1, 8)]
        print(f"  点击卡片 ({target.cx}, {target.cy})")
        dev.click(int(target.cx), int(target.cy))
        time.sleep(2.0)

        frame = ss(dev)
        cv2.imwrite(os.path.join(OUT, "card_detail.png"), frame)

        # 查看所有按钮OCR
        yd = det(frame)
        from src.entity.Game.Components.Button import ButtonList
        from src.utils.string_tools import MatchConfig
        _FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)
        btns = ButtonList(yd)
        lb = btns.get_button_by_text("上限解放", _FUZZ)
        lv = btns.get_button_by_text("Lv強化", _FUZZ)
        cv = btns.get_button_by_text("サポート変換", _FUZZ)
        print(f"  Lv強化: {'✓' if lv else '✗'}{' disabled' if lv and lv.is_disabled() else ''}")
        print(f"  上限解放: {'✓' if lb else '✗'}{' disabled' if lb and lb.is_disabled() else ''}")
        print(f"  サポート変換: {'✓' if cv else '✗'}{' disabled' if cv and cv.is_disabled() else ''}")

        # 如果 Lv強化 可用，进入强化页面触发模态
        if lv and not lv.is_disabled():
            print(f"\n--- 点击 Lv強化 ---")
            dev.click(int(lv.cx), int(lv.cy))
            time.sleep(2.0)

            frame = ss(dev)
            cv2.imwrite(os.path.join(OUT, "enhance_page.png"), frame)
            yd2 = det(frame)
            labels2 = {}
            for b in yd2.boxes:
                labels2[b.label] = labels2.get(b.label, 0) + 1
            print(f"  强化页 labels: {labels2}")

            # 找强化按钮
            btns2 = ButtonList(yd2)
            confirm = btns2.get_button_by_text("強化する", _FUZZ)
            if not confirm:
                confirm = btns2.get_button_by_text("強化", _FUZZ)
            if confirm and not confirm.is_disabled():
                print(f"  点击 強化する ({confirm.cx}, {confirm.cy})")
                dev.click(int(confirm.cx), int(confirm.cy))
                time.sleep(2.0)

                # 反复截图分析模态框
                for i in range(8):
                    time.sleep(0.5)
                    frame = ss(dev)
                    cv2.imwrite(os.path.join(OUT, f"enhance_modal_cap{i}.png"), frame)
                    print(f"\n强化模态 cap{i}:")
                    r = do_analysis(frame, f"enhance_{i}")
                    all_results[f"enhance_cap{i}"] = r

                    # 如果成功检测到模态框，尝试取消
                    if r.get("modal_title"):
                        print(f"  取消（点返回键）...")
                        dev.keyevent(4)
                        time.sleep(1)
                        break
                else:
                    # 没有检测到模态，返回
                    dev.keyevent(4)
                    time.sleep(1)
            else:
                print(f"  強化する: {'disabled' if confirm else '未找到'}")

            # 返回详情
            dev.keyevent(4)
            time.sleep(1)

        # 返回列表
        dev.keyevent(4)
        time.sleep(1)
    else:
        print("  没有找到支援卡")

    # 回主页
    dev.keyevent(4)
    time.sleep(1)

    # 保存报告
    rpath = os.path.join(OUT, "modal_trigger_report.json")
    with open(rpath, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {rpath}")
    print(f"截图: {OUT}")


if __name__ == "__main__":
    main()
