"""Test chevron glyph detection on captured button frames."""
import cv2
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Button import ButtonList


def detect_chevron_count(frame: np.ndarray) -> int:
    """检测按钮帧中的 chevron (>) 字形数量。
    
    通过二值化 + 轮廓多边形近似，识别向右的 V 字形。
    Returns: 0=没有, 1=单箭头(>), 2=双箭头(>>)
    """
    if frame is None or frame.size == 0:
        return 0
    h, w = frame.shape[:2]
    if h < 10 or w < 10:
        return 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Otsu 二值化: BINARY_INV 使深色字形变白（前景）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 裁掉边缘 margin（避免圆形按钮边框被误识别为轮廓）
    margin = max(2, int(min(h, w) * 0.08))
    roi = binary[margin:h - margin, margin:w - margin]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = h * w * 0.02
    roi_h, roi_w = roi.shape[:2]
    chevron_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 多边形近似
        eps = 0.04 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True)
        n = len(poly)
        pts = poly.reshape(-1, 2)

        # 尺寸过滤: 字形高度需要占 ROI 的 30% 以上
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        shape_h = y_max - y_min
        if shape_h < roi_h * 0.3:
            continue

        # chevron 特征:
        # 1. 顶点数 3~8（简单 V 形或带圆角的箭头）
        # 2. 最右端的点接近垂直中心（箭头尖端指向右侧）
        rightmost_pt = pts[pts[:, 0].argmax()]
        center_y = roi_h / 2.0

        print(f"  contour: area={area:.0f} vertices={n} shape_h={shape_h} "
              f"rightmost=({rightmost_pt[0]},{rightmost_pt[1]}) cy={center_y:.0f}")

        if 3 <= n <= 8 and abs(rightmost_pt[1] - center_y) < roi_h * 0.3:
            chevron_count += 1
            print(f"    -> CHEVRON detected!")

    return chevron_count


def main():
    print("=== Chevron detection test ===\n")

    # Test 1: From saved button crops
    print("--- Test from individual button crops ---")
    for i in range(4):
        path = f"/tmp/btn_{i}.png"
        if not os.path.exists(path):
            continue
        frame = cv2.imread(path)
        count = detect_chevron_count(frame)
        print(f"Btn {i}: chevron_count = {count}\n")

    # Test 2: From full screenshots via YOLO
    print("\n--- Test from YOLO-detected buttons on full screenshots ---")
    model = YoloModelFromONNX("model/base_ui.onnx")

    test_images = [
        "logs/debug/test_captures/support_card/enhance_nonmax_0_enhance_page.png",
        "logs/debug/test_captures/support_card/step4_enhance_page_card0.png",
        "logs/debug/test_captures/support_card/step4_enhance_page_card1.png",
        "logs/debug/test_captures/support_card/step4_enhance_page_card2.png",
    ]

    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        print(f"\nImage: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        yr = Yolo_Results(model(img), img)
        btns = ButtonList(yr)

        for j, b in enumerate(btns.buttons):
            chevrons = detect_chevron_count(b.frame)
            glyph = ">>" if chevrons == 2 else (">" if chevrons == 1 else "-")
            print(f"  Btn {j}: text=\"{b.text}\" size={b.frame.shape[1]}x{b.frame.shape[0]} "
                  f"chevrons={chevrons} glyph=\"{glyph}\"")
        print()

    # Test 3: From non-enhance pages (should detect 0 chevrons)
    print("\n--- Negative test: non-enhance pages ---")
    neg_images = [
        "logs/debug/test_captures/support_card/step3_detail_page_card0.png",
        "logs/debug/test_captures/support_card/step1_card_list_card0.png",
    ]
    for img_path in neg_images:
        if not os.path.exists(img_path):
            continue
        print(f"\nImage: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        yr = Yolo_Results(model(img), img)
        btns = ButtonList(yr)
        for j, b in enumerate(btns.buttons):
            chevrons = detect_chevron_count(b.frame)
            print(f"  Btn {j}: text=\"{b.text}\" chevrons={chevrons}")


if __name__ == "__main__":
    main()
