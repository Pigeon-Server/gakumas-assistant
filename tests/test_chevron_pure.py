"""Pure OpenCV chevron glyph detection test - no OCR/YOLO dependency."""
import cv2
import numpy as np


def detect_chevron_count(frame: np.ndarray) -> int:
    """检测按钮帧中的 chevron (>) 字形数量。

    通过 Otsu 二值化 + 轮廓多边形近似，识别向右的 V 字形。
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
    roi_h = roi.shape[0]
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

        # 字形高度需占 ROI 30% 以上
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        shape_h = y_max - y_min
        if shape_h < roi_h * 0.3:
            continue

        # chevron 特征:
        # 1. 顶点数 3~8
        # 2. 最右端点接近垂直中心（箭头尖端指向右侧）
        rightmost_pt = pts[pts[:, 0].argmax()]
        center_y = roi_h / 2.0

        print(f"  contour: area={area:.0f} vertices={n} shape_h={shape_h} "
              f"rightmost=({rightmost_pt[0]},{rightmost_pt[1]}) cy={center_y:.0f}")

        if 3 <= n <= 8 and abs(rightmost_pt[1] - center_y) < roi_h * 0.3:
            chevron_count += 1
            print(f"    -> CHEVRON detected!")

    return chevron_count


if __name__ == "__main__":
    print("=== Chevron glyph detection test ===\n")

    # Test from saved button crops
    for i in range(4):
        path = f"/tmp/btn_{i}.png"
        frame = cv2.imread(path)
        if frame is None:
            print(f"Btn {i}: file not found")
            continue
        h, w = frame.shape[:2]
        count = detect_chevron_count(frame)
        glyph = ">>" if count == 2 else (">" if count == 1 else "none")
        print(f"Btn {i} ({w}x{h}): chevron_count={count} glyph=\"{glyph}\"\n")

    # Also test with キャンセル and Lv強化 buttons to ensure no false positives
    print("=== Expected: Btn 0 = > (1), Btn 1 = >> (2), Btn 2 = none (0), Btn 3 = none (0) ===")
