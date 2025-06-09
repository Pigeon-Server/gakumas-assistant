import colorsys
from typing import Tuple

import cv2
import numpy as np

from src.utils.logger import logger
from sklearn.metrics.pairwise import cosine_similarity


def get_mask_contours(img, lower_color, upper_color):
    """从图像中提取指定颜色范围的轮廓"""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_max_contour(contours):
    """返回最大轮廓和其边界框"""
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour

def extract_roi_from_mask(img, lower_color, upper_color):
    """提取最大轮廓的ROI"""
    contours = get_mask_contours(img, lower_color, upper_color)
    max_contour = get_max_contour(contours)

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        return x, y, w, h
    return None

def get_mark_y_position(img, lower_color, upper_color, roi_y, roi_h):
    """提取mark区域的Y位置"""
    contours = get_mask_contours(img[roi_y + roi_h:], lower_color, upper_color)
    mark_y = 0
    for contour in contours:
        _x, _y, _w, _h = cv2.boundingRect(contour)
        if _h > 5 and _w > 5:
            mark_y = min(_y, mark_y)
    return mark_y

def hsv_range_to_image_cv(lower, upper, height=50, width=300):
    """
    用 OpenCV HSV 范围的 lower 和 upper 生成一张条形图，表示色调范围。
    """
    h_vals = np.linspace(lower[0], upper[0], width)
    s_val = (lower[1] + upper[1]) / 2
    v_val = (lower[2] + upper[2]) / 2

    img = np.zeros((height, width, 3), dtype=np.uint8)

    for i, h in enumerate(h_vals):
        h_norm = h / 179
        s_norm = s_val / 255
        v_norm = v_val / 255
        r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
        img[:, i, 0] = int(b * 255)
        img[:, i, 1] = int(g * 255)
        img[:, i, 2] = int(r * 255)

    cv2.imshow(f"HSV Range {upper} - {lower}", img)
    # cv2.waitKey(0)

def check_color_in_region(
        frame: np.array,
        region: Tuple[int, int, int, int],
        lower_color: Tuple[int, int, int],
        upper_color: Tuple[int, int, int],
        threshold=1
):
    """
    检查图像某区域是否存在指定 RGB 范围的颜色
    """
    if frame.size == 0:
        return False
    x, y, w, h = map(int, region)
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return False
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, lower_color, upper_color)
    return cv2.countNonZero(mask) >= threshold

@logger.catch
def check_status_detection(
        frame: np.array,
        threshold=0.15,
        upper_color: Tuple[int, int, int] = (22, 255, 255),
        lower_color: Tuple[int, int, int] = (8, 100, 100),
        black_background_threshold=0.3  # 黑色背景占比阈值
):
    """
    选中状态检测：默认屏蔽白色背景，黑色背景占比大时屏蔽黑色背景
    """
    if frame.size == 0:
        return False

    lower_color = np.array(lower_color)
    upper_color = np.array(upper_color)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]
    total_area = height * width

    # 白色背景掩码
    white_mask = cv2.inRange(gray, 220, 255)
    # 黑色背景掩码
    black_mask = cv2.inRange(gray, 0, 30)

    black_ratio = cv2.countNonZero(black_mask) / total_area

    if black_ratio > black_background_threshold:
        # 屏蔽黑色背景，检测非黑色区域橙色
        combined_mask = cv2.bitwise_and(
            cv2.inRange(hsv, lower_color, upper_color),
            cv2.bitwise_not(black_mask)
        )
        non_black_area = cv2.countNonZero(cv2.bitwise_not(black_mask))
        if non_black_area == 0:
            return False
        orange_ratio = cv2.countNonZero(combined_mask) / non_black_area
        return orange_ratio > threshold
    else:
        # 默认屏蔽白色背景，检测非白色区域橙色
        combined_mask = cv2.bitwise_and(
            cv2.inRange(hsv, lower_color, upper_color),
            cv2.bitwise_not(white_mask)
        )
        non_white_area = cv2.countNonZero(cv2.bitwise_not(white_mask))
        if non_white_area == 0:
            return False
        orange_ratio = cv2.countNonZero(combined_mask) / non_white_area
        return orange_ratio > threshold

def extract_feature(image: np.ndarray) -> np.ndarray:
    # 简单处理方式：resize + flatten 成向量
    resized = cv2.resize(image, (64, 64))
    return resized.flatten().astype(np.float32).reshape(1, -1)