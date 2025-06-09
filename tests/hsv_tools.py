import colorsys

import cv2
import numpy as np

# 回调函数：实时更新图像
def nothing(x):
    pass

def hsv_range_to_image_cv(lower, upper, height=50, width=300):
    """
    用 OpenCV HSV 范围的 lower 和 upper 生成一张条形图，表示色调范围。
    lower, upper: (H:0-179, S:0-255, V:0-255)
    """
    h_vals = np.linspace(lower[0], upper[0], width)
    s_val = (lower[1] + upper[1]) / 2
    v_val = (lower[2] + upper[2]) / 2

    img = np.zeros((height, width, 3), dtype=np.uint8)

    for i, h in enumerate(h_vals):
        # colorsys hsv 输入范围是 H:0-1, S/V:0-1，需要转换
        h_norm = h / 179
        s_norm = s_val / 255
        v_norm = v_val / 255
        r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
        img[:, i, 0] = int(b * 255)  # OpenCV 是 BGR 顺序
        img[:, i, 1] = int(g * 255)
        img[:, i, 2] = int(r * 255)

    cv2.imshow(f"HSV Range", img)

# 获取点击位置的HSV值
def get_hsv_value(event, x, y, flags, param):
    global lower_h, upper_h, lower_s, upper_s, lower_v, upper_v

    if event == cv2.EVENT_LBUTTONDOWN:
        # 读取图像
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 获取点击位置的HSV值
        hsv_value = hsv_img[y, x]
        h, s, v = hsv_value

        # 更新滑动条的值
        cv2.setTrackbarPos('Lower H', 'Trackbars', h - 10)
        cv2.setTrackbarPos('Upper H', 'Trackbars', h + 10)
        cv2.setTrackbarPos('Lower S', 'Trackbars', max(s - 50, 0))
        cv2.setTrackbarPos('Upper S', 'Trackbars', min(s + 50, 255))
        cv2.setTrackbarPos('Lower V', 'Trackbars', max(v - 50, 0))
        cv2.setTrackbarPos('Upper V', 'Trackbars', min(v + 50, 255))

# 读取图像
img = cv2.imread(r"E:\Projects\gkmas-auto\tests\Trade Confirm.png")

# 创建窗口
cv2.namedWindow('Trackbars')

# 创建滑动条，调节HSV的上下限
cv2.createTrackbar('Lower H', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('Upper H', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('Lower S', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Upper S', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Lower V', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Upper V', 'Trackbars', 255, 255, nothing)

# 设置鼠标点击事件
cv2.namedWindow('Original Image')
cv2.moveWindow("Original Image", 100, 100)
cv2.setMouseCallback('Original Image', get_hsv_value)

while True:
    # 读取滑动条的当前值
    lower_h = cv2.getTrackbarPos('Lower H', 'Trackbars')
    upper_h = cv2.getTrackbarPos('Upper H', 'Trackbars')
    lower_s = cv2.getTrackbarPos('Lower S', 'Trackbars')
    upper_s = cv2.getTrackbarPos('Upper S', 'Trackbars')
    lower_v = cv2.getTrackbarPos('Lower V', 'Trackbars')
    upper_v = cv2.getTrackbarPos('Upper V', 'Trackbars')

    # 转换为HSV色彩空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 设置HSV的范围
    lower_hsv = np.array([lower_h, lower_s, lower_v])
    upper_hsv = np.array([upper_h, upper_s, upper_v])

    # 创建蒙版
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    # 提取目标区域
    result = cv2.bitwise_and(img, img, mask=mask)

    # 显示结果
    cv2.imshow('Original Image', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    hsv_range_to_image_cv(lower_hsv, upper_hsv)

    # 按Esc键退出
    if cv2.waitKey(1) & 0xFF == 27:
        # 输出当前HSV范围数组
        print("Lower HSV: ", lower_hsv)
        print("Upper HSV: ", upper_hsv)
        break

cv2.destroyAllWindows()
