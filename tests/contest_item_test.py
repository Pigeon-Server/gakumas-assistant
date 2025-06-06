import cv2
import numpy as np
from src.utils.opencv_tools import *

# 读取图像
img = cv2.imread("2025-06-05 190438.png")

height, width = img.shape[:2]

# 颜色阈值：123,130,131 为 BGR 格式（你可以微调容差）
target = np.array([123,130,131])
lower = target-20  # 允许一定偏差
upper = target+20

# 颜色掩膜：只保留接近指定颜色的区域
mask = cv2.inRange(img, lower, upper)

# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 依次提取每个区域
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)

    # 可加筛选条件，比如最小宽高或面积
    if w > width//2 and h > 10:
        roi = img[y:y+h, x:x+w]
        cv2.imshow(f"element_{i}.png", roi)
        # print(f"保存区域 element_{i}.png")

# 可视化检查
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
