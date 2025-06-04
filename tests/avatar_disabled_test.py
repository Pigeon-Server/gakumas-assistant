import os.path

import cv2
import numpy as np

def is_desaturated(image_path, saturation_threshold=30, ratio_threshold=0.9):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]  # S 通道

    low_sat_ratio = np.sum(saturation < saturation_threshold) / saturation.size
    return low_sat_ratio > ratio_threshold

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

def is_desaturated_with_vis(image_path, saturation_threshold=40, ratio_threshold=0.9, show_vis=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像无法读取")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]  # S 通道

    # 判断低饱和度区域
    low_sat_mask = saturation < saturation_threshold
    low_sat_ratio = np.sum(low_sat_mask) / saturation.size
    is_gray = low_sat_ratio > ratio_threshold

    if show_vis:
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))

        # 原图（BGR 转 RGB）
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("原图")
        axs[0].axis('off')

        # 饱和度通道可视化
        axs[1].imshow(saturation, cmap='viridis')
        axs[1].set_title("饱和度 (S通道)")
        axs[1].axis('off')

        # 低饱和度掩膜图
        axs[2].imshow(low_sat_mask, cmap='gray')
        axs[2].set_title(f"低饱和度区域\n占比: {low_sat_ratio:.2%}")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

    return is_gray

# 读取抠图后的按钮图像
base_path = os.path.join(os.getcwd(), "avatar_disabled_test")
for filename in os.listdir(base_path):
    if filename.endswith(".png"):
        print(filename, is_desaturated_with_vis(os.path.join(base_path, filename), ratio_threshold=0.6))
