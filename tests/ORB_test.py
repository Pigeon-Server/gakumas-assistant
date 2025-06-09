import os
import cv2
import numpy as np
import pickle

# 初始化记忆库（字典形式）
image_memory = {}

# 从磁盘加载记忆库
def load_memory(file_path):
    global image_memory
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            image_memory = pickle.load(f)

# 保存记忆库到磁盘
def save_memory(file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(image_memory, f)

def extract_orb_features(img):
    # 加载图像并转换为灰度图像
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img_gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测关键点和计算描述符
    keypoints, descriptors = orb.detectAndCompute(img, None)

    return keypoints, descriptors

def add_to_memory(image_name, img):
    keypoints, descriptors = extract_orb_features(img)

    if descriptors is None:
        print(f"图像 '{image_name}' 无特征，跳过记忆。")
        return

    image_memory[image_name] = {'features': descriptors}
    save_memory('image_memory.pkl')


def match_images(img):
    query_keypoints, query_descriptors = extract_orb_features(img)

    if query_descriptors is None:
        return None, 0  # 查询图像没有特征

    best_match = None
    best_match_count = 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for card_name, card_data in image_memory.items():
        stored_descriptors = card_data['features']

        if stored_descriptors is None:
            continue  # 跳过没有特征的图像

        # 确保数据类型一致
        if query_descriptors.dtype != stored_descriptors.dtype:
            continue

        matches = bf.match(query_descriptors, stored_descriptors)

        if len(matches) > best_match_count:
            best_match_count = len(matches)
            best_match = card_name

    return best_match, best_match_count


def handle_new_image(image_name, img):
    # 检查图像是否已记忆
    best_match, match_count = match_images(img)

    if best_match is None or match_count < 10:  # 认为没有足够匹配时，视为新图像
        # 如果未找到匹配，添加到记忆库
        print(f"图像 '{image_name}' 尚未记忆，正在添加到记忆库。")
        add_to_memory(image_name, img)
    else:
        print(f"图像 '{image_name}' 已记忆，与 '{best_match}' 匹配。")

# 示例：处理新卡牌图像

# 读取抠图后的按钮图像
# base_path = os.path.join(os.getcwd(), "button_disabled_test")
# 加载之前保存的记忆库
load_memory('image_memory.pkl')

# for filename in os.listdir(base_path):
#     if filename.endswith(".png"):
#         image_path = os.path.join(base_path, filename)
#         img = cv2.imread(image_path)
#         print(f"正在处理 {filename}")
#         print(handle_new_image(filename, img))
