import cv2
import numpy as np

img = cv2.imread("skill_card_list__error1.png")
img_w, img_h = img.shape[:2]


hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 信息边框颜色
lower_color = np.array([0, 0, 180])
upper_color = np.array([0, 0, 220])
mask = cv2.inRange(hsv_img, lower_color, upper_color)
# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 技能卡边框颜色
skill_card_lower_color = np.array([104, 32, 87])
skill_card_upper_color = np.array([115, 87, 142])

# 提取每个区域
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # 筛选条件 宽度必须大于图像宽度的一半
    if w > img_w // 2 and h > img_h // 4:
        roi = img[y:y + h, x:x + w]
        cv2.imshow(f"ROI (x: {x} y: {y})", roi)

        # 在 ROI 中创建新的掩码
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(roi_hsv, skill_card_lower_color, skill_card_upper_color)

        # 查找技能卡轮廓
        skill_card_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到技能卡最大宽度
        for skill_card_cnt in skill_card_contours:
            x_skill, y_skill, w_skill, h_skill = cv2.boundingRect(skill_card_cnt)
            if h_skill >= h // 3:
                skill_card = roi[y_skill:y_skill + h_skill, x_skill:x_skill + w_skill]
                skill_card_info = roi[:, x_skill + w_skill:]
                skill_card_info = cv2.cvtColor(skill_card_info, cv2.COLOR_BGR2GRAY)
                _, skill_card_info = cv2.threshold(skill_card_info, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imshow(f"Extracted Skill Card (x: {x_skill} y: {y_skill})", skill_card)
                cv2.imshow(f"Skill Info (x: {x} y: {y})", skill_card_info)
                break
        break  # 只处理第一个符合条件的区域

cv2.waitKey(0)
