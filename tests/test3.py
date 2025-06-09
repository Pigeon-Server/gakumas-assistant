import cv2
import numpy as np


# 从掩码中提取最大轮廓的 ROI

def get_mask_contours(img, lower_color, upper_color):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_roi_from_mask(img, lower_color, upper_color):
    contours = get_mask_contours(img, lower_color, upper_color)
    max_area = 0
    max_contour = None

    # 找到最大轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        # 获取最大轮廓的边界框
        x, y, w, h = cv2.boundingRect(max_contour)
        # roi = img[y:y + h, x:x + w]
        return x, y, w, h
    return None


def modal_body_extract_item_info(img):
    """
    在模态框中提取物品信息
    :param img:
    :return:
    """
    item_lower = np.array([81,0,95])
    item_upper = np.array([110,19,128])

    mark_lower = np.array([96,75,231])
    mark_upper = np.array([98,145,250])

    item_marks = extract_roi_from_mask(img, item_lower, item_upper)

    if not item_marks:
        return None, None

    item_x,item_y,item_w,item_h = item_marks

    item = img[item_y:item_y+item_h, item_x:item_x+item_w]

    mark_y = 0
    contours = get_mask_contours(img[item_y+item_h:], mark_lower, mark_upper)
    for contour in contours:
        _x,_y,_w,_h = cv2.boundingRect(contour)
        if _h > 5 and _w > 5:
            mark_y = min(_y, mark_y)
    mark_y = item_y+item_h+mark_y
    if mark_y < item_y + item_h:
        item_info = img[item_y:item_y+item_h, item_x+item_w:]
    else:
        item_info = img[item_y:mark_y, item_x+item_w:]

    return item, item_info


item_lower = np.array([81,0,95])
item_upper = np.array([110,19,128])

mark_lower = np.array([96,75,231])
mark_upper = np.array([98,145,250])
if __name__ == '__main__':

    img = cv2.imread("Trade Confirm Modal Body.png")

    # item_x,item_y,item_w,item_h = extract_roi_from_mask(img, item_lower, item_upper)
    #
    # item = img[item_y:item_y+item_h, item_x:item_x+item_w]
    #
    # # mark_x, mark_y, mark_w, mark_h = 0,0,0,0
    # mark_y = 0
    # contours = get_mask_contours(img[item_y+item_h:], mark_lower, mark_upper)
    # for contour in contours:
    #     _x,_y,_w,_h = cv2.boundingRect(contour)
    #     if _h > 5 and _w > 5:
    #         mark_y = min(_y, mark_y)
    # # mark = img[mark_y:mark_y+mark_h, mark_x:mark_x+mark_w]
    # item_info = img[item_y:item_y+item_h+mark_y, item_x+item_w:]

    item, item_info = modal_body_extract_item_info(img)

    cv2.imshow("item", item)
    # cv2.imshow("mark", mark)
    cv2.imshow("item_info", item_info)

    cv2.waitKey(0)
    cv2.destroyAllWindows()