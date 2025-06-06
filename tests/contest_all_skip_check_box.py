import os

import cv2

from src.utils.opencv_tools import check_status_detection

base_path = os.path.join(os.getcwd(), "contest_all_skip_check_box")
for filename in os.listdir(base_path):
    if filename.endswith(".png"):
        image = cv2.imread(os.path.join(base_path, filename))
        print(filename, check_status_detection(image))
