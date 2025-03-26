import cv2
import numpy as np
import scrcpy
import torch
from adbutils import adb
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# 检查 CUDA 是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")  # 输出当前使用的设备
model = YOLO('gkmas-menu_2025-3-13_best.pt').to(device)
client = scrcpy.Client(device=adb.device_list()[0])
cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)

def on_frame(frame):
    # If you set non-blocking (default) in constructor, the frame event receiver
    # may receive None to avoid blocking event.
    # print(frame)
    if frame is not None and model is not None:
        # frame is an bgr numpy ndarray (cv2' default format)
        # cv2.imshow("viz", frame)

        # # 解码
        # img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        # 记录原始分辨率
        original_h, original_w = frame.shape[:2]
        # results = model(frame, imgsz=(original_h, original_w), verbose=False)
        results = model(frame, imgsz=1088, verbose=False, device=device)  # 以 1/2 分辨率运行
        # 创建标注器并绘制结果
        annotator = Annotator(frame)
        if results[0].boxes:
            # 获取实际推理尺寸
            model_h = results[0].orig_shape[0]
            model_w = results[0].orig_shape[1]

            # 计算缩放比例
            scale_x = original_w / model_w
            scale_y = original_h / model_h

            for box in results[0].boxes:
                # 缩放坐标到原始尺寸
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                scaled_coords = [
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y
                ]
                # 确保 box.conf 是 float
                annotator.box_label(
                    scaled_coords,
                    f"{model.names[int(box.cls)]} {float(box.conf):.2f}",
                    color=(255, 0, 0)
                )

        # 获取标注后的图像
        annotated_img = annotator.result()

        # 获取当前窗口大小
        win_w, win_h = cv2.getWindowImageRect("YOLO Detection")[2:4]

        # 计算缩放比例，保持等比例
        scale = min(win_w / original_w, win_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        # 重新缩放图像
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 在窗口中间绘制 (使用黑色背景填充)
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        start_x = (win_w - new_w) // 2
        start_y = (win_h - new_h) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_frame

        # 保持原始分辨率显示
        cv2.imshow('YOLO Detection', canvas)
    cv2.waitKey(1)

def on_init():
    # Print device name
    print("已连接到设备：", client.device_name)

client.add_listener(scrcpy.EVENT_INIT, on_init)
client.add_listener(scrcpy.EVENT_FRAME, on_frame)

client.start()