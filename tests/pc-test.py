import torch
import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
from ctypes import windll
import win32gui
# from paddleocr import PaddleOCR
from src.entity.Yolo import Yolo_Box

# 目标窗口名
window_name = "gakumas"
debug_window_name = f"{window_name} yolo debug"

# 修复DPI缩放问题
windll.user32.SetProcessDPIAware()

def capture_window(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if not hwnd:
        raise Exception(f'窗口 "{window_name}" 未找到')

    # 获取客户区尺寸和屏幕坐标
    client_rect = win32gui.GetClientRect(hwnd)
    client_left, client_top = win32gui.ClientToScreen(hwnd, (0, 0))
    client_width = client_rect[2]  # 客户区宽度
    client_height = client_rect[3]  # 客户区高度

    # 截取客户区域
    screenshot = pyautogui.screenshot(
        region=(client_left, client_top, client_width, client_height)
    )

    # 转换为 OpenCV 格式
    img = np.array(screenshot)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"
print("device:", device)
model = YOLO('../model/producer.pt').to(device).eval()
model_imgsz = model.args['imgsz'] if hasattr(model, 'args') else 640
print(f"模型输入尺寸: {model_imgsz}")
# ocr = PaddleOCR(lang='japan', debug=False, show_log=False, use_angle_cls=True)

# 设置模型参数，提高推理精度和速度
model.conf = 0.5  # 置信度阈值
model.iou = 0.5   # IoU 阈值
model.half = torch.cuda.is_available()  # 使用半精度（FP16）加快推理

try:
    cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL)

    while True:
        # 捕获窗口图像
        frame = capture_window(window_name)
        if frame is None or frame.size == 0:
            continue

        h, w = frame.shape[:2]

        # 模型推理
        results = model(frame, imgsz=model_imgsz, verbose=False, stream=True)

        cv2.resizeWindow(debug_window_name, w, h)
        # 结果处理
        for result in results:
            if len(result.boxes) == 0:
                cv2.imshow(debug_window_name, frame)
                continue

            boxs = []
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxs.append(Yolo_Box(x1, y1, x2, y2, class_name, frame[y1:y2, x1:x2]))
            # print(boxs)
                # if class_name in ["Universal Confirm button", "Universal Cancel button", "Universal button", "Universal disable button", "Current location"]:
                #     ocr_result = ocr.ocr(frame_box, cls=True)
                #     print(ocr_result)

            # 绘制结果
            annotated_frame = result.plot(
                conf=False,
                line_width=max(1, int(h / 600)),
                font_size=max(0.5, h / 1200),
                pil=False
            )

            # 显示结果
            cv2.imshow(debug_window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"发生错误: {str(e)}")
finally:
    cv2.destroyAllWindows()
