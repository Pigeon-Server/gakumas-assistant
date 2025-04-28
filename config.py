# 运行模式（Phone | PC）
from src.entity.YOLO_Model_Type import YoloModelType

mode = "PC"
# 游戏窗口名称（仅PC）
window_name = "gakumas"
# 强制使用CPU推理
use_cpu = False
# 是否启用Debug模式
debug = True
# Debug模式预览窗口名
debug_window_name = f"{window_name} yolo debug"

model_config = {
    YoloModelType.BASE_UI: {
        "model_path": "model/base_ui.pt",
        "conf_threshold": 0.5,
        "iou_threshold": 0.5
    },
    YoloModelType.PRODUCER: {
        "model_path": "model/producer.pt",
        "conf_threshold": 0.5,
        "iou_threshold": 0.5
    },
}