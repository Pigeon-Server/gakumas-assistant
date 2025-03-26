# 运行模式（Phone | PC）
mode = "PC"
# 游戏窗口名称（仅PC）
window_name = "gakumas"
# 强制使用CPU推理
use_cpu = False
# 是否启用Debug模式
debug = True
# Debug模式预览窗口名
debug_window_name = f"{window_name} yolo debug"

model_path = './model/last.pt'
conf_threshold = 0.5
iou_threshold = 0.5