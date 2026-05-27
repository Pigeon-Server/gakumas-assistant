import json
import os

import clip
import onnx
import torch
from ultralytics import YOLO

BASE_PATH = os.path.join(os.getcwd(), "model")

print("Exporting YOLO model...")
for filename in os.listdir(BASE_PATH):
    if filename.endswith(".pt"):
        file_path = os.path.join(BASE_PATH, filename)
        print(f"Loading model: {file_path}")
        model = YOLO(file_path)

        # 获取模型导出前的关键信息
        model_name = os.path.splitext(filename)[0]
        model.export(format="onnx", dynamic=True)
        # 写入元信息到 JSON 文件
        model = onnx.load(os.path.join(BASE_PATH, f"{model_name}.onnx"))
        meta = {p.key: p.value for p in model.metadata_props}
        print(meta)
        meta_path = os.path.join(BASE_PATH, f"{model_name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"Exported {model_name}.onnx and {model_name}_meta.json")

print("Export CLIP model...")
# 载入 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()

# 获取图像子模型
visual_model = model.visual.eval()

# 构造 dummy 输入
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # CLIP 的标准输入尺寸

# 导出路径
os.makedirs("onnx_models", exist_ok=True)
output_path = os.path.join(BASE_PATH, "clip_visual.onnx")

# 导出 ONNX 模型
torch.onnx.export(
    visual_model,                    # CLIP 的视觉子模块
    dummy_input,                     # 输入张量
    output_path,
    input_names=["input"],           # ONNX 输入名
    output_names=["image_features"], # ONNX 输出名
    dynamic_axes={"input": {0: "batch_size"}, "image_features": {0: "batch_size"}},
    # opset_version=11,
    do_constant_folding=True
)