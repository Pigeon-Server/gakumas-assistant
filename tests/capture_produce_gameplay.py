#!/usr/bin/env python3
"""
采集培育游戏流程截图 & 双模型推理验证脚本。

用法：
  python tests/capture_produce_gameplay.py [step_name]

每次调用会：
  1. 截图保存 PNG + JPG（测试 JPG 噪点抗性）
  2. 分别用 BASE_UI 和 PRODUCER 两个 YOLO 模型推理
  3. 输出检测结果对比
  4. 可选地将检测框画在图上保存

所有截图和元数据存放在 tests/produce_gameplay_captures/<step_name>/ 下。
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Setup path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.constants.yolo.model_type import YoloModelType
from src.core.inference.ONNX import YoloModelFromONNX

CAPTURE_DIR = PROJECT_ROOT / "tests" / "produce_gameplay_captures"

# ── Model cache ─────────────────────────────────────────────
_models = {}

def get_model(model_type: str) -> YoloModelFromONNX:
    if model_type not in _models:
        _models[model_type] = YoloModelFromONNX(config.model_config[model_type])
    return _models[model_type]


def capture_adb(serial: str = "48c89aa") -> np.ndarray:
    """从 ADB 截图并返回 BGR ndarray。"""
    import subprocess
    raw = subprocess.check_output(
        ["adb", "-s", serial, "exec-out", "screencap", "-p"],
        timeout=10,
    )
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("ADB screencap decode failed")
    return img


def run_inference(img: np.ndarray, model_type: str, conf: float = 0.5):
    model = get_model(model_type)
    result = model(img, conf_threshold=conf)
    detections = []
    for box, score, cls_id in zip(result.boxes, result.scores, result.class_ids):
        cls_name = model._model_meta.names.get(int(cls_id), str(cls_id))
        detections.append({
            "label": cls_name,
            "confidence": float(score),
            "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "class_id": int(cls_id),
        })
    return detections


def draw_detections(img: np.ndarray, detections: list, color=(0, 255, 0)) -> np.ndarray:
    canvas = img.copy()
    for det in detections:
        x, y, w, h = det["box"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.putText(canvas, label, (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return canvas


def capture_step(step_name: str, variant: int = 0, serial: str = "48c89aa"):
    """截图 + 双模型推理 + 保存。"""
    step_dir = CAPTURE_DIR / step_name
    step_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    prefix = f"{step_name}_{variant}_{ts}"

    # 截图
    img = capture_adb(serial)
    png_path = step_dir / f"{prefix}.png"
    jpg_path = step_dir / f"{prefix}.jpg"
    cv2.imwrite(str(png_path), img)
    cv2.imwrite(str(jpg_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"\n{'='*60}")
    print(f"Step: {step_name} | Variant: {variant}")
    print(f"  PNG: {png_path}")
    print(f"  JPG: {jpg_path}")

    results = {}
    for model_type in [YoloModelType.BASE_UI, YoloModelType.PRODUCER]:
        # 同时对 PNG 和 JPG 做推理
        for img_path, fmt in [(png_path, "png"), (jpg_path, "jpg")]:
            test_img = cv2.imread(str(img_path))
            dets = run_inference(test_img, model_type)
            key = f"{model_type}_{fmt}"
            results[key] = dets
            print(f"\n  [{model_type}] ({fmt}) => {len(dets)} detections:")
            for d in dets:
                print(f"    {d['label']:45s} conf={d['confidence']:.3f}  box={d['box']}")

        # 画检测框保存
        annotated = draw_detections(img, results[f"{model_type}_png"],
                                    color=(0, 255, 0) if model_type == YoloModelType.BASE_UI else (0, 165, 255))
        cv2.imwrite(str(step_dir / f"{prefix}_{model_type}_annotated.jpg"), annotated)

    # 保存元数据
    meta_path = step_dir / f"{prefix}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 对比 PNG vs JPG 检测一致性
    for model_type in [YoloModelType.BASE_UI, YoloModelType.PRODUCER]:
        png_labels = {d["label"] for d in results[f"{model_type}_png"]}
        jpg_labels = {d["label"] for d in results[f"{model_type}_jpg"]}
        if png_labels != jpg_labels:
            diff = png_labels.symmetric_difference(jpg_labels)
            print(f"\n  ⚠ [{model_type}] PNG/JPG label mismatch: {diff}")
        else:
            print(f"\n  ✓ [{model_type}] PNG/JPG labels consistent ({len(png_labels)} labels)")

    return results


def adb_click(x: int, y: int, serial: str = "48c89aa"):
    import subprocess
    subprocess.check_call(["adb", "-s", serial, "shell", "input", "tap", str(x), str(y)])


def adb_swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 400, serial: str = "48c89aa"):
    import subprocess
    subprocess.check_call(["adb", "-s", serial, "shell", "input", "swipe",
                           str(x1), str(y1), str(x2), str(y2), str(duration_ms)])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    variant = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    capture_step(step, variant)
