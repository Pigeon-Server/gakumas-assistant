#!/usr/bin/env python3
"""使用程序当前截图链路抓取一帧，并保存 YOLO 结果。"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.constants.yolo.model_type import YoloModelType
from src.main import AppProcessor
from src.utils.logger import logger


def _wait_for_frame(app: AppProcessor, timeout: float = 15.0):
    """等待首帧和首份推理结果就绪。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = getattr(app, "latest_frame", None)
        results = getattr(app, "latest_results", None)
        if frame is not None and getattr(frame, "size", 0) > 0 and results is not None:
            return frame, results
        time.sleep(0.2)
    raise TimeoutError("等待最新画面超时")


def _wait_for_fresh_results(app: AppProcessor, previous_results, timeout: float = 3.0):
    """等待模型切换后的新推理结果。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        results = getattr(app, "latest_results", None)
        if results is not None and results is not previous_results:
            return results
        time.sleep(0.15)
    raise TimeoutError("等待新推理结果超时")


def _serialize_results(results) -> list[dict]:
    """将 YOLO 结果序列化为 JSON 友好结构。"""
    payload = []
    for item in results:
        payload.append(
            {
                "label": getattr(item, "label", ""),
                "confidence": float(getattr(item, "confidence", 0.0)),
                "box": [
                    int(getattr(item, "x", 0)),
                    int(getattr(item, "y", 0)),
                    int(getattr(item, "w", 0)),
                    int(getattr(item, "h", 0)),
                ],
            }
        )
    return payload


def _draw_results(frame, results):
    """绘制 YOLO 检测框。"""
    canvas = frame.copy()
    for item in results:
        x = int(getattr(item, "x", 0))
        y = int(getattr(item, "y", 0))
        w = int(getattr(item, "w", 0))
        h = int(getattr(item, "h", 0))
        label = getattr(item, "label", "?")
        confidence = float(getattr(item, "confidence", 0.0))
        cv2.rectangle(canvas, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{label} {confidence:.2f}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return canvas


def main():
    parser = argparse.ArgumentParser(description="抓取当前真机帧并保存 YOLO 结果")
    parser.add_argument(
        "--model",
        choices=[YoloModelType.BASE_UI, YoloModelType.PRODUCER],
        default=YoloModelType.PRODUCER,
        help="抓图前切换到指定 YOLO 模型",
    )
    parser.add_argument(
        "--prefix",
        default="manual_capture",
        help="输出文件名前缀",
    )
    args = parser.parse_args()

    logger.info("初始化 AppProcessor，用程序当前截图链路抓取画面 ...")
    app = AppProcessor()
    if not app.is_resource_ready():
        app.ensure_resource_dependencies_initialized()
    if not app.yolo_engine.running:
        app.yolo_engine.start()
    app.yolo_engine.resume()

    _wait_for_frame(app)
    previous_results = getattr(app, "latest_results", None)
    if getattr(app.yolo_engine, "model_type", None) != args.model:
        logger.info(f"切换 YOLO 模型到 {args.model}")
        app.yolo_engine.load_model(args.model)
        results = _wait_for_fresh_results(app, previous_results)
    else:
        results = getattr(app, "latest_results", None)

    frame = getattr(results, "frame", None)
    if frame is None or getattr(frame, "size", 0) == 0:
        frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) == 0:
        raise RuntimeError("无法获取当前画面")

    out_dir = ROOT / "out" / "debug_captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    stem = f"{args.prefix}_{args.model.lower()}_{ts}"
    raw_path = out_dir / f"{stem}.png"
    annotated_path = out_dir / f"{stem}_annotated.jpg"
    meta_path = out_dir / f"{stem}_meta.json"

    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(annotated_path), _draw_results(frame, results))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "captured_at": ts,
                "detections": _serialize_results(results),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.success(f"原图已保存: {raw_path}")
    logger.success(f"标注图已保存: {annotated_path}")
    logger.success(f"元数据已保存: {meta_path}")


if __name__ == "__main__":
    main()
