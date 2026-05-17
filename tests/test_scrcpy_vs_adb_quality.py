"""
scrcpy vs ADB 截图质量对比诊断

同时截图用 scrcpy(H264) 和 ADB screencap，对比：
  - 分辨率
  - PSNR / SSIM
  - YOLO 检测结果差异（尤其是 MODAL_HEADER）
  - OCR 结果差异

用法:
  python -m tests.test_scrcpy_vs_adb_quality
"""
import os
import sys
import time
import json
import traceback

import cv2
import numpy as np
import adbutils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.inference.ocr_engine import OCRService
from src.entity.Yolo import Yolo_Results
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels

OUT_DIR = os.path.join("logs", "debug", "test_captures", "quality_compare")
os.makedirs(OUT_DIR, exist_ok=True)

model = YoloModelFromONNX(config.model_config["BASE_UI"])
ocr_service = OCRService()

CAPTURE_ROUNDS = 5


def adb_screenshot(dev) -> np.ndarray:
    """ADB screencap: 无压缩、原生像素"""
    pil_img = dev.screenshot()
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def scrcpy_screenshot(adapter, wait: float = 0.5) -> np.ndarray | None:
    """从 scrcpy H264 流取最新帧"""
    return adapter.capture(wait_timeout=wait)


def compute_psnr_ssim(ref: np.ndarray, target: np.ndarray):
    """计算 PSNR 和 SSIM（灰度）"""
    if ref.shape != target.shape:
        target = cv2.resize(target, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # PSNR
    mse = np.mean((gray_ref.astype(np.float64) - gray_tgt.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-10)) if mse > 0 else float("inf")

    # SSIM (simplified)
    C1, C2 = 6.5025, 58.5225
    mu1 = cv2.GaussianBlur(gray_ref.astype(np.float64), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray_tgt.astype(np.float64), (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(gray_ref.astype(np.float64) ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray_tgt.astype(np.float64) ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(
        gray_ref.astype(np.float64) * gray_tgt.astype(np.float64), (11, 11), 1.5
    ) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim = float(ssim_map.mean())
    return psnr, ssim


def compute_laplacian_var(img: np.ndarray) -> float:
    """拉普拉斯方差 — 边缘清晰度指标"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect(frame: np.ndarray) -> Yolo_Results:
    raw = model(frame, conf_threshold=0.5, iou_threshold=0.5)
    return Yolo_Results(raw, frame)


def compare_yolo(adb_res: Yolo_Results, scrcpy_res: Yolo_Results) -> dict:
    """对比 YOLO 检测结果"""
    def summarize(yolo_res: Yolo_Results):
        labels = {}
        for box in yolo_res.boxes:
            lbl = box.label
            if lbl not in labels:
                labels[lbl] = {"count": 0, "confs": []}
            labels[lbl]["count"] += 1
            labels[lbl]["confs"].append(round(box.confidence, 4) if hasattr(box, "confidence") else None)
        return labels

    adb_summary = summarize(adb_res)
    scrcpy_summary = summarize(scrcpy_res)

    all_labels = sorted(set(list(adb_summary.keys()) + list(scrcpy_summary.keys())))
    diff = {}
    for lbl in all_labels:
        a = adb_summary.get(lbl, {"count": 0, "confs": []})
        s = scrcpy_summary.get(lbl, {"count": 0, "confs": []})
        diff[lbl] = {
            "adb_count": a["count"],
            "scrcpy_count": s["count"],
            "adb_confs": a["confs"],
            "scrcpy_confs": s["confs"],
            "match": a["count"] == s["count"],
        }
    return diff


def main():
    print("=" * 60)
    print("  scrcpy vs ADB 截图质量对比诊断")
    print("=" * 60)

    adb = adbutils.AdbClient()
    devices = adb.device_list()
    if not devices:
        print("ERROR: 没有检测到 ADB 设备")
        return
    dev = devices[0]
    print(f"设备: {dev.serial}")

    # 初始化 scrcpy adapter
    from src.core.device.Android.adapters.scrcpy_adapter import ScrcpyAdapter
    adapter = ScrcpyAdapter(dev, max_width=0, bitrate=20000000, max_fps=30)
    if not adapter.start():
        print("ERROR: scrcpy adapter 启动失败，仅进行 ADB 截图测试")
        adapter = None
    else:
        print(f"scrcpy adapter 已启动 (bitrate={adapter._bitrate}, max_width={adapter._max_width})")
        # 等一下让 decoder 产生帧
        time.sleep(1.0)

    results = []

    for i in range(CAPTURE_ROUNDS):
        print(f"\n--- Round {i + 1}/{CAPTURE_ROUNDS} ---")
        time.sleep(0.3)

        # ADB screenshot
        t0 = time.time()
        adb_frame = adb_screenshot(dev)
        adb_time = time.time() - t0
        print(f"  ADB: {adb_frame.shape[1]}x{adb_frame.shape[0]}, 耗时 {adb_time:.3f}s")

        # scrcpy screenshot
        scrcpy_frame = None
        scrcpy_time = None
        if adapter:
            t0 = time.time()
            scrcpy_frame = scrcpy_screenshot(adapter)
            scrcpy_time = time.time() - t0
            if scrcpy_frame is not None:
                print(f"  scrcpy: {scrcpy_frame.shape[1]}x{scrcpy_frame.shape[0]}, 耗时 {scrcpy_time:.3f}s")
            else:
                print("  scrcpy: 帧获取失败（None）")

        # 保存原图
        cv2.imwrite(os.path.join(OUT_DIR, f"round{i}_adb.png"), adb_frame)
        if scrcpy_frame is not None:
            cv2.imwrite(os.path.join(OUT_DIR, f"round{i}_scrcpy.png"), scrcpy_frame)

        # 质量指标
        adb_lap = compute_laplacian_var(adb_frame)
        scrcpy_lap = compute_laplacian_var(scrcpy_frame) if scrcpy_frame is not None else None
        print(f"  Laplacian variance: ADB={adb_lap:.1f}, scrcpy={scrcpy_lap:.1f}" if scrcpy_lap else f"  Laplacian variance: ADB={adb_lap:.1f}")

        psnr, ssim = (None, None)
        if scrcpy_frame is not None:
            psnr, ssim = compute_psnr_ssim(adb_frame, scrcpy_frame)
            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

        # YOLO 检测
        adb_yolo = detect(adb_frame)
        scrcpy_yolo = detect(scrcpy_frame) if scrcpy_frame is not None else None

        # MODAL_HEADER 特别检查
        adb_mh = adb_yolo.filter_by_label(BaseUILabels.MODAL_HEADER)
        scrcpy_mh = scrcpy_yolo.filter_by_label(BaseUILabels.MODAL_HEADER) if scrcpy_yolo else None
        print(f"  MODAL_HEADER: ADB={len(adb_mh) if adb_mh else 0}, scrcpy={len(scrcpy_mh) if scrcpy_mh else 0}")

        # YOLO 对比
        yolo_diff = compare_yolo(adb_yolo, scrcpy_yolo) if scrcpy_yolo else None

        round_result = {
            "round": i,
            "adb": {
                "resolution": f"{adb_frame.shape[1]}x{adb_frame.shape[0]}",
                "capture_time": round(adb_time, 3),
                "laplacian_var": round(adb_lap, 1),
                "yolo_total": len(adb_yolo.boxes) if adb_yolo else 0,
                "modal_header_count": len(adb_mh) if adb_mh else 0,
            },
            "scrcpy": None,
            "psnr": psnr,
            "ssim": ssim,
            "yolo_diff": yolo_diff,
        }
        if scrcpy_frame is not None:
            round_result["scrcpy"] = {
                "resolution": f"{scrcpy_frame.shape[1]}x{scrcpy_frame.shape[0]}",
                "capture_time": round(scrcpy_time, 3) if scrcpy_time else None,
                "laplacian_var": round(scrcpy_lap, 1) if scrcpy_lap else None,
                "yolo_total": len(scrcpy_yolo.boxes) if scrcpy_yolo else 0,
                "modal_header_count": len(scrcpy_mh) if scrcpy_mh else 0,
            }

        results.append(round_result)

    # 汇总
    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)

    if results:
        adb_laps = [r["adb"]["laplacian_var"] for r in results]
        scrcpy_laps = [r["scrcpy"]["laplacian_var"] for r in results if r["scrcpy"] and r["scrcpy"]["laplacian_var"]]
        psnrs = [r["psnr"] for r in results if r["psnr"] is not None]
        ssims = [r["ssim"] for r in results if r["ssim"] is not None]
        adb_times = [r["adb"]["capture_time"] for r in results]
        scrcpy_times = [r["scrcpy"]["capture_time"] for r in results if r["scrcpy"] and r["scrcpy"]["capture_time"]]

        print(f"  ADB 分辨率: {results[0]['adb']['resolution']}")
        if results[0]["scrcpy"]:
            print(f"  scrcpy 分辨率: {results[0]['scrcpy']['resolution']}")
        print(f"  ADB 截图耗时: {np.mean(adb_times):.3f}s (avg)")
        if scrcpy_times:
            print(f"  scrcpy 截图耗时: {np.mean(scrcpy_times):.3f}s (avg)")
        print(f"  ADB Laplacian var: {np.mean(adb_laps):.1f} (avg), range [{min(adb_laps):.1f}, {max(adb_laps):.1f}]")
        if scrcpy_laps:
            print(f"  scrcpy Laplacian var: {np.mean(scrcpy_laps):.1f} (avg), range [{min(scrcpy_laps):.1f}, {max(scrcpy_laps):.1f}]")
        if psnrs:
            print(f"  PSNR: {np.mean(psnrs):.2f} dB (avg), range [{min(psnrs):.2f}, {max(psnrs):.2f}]")
        if ssims:
            print(f"  SSIM: {np.mean(ssims):.4f} (avg), range [{min(ssims):.4f}, {max(ssims):.4f}]")

        # MODAL_HEADER 统计
        adb_mh_total = sum(r["adb"]["modal_header_count"] for r in results)
        scrcpy_mh_total = sum(r["scrcpy"]["modal_header_count"] for r in results if r["scrcpy"])
        print(f"  MODAL_HEADER 总检出: ADB={adb_mh_total}/{len(results)}, scrcpy={scrcpy_mh_total}/{len(results)}")

        # YOLO label 差异统计
        if any(r["yolo_diff"] for r in results):
            mismatch_labels = set()
            for r in results:
                if r["yolo_diff"]:
                    for lbl, info in r["yolo_diff"].items():
                        if not info["match"]:
                            mismatch_labels.add(lbl)
            if mismatch_labels:
                print(f"  YOLO检测不一致的label: {sorted(mismatch_labels)}")
            else:
                print(f"  YOLO检测结果: ADB 与 scrcpy 完全一致")

    # 保存结果
    report_path = os.path.join(OUT_DIR, "quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告: {report_path}")
    print(f"截图保存: {OUT_DIR}")

    if adapter:
        adapter.stop()


if __name__ == "__main__":
    main()
