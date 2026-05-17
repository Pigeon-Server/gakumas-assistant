from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Game.Components.SupportCard import SupportCardLevelStars
from src.utils.game_tools import extract_support_card_level_stars
from src.utils.opencv_tools import compute_ssim_score

ARTIFACT_DIR = TESTS_DIR / "_artifacts" / "card_level_stars"


@dataclass
class AdbScanPageResult:
    index: int
    screenshot_path: Path
    debug_path: Path
    cards: SupportCardLevelStars
    similarity_to_previous: float | None = None


def extract_card_level_stars(
        image: np.ndarray,
        model: YoloModelFromONNX | None = None,
        conf_threshold: float = 0.7,
) -> SupportCardLevelStars:
    return extract_support_card_level_stars(image, model=model, conf_threshold=conf_threshold)


def render_debug_image(
        image: np.ndarray,
        cards: SupportCardLevelStars,
        output_path: Path | None = None,
) -> np.ndarray:
    annotated = cards.draw_debug(image)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
    return annotated


def _adb_base_command(serial: str | None = None) -> list[str]:
    command = ["adb"]
    if serial:
        command.extend(["-s", serial])
    return command


def _run_adb_command(args: list[str], serial: str | None = None, capture_output: bool = True) -> subprocess.CompletedProcess:
    command = _adb_base_command(serial) + args
    result = subprocess.run(command, capture_output=capture_output)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ADB command failed: {' '.join(command)}\n{stderr}")
    return result


def capture_adb_image(output_path: Path | None = None, serial: str | None = None) -> np.ndarray:
    result = _run_adb_command(["exec-out", "screencap", "-p"], serial=serial)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(result.stdout)
    image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode ADB screenshot")
    return image


def adb_swipe_page(
        width: int,
        height: int,
        serial: str | None = None,
        start_ratio: float = 0.78,
        end_ratio: float = 0.28,
        duration_ms: int = 360,
) -> None:
    center_x = width // 2
    start_y = int(height * start_ratio)
    end_y = int(height * end_ratio)
    _run_adb_command(
        ["shell", "input", "swipe", str(center_x), str(start_y), str(center_x), str(end_y), str(duration_ms)],
        serial=serial,
    )


def adb_reset_to_top(
        width: int,
        height: int,
        serial: str | None = None,
        swipes: int = 4,
        pause_seconds: float = 0.6,
) -> None:
    for _ in range(swipes):
        adb_swipe_page(
            width,
            height,
            serial=serial,
            start_ratio=0.32,
            end_ratio=0.84,
            duration_ms=420,
        )
        time.sleep(pause_seconds)


def _print_card_summary(cards: SupportCardLevelStars, prefix: str = "") -> None:
    print(f"{prefix}检测到 {len(cards)} 张支援卡")
    for item in cards:
        level = "?" if item.level is None else str(item.level)
        stars = "?" if item.stars is None else str(item.stars)
        limit_break = " [上限解放可能]" if item.limit_break else ""
        print(
            f"{prefix}  卡片#{item.index:02d} "
            f"Lv={level:>3} ★={stars} 置信度={item.confidence:.3f}{limit_break}"
        )


def _get_scroll_area(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    top = int(height * 0.12)
    bottom = int(height * 0.92)
    left = int(width * 0.04)
    right = int(width * 0.96)
    return frame[top:bottom, left:right]


def _page_signature(cards: SupportCardLevelStars) -> tuple[tuple[int | None, int | None, bool, int, int], ...]:
    return tuple(
        (
            item.level,
            item.stars,
            item.limit_break,
            int(item.box.y),
            int(item.box.h),
        )
        for item in cards
    )


def run_adb_full_scan(
        model: YoloModelFromONNX,
        serial: str | None = None,
        conf_threshold: float = 0.7,
        max_pages: int = 12,
        pause_seconds: float = 1.0,
        stop_ssim: float = 0.999,
        reset_to_top_swipes: int = 0,
) -> tuple[list[AdbScanPageResult], Path]:
    output_dir = ARTIFACT_DIR / "adb_full_scan"
    output_dir.mkdir(parents=True, exist_ok=True)

    page_results: list[AdbScanPageResult] = []
    previous_frame: np.ndarray | None = None
    previous_signature: tuple[tuple[int | None, int | None, bool, int, int], ...] | None = None

    if reset_to_top_swipes > 0:
        seed_image = capture_adb_image(serial=serial)
        adb_reset_to_top(
            seed_image.shape[1],
            seed_image.shape[0],
            serial=serial,
            swipes=reset_to_top_swipes,
            pause_seconds=min(pause_seconds, 0.6),
        )
        time.sleep(pause_seconds)

    for page_index in range(max_pages):
        screenshot_path = output_dir / f"page_{page_index:02d}.png"
        debug_path = output_dir / f"page_{page_index:02d}_debug.png"
        image = capture_adb_image(screenshot_path, serial=serial)
        cards = extract_card_level_stars(image, model=model, conf_threshold=conf_threshold)
        render_debug_image(image, cards, debug_path)

        similarity = None
        signature = _page_signature(cards)
        if previous_frame is not None:
            similarity = compute_ssim_score(_get_scroll_area(previous_frame), _get_scroll_area(image))
            if similarity >= stop_ssim and signature == previous_signature:
                print(
                    f"\n第 {page_index + 1} 页与上一页高度相似 (SSIM={similarity:.4f})，"
                    "认为已滑到底部，停止继续采样。"
                )
                break

        page_result = AdbScanPageResult(page_index, screenshot_path, debug_path, cards, similarity)
        page_results.append(page_result)

        print(f"\n第 {page_index + 1} 页")
        if similarity is not None:
            print(f"与上一页滚动区域 SSIM: {similarity:.4f}")
        _print_card_summary(cards, prefix="  ")
        print(f"  调试图像: {debug_path}")

        previous_frame = image
        previous_signature = signature

        if page_index >= max_pages - 1:
            break
        adb_swipe_page(image.shape[1], image.shape[0], serial=serial)
        time.sleep(pause_seconds)

    return page_results, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="支援卡等级和星级提取")
    parser.add_argument("--image", type=str, default=None, help="输入图像路径")
    parser.add_argument("--adb", action="store_true", default=False, help="从 ADB 截图检测当前页")
    parser.add_argument("--adb-full", action="store_true", default=False, help="从当前页开始 ADB 滑动全测")
    parser.add_argument("--serial", type=str, default=None, help="ADB 设备 serial，可选")
    parser.add_argument("--conf-threshold", type=float, default=0.7, help="BASE_UI 模型阈值")
    parser.add_argument("--max-pages", type=int, default=12, help="ADB 全测最大页数")
    parser.add_argument("--pause", type=float, default=1.0, help="ADB 每次滑动后的等待秒数")
    parser.add_argument("--stop-ssim", type=float, default=0.999, help="判定滑动到底部的相似度阈值")
    parser.add_argument("--reset-to-top", type=int, default=0, help="全测前先向上翻回顶部的下滑次数")
    args = parser.parse_args()

    model = YoloModelFromONNX(config.model_config["BASE_UI"])

    if args.adb_full:
        print("正在执行 ADB 滑动全测...")
        page_results, output_dir = run_adb_full_scan(
            model=model,
            serial=args.serial,
            conf_threshold=args.conf_threshold,
            max_pages=args.max_pages,
            pause_seconds=args.pause,
            stop_ssim=args.stop_ssim,
            reset_to_top_swipes=args.reset_to_top,
        )
        print(f"\n全测完成，共采样 {len(page_results)} 页，输出目录: {output_dir}")
        return

    if args.adb or args.image is None:
        print("正在从 ADB 截图...")
        screenshot_path = ARTIFACT_DIR / "adb_capture.png"
        image = capture_adb_image(screenshot_path, serial=args.serial)
        print(f"截图尺寸: {image.shape[1]}x{image.shape[0]}")
    else:
        image = cv2.imread(args.image)
        if image is None:
            raise RuntimeError(f"无法读取图像: {args.image}")

    cards = extract_card_level_stars(image, model=model, conf_threshold=args.conf_threshold)
    print()
    _print_card_summary(cards)

    debug_path = ARTIFACT_DIR / "debug_output.png"
    render_debug_image(image, cards, debug_path)
    print(f"\n调试图像已保存: {debug_path}")


if __name__ == "__main__":
    main()
