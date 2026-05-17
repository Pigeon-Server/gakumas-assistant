from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.tasks.base_ui.goto_pages import goto_idol_card_list_page
from src.core.tasks.base_ui.learn_idol_card_clip import action__learn_idol_card_clip
from src.main import AppProcessor

ARTIFACT_DIR = TESTS_DIR / "_artifacts" / "idol_card_learning" / "live_task_runs"


def _wait_for_results(app: AppProcessor, timeout: float = 15.0) -> None:
    start = time.time()
    while time.time() - start <= timeout:
        results = app.latest_results
        if results is not None and getattr(results, "frame", None) is not None and results.frame.size > 0:
            return
        time.sleep(0.2)
    raise TimeoutError("Timed out waiting for inference results")


def _save_frame(path: Path, frame) -> None:
    if frame is None or frame.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def run_live_task() -> Path:
    run_dir = ARTIFACT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    app = AppProcessor()
    try:
        if not app.ensure_resource_dependencies_initialized():
            raise RuntimeError("Required resources are not ready")
        if not app.ensure_device_ready(restart_inference=True):
            raise RuntimeError(f"Device not ready: {app.get_device_status()}")
        if not app.start_inference_if_possible():
            raise RuntimeError("Failed to start inference")

        _wait_for_results(app)
        _save_frame(run_dir / "before_goto.png", app.latest_frame)

        goto_idol_card_list_page(app)
        _wait_for_results(app)
        _save_frame(run_dir / "idol_page_before_task.png", app.latest_frame)

        ok = action__learn_idol_card_clip(app)
        _save_frame(run_dir / "after_task.png", app.latest_frame)

        (run_dir / "result.txt").write_text(
            f"ok={ok}\n"
            f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            encoding="utf-8",
        )
        return run_dir
    finally:
        try:
            app.shutdown()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learn_idol_card_clip live on the connected device.")
    parser.parse_args()
    run_dir = run_live_task()
    print(f"live_run_dir={run_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
