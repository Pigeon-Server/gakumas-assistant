import argparse
from pathlib import Path
import sys

import cv2

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Game.Components.Modal import ModalParser
from src.entity.Yolo import Yolo_Results

MODAL_IMAGE_DIR = TESTS_DIR / "modal_test_images"
DEFAULT_OUTPUT_DIR = TESTS_DIR / "_artifacts" / "modal_debug"


def _normalize_image_names(image_names: list[str] | None) -> list[Path]:
    if not image_names:
        return sorted(MODAL_IMAGE_DIR.glob("*.PNG"))

    resolved_paths: list[Path] = []
    for image_name in image_names:
        candidate = Path(image_name)
        if not candidate.suffix:
            candidate = candidate.with_suffix(".PNG")
        if not candidate.is_absolute():
            candidate = MODAL_IMAGE_DIR / candidate.name
        resolved_paths.append(candidate)
    return resolved_paths


def render_modal_debug_images(image_names: list[str] | None = None, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_ui_model = YoloModelFromONNX(config.model_config["BASE_UI"])
    written_paths: list[Path] = []

    for image_path in _normalize_image_names(image_names):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Modal test image not found: {image_path}")

        results = base_ui_model(image, conf_threshold=0.7)
        parser = ModalParser(Yolo_Results(results, image), no_body=True)
        modal = parser.parse()
        if modal is None:
            raise RuntimeError(f"Failed to parse modal image: {image_path.name}")

        output_path = output_dir / f"{image_path.stem}_debug.png"
        if not cv2.imwrite(str(output_path), parser.draw_debug(modal)):
            raise RuntimeError(f"Failed to write debug image: {output_path}")
        written_paths.append(output_path)

    return written_paths


def main():
    parser = argparse.ArgumentParser(description="Render modal debug overlays into the repository.")
    parser.add_argument(
        "image_names",
        nargs="*",
        help="Modal image file names under tests/modal_test_images. Defaults to all PNG samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered debug images.",
    )
    args = parser.parse_args()

    written_paths = render_modal_debug_images(args.image_names, args.output_dir)
    print(f"Rendered {len(written_paths)} modal debug images to {args.output_dir}")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
