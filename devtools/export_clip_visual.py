from __future__ import annotations

import argparse
from pathlib import Path

import clip
import torch


def main():
    parser = argparse.ArgumentParser(description="Export OpenAI CLIP ViT-B/32 image tower to ONNX")
    parser.add_argument(
        "--output",
        default="model/clip_visual.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--download-root",
        default=".cache/clip",
        help="Directory used by clip.load to cache downloaded checkpoints",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    download_root = Path(args.download_root).resolve()
    download_root.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    model, _ = clip.load("ViT-B/32", device=device, download_root=str(download_root))
    visual_model = model.visual.eval().float()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        visual_model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["image_features"],
        dynamic_axes={"input": {0: "batch_size"}, "image_features": {0: "batch_size"}},
        do_constant_folding=True,
    )
    print(f"exported: {output_path}")
    print(f"size_bytes: {output_path.stat().st_size}")
    print(f"size_mib: {output_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
