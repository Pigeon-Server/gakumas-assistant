from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
import timm


MODEL_MAP = {
    "s0": "hf-hub:timm/fastvit_mci0.apple_mclip2_dfndr2b",
    "s2": "hf-hub:timm/fastvit_mci2.apple_mclip2_dfndr2b",
}


class ImageTowerWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def export_model(variant: str, output_dir: Path, cache_dir: Path):
    model_name = MODEL_MAP[variant]
    model = timm.create_model(
        model_name,
        pretrained=True,
        cache_dir=cache_dir,
    )
    wrapper = ImageTowerWrapper(model).eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    output_path = output_dir / f"mobileclip2_{variant}_visual.onnx"
    with torch.no_grad():
        sample = wrapper(dummy_input)
    print(f"{variant}: sample output shape = {tuple(sample.shape)}")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["image_features"],
        dynamic_axes={"input": {0: "batch_size"}, "image_features": {0: "batch_size"}},
        do_constant_folding=True,
        opset_version=18,
    )
    # PyTorch may emit external tensor data for large models; fold it back into
    # a single ONNX file so size comparison and packaging are straightforward.
    model_proto = onnx.load(output_path)
    onnx.save(
        model_proto,
        output_path,
        save_as_external_data=False,
    )
    print(f"exported: {output_path}")
    print(f"size_bytes: {output_path.stat().st_size}")
    print(f"size_mib: {output_path.stat().st_size / 1024 / 1024:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Export MobileCLIP2 image towers to ONNX")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["s0", "s2"],
        choices=sorted(MODEL_MAP),
    )
    parser.add_argument(
        "--output-dir",
        default="model",
        help="Directory for exported ONNX files",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/hf-models",
        help="Directory for timm / HF cache",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        export_model(variant, output_dir, cache_dir)


if __name__ == "__main__":
    main()
