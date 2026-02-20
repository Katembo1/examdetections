import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "onnx_flask" / "models"
TARGET_ONNX = TARGET_DIR / "best.onnx"
SOURCE_PT_CANDIDATES = [
    TARGET_DIR / "weights" / "best.pt",
    ROOT / "best.pt",
]

if str(TARGET_DIR) not in sys.path:
    sys.path.insert(0, str(TARGET_DIR))


def _export_with_ultralytics(source_pt: Path) -> Path | None:
    try:
        model = YOLO(str(source_pt))
        exported = Path(
            model.export(
                format="onnx",
                imgsz=640,
                simplify=True,
                dynamic=True,
                opset=12,
            )
        )
        return exported if exported.exists() else None
    except Exception:
        return None


def _export_with_torch(source_pt: Path) -> Path:
    ckpt = torch.load(str(source_pt), map_location="cpu", weights_only=False)
    checkpoint_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if checkpoint_model is None:
        raise RuntimeError("Checkpoint does not contain a model for ONNX export")

    checkpoint_model = checkpoint_model.float().eval()
    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)

    torch.onnx.export(
        checkpoint_model,
        dummy,
        str(TARGET_ONNX),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        },
    )
    return TARGET_ONNX


def main() -> None:
    source_pt = next((path for path in SOURCE_PT_CANDIDATES if path.exists()), None)
    if source_pt is None:
        raise FileNotFoundError(
            "Missing source model. Checked: " + ", ".join(str(path) for path in SOURCE_PT_CANDIDATES)
        )

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    exported_path = _export_with_ultralytics(source_pt)
    if exported_path is None:
        exported_path = _export_with_torch(source_pt)

    if not exported_path.exists():
        raise RuntimeError("ONNX export failed: output file not found")

    if exported_path.resolve() != TARGET_ONNX.resolve():
        TARGET_ONNX.write_bytes(exported_path.read_bytes())

    print(f"ONNX model ready: {TARGET_ONNX}")


if __name__ == "__main__":
    main()
