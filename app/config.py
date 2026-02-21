"""Configuration constants and paths."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "models"
UPLOADS_ROOT = ROOT / "uploads"

if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

MODEL_PATH = Path(os.getenv("ONNX_MODEL_PATH", MODELS_ROOT / "best.onnx"))
PT_FALLBACK_PATH = Path(os.getenv("PT_MODEL_FALLBACK_PATH", MODELS_ROOT / "weights" / "best.pt"))
DEFAULT_CAMERA_REF = os.getenv("ONNX_CAMERA_REFERENCE", os.getenv("VIDEO_REFERENCE", "0"))
MAX_CAMERAS = 5


def get_active_model_path() -> str:
    """Get the path to the active model (ONNX or PT fallback)."""
    return str(MODEL_PATH if MODEL_PATH.exists() else PT_FALLBACK_PATH)
