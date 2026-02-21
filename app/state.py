"""Application runtime state."""
import os
import threading
from typing import Any


class RuntimeState:
    """Global runtime state for the application."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.confidence = float(os.getenv("ONNX_CONFIDENCE", "0.3"))
        self.imgsz = int(os.getenv("ONNX_IMGSZ", "640"))
        self.infer_every_n = int(os.getenv("ONNX_INFER_EVERY_N", "2"))
        self.jpeg_quality = int(os.getenv("ONNX_JPEG_QUALITY", "85"))
        self.inference_enabled = os.getenv("ONNX_INFERENCE_ENABLED", "false").strip().lower() == "true"

        self.cameras: list[dict[str, str]] = []
        self.camera_stats: dict[str, dict[str, Any]] = {}
        self.camera_workers: dict[str, Any] = {}  # Will store CameraWorker instances
        self.active_camera_id = ""
        self._next_camera_id = 1


# Global runtime state instance
state = RuntimeState()
