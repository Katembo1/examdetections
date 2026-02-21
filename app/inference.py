"""Model loading and inference logic."""
import threading
import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .config import MODEL_PATH, PT_FALLBACK_PATH

_model: YOLO | None = None
_model_lock = threading.Lock()


def _load_model() -> YOLO:
    """Load the YOLO model (ONNX or PT fallback)."""
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            _model = YOLO(str(MODEL_PATH))
        elif PT_FALLBACK_PATH.exists():
            _model = YOLO(str(PT_FALLBACK_PATH))
        else:
            raise FileNotFoundError(
                f"Model not found. Expected ONNX: {MODEL_PATH} or PT fallback: {PT_FALLBACK_PATH}"
            )
    return _model


def run_detection(frame: Any, confidence: float, imgsz: int) -> tuple[list[dict[str, Any]], float]:
    """
    Run object detection on a frame.
    
    Returns:
        Tuple of (predictions, elapsed_ms)
    """
    model = _load_model()
    start = time.perf_counter()
    with _model_lock:
        result = model(frame, conf=confidence, imgsz=imgsz, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    predictions: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []

        for idx, coords in enumerate(xyxy):
            x1, y1, x2, y2 = [int(value) for value in coords]
            conf_value = float(confs[idx]) if len(confs) > idx else 0.0
            cls_id = int(classes[idx]) if len(classes) > idx else -1
            label = model.names.get(cls_id, str(cls_id)) if cls_id >= 0 else "object"
            predictions.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf_value,
                    "label": label,
                }
            )

    return predictions, elapsed_ms


def draw_overlay(frame: Any, predictions: list[dict[str, Any]], fps: float, inference_ms: float) -> Any:
    """Draw bounding boxes and stats overlay on frame."""
    color_map = {
        "cheating": (40, 40, 255),
        "not_cheating": (60, 220, 90),
    }

    for prediction in predictions:
        x1, y1, x2, y2 = prediction["x1"], prediction["y1"], prediction["x2"], prediction["y2"]
        confidence = float(prediction["confidence"])
        label = str(prediction["label"])
        text = f"{label} {confidence:.2f}"

        color = color_map.get(label.lower(), (0, 200, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            text,
            (max(x1, 0), max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
    cv2.putText(
        frame,
        f"Inference: {inference_ms:.1f} ms",
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (50, 255, 50),
        2,
    )
    return frame


def build_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    """Build a dictionary of object counts from predictions."""
    counts: dict[str, int] = {}
    for prediction in predictions:
        label = str(prediction.get("label", "object"))
        counts[label] = counts.get(label, 0) + 1
    return counts
