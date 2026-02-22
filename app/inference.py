"""Model loading and inference logic (pure ONNX Runtime — no torch/ultralytics)."""
import ast
import threading
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .config import MODEL_PATH

_session: ort.InferenceSession | None = None
_class_names: dict[int, str] = {}
_input_name: str = ""
_model_lock = threading.Lock()


def _load_model() -> ort.InferenceSession:
    global _session, _class_names, _input_name
    if _session is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        meta = _session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            _class_names = ast.literal_eval(meta["names"])
        else:
            _class_names = {0: "cheating", 1: "not_cheating"}
    return _session


def _letterbox(img: np.ndarray, new_shape: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize + pad to square while keeping aspect ratio."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2
    img = cv2.copyMakeBorder(
        img, pad_h, new_shape - new_h - pad_h,
        pad_w, new_shape - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return img, r, (pad_w, pad_h)


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def run_detection(frame: Any, confidence: float, imgsz: int) -> tuple[list[dict[str, Any]], float]:
    """Run object detection on a frame. Returns (predictions, elapsed_ms)."""
    session = _load_model()
    orig_h, orig_w = frame.shape[:2]
    img, ratio, (pad_w, pad_h) = _letterbox(frame, imgsz)
    inp = img[..., ::-1].astype(np.float32) / 255.0   # BGR→RGB, normalise
    inp = inp.transpose(2, 0, 1)[np.newaxis]            # HWC → NCHW

    start = time.perf_counter()
    with _model_lock:
        outputs = session.run(None, {_input_name: inp})
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # YOLOv8 ONNX output shape: [1, 4+nc, 8400]
    preds = outputs[0][0].T                      # (8400, 4+nc)
    boxes_raw = preds[:, :4]
    cls_scores = preds[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    cls_confs = cls_scores.max(axis=1)

    mask = cls_confs >= confidence
    boxes_raw = boxes_raw[mask]
    cls_ids = cls_ids[mask]
    cls_confs = cls_confs[mask]

    boxes_xyxy = _xywh2xyxy(boxes_raw)
    keep = _nms(boxes_xyxy, cls_confs)

    predictions: list[dict[str, Any]] = []
    for i in keep:
        x1 = int((boxes_xyxy[i, 0] - pad_w) / ratio)
        y1 = int((boxes_xyxy[i, 1] - pad_h) / ratio)
        x2 = int((boxes_xyxy[i, 2] - pad_w) / ratio)
        y2 = int((boxes_xyxy[i, 3] - pad_h) / ratio)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        cls_id = int(cls_ids[i])
        label = _class_names.get(cls_id, str(cls_id))
        predictions.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": float(cls_confs[i]),
            "label": label,
        })

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
