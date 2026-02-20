import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "models"
UPLOADS_ROOT = ROOT / "uploads"

if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

MODEL_PATH = Path(os.getenv("ONNX_MODEL_PATH", MODELS_ROOT / "best.onnx"))
PT_FALLBACK_PATH = Path(os.getenv("PT_MODEL_FALLBACK_PATH", MODELS_ROOT / "weights" / "best.pt"))
DEFAULT_CAMERA_REF = os.getenv("ONNX_CAMERA_REFERENCE", os.getenv("VIDEO_REFERENCE", "0"))

_model: YOLO | None = None


def _parse_camera_reference(value: str) -> int | str:
    value = str(value).strip()
    if value.isdigit():
        return int(value)
    return value


def _load_model() -> YOLO:
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


def get_active_model_path() -> str:
    return str(MODEL_PATH if MODEL_PATH.exists() else PT_FALLBACK_PATH)


class RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.confidence = float(os.getenv("ONNX_CONFIDENCE", "0.3"))
        self.imgsz = int(os.getenv("ONNX_IMGSZ", "640"))
        self.infer_every_n = int(os.getenv("ONNX_INFER_EVERY_N", "2"))
        self.jpeg_quality = int(os.getenv("ONNX_JPEG_QUALITY", "85"))

        self.predictions: list[dict[str, Any]] = []
        self.counts_text = "No objects detected."
        self.inference_ms = 0.0
        self.stream_fps = 0.0

        self.camera_ref = str(DEFAULT_CAMERA_REF)
        self.cameras: list[dict[str, str]] = [
            {"label": "Primary", "ref": self.camera_ref}
        ]


state = RuntimeState()


def _summarize_counts(predictions: list[dict[str, Any]]) -> str:
    if not predictions:
        return "No objects detected."
    counts: dict[str, int] = {}
    for prediction in predictions:
        label = str(prediction.get("label", "object"))
        counts[label] = counts.get(label, 0) + 1
    return "\n".join(f"{label}: {count}" for label, count in sorted(counts.items()))


def _run_detection(frame: Any, confidence: float, imgsz: int) -> list[dict[str, Any]]:
    model = _load_model()
    start = time.perf_counter()
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

    with state.lock:
        state.inference_ms = elapsed_ms
        state.predictions = predictions
        state.counts_text = _summarize_counts(predictions)

    return predictions


def _draw_overlay(frame: Any, predictions: list[dict[str, Any]]) -> Any:
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

    with state.lock:
        fps = state.stream_fps
        inference_ms = state.inference_ms

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


def generate_frames() -> Any:
    cap: cv2.VideoCapture | None = None
    current_ref: str | None = None
    last_ts = time.perf_counter()
    frame_index = 0

    while True:
        with state.lock:
            desired_ref = state.camera_ref

        if cap is None or desired_ref != current_ref:
            if cap is not None:
                cap.release()
            current_ref = desired_ref
            cap = cv2.VideoCapture(_parse_camera_reference(current_ref))

        if cap is None or not cap.isOpened():
            time.sleep(0.2)
            continue

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        now = time.perf_counter()
        delta = max(now - last_ts, 1e-6)
        last_ts = now

        with state.lock:
            state.stream_fps = 1.0 / delta
            confidence = state.confidence
            imgsz = state.imgsz
            infer_every_n = max(1, state.infer_every_n)
            predictions = list(state.predictions)
            jpeg_quality = state.jpeg_quality

        if frame_index % infer_every_n == 0:
            predictions = _run_detection(frame, confidence=confidence, imgsz=imgsz)

        rendered = _draw_overlay(frame.copy(), predictions)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(60, min(95, jpeg_quality)))]
        success, buffer = cv2.imencode(".jpg", rendered, encode_param)
        if not success:
            frame_index += 1
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        frame_index += 1


def get_cameras_snapshot() -> list[dict[str, str]]:
    with state.lock:
        return [dict(camera) for camera in state.cameras]


def add_camera(label: str, ref: str) -> list[dict[str, str]]:
    with state.lock:
        state.cameras.append({"label": label, "ref": ref})
        return [dict(camera) for camera in state.cameras]


def remove_camera(index: int) -> list[dict[str, str]]:
    with state.lock:
        if 0 <= index < len(state.cameras):
            state.cameras.pop(index)
        return [dict(camera) for camera in state.cameras]


def set_active_camera(ref: str) -> str:
    with state.lock:
        state.camera_ref = ref
        return state.camera_ref


def test_camera_reference(ref: str) -> bool:
    cap = cv2.VideoCapture(_parse_camera_reference(ref))
    if not cap.isOpened():
        cap.release()
        return False
    ok, _frame = cap.read()
    cap.release()
    return bool(ok)


def save_upload(file_storage: Any) -> str:
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file_storage.filename or "upload")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_name = f"{timestamp}_{filename}"
    dest = UPLOADS_ROOT / final_name
    file_storage.save(dest)
    return str(dest)
