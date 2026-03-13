"""Camera management and worker threads."""
import base64
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DEFAULT_CAMERA_REF, MAX_CAMERAS, UPLOADS_ROOT
from .db import db
from .inference import build_counts, draw_overlay, run_detection
from .models import CameraRecord
from .state import state
from .utils import (
    enumerate_available_cameras,
    format_counts,
    get_placeholder_frame,
    is_hardware_camera,
    is_network_stream_reference,
    open_video_source,
    parse_camera_reference,
    try_open_camera_with_backends,
)


INCIDENT_WINDOW_SEC = 120.0
INCIDENT_CLIP_SEC = 10.0
INCIDENT_SAMPLE_INTERVAL_SEC = 1.0
INCIDENT_SAVE_COOLDOWN_SEC = 120.0
INCIDENT_BASE64_PREVIEW_CHARS = 320
INCIDENT_CLIP_FPS = 6.0
INCIDENTS_DIR = UPLOADS_ROOT / "incidents"
INCIDENTS_INDEX = INCIDENTS_DIR / "incidents.jsonl"
_INCIDENT_IO_LOCK = threading.Lock()


def _prediction_is_cheating(prediction: dict[str, Any]) -> bool:
    """Return True if prediction label indicates cheating-like behavior."""
    label = str(prediction.get("label", "")).strip().lower()
    if not label:
        return False
    if "not_cheat" in label or "not cheating" in label:
        return False
    return "cheat" in label


def _has_cheating_detection(predictions: list[dict[str, Any]]) -> bool:
    """Check whether the current frame contains a cheating prediction."""
    return any(_prediction_is_cheating(pred) for pred in predictions)


def _window_counts(history: list[tuple[float, dict[str, int]]], now_ts: float) -> dict[str, int]:
    """Aggregate detection counts over the last INCIDENT_WINDOW_SEC window."""
    out: dict[str, int] = {}
    cutoff = now_ts - INCIDENT_WINDOW_SEC
    for ts, counts in history:
        if ts < cutoff:
            continue
        for label, value in counts.items():
            out[label] = out.get(label, 0) + int(value)
    return out


def _window_has_cheating(counts: dict[str, int]) -> bool:
    """Return True if cheating appears in the aggregated window counts."""
    for label, value in counts.items():
        key = str(label).lower()
        if value > 0 and "cheat" in key and "not" not in key:
            return True
    return False


def _append_incident_frame(stats: dict[str, Any], now_ts: float, frame_bytes: bytes) -> None:
    """Keep a low-rate rolling frame buffer for incident clip extraction."""
    last_sample = float(stats.get("incident_last_sample_ts", 0.0))
    if now_ts - last_sample < INCIDENT_SAMPLE_INTERVAL_SEC:
        return

    buffer = list(stats.get("incident_buffer", []))
    buffer.append((now_ts, frame_bytes))
    cutoff = now_ts - INCIDENT_WINDOW_SEC
    buffer = [item for item in buffer if item[0] >= cutoff]

    stats["incident_buffer"] = buffer
    stats["incident_last_sample_ts"] = now_ts


def _write_incident_clip(camera_id: str, samples: list[tuple[float, bytes]]) -> tuple[Path | None, str]:
    """Write a short MP4 clip and return (path, base64_preview)."""
    if not samples:
        return None, ""

    first_arr = np.frombuffer(samples[0][1], dtype="uint8")
    first = cv2.imdecode(first_arr, cv2.IMREAD_COLOR)
    if first is None:
        return None, ""

    h, w = first.shape[:2]
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    clip_path = INCIDENTS_DIR / f"camera_{camera_id}_{ts}.mp4"

    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        INCIDENT_CLIP_FPS,
        (w, h),
    )
    if not writer.isOpened():
        return None, ""

    try:
        for _, frame_bytes in samples:
            arr = np.frombuffer(frame_bytes, dtype="uint8")
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
    finally:
        writer.release()

    if not clip_path.exists():
        return None, ""

    clip_bytes = clip_path.read_bytes()
    b64 = base64.b64encode(clip_bytes).decode("ascii")
    return clip_path, b64[:INCIDENT_BASE64_PREVIEW_CHARS]


def _save_incident_reference(
    camera_id: str,
    camera_label: str,
    counts_window: dict[str, int],
    samples: list[tuple[float, bytes]],
) -> dict[str, Any]:
    """Persist cheating incident metadata and short base64 video reference."""
    clip_path, preview_b64 = _write_incident_clip(camera_id, samples)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "camera_label": camera_label,
        "window_seconds": int(INCIDENT_WINDOW_SEC),
        "counts_window": counts_window,
        "clip_path": str(clip_path) if clip_path is not None else None,
        "clip_b64_short": preview_b64,
    }

    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    with _INCIDENT_IO_LOCK:
        with INCIDENTS_INDEX.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return payload


def _get_camera_by_id_nolock(camera_id: str) -> dict[str, str] | None:
    """Get camera by ID without locking (caller must hold lock)."""
    for camera in state.cameras:
        if camera["id"] == camera_id:
            return camera
    return None


def _camera_dict(record: CameraRecord) -> dict[str, str]:
    """Convert CameraRecord to dictionary."""
    return {
        "id": str(record.id),
        "label": record.label,
        "ref": record.ref,
    }


def _init_camera_stats(camera_id: str, label: str, ref: str) -> None:
    """Initialize camera stats dictionary."""
    state.camera_stats[camera_id] = {
        "id": camera_id,
        "label": label,
        "ref": ref,
        "fps": 0.0,
        "inference_ms": 0.0,
        "counts": {},
        "counts_text": "No objects detected.",
        "counts_history": [],
        "incident_buffer": [],
        "incident_last_sample_ts": 0.0,
        "last_incident_save_ts": 0.0,
        "last_incident_ref": "",
        "last_frame": None,
        "running": False,
        "error": None,
    }


class CameraWorker(threading.Thread):
    """Worker thread for camera capture and inference."""

    def __init__(self, camera_id: str) -> None:
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()

    def run(self) -> None:
        """Main worker loop."""
        cap: cv2.VideoCapture | None = None
        current_ref: str | None = None
        last_ts = time.perf_counter()
        frame_index = 0
        predictions: list[dict[str, Any]] = []
        last_inference_ms = 0.0
        last_counts: dict[str, int] = {}
        failed_attempts = 0
        max_failed_attempts = 10

        while not self._stop_event.is_set():
            with state.lock:
                camera = _get_camera_by_id_nolock(self.camera_id)
                if camera is None:
                    break
                desired_ref = camera["ref"]
                confidence = state.confidence
                imgsz = state.imgsz
                infer_every_n = max(1, state.infer_every_n)
                jpeg_quality = state.jpeg_quality
                inference_enabled = state.inference_enabled

            if cap is None or desired_ref != current_ref:
                if cap is not None:
                    cap.release()
                current_ref = desired_ref
                parsed_ref = parse_camera_reference(current_ref)
                
                # Try opening camera with appropriate method
                if is_hardware_camera(current_ref):
                    # Hardware camera - try multiple backends
                    print(f"[Camera {self.camera_id}] Attempting to open hardware camera index {parsed_ref}")
                    cap = try_open_camera_with_backends(parsed_ref)
                    if cap is None:
                        # Show available cameras for debugging
                        available = enumerate_available_cameras(5)
                        if available:
                            print(f"[Camera {self.camera_id}] Available camera indices: {available}")
                        else:
                            print(f"[Camera {self.camera_id}] No hardware cameras detected")
                else:
                    # Video file or stream
                    print(f"[Camera {self.camera_id}] Attempting to open video source: {current_ref}")
                    cap = open_video_source(parsed_ref)
                
                if cap is None or not cap.isOpened():
                    failed_attempts += 1
                    
                    if is_hardware_camera(current_ref):
                        available = enumerate_available_cameras(5)
                        if available:
                            error_msg = f"Camera index {parsed_ref} not found. Available cameras: {available}"
                        else:
                            error_msg = f"No hardware cameras detected. Use video file/stream or check camera permissions."
                    else:
                        error_msg = f"Failed to open '{current_ref}'. Check file path or stream URL."
                    
                    print(f"[Camera {self.camera_id}] {error_msg} (attempt {failed_attempts}/{max_failed_attempts})")
                    
                    with state.lock:
                        stats = state.camera_stats.get(self.camera_id)
                        if stats is not None:
                            stats["error"] = error_msg
                    
                    if failed_attempts >= max_failed_attempts:
                        final_error = error_msg + " — Max attempts reached."
                        print(f"[Camera {self.camera_id}] {final_error}")
                        
                        with state.lock:
                            stats = state.camera_stats.get(self.camera_id)
                            if stats is not None:
                                stats["error"] = final_error
                                stats["running"] = False
                        break
                    
                    cap.release()
                    cap = None
                    time.sleep(min(2.0 * failed_attempts, 10.0))  # Exponential backoff
                    continue
                else:
                    failed_attempts = 0  # Reset on success
                    with state.lock:
                        stats = state.camera_stats.get(self.camera_id)
                        if stats is not None:
                            stats["error"] = None

            if cap is None or not cap.isOpened():
                time.sleep(0.2)
                continue

            ok, frame = cap.read()
            if not ok:
                # Loop local files, but reconnect live network streams.
                if current_ref and is_network_stream_reference(current_ref):
                    failed_attempts += 1
                    error_msg = f"Stream read timed out for '{current_ref}'. Reconnecting..."
                    print(f"[Camera {self.camera_id}] {error_msg} (attempt {failed_attempts}/{max_failed_attempts})")
                    with state.lock:
                        stats = state.camera_stats.get(self.camera_id)
                        if stats is not None:
                            stats["error"] = error_msg
                    cap.release()
                    cap = None
                    time.sleep(min(1.5 * failed_attempts, 5.0))
                    continue

                if isinstance(parse_camera_reference(current_ref), str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue

            failed_attempts = 0
            with state.lock:
                stats = state.camera_stats.get(self.camera_id)
                if stats is not None and stats.get("error"):
                    stats["error"] = None

            now = time.perf_counter()
            delta = max(now - last_ts, 1e-6)
            last_ts = now
            fps = 1.0 / delta

            if not inference_enabled:
                predictions = []
                last_inference_ms = 0.0
                last_counts = {}
            elif frame_index % infer_every_n == 0:
                predictions, last_inference_ms = run_detection(frame, confidence=confidence, imgsz=imgsz)
                last_counts = build_counts(predictions)

            rendered = draw_overlay(frame.copy(), predictions, fps, last_inference_ms)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(60, min(95, jpeg_quality)))]
            success, buffer = cv2.imencode(".jpg", rendered, encode_param)
            if success:
                with state.lock:
                    stats = state.camera_stats.get(self.camera_id)
                    if stats is not None:
                        stats["fps"] = fps
                        stats["inference_ms"] = last_inference_ms
                        stats["counts"] = dict(last_counts)
                        stats["counts_text"] = format_counts(last_counts)
                        now_ts = time.time()
                        history = stats.get("counts_history", [])
                        history.append((now_ts, dict(last_counts)))
                        # keep only last 2 minutes
                        cutoff = now_ts - INCIDENT_WINDOW_SEC
                        history = [item for item in history if item[0] >= cutoff]
                        stats["counts_history"] = history
                        stats["last_frame"] = buffer.tobytes()
                        _append_incident_frame(stats, now_ts, stats["last_frame"])

                        if inference_enabled and _has_cheating_detection(predictions):
                            last_saved = float(stats.get("last_incident_save_ts", 0.0))
                            if now_ts - last_saved >= INCIDENT_SAVE_COOLDOWN_SEC:
                                window_counts = _window_counts(history, now_ts)
                                if _window_has_cheating(window_counts):
                                    clip_cutoff = now_ts - INCIDENT_CLIP_SEC
                                    clip_samples = [
                                        item for item in stats.get("incident_buffer", [])
                                        if item[0] >= clip_cutoff
                                    ]
                                    incident = _save_incident_reference(
                                        camera_id=self.camera_id,
                                        camera_label=str(stats.get("label", self.camera_id)),
                                        counts_window=window_counts,
                                        samples=clip_samples,
                                    )
                                    stats["last_incident_save_ts"] = now_ts
                                    stats["last_incident_ref"] = incident.get("clip_b64_short", "")
                                    print(
                                        f"[Camera {self.camera_id}] Saved cheating incident reference "
                                        f"({len(clip_samples)} frames, 2-minute window)"
                                    )

                        stats["ref"] = current_ref or stats["ref"]

            frame_index += 1

        if cap is not None:
            cap.release()
        with state.lock:
            stats = state.camera_stats.get(self.camera_id)
            if stats is not None:
                stats["running"] = False
                stats["last_frame"] = None


def _start_camera_worker(camera_id: str) -> None:
    """Start a camera worker thread if not already running."""
    existing = state.camera_workers.get(camera_id)
    if existing is not None and existing.is_alive():
        return
    worker = CameraWorker(camera_id)
    state.camera_workers[camera_id] = worker
    with state.lock:
        stats = state.camera_stats.get(camera_id)
        if stats is not None:
            stats["running"] = True
    worker.start()


def generate_frames(camera_id: str) -> Any:
    """Generate MJPEG frames for streaming."""
    while True:
        with state.lock:
            stats = state.camera_stats.get(camera_id)
            if stats is None:
                break
            frame_bytes = stats.get("last_frame")
            running = stats.get("running", False)
        if not frame_bytes:
            # Keep the MJPEG stream alive with a placeholder until the first real frame arrives.
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + get_placeholder_frame() + b"\r\n"
            )
            time.sleep(0.2 if running else 1.0)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


def get_cameras_snapshot() -> list[dict[str, str]]:
    """Get a snapshot of all cameras."""
    with state.lock:
        return [dict(camera) for camera in state.cameras]


def add_camera(label: str, ref: str, set_active: bool = False) -> list[dict[str, str]]:
    """Add a new camera."""
    with state.lock:
        if len(state.cameras) >= MAX_CAMERAS:
            raise ValueError("max cameras reached")
    record = CameraRecord(label=label, ref=ref)
    db.session.add(record)
    db.session.commit()
    camera = _camera_dict(record)

    with state.lock:
        state.cameras.append(camera)
        _init_camera_stats(camera["id"], camera["label"], camera["ref"])
        state._next_camera_id = max(state._next_camera_id, int(record.id) + 1)
        if set_active or not state.active_camera_id:
            state.active_camera_id = camera["id"]

    return get_cameras_snapshot()


def remove_camera(camera_id: str) -> list[dict[str, str]]:
    """Remove a camera."""
    worker: CameraWorker | None = None
    try:
        record_id = int(camera_id)
    except (TypeError, ValueError):
        record_id = None
    if record_id is not None:
        record = db.session.get(CameraRecord, record_id)
        if record is not None:
            db.session.delete(record)
            db.session.commit()
    with state.lock:
        state.cameras = [camera for camera in state.cameras if camera["id"] != camera_id]
        state.camera_stats.pop(camera_id, None)
        worker = state.camera_workers.pop(camera_id, None)
        if state.active_camera_id == camera_id:
            state.active_camera_id = state.cameras[0]["id"] if state.cameras else ""
    if worker is not None:
        worker.stop()
    return get_cameras_snapshot()


def set_active_camera(camera_id: str) -> str:
    """Set the active camera."""
    with state.lock:
        if _get_camera_by_id_nolock(camera_id) is None:
            raise ValueError("camera not found")
        state.active_camera_id = camera_id
        return state.active_camera_id


def get_active_camera_id() -> str:
    """Get the active camera ID."""
    with state.lock:
        return state.active_camera_id


def start_camera(camera_id: str) -> None:
    """Start a camera."""
    with state.lock:
        if _get_camera_by_id_nolock(camera_id) is None:
            raise ValueError("camera not found")
    _start_camera_worker(camera_id)


def stop_camera(camera_id: str) -> None:
    """Stop a camera."""
    worker: CameraWorker | None = None
    with state.lock:
        if _get_camera_by_id_nolock(camera_id) is None:
            raise ValueError("camera not found")
        worker = state.camera_workers.pop(camera_id, None)
        stats = state.camera_stats.get(camera_id)
        if stats is not None:
            stats["running"] = False
            stats["last_frame"] = None
    if worker is not None:
        worker.stop()


def get_camera_stats_snapshot() -> list[dict[str, Any]]:
    """Get a snapshot of all camera stats."""
    snapshot: list[dict[str, Any]] = []
    with state.lock:
        for stats in state.camera_stats.values():
            snapshot.append(
                {
                    "id": stats.get("id"),
                    "label": stats.get("label"),
                    "ref": stats.get("ref"),
                    "fps": stats.get("fps", 0.0),
                    "inference_ms": stats.get("inference_ms", 0.0),
                    "counts_text": stats.get("counts_text", "No objects detected."),
                    "running": stats.get("running", False),
                    "error": stats.get("error"),
                }
            )
    return snapshot


def get_totals_snapshot() -> dict[str, Any]:
    """Get aggregated totals across all cameras."""
    with state.lock:
        stats_list = list(state.camera_stats.values())
    totals: dict[str, int] = {}
    fps_values: list[float] = []
    inference_values: list[float] = []
    for stats in stats_list:
        counts = stats.get("counts", {})
        for label, count in counts.items():
            totals[label] = totals.get(label, 0) + int(count)
        fps = float(stats.get("fps", 0.0))
        inference = float(stats.get("inference_ms", 0.0))
        if fps > 0:
            fps_values.append(fps)
        if inference > 0:
            inference_values.append(inference)

    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
    avg_inference = sum(inference_values) / len(inference_values) if inference_values else 0.0

    return {
        "counts": format_counts(totals),
        "objects": sum(totals.values()),
        "fps": round(avg_fps, 2),
        "inference_ms": round(avg_inference, 2),
    }


def set_inference_enabled(enabled: bool) -> bool:
    """Set inference enabled state."""
    with state.lock:
        state.inference_enabled = bool(enabled)
        return state.inference_enabled


def get_inference_enabled() -> bool:
    """Get inference enabled state."""
    with state.lock:
        return state.inference_enabled


def init_camera_store() -> None:
    """Initialize camera store from database."""
    records = CameraRecord.query.order_by(CameraRecord.id).all()
    if not records:
        record = CameraRecord(label="Primary", ref=str(DEFAULT_CAMERA_REF))
        db.session.add(record)
        db.session.commit()
        records = [record]

    with state.lock:
        state.cameras = []
        state.camera_stats = {}
        state.camera_workers = {}
        state.active_camera_id = ""
        state._next_camera_id = 1
        for record in records:
            camera = _camera_dict(record)
            state.cameras.append(camera)
            _init_camera_stats(camera["id"], camera["label"], camera["ref"])
            state._next_camera_id = max(state._next_camera_id, record.id + 1)
        state.active_camera_id = state.cameras[0]["id"] if state.cameras else ""

    # Camera workers are started manually via start_camera()
