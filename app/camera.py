"""Camera management and worker threads."""
import threading
import time
from typing import Any

import cv2

from .config import DEFAULT_CAMERA_REF, MAX_CAMERAS
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
                        # keep only last 60s
                        cutoff = now_ts - 60.0
                        history = [item for item in history if item[0] >= cutoff]
                        stats["counts_history"] = history
                        stats["last_frame"] = buffer.tobytes()
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
