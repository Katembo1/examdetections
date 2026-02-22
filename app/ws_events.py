"""WebSocket events — camera frame streaming via Socket.IO."""
from __future__ import annotations

import base64
import threading
import time

from flask_socketio import emit, join_room, leave_room

from .camera import get_cameras_snapshot, start_camera
from .extensions import socketio
from .state import state


@socketio.on("connect")
def on_connect() -> None:
    cameras = get_cameras_snapshot()
    emit("cameras_list", {"cameras": cameras})


@socketio.on("subscribe_camera")
def on_subscribe(data: dict) -> None:
    """Client sends {camera_id} to start receiving frames for that camera."""
    camera_id = str(data.get("camera_id", ""))
    if not camera_id:
        emit("error", {"msg": "camera_id required"})
        return

    # Ensure worker is running
    try:
        start_camera(camera_id)
    except ValueError:
        pass  # already running or not found — continue anyway

    join_room(camera_id)
    emit("subscribed", {"camera_id": camera_id})

    # Push frames in a background thread for this client's room
    t = threading.Thread(target=_stream_frames, args=(camera_id,), daemon=True)
    t.start()


@socketio.on("unsubscribe_camera")
def on_unsubscribe(data: dict) -> None:
    camera_id = str(data.get("camera_id", ""))
    leave_room(camera_id)


def _stream_frames(camera_id: str) -> None:
    """Greenlet: push JPEG frames as base64 to the room at ~25 fps."""
    target_interval = 1.0 / 25
    last_frame: bytes | None = None

    while True:
        t_start = time.monotonic()

        with state.lock:
            stats = state.camera_stats.get(camera_id)
            if stats is None:
                break
            frame_bytes: bytes | None = stats.get("last_frame")
            fps = stats.get("fps", 0.0)
            inference_ms = stats.get("inference_ms", 0.0)
            running = stats.get("running", False)

        if frame_bytes and frame_bytes != last_frame:
            last_frame = frame_bytes
            b64 = base64.b64encode(frame_bytes).decode("ascii")
            socketio.emit(
                "frame",
                {"camera_id": camera_id, "data": b64, "fps": fps, "inference_ms": inference_ms},
                room=camera_id,
            )
        elif not running and not frame_bytes:
            # Camera stopped — let the room know
            socketio.emit("camera_stopped", {"camera_id": camera_id}, room=camera_id)
            break

        elapsed = time.monotonic() - t_start
        sleep_time = max(0.0, target_interval - elapsed)
        time.sleep(sleep_time)
