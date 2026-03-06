"""WebSocket events — camera frame streaming via Socket.IO."""
from __future__ import annotations

import base64
import threading
import time

import cv2
import numpy as np
from flask import request
from flask_socketio import emit, join_room, leave_room

from .camera import get_cameras_snapshot, start_camera
from .extensions import socketio
from .inference import build_counts, draw_overlay, run_detection
from .state import state
from .utils import format_counts


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


# ── Client-side camera streaming ──

@socketio.on("client_stream_start")
def on_client_stream_start() -> None:
    """Client requests to start streaming from their device camera."""
    client_id = request.sid
    join_room(f"client_{client_id}")
    
    # Initialize client camera stats
    with state.lock:
        if not hasattr(state, "client_streams"):
            state.client_streams = {}
        state.client_streams[client_id] = {
            "fps": 0.0,
            "inference_ms": 0.0,
            "counts": {},
            "frame_count": 0,
            "last_frame_time": time.time(),
        }
    
    emit("client_stream_ready", {"client_id": client_id})


@socketio.on("client_frame")
def on_client_frame(data: dict) -> None:
    """Receive frame from client, run inference, and send back results."""
    client_id = request.sid
    
    try:
        # Decode base64 frame
        frame_data = data.get("data", "")
        if not frame_data:
            emit("error", {"msg": "No frame data"})
            return
        
        # Remove data URL prefix if present
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit("error", {"msg": "Failed to decode frame"})
            return
        
        # Get configuration
        with state.lock:
            confidence = state.confidence
            imgsz = state.imgsz
            inference_enabled = state.inference_enabled
            
            if not hasattr(state, "client_streams"):
                state.client_streams = {}
            
            client_stats = state.client_streams.get(client_id, {})
            frame_count = client_stats.get("frame_count", 0) + 1
            last_time = client_stats.get("last_frame_time", time.time())
        
        # Calculate FPS
        now = time.time()
        fps = 1.0 / max(now - last_time, 0.001)
        
        # Run inference if enabled
        predictions = []
        inference_ms = 0.0
        counts = {}
        counts_text = "Inference disabled"
        
        if inference_enabled:
            predictions, inference_ms = run_detection(frame, confidence, imgsz)
            counts = build_counts(predictions)
            counts_text = format_counts(counts) if counts else "No objects detected"
            
            # Draw overlay on frame
            frame = draw_overlay(frame, predictions, fps, inference_ms)
        
        # Encode processed frame
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_b64 = base64.b64encode(buffer).decode("ascii")
        
        # Update client stats
        with state.lock:
            if not hasattr(state, "client_streams"):
                state.client_streams = {}
            state.client_streams[client_id] = {
                "fps": fps,
                "inference_ms": inference_ms,
                "counts": counts,
                "counts_text": counts_text,
                "frame_count": frame_count,
                "last_frame_time": now,
            }
        
        # Send processed frame and stats back to client
        emit("client_frame_result", {
            "data": processed_b64,
            "fps": round(fps, 1),
            "inference_ms": round(inference_ms, 1),
            "counts": counts,
            "counts_text": counts_text,
        })
        
    except Exception as e:
        emit("error", {"msg": f"Frame processing error: {str(e)}"})


@socketio.on("client_stream_stop")
def on_client_stream_stop() -> None:
    """Client stops streaming from their device camera."""
    client_id = request.sid
    leave_room(f"client_{client_id}")
    
    # Clean up client stats
    with state.lock:
        if hasattr(state, "client_streams") and client_id in state.client_streams:
            del state.client_streams[client_id]
    
    emit("client_stream_stopped")
