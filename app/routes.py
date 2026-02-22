from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, jsonify, render_template, request

from .camera import (
    add_camera,
    generate_frames,
    get_active_camera_id,
    get_cameras_snapshot,
    get_camera_stats_snapshot,
    get_inference_enabled,
    get_totals_snapshot,
    remove_camera,
    set_active_camera,
    set_inference_enabled,
    start_camera,
    stop_camera,
)
from .config import MAX_CAMERAS, get_active_model_path
from .state import state
from .utils import save_upload, test_camera_reference

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index() -> str:
    return render_template(
        "index.html",
        confidence=state.confidence,
        model_path=get_active_model_path(),
        camera_ref=get_active_camera_id(),
    )


@main_bp.route("/video_feed/<camera_id>")
def video_feed(camera_id: str) -> Response:
    if not any(camera.get("id") == camera_id for camera in get_cameras_snapshot()):
        return Response("Camera not found", status=404)
    
    response = Response(
        generate_frames(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    
    # Security and caching headers for camera stream
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    return response


@main_bp.route("/stats")
def stats() -> Any:
    return jsonify(
        {
            "totals": get_totals_snapshot(),
            "cameras": get_camera_stats_snapshot(),
            "confidence": state.confidence,
            "active_camera_id": get_active_camera_id(),
            "inference_enabled": get_inference_enabled(),
        }
    )


@main_bp.route("/config", methods=["POST"])
def config() -> Any:
    payload = request.get_json(silent=True) or {}
    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            parsed = float(confidence)
            parsed = max(0.0, min(1.0, parsed))
            with state.lock:
                state.confidence = parsed
        except (TypeError, ValueError):
            pass
    return stats()


@main_bp.route("/inference", methods=["POST"])
def inference_toggle() -> Any:
    payload = request.get_json(silent=True) or {}
    enabled = payload.get("enabled")
    if enabled is None:
        return jsonify({"error": "enabled required"}), 400
    status = set_inference_enabled(bool(enabled))
    return jsonify({"inference_enabled": status})


@main_bp.route("/cameras", methods=["GET", "POST"])
def cameras() -> Any:
    if request.method == "GET":
        return jsonify(
            {
                "cameras": get_cameras_snapshot(),
                "active": get_active_camera_id(),
                "max": MAX_CAMERAS,
            }
        )

    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip()
    ref = str(payload.get("ref", "")).strip()
    if not label or not ref:
        return jsonify({"error": "label and ref required"}), 400
    try:
        cameras_list = add_camera(label, ref)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"cameras": cameras_list, "active": get_active_camera_id(), "max": MAX_CAMERAS})


@main_bp.route("/cameras/<camera_id>", methods=["DELETE"])
def camera_remove(camera_id: str) -> Any:
    return jsonify({"cameras": remove_camera(camera_id), "active": get_active_camera_id()})


@main_bp.route("/cameras/active", methods=["POST"])
def camera_active() -> Any:
    payload = request.get_json(silent=True) or {}
    camera_id = payload.get("id")
    if camera_id is None:
        return jsonify({"error": "id required"}), 400
    try:
        active = set_active_camera(str(camera_id))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"active": active})


@main_bp.route("/cameras/<camera_id>/start", methods=["POST"])
def camera_start(camera_id: str) -> Any:
    try:
        start_camera(camera_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"status": "started", "id": camera_id})


@main_bp.route("/cameras/<camera_id>/stop", methods=["POST"])
def camera_stop(camera_id: str) -> Any:
    try:
        stop_camera(camera_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"status": "stopped", "id": camera_id})


@main_bp.route("/cameras/test", methods=["POST"])
def camera_test() -> Any:
    payload = request.get_json(silent=True) or {}
    ref = payload.get("ref")
    if ref is None:
        camera_id = payload.get("id") or get_active_camera_id()
        ref = None
        for camera in get_cameras_snapshot():
            if camera.get("id") == camera_id:
                ref = camera.get("ref")
                break
    if ref is None:
        return jsonify({"status": "error"}), 400
    ok = test_camera_reference(str(ref))
    status = "ok" if ok else "error"
    return jsonify({"status": status})


@main_bp.route("/upload", methods=["POST"])
def upload() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400
    file_storage = request.files["file"]
    if not file_storage.filename:
        return jsonify({"error": "empty filename"}), 400
    saved_path = save_upload(file_storage)
    return jsonify({"status": "ok", "path": saved_path})
