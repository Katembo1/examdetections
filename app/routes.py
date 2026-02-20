from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, jsonify, render_template, request

from .models import (
    add_camera,
    generate_frames,
    get_active_model_path,
    get_cameras_snapshot,
    remove_camera,
    save_upload,
    set_active_camera,
    state,
    test_camera_reference,
)

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index() -> str:
    return render_template(
        "index.html",
        confidence=state.confidence,
        model_path=get_active_model_path(),
        camera_ref=state.camera_ref,
    )


@main_bp.route("/video_feed")
def video_feed() -> Response:
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@main_bp.route("/stats")
def stats() -> Any:
    with state.lock:
        return jsonify(
            {
                "counts": state.counts_text,
                "fps": round(state.stream_fps, 2),
                "inference_ms": round(state.inference_ms, 2),
                "confidence": state.confidence,
                "camera_ref": state.camera_ref,
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


@main_bp.route("/cameras", methods=["GET", "POST"])
def cameras() -> Any:
    if request.method == "GET":
        return jsonify({"cameras": get_cameras_snapshot(), "active": state.camera_ref})

    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip()
    ref = str(payload.get("ref", "")).strip()
    if not label or not ref:
        return jsonify({"error": "label and ref required"}), 400
    return jsonify({"cameras": add_camera(label, ref), "active": state.camera_ref})


@main_bp.route("/cameras/<int:index>", methods=["DELETE"])
def camera_remove(index: int) -> Any:
    return jsonify({"cameras": remove_camera(index), "active": state.camera_ref})


@main_bp.route("/cameras/active", methods=["POST"])
def camera_active() -> Any:
    payload = request.get_json(silent=True) or {}
    ref = payload.get("ref")
    if ref is None:
        return jsonify({"error": "ref required"}), 400
    active = set_active_camera(str(ref))
    return jsonify({"active": active})


@main_bp.route("/cameras/test", methods=["POST"])
def camera_test() -> Any:
    payload = request.get_json(silent=True) or {}
    ref = payload.get("ref") or state.camera_ref
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
