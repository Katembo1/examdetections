from __future__ import annotations

from typing import Any

import base64
import json
import subprocess
import time
from pathlib import Path

import cv2
from flask import Blueprint, Response, jsonify, render_template, request, send_from_directory, current_app
from sqlalchemy.exc import OperationalError

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
from .config import MAX_CAMERAS, UPLOADS_ROOT, get_active_model_path
from .inference import build_counts, draw_overlay, run_detection
from .models import UploadAnalytics, UploadRecord
from .state import state
from .utils import enumerate_available_cameras, format_counts, format_counts_rate, save_upload, test_camera_reference

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
    now_ts = time.time()
    live_counts: dict[str, float] = {}
    live_duration = 0.0
    with state.lock:
        for stats in state.camera_stats.values():
            history = stats.get("counts_history", [])
            if not history:
                continue
            min_ts = min(item[0] for item in history)
            live_duration = max(live_duration, now_ts - min_ts)
            for _, counts in history:
                for label, count in counts.items():
                    live_counts[label] = live_counts.get(label, 0.0) + float(count)

    live_minutes = max(live_duration / 60.0, 1e-6)
    live_per_minute = {k: v / live_minutes for k, v in live_counts.items()}

    try:
        upload_records = UploadAnalytics.query.all()
    except OperationalError:
        upload_records = []
    upload_counts: dict[str, float] = {}
    upload_minutes_total = 0.0
    upload_avg_inference = 0.0
    if upload_records:
        for record in upload_records:
            upload_minutes_total += max(record.duration_sec / 60.0, 0.0)
            try:
                counts = json.loads(record.counts_json)
            except Exception:
                counts = {}
            for label, value in counts.items():
                upload_counts[label] = upload_counts.get(label, 0.0) + float(value)
            upload_avg_inference += float(record.avg_inference_ms)
        upload_avg_inference = upload_avg_inference / max(len(upload_records), 1)

    upload_minutes_total = max(upload_minutes_total, 1e-6)
    uploads_per_minute = {k: v / upload_minutes_total for k, v in upload_counts.items()}

    combined_counts: dict[str, float] = {}
    for label in set(live_per_minute.keys()) | set(uploads_per_minute.keys()):
        combined_counts[label] = live_per_minute.get(label, 0.0) + uploads_per_minute.get(label, 0.0)

    return jsonify(
        {
            "totals": get_totals_snapshot(),
            "cameras": get_camera_stats_snapshot(),
            "confidence": state.confidence,
            "active_camera_id": get_active_camera_id(),
            "inference_enabled": get_inference_enabled(),
            "analytics": {
                "live_per_minute": live_per_minute,
                "uploads_per_minute": uploads_per_minute,
                "combined_per_minute": combined_counts,
                "live_counts_text": format_counts_rate(live_per_minute),
                "uploads_counts_text": format_counts_rate(uploads_per_minute),
                "combined_counts_text": format_counts_rate(combined_counts),
                "uploads_avg_inference_ms": round(upload_avg_inference, 2),
            },
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
    except Exception as exc:
        current_app.logger.exception("Unexpected error adding camera: %s", exc)
        return jsonify({"error": "Failed to add camera"}), 500
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


@main_bp.route("/cameras/available", methods=["GET"])
def cameras_available() -> Any:
    """List available hardware cameras."""
    available = enumerate_available_cameras(10)
    return jsonify({
        "available": available,
        "count": len(available),
        "message": "Hardware cameras detected" if available else "No hardware cameras found. Use video files or streams for cloud deployment."
    })


@main_bp.route("/upload", methods=["POST"])
def upload() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400
    file_storage = request.files["file"]
    if not file_storage.filename:
        return jsonify({"error": "empty filename"}), 400
    record = save_upload(file_storage)
    return jsonify({"status": "ok", "path": record.path, "upload_id": record.id})


@main_bp.route("/uploads/<path:filename>")
def uploads(filename: str) -> Any:
    return send_from_directory(UPLOADS_ROOT, filename)


@main_bp.route("/upload/infer", methods=["POST"])
def upload_infer() -> Any:
    payload = request.get_json(silent=True) or {}
    raw_path = payload.get("path")
    if not raw_path:
        return jsonify({"error": "path required"}), 400

    try:
        resolved = Path(str(raw_path)).resolve()
        uploads_root = UPLOADS_ROOT.resolve()
    except Exception:
        return jsonify({"error": "invalid path"}), 400

    if uploads_root not in resolved.parents and resolved != uploads_root:
        return jsonify({"error": "path not allowed"}), 400

    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        return jsonify({"error": "unable to open video"}), 400

    max_frames = int(payload.get("max_frames", 30))
    every_n = max(1, int(payload.get("every_n", 2)))
    write_video = bool(payload.get("write_video", True))

    confidence = state.confidence
    imgsz = state.imgsz
    jpeg_quality = int(max(60, min(95, state.jpeg_quality)))

    processed = 0
    total_inference = 0.0
    counts_total: dict[str, int] = {}
    last_frame = None
    video_path = None
    mp4_path = None
    writer = None
    writer_ok = False
    frame_index = 0

    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % every_n != 0:
            continue

        preds, inf_ms = run_detection(frame, confidence=confidence, imgsz=imgsz)
        total_inference += inf_ms
        processed += 1

        frame_counts = build_counts(preds)
        for key, value in frame_counts.items():
            counts_total[key] = counts_total.get(key, 0) + value

        last_frame = draw_overlay(frame.copy(), preds, 0.0, inf_ms)

        if write_video:
            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                height, width = frame.shape[:2]
                output_name = f"{resolved.stem}_annotated_{int(time.time())}.avi"
                video_path = uploads_root / output_name
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                writer_ok = writer.isOpened()
            if writer_ok and last_frame is not None:
                writer.write(last_frame)

    cap.release()
    if writer is not None:
        writer.release()

    if processed == 0:
        return jsonify({"error": "no frames processed"}), 400

    avg_inference = total_inference / max(1, processed)
    counts_text = format_counts(counts_total) if counts_total else "No objects detected."

    preview_b64 = ""
    if last_frame is not None:
        success, buffer = cv2.imencode(".jpg", last_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if success:
            preview_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

    video_url = ""
    if video_path is not None and writer_ok and video_path.exists() and video_path.stat().st_size > 0:
        # Try converting to MP4 for browser playback if ffmpeg is available
        mp4_path = video_path.with_suffix(".mp4")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-movflags",
            "faststart",
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if mp4_path.exists() and mp4_path.stat().st_size > 0:
                video_url = f"/uploads/{mp4_path.name}"
            else:
                video_url = f"/uploads/{video_path.name}"
        except Exception:
            video_url = f"/uploads/{video_path.name}"

    return jsonify({
        "status": "ok",
        "frames": processed,
        "avg_inference_ms": avg_inference,
        "counts": counts_total,
        "counts_text": counts_text,
        "preview_jpeg": preview_b64,
        "video_url": video_url,
        "video_format": "mp4" if video_url.endswith(".mp4") else ("avi" if video_url else ""),
    })


@main_bp.route("/upload/stream")
def upload_stream() -> Response:
    raw_path = request.args.get("path")
    if not raw_path:
        return Response("path required", status=400)

    try:
        resolved = Path(str(raw_path)).resolve()
        uploads_root = UPLOADS_ROOT.resolve()
    except Exception:
        return Response("invalid path", status=400)

    if uploads_root not in resolved.parents and resolved != uploads_root:
        return Response("path not allowed", status=400)

    fps = float(request.args.get("fps", "15"))
    fps = max(1.0, min(30.0, fps))
    max_seconds = float(request.args.get("seconds", "0"))
    max_seconds = max(0.0, max_seconds)

    confidence = state.confidence
    imgsz = state.imgsz
    jpeg_quality = int(max(60, min(95, state.jpeg_quality)))

    upload_id = None
    try:
        record = UploadRecord.query.filter_by(path=str(resolved)).first()
        upload_id = record.id if record is not None else None
    except Exception:
        upload_id = None

    def generate() -> Any:
        cap = cv2.VideoCapture(str(resolved))
        if not cap.isOpened():
            payload = json.dumps({"error": "unable to open video"})
            yield f"event: error\ndata: {payload}\n\n"
            return

        start_time = time.monotonic()
        fps_source = cap.get(cv2.CAP_PROP_FPS) or fps
        frame_index = 0
        target_interval = 1.0 / fps
        total_inference = 0.0
        total_counts: dict[str, int] = {}

        while True:
            t0 = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            preds, inf_ms = run_detection(frame, confidence=confidence, imgsz=imgsz)
            total_inference += inf_ms
            counts = build_counts(preds)
            for label, value in counts.items():
                total_counts[label] = total_counts.get(label, 0) + int(value)
            counts_text = format_counts(counts) if counts else "No objects detected."
            rendered = draw_overlay(frame.copy(), preds, 0.0, inf_ms)

            success, buffer = cv2.imencode(
                ".jpg",
                rendered,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if success:
                b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
                payload = json.dumps(
                    {
                        "frame": frame_index,
                        "jpeg": b64,
                        "counts_text": counts_text,
                        "inference_ms": inf_ms,
                    }
                )
                yield f"event: frame\ndata: {payload}\n\n"

            if max_seconds > 0 and (time.monotonic() - start_time) >= max_seconds:
                break

            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, target_interval - elapsed)
            time.sleep(sleep_time)

        cap.release()
        duration_sec = frame_index / max(fps_source, 1.0)
        avg_inference = total_inference / max(frame_index, 1)
        # persist analytics for uploaded video
        if upload_id is not None:
            db_record = UploadAnalytics(
                upload_id=upload_id,
                duration_sec=duration_sec,
                frames=frame_index,
                avg_inference_ms=avg_inference,
                counts_json=json.dumps(total_counts),
            )
            try:
                from .db import db

                with current_app.app_context():
                    db.session.add(db_record)
                    db.session.commit()
            except Exception:
                pass
        yield "event: done\ndata: {}\n\n"

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response
