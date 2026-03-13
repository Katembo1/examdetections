"""Utility helper functions."""
import socket
import time
import uuid
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from werkzeug.utils import secure_filename

from .config import UPLOADS_ROOT
from .db import db
from .models import UploadRecord


def _extract_scope_name(scopes: list[str]) -> str | None:
    """Extract human-readable camera name from ONVIF scopes."""
    for scope in scopes:
        marker = "onvif://www.onvif.org/name/"
        idx = scope.find(marker)
        if idx >= 0:
            name = scope[idx + len(marker):].replace("_", " ").strip()
            if name:
                return name
    return None


def _guess_rtsp_candidates(ip: str) -> list[str]:
    """Build common RTSP URL templates for quick manual testing."""
    return [
        f"rtsp://username:password@{ip}:554/stream1",
        f"rtsp://username:password@{ip}:554/cam/realmonitor?channel=1&subtype=0",
        f"rtsp://username:password@{ip}:554/h264Preview_01_main",
        f"rtsp://username:password@{ip}:554/live/ch00_00",
    ]


def discover_onvif_devices(timeout_sec: float = 2.5, max_results: int = 32) -> list[dict[str, Any]]:
    """Discover ONVIF cameras on LAN using WS-Discovery multicast."""
    timeout_sec = max(0.5, min(float(timeout_sec), 10.0))
    max_results = max(1, min(int(max_results), 128))

    message_id = f"uuid:{uuid.uuid4()}"
    probe_xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<e:Envelope xmlns:e=\"http://www.w3.org/2003/05/soap-envelope\"
            xmlns:w=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\"
            xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\"
            xmlns:dn=\"http://www.onvif.org/ver10/network/wsdl\">
  <e:Header>
    <w:MessageID>{message_id}</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>""".encode("utf-8")

    discovered: dict[str, dict[str, Any]] = {}
    sock: socket.socket | None = None

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout_sec)
        sock.sendto(probe_xml, ("239.255.255.250", 3702))

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and len(discovered) < max_results:
            remaining = max(0.05, deadline - time.monotonic())
            sock.settimeout(remaining)
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                break
            except OSError:
                break

            sender_ip = str(addr[0])
            try:
                root = ET.fromstring(data)
            except ET.ParseError:
                continue

            xaddrs: list[str] = []
            scopes: list[str] = []

            for elem in root.iter():
                tag = elem.tag.lower()
                text = (elem.text or "").strip()
                if not text:
                    continue
                if tag.endswith("xaddrs"):
                    xaddrs.extend(part for part in text.split() if part)
                elif tag.endswith("scopes"):
                    scopes.extend(part for part in text.split() if part)

            host_ip = sender_ip
            for xaddr in xaddrs:
                parsed = urlparse(xaddr)
                if parsed.hostname:
                    host_ip = parsed.hostname
                    break

            name = _extract_scope_name(scopes) or f"ONVIF {host_ip}"
            entry = discovered.get(host_ip, {
                "ip": host_ip,
                "label": name,
                "xaddrs": [],
                "scopes": [],
            })

            entry["label"] = entry.get("label") or name
            entry["xaddrs"] = sorted(set([*entry.get("xaddrs", []), *xaddrs]))
            entry["scopes"] = sorted(set([*entry.get("scopes", []), *scopes]))
            entry["rtsp_candidates"] = _guess_rtsp_candidates(host_ip)
            discovered[host_ip] = entry

    finally:
        if sock is not None:
            sock.close()

    return sorted(discovered.values(), key=lambda item: item.get("ip", ""))


def parse_camera_reference(value: str) -> int | str:
    """Parse camera reference (convert to int if numeric)."""
    value = str(value).strip()
    if value.isdigit():
        return int(value)
    return value


def is_hardware_camera(ref: str) -> bool:
    """Check if reference is a hardware camera (numeric index)."""
    return str(ref).strip().isdigit()


def _frame_has_visual_content(frame: np.ndarray | None) -> bool:
    """Return True when a frame appears valid and not an all-black capture."""
    if frame is None or frame.size == 0:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Reject frames that are almost entirely black and have no texture.
    return mean_val > 4.0 or std_val > 2.0


def _configure_capture(cap: cv2.VideoCapture) -> None:
    """Apply camera properties that improve compatibility on common webcams."""
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Prefer MJPG when available; it avoids black frames on some USB cameras.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def try_open_camera_with_backends(camera_index: int) -> cv2.VideoCapture | None:
    """Try to open a hardware camera with multiple backends."""
    # Try different backends that work with USB/built-in cameras
    backends = [
        cv2.CAP_DSHOW,      # DirectShow (Windows)
        cv2.CAP_MSMF,       # Microsoft Media Foundation (Windows)
        cv2.CAP_V4L2,       # Video4Linux2 (Linux)
        cv2.CAP_AVFOUNDATION, # AVFoundation (macOS)
        cv2.CAP_ANY,        # Auto-detect
    ]
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            _configure_capture(cap)
            if cap.isOpened():
                # Warm up and verify frame quality. Some backends return black frames while reporting success.
                valid_reads = 0
                for _ in range(8):
                    ok, frame = cap.read()
                    if ok and _frame_has_visual_content(frame):
                        valid_reads += 1
                    time.sleep(0.03)

                if valid_reads >= 2:
                    return cap
                cap.release()
        except Exception:
            continue
    
    return None


def enumerate_available_cameras(max_cameras: int = 10) -> list[int]:
    """Enumerate available hardware camera indices."""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(i)
        cap.release()
    return available


def format_counts(counts: dict[str, int]) -> str:
    """Format object counts as a readable string."""
    if not counts:
        return "No objects detected."
    return "\n".join(f"{label}: {count}" for label, count in sorted(counts.items()))


def format_counts_rate(counts: dict[str, float], suffix: str = "/min") -> str:
    """Format per-minute counts as a readable string."""
    if not counts:
        return "No objects detected."
    return "\n".join(f"{label}: {value:.2f}{suffix}" for label, value in counts.items())

def make_placeholder_frame() -> bytes:
    """Create a placeholder frame for stopped cameras."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Camera Stopped", (145, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press Start Camera to begin", (95, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (55, 55, 55), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


_PLACEHOLDER_FRAME: bytes = make_placeholder_frame()


def get_placeholder_frame() -> bytes:
    """Get the cached placeholder frame."""
    return _PLACEHOLDER_FRAME


def test_camera_reference(ref: str) -> bool:
    """Test if a camera reference can be opened and read from."""
    parsed_ref = parse_camera_reference(ref)
    
    # Use improved backend detection for hardware cameras
    if is_hardware_camera(ref):
        cap = try_open_camera_with_backends(parsed_ref)
        if cap is None:
            return False
        ok, _ = cap.read()
        cap.release()
        return bool(ok)
    else:
        # Video file or stream
        cap = cv2.VideoCapture(parsed_ref)
        if not cap.isOpened():
            cap.release()
            return False
        ok, _ = cap.read()
        cap.release()
        return bool(ok)


def save_upload(file_storage: Any) -> UploadRecord:
    """Save an uploaded file and create a database record."""
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file_storage.filename or "upload")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_name = f"{timestamp}_{filename}"
    dest = UPLOADS_ROOT / final_name
    file_storage.save(dest)
    record = UploadRecord(filename=filename, path=str(dest))
    db.session.add(record)
    db.session.commit()
    return record
