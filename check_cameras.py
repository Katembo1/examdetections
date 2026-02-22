#!/usr/bin/env python
"""List available camera devices."""
import sys
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.utils import enumerate_available_cameras

if __name__ == "__main__":
    print("Scanning for available cameras...")
    print("-" * 40)
    
    available = enumerate_available_cameras(10)
    
    if available:
        print(f"✓ Found {len(available)} camera(s):")
        for idx in available:
            print(f"  • Camera index: {idx}")
        print()
        print("To use a camera, set in .env:")
        print(f"  ONNX_CAMERA_REFERENCE={available[0]}")
    else:
        print("✗ No hardware cameras detected.")
        print()
        print("Options:")
        print("  1. Check camera permissions")
        print("  2. Connect a USB camera")
        print("  3. Use a video file:")
        print("     ONNX_CAMERA_REFERENCE=/path/to/video.mp4")
        print("  4. Use a video stream:")
        print("     ONNX_CAMERA_REFERENCE=rtsp://stream-url")
    
    print("-" * 40)
