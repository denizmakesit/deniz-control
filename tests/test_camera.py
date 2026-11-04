"""Step 1 - camera-only test. No MediaPipe, no mouse.

    python -m tests.test_camera                 # live mirrored preview, Q/ESC to quit
    python -m tests.test_camera --frames 90 --headless   # grab 90 frames, print stats
"""

import argparse
import logging
import sys
import time

import cv2

from deniz_control.camera import Camera, CameraError
from deniz_control.config import CameraConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=None, help="stop after N frames")
    ap.add_argument("--headless", action="store_true", help="no preview window")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        cam = Camera(CameraConfig())
        cam.open()
    except CameraError as exc:
        print(f"FAIL: {exc}")
        return 1

    count, t0 = 0, time.perf_counter()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            count += 1
            if not args.headless:
                fps = count / (time.perf_counter() - t0)
                cv2.putText(frame, f"CAMERA TEST  {fps:4.1f} FPS  (Q to quit)", (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 217, 101), 2, cv2.LINE_AA)
                cv2.imshow("DENIZ CONTROL - camera test", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.frames and count >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    print(f"PASS: {count} frames at {cam.actual_size[0]}x{cam.actual_size[1]}, "
          f"{count / elapsed:.1f} FPS average")
    return 0


if __name__ == "__main__":
    sys.exit(main())
