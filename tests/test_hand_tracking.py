"""Step 2 - hand-tracking test. Camera + MediaPipe + HUD, but NEVER touches the mouse.

    python -m tests.test_hand_tracking            # live preview, Q/ESC to quit
    python -m tests.test_hand_tracking --frames 120 --headless   # print detection stats

Check that: landmarks follow your hand, the state shows MOVING when you point
with only your index finger, and CLICKING when you pinch thumb + index.
"""

import argparse
import logging
import sys
import time

from deniz_control.app import run
from deniz_control.camera import Camera, CameraError
from deniz_control.config import AppConfig
from deniz_control.gestures import GestureStateMachine, extract_features
from deniz_control.hand_tracker import HandTracker


def headless(frames: int) -> int:
    cfg = AppConfig()
    try:
        cam = Camera(cfg.camera)
        cam.open()
    except CameraError as exc:
        print(f"FAIL: {exc}")
        return 1
    machine = GestureStateMachine(cfg.gesture)
    detected = n = 0
    t0 = time.perf_counter()
    with HandTracker(cfg.tracker) as tracker:
        try:
            while n < frames:
                frame = cam.read()
                if frame is None:
                    continue
                n += 1
                hand = tracker.detect(frame)
                feats = None
                if hand:
                    detected += 1
                    feats = extract_features(hand.landmarks, frame.shape[1] / frame.shape[0], cfg.gesture)
                upd = machine.update(feats, time.monotonic())
                if n % 15 == 0:
                    print(f"frame {n:4d}  hand={'yes' if hand else 'no '}  state={upd.state.value}")
        finally:
            cam.release()
    fps = n / (time.perf_counter() - t0)
    print(f"PASS: pipeline ran {n} frames at {fps:.1f} FPS; hand detected in {detected} frames")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.headless:
        return headless(args.frames or 120)
    return run(AppConfig(), enable_mouse=False, max_frames=args.frames)


if __name__ == "__main__":
    sys.exit(main())
