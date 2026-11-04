"""Hand landmark detection using the MediaPipe Tasks HandLandmarker API.

The legacy `mediapipe.solutions.hands` API no longer exists in MediaPipe 1.x;
this module uses `mediapipe.tasks.python.vision.HandLandmarker` in VIDEO mode.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .config import TrackerConfig

log = logging.getLogger(__name__)

# Pairs of landmark indices to draw as the hand skeleton.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]


@dataclass
class HandObservation:
    """One detected hand.

    landmarks:  (21, 3) float array of normalized x, y (0..1 of frame size) and
                relative depth z.
    handedness: "Left" / "Right" as reported by MediaPipe. Because the frame is
                mirrored before detection, this label is swapped relative to
                your real hand.
    score:      handedness classification score (0..1). MediaPipe Tasks does not
                expose the palm detector score directly; this is the confidence
                value it does report per hand.
    """
    landmarks: np.ndarray
    handedness: str
    score: float


def ensure_model(cfg: TrackerConfig) -> None:
    """Download the official hand landmarker model if it is missing."""
    if cfg.model_path.exists():
        return
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading hand landmarker model to %s", cfg.model_path)
    tmp = cfg.model_path.with_suffix(".part")
    urllib.request.urlretrieve(cfg.model_url, tmp)
    tmp.rename(cfg.model_path)


class HandTracker:
    def __init__(self, cfg: TrackerConfig) -> None:
        ensure_model(cfg)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(cfg.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray) -> HandObservation | None:
        """Detect the most prominent hand in a BGR frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps.
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.hand_landmarks:
            return None

        pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        label, score = "Unknown", 0.0
        if result.handedness and result.handedness[0]:
            cat = result.handedness[0][0]
            label, score = cat.category_name, float(cat.score)
        return HandObservation(landmarks=pts, handedness=label, score=score)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
