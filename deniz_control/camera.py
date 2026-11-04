"""Webcam capture with mirroring, FPS targeting and automatic recovery.

Frames are grabbed on a background thread so camera I/O overlaps with hand
detection; read() always returns the newest frame (never a stale backlog).
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or keeps failing."""


class Camera:
    """cv2.VideoCapture (macOS AVFoundation backend) with a reader thread.

    Usage:
        with Camera(cfg) as cam:
            frame = cam.read()   # BGR, mirrored if configured, or None on timeout
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_t = 0.0
        self.last_frame_time = 0.0   # time.monotonic() when the returned frame arrived
        self._seq = 0            # increments for every new frame
        self._read_seq = 0       # last seq handed out by read()
        self._failed_reads = 0
        self.actual_size: tuple[int, int] = (0, 0)
        self.actual_fps: float = 0.0

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        self.release()
        cap = cv2.VideoCapture(self.cfg.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Could not open camera {self.cfg.index}. Check System Settings > "
                "Privacy & Security > Camera and allow your terminal / editor app."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.target_fps)

        # The first frames after opening can be empty while the sensor warms up.
        ok, frame = False, None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if not ok:
            cap.release()
            raise CameraError(
                "Camera opened but returned no frames. It may be in use by another "
                "app, or camera permission was just requested - run again."
            )

        self._cap = cap
        self._failed_reads = 0
        self.actual_size = (frame.shape[1], frame.shape[0])
        self.actual_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()
        log.info("Camera %d opened at %dx%d (reported %.0f FPS)",
                 self.cfg.index, *self.actual_size, self.actual_fps)

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # -- background grabbing ----------------------------------------------
    def _reader(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            t = time.monotonic()
            with self._cond:
                self._frame, self._frame_t = frame, t
                self._seq += 1
                self._cond.notify_all()

    # -- reading -----------------------------------------------------------
    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        """Return the newest unseen BGR frame, or None if none arrived in time.

        After `max_failed_reads` consecutive timeouts the camera is reopened;
        if that also fails, CameraError is raised.
        """
        if self._cap is None:
            raise CameraError("Camera is not open")

        with self._cond:
            got = self._cond.wait_for(lambda: self._seq > self._read_seq, timeout=timeout)
            if got:
                frame, self._read_seq = self._frame, self._seq
                self.last_frame_time = self._frame_t

        if not got:
            self._failed_reads += 1
            log.debug("No camera frame within %.1fs (%d)", timeout, self._failed_reads)
            if self._failed_reads * timeout >= self.cfg.reopen_after_s:
                self._recover()
            return None

        self._failed_reads = 0
        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def _recover(self) -> None:
        log.warning("Camera stopped delivering frames; trying to reopen")
        for attempt in range(1, self.cfg.reopen_attempts + 1):
            try:
                self.open()
                self._read_seq = self._seq
                log.info("Camera recovered on attempt %d", attempt)
                return
            except CameraError as exc:
                log.warning("Reopen attempt %d failed: %s", attempt, exc)
                time.sleep(0.5)
        raise CameraError("Camera lost and could not be reopened")
