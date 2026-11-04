"""Gesture recognition: per-frame hand features plus a temporal state machine.

Pure logic with no camera or OS calls, so it can be unit-tested with
synthetic landmarks (see tests/test_gestures.py).

Poses
-----
POINT  index extended; middle, ring, pinky folded
V      index + middle extended; ring, pinky folded ("peace sign")
PALM   all four fingers extended (open hand)
OTHER  anything else

States
------
IDLE         no hand in view
TRACKING     hand in view, no control pose
MOVING       POINT confirmed: cursor follows the index fingertip
CLICKING     pinch while MOVING: left button pressed, cursor frozen
DRAGGING     pinch held longer than drag_hold_s: left button held, cursor follows
SCROLLING    V confirmed: vertical hand movement scrolls
RIGHT_CLICK  pinch while SCROLLING: one right click
LISTENING    open palm held for listen_confirm_s: voice input is recorded

Events (GestureUpdate fields): left_down, left_up, right_click, scroll_dy.
A left_down is always followed by a left_up, including when the hand is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import GestureConfig

# MediaPipe landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20


class GestureState(str, Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    MOVING = "MOVING"
    CLICKING = "CLICKING"
    DRAGGING = "DRAGGING"
    SCROLLING = "SCROLLING"
    RIGHT_CLICK = "RIGHT CLICK"
    LISTENING = "LISTENING"


class Pose(str, Enum):
    POINT = "POINT"
    V = "V"
    PALM = "PALM"
    OTHER = "OTHER"


@dataclass
class HandFeatures:
    """Per-frame measurements derived from the 21 landmarks."""
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool
    pinch_ratio: float               # thumb-index distance / hand size
    index_tip: tuple[float, float]   # normalized (x, y)
    middle_tip: tuple[float, float] = (0.0, 0.0)

    @property
    def pose(self) -> Pose:
        if (self.index_extended and self.middle_extended
                and self.ring_extended and self.pinky_extended):
            return Pose.PALM
        if not self.index_extended or self.ring_extended or self.pinky_extended:
            return Pose.OTHER
        return Pose.V if self.middle_extended else Pose.POINT

    @property
    def is_pointing(self) -> bool:
        return self.pose == Pose.POINT


def _xy(landmarks: np.ndarray, aspect: float) -> np.ndarray:
    """Landmark x, y scaled so one unit is equal in both axes (aspect = w / h)."""
    pts = landmarks[:, :2].astype(np.float64).copy()
    pts[:, 0] *= aspect
    return pts


def extract_features(landmarks: np.ndarray, aspect: float, cfg: GestureConfig) -> HandFeatures:
    """Compute finger states and pinch ratio from (21, 3) normalized landmarks."""
    pts = _xy(landmarks, aspect)
    wrist = pts[WRIST]

    def extended(tip: int, pip: int) -> bool:
        d_tip = np.linalg.norm(pts[tip] - wrist)
        d_pip = np.linalg.norm(pts[pip] - wrist)
        return d_tip > d_pip * cfg.finger_extended_ratio

    hand_size = max(float(np.linalg.norm(pts[MIDDLE_MCP] - wrist)), 1e-6)
    pinch = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])) / hand_size

    return HandFeatures(
        index_extended=extended(INDEX_TIP, INDEX_PIP),
        middle_extended=extended(MIDDLE_TIP, MIDDLE_PIP),
        ring_extended=extended(RING_TIP, RING_PIP),
        pinky_extended=extended(PINKY_TIP, PINKY_PIP),
        pinch_ratio=pinch,
        index_tip=(float(landmarks[INDEX_TIP, 0]), float(landmarks[INDEX_TIP, 1])),
        middle_tip=(float(landmarks[MIDDLE_TIP, 0]), float(landmarks[MIDDLE_TIP, 1])),
    )


class Debouncer:
    """True only after the input has been True continuously for `hold_s` seconds."""

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s
        self._since: float | None = None

    def update(self, value: bool, now: float) -> bool:
        if not value:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return (now - self._since) >= self.hold_s

    def reset(self) -> None:
        self._since = None


class PoseFilter:
    """Reports a pose only after it has been seen continuously for `hold_s`."""

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s
        self.confirmed = Pose.OTHER
        self._candidate = Pose.OTHER
        self._since = 0.0

    def update(self, pose: Pose, now: float) -> Pose:
        if pose != self._candidate:
            self._candidate, self._since = pose, now
        if now - self._since >= self.hold_s:
            self.confirmed = self._candidate
        return self.confirmed

    def reset(self) -> None:
        self.confirmed = self._candidate = Pose.OTHER
        self._since = 0.0


@dataclass
class GestureUpdate:
    state: GestureState
    features: HandFeatures | None
    pinch_raw: bool = False      # hysteresis output before temporal confirmation
    left_down: bool = False      # press the left button this frame
    left_up: bool = False        # release the left button this frame
    right_click: bool = False    # one right click this frame
    scroll_dy: float = 0.0       # normalized vertical hand movement to scroll by

    @property
    def click(self) -> bool:
        """Backwards-compatible name: a left click starts this frame."""
        return self.left_down


_POSE_STATE = {Pose.POINT: GestureState.MOVING, Pose.V: GestureState.SCROLLING}


class GestureStateMachine:
    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self.state = GestureState.IDLE
        self._pinch_raw = False
        self._pinch_armed = False      # pinch must start inside a control state
        self._button_down = False
        self._pinch_start_t = 0.0
        self._last_click_t = -1e9
        self._last_hand_t: float | None = None
        self._last_features: HandFeatures | None = None
        self._scroll_y: float | None = None
        self._pose = PoseFilter(cfg.pose_confirm_s)
        self._pinch = Debouncer(cfg.pinch_confirm_s)
        self._release = Debouncer(cfg.release_confirm_s)
        self._palm = Debouncer(cfg.listen_confirm_s)

    @property
    def button_down(self) -> bool:
        return self._button_down

    # -- helpers -----------------------------------------------------------
    def _update_pinch_hysteresis(self, ratio: float) -> bool:
        if self._pinch_raw:
            if ratio > self.cfg.pinch_off_ratio:
                self._pinch_raw = False
        elif ratio < self.cfg.pinch_on_ratio:
            self._pinch_raw = True
        return self._pinch_raw

    def _cooldown_ok(self, now: float) -> bool:
        return now - self._last_click_t >= self.cfg.click_cooldown_s

    def _enter(self, state: GestureState) -> None:
        if state == GestureState.SCROLLING and self.state != GestureState.RIGHT_CLICK:
            self._scroll_y = None
        if state in (GestureState.MOVING, GestureState.SCROLLING):
            # A pinch already closed on entry must be released before it counts.
            self._pinch_armed = not self._pinch_raw
            self._pinch.reset()
        self.state = state

    def _scroll_step(self, f: HandFeatures) -> float:
        y = (f.index_tip[1] + f.middle_tip[1]) / 2
        if self._scroll_y is None:
            self._scroll_y = y
            return 0.0
        prev = self._scroll_y
        self._scroll_y = prev + self.cfg.scroll_smoothing * (y - prev)
        return self._scroll_y - prev

    def reset(self) -> GestureUpdate:
        """Go to IDLE, releasing the left button if it was held."""
        release = self._button_down
        self._button_down = False
        self.state = GestureState.IDLE
        self._pinch_raw = self._pinch_armed = False
        self._last_features = None
        self._scroll_y = None
        self._pose.reset()
        self._pinch.reset()
        self._release.reset()
        self._palm.reset()
        return GestureUpdate(self.state, None, left_up=release)

    # -- main update -------------------------------------------------------
    def update(self, features: HandFeatures | None, now: float) -> GestureUpdate:
        # Hand presence with a short grace period for detection dropouts.
        if features is None:
            if self._last_hand_t is None or now - self._last_hand_t > self.cfg.hand_lost_grace_s:
                return self.reset()
            # Within grace: hold the current state, fire nothing.
            return GestureUpdate(self.state, self._last_features, self._pinch_raw)

        self._last_hand_t = now
        self._last_features = features
        pinch_raw = self._update_pinch_hysteresis(features.pinch_ratio)
        if not pinch_raw:
            self._pinch_armed = True
        pose = self._pose.update(features.pose, now)
        pinch_ok = self._pinch_armed and self._pinch.update(pinch_raw, now)
        release_ok = self._release.update(not pinch_raw, now)
        palm_ok = self._palm.update(features.pose == Pose.PALM, now)

        upd = GestureUpdate(self.state, features, pinch_raw)
        S = GestureState

        # An open palm starts listening from any state without a held button.
        if palm_ok and self.state in (S.IDLE, S.TRACKING, S.MOVING, S.SCROLLING):
            self._enter(S.LISTENING)

        elif self.state == S.LISTENING:
            if pose != Pose.PALM:
                self._palm.reset()
                self._enter(_POSE_STATE.get(pose, S.TRACKING))

        elif self.state in (S.IDLE, S.TRACKING):
            self._enter(_POSE_STATE.get(pose, S.TRACKING))

        elif self.state == S.MOVING:
            # Pinching bends the index finger, so a pinch beats the pose check.
            if pinch_ok:
                self._enter(S.CLICKING)
                self._pinch_start_t = now
                self._release.reset()
                if self._cooldown_ok(now):
                    upd.left_down = self._button_down = True
                    self._last_click_t = now
            elif not pinch_raw and pose != Pose.POINT:
                self._enter(_POSE_STATE.get(pose, S.TRACKING))

        elif self.state in (S.CLICKING, S.DRAGGING):
            if release_ok:
                upd.left_up = self._button_down
                self._button_down = False
                self._enter(S.MOVING)
            elif (self.state == S.CLICKING and self._button_down
                  and now - self._pinch_start_t >= self.cfg.drag_hold_s):
                self._enter(S.DRAGGING)

        elif self.state == S.SCROLLING:
            dy = self._scroll_step(features)
            if pinch_ok:
                self._enter(S.RIGHT_CLICK)
                self._release.reset()
                if self._cooldown_ok(now):
                    upd.right_click = True
                    self._last_click_t = now
            elif not pinch_raw and pose != Pose.V:
                self._enter(_POSE_STATE.get(pose, S.TRACKING))
            elif not pinch_raw:
                upd.scroll_dy = dy

        elif self.state == S.RIGHT_CLICK:
            self._scroll_step(features)   # keep the reference fresh, no scrolling
            if release_ok:
                self._enter(S.SCROLLING)

        upd.state = self.state
        return upd
