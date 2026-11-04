"""Dark-theme OpenCV HUD with cinematic effects.

Effects (fingertip trail, click ripples, scroll arrows) are drawn on a black
layer that is blurred and added on top of the video, which gives a soft glow.
Press H in the app for a clean mode that hides the stats panel (for filming).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .config import GestureConfig, MouseConfig, UIConfig
from .gestures import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, GestureState
from .hand_tracker import HAND_CONNECTIONS, HandObservation

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class HudInfo:
    fps: float
    state: GestureState
    control_enabled: bool
    cursor: tuple[int, int] | None
    hand: HandObservation | None
    pinch_ratio: float | None
    clicks: int
    notice: str = ""            # e.g. permission warnings / fail-safe message
    mouse_enabled_allowed: bool = True
    left_down: bool = False     # events this frame, for effects
    right_click: bool = False
    scroll_dy: float = 0.0
    voice: object = None        # VoiceController or None


class Overlay:
    def __init__(self, ui: UIConfig, gesture: GestureConfig, mouse: MouseConfig | None) -> None:
        self.ui = ui
        self.gesture = gesture
        self.mouse = mouse
        self.clean_mode = False
        self._trail: deque[tuple[float, tuple[int, int], tuple[int, int, int]]] = deque()
        self._ripples: list[tuple[float, tuple[int, int], tuple[int, int, int]]] = []
        self._scroll_vis = 0.0     # smoothed scroll speed for the arrow indicator
        self.state_colors = {
            GestureState.IDLE: ui.muted_bgr,
            GestureState.TRACKING: ui.landmark_color_bgr,
            GestureState.MOVING: ui.accent_on_bgr,
            GestureState.CLICKING: ui.left_click_bgr,
            GestureState.DRAGGING: ui.drag_bgr,
            GestureState.SCROLLING: ui.scroll_bgr,
            GestureState.RIGHT_CLICK: ui.right_click_bgr,
            GestureState.LISTENING: ui.listening_bgr,
        }

    # -- helpers -----------------------------------------------------------
    def _text(self, img, text, org, scale=0.55, color=None, thick=1):
        cv2.putText(img, text, org, FONT, scale, color or self.ui.text_bgr, thick, cv2.LINE_AA)

    def _panel(self, img, x, y, w, h, alpha=0.72):
        x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
        roi = img[y:y2, x:x2]
        bg = np.full_like(roi, self.ui.panel_bgr)
        cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0, dst=roi)

    @staticmethod
    def _scaled(color, k: float) -> tuple[int, int, int]:
        return tuple(int(c * k) for c in color)

    # -- main entry --------------------------------------------------------
    def draw(self, frame: np.ndarray, info: HudInfo) -> np.ndarray:
        h, w = frame.shape[:2]
        now = time.monotonic()
        out = cv2.convertScaleAbs(frame, alpha=self.ui.video_gain, beta=self.ui.video_lift)
        fx = np.zeros_like(out)   # glow layer

        pts = None
        if info.hand is not None:
            pts = [(int(x * w), int(y * h)) for x, y, _ in info.hand.landmarks]

        self._update_effects(info, pts, now)
        self._draw_trail(fx, now)
        self._draw_ripples(fx, now)
        if pts is not None and info.state in (GestureState.SCROLLING, GestureState.RIGHT_CLICK):
            self._draw_scroll_indicator(fx, pts, info)
        if pts is not None and info.voice is not None and info.voice.status == "listening":
            self._draw_listening_rings(fx, pts, info.voice.level, now)

        if self.mouse is not None and not self.clean_mode:
            self._draw_active_region(out, w, h, info.control_enabled)
        if pts is not None:
            self._draw_hand(out, fx, pts, info)

        if self.ui.glow:
            small = cv2.resize(fx, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 3)
            out = cv2.add(out, cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))
        out = cv2.add(out, fx)

        if info.voice is not None:
            self._draw_voice(out, w, h, info.voice, now)
        if self.clean_mode:
            self._draw_minimal(out, w, h, info)
        else:
            self._draw_status_panel(out, info)
            self._draw_footer(out, w, h, info)
        return out

    # -- recording badge (drawn on the preview only, never recorded) -------
    def draw_record_badge(self, img, rec) -> None:
        w = img.shape[1]
        now = time.monotonic()
        red = (70, 60, 235)
        if rec.status == "recording":
            t = int(rec.elapsed)
            label = f"REC  {t // 60:02d}:{t % 60:02d}"
            color = red
        elif rec.status == "saving":
            label, color = "SAVING" + "." * (1 + int(now * 3) % 3), self.ui.text_bgr
        elif rec.message and now - rec.message_t < 5:
            label, color = rec.message, self.ui.muted_bgr
        else:
            return
        size = cv2.getTextSize(label, FONT, 0.55, 2)[0]
        x = (w - size[0]) // 2
        self._panel(img, x - 34, 12, size[0] + 52, 34, alpha=0.75)
        if rec.status == "recording" and int(now * 2) % 2 == 0:
            cv2.circle(img, (x - 16, 29), 7, red, -1, cv2.LINE_AA)
        self._text(img, label, (x, 35), 0.55, color, 2)

    # -- voice -------------------------------------------------------------
    def _draw_listening_rings(self, fx, pts, level: float, now: float) -> None:
        """Pulsing rings around the palm that swell with voice loudness."""
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        color = self.ui.listening_bgr
        for i in range(3):
            phase = ((now * 0.9) + i / 3) % 1.0
            r = int(60 + 90 * phase + 40 * level)
            cv2.circle(fx, (cx, cy), r, self._scaled(color, (1 - phase) * 0.9), 2, cv2.LINE_AA)
        cv2.circle(fx, (cx, cy), int(36 + 50 * level), self._scaled(color, 0.6 + 0.4 * level),
                   3, cv2.LINE_AA)

    def _draw_voice(self, img, w, h, voice, now: float) -> None:
        status = voice.status
        cx = w // 2
        base_y = h - 70
        color = self.ui.listening_bgr

        if status == "listening":
            # level meter: mirrored bars from recent mic levels
            levels = list(voice.levels)[-32:]
            n = len(levels)
            bw = 6
            x0 = cx - (n * (bw + 3)) // 2
            self._panel(img, x0 - 16, base_y - 38, n * (bw + 3) + 32, 64, alpha=0.6)
            for i, lv in enumerate(levels):
                bh = int(3 + 26 * lv)
                x = x0 + i * (bw + 3)
                cv2.rectangle(img, (x, base_y - bh // 2 - 8), (x + bw, base_y + bh // 2 - 8),
                              color, -1, cv2.LINE_AA)
            label = f"LISTENING  {voice.record_seconds:4.1f}s"
            size = cv2.getTextSize(label, FONT, 0.45, 1)[0]
            blink = 0.6 + 0.4 * abs(np.sin(now * 4))
            cv2.circle(img, (cx - size[0] // 2 - 12, base_y + 17), 5,
                       self._scaled(color, blink), -1, cv2.LINE_AA)
            self._text(img, label, (cx - size[0] // 2, base_y + 21), 0.45, color, 1)
            return

        if status == "thinking":
            dots = "." * (1 + int(now * 4) % 3)
            label = f"TRANSCRIBING{dots}"
            size = cv2.getTextSize(label, FONT, 0.5, 1)[0]
            self._panel(img, cx - size[0] // 2 - 16, base_y - 16, size[0] + 32, 34, alpha=0.6)
            self._text(img, label, (cx - size[0] // 2, base_y + 6), 0.5, self.ui.text_bgr, 1)
            return

        cap = voice.caption
        if cap is not None and now - cap.t < voice.cfg.feedback_s:
            fade = min(1.0, (voice.cfg.feedback_s - (now - cap.t)) / 0.6)
            heard = f'"{cap.heard}"' if cap.heard else ""
            res_color = self.ui.accent_on_bgr if cap.ok else self.ui.accent_off_bgr
            res = cap.result
            s1 = cv2.getTextSize(heard, FONT, 0.6, 1)[0] if heard else (0, 0)
            s2 = cv2.getTextSize(res, FONT, 0.55, 2)[0]
            gap = 28 if heard else 0
            total = s1[0] + gap + s2[0]
            x = cx - total // 2
            self._panel(img, x - 18, base_y - 22, total + 36, 40, alpha=0.7 * fade)
            if heard:
                self._text(img, heard, (x, base_y + 5), 0.6, self._scaled(self.ui.text_bgr, fade))
                ax = x + s1[0] + 8
                cv2.arrowedLine(img, (ax, base_y), (ax + 14, base_y), self._scaled(self.ui.muted_bgr, fade),
                                1, cv2.LINE_AA, tipLength=0.4)
            self._text(img, res, (x + s1[0] + gap, base_y + 5), 0.55,
                       self._scaled(res_color, fade), 2)

    # -- effects -----------------------------------------------------------
    def _update_effects(self, info: HudInfo, pts, now: float) -> None:
        if pts is not None:
            tip = pts[INDEX_TIP]
            if info.left_down:
                self._ripples.append((now, tip, self.ui.left_click_bgr))
            if info.right_click:
                mid = ((pts[INDEX_TIP][0] + pts[MIDDLE_TIP][0]) // 2,
                       (pts[INDEX_TIP][1] + pts[MIDDLE_TIP][1]) // 2)
                self._ripples.append((now, mid, self.ui.right_click_bgr))
            if info.state in (GestureState.MOVING, GestureState.DRAGGING, GestureState.CLICKING):
                color = self.ui.drag_bgr if info.state == GestureState.DRAGGING \
                    else self.ui.landmark_color_bgr
                self._trail.append((now, tip, color))

        while self._trail and now - self._trail[0][0] > self.ui.trail_s:
            self._trail.popleft()
        self._ripples = [r for r in self._ripples if now - r[0] <= self.ui.ripple_s]
        self._scroll_vis += 0.3 * (info.scroll_dy - self._scroll_vis)

    def _draw_trail(self, fx, now: float) -> None:
        pts = list(self._trail)
        for (t0, p0, _), (t1, p1, c1) in zip(pts, pts[1:]):
            k = 1.0 - (now - t1) / self.ui.trail_s       # 1 = newest
            if k <= 0:
                continue
            cv2.line(fx, p0, p1, self._scaled(c1, k), max(1, int(1 + 5 * k)), cv2.LINE_AA)

    def _draw_ripples(self, fx, now: float) -> None:
        for t0, center, color in self._ripples:
            p = (now - t0) / self.ui.ripple_s            # 0 -> 1
            ease = 1 - (1 - p) ** 3
            k = 1.0 - p
            cv2.circle(fx, center, int(10 + 70 * ease), self._scaled(color, k),
                       max(1, int(4 * k + 1)), cv2.LINE_AA)
            if p < 0.5:   # bright core flash
                cv2.circle(fx, center, int(8 * (1 - 2 * p)) + 2,
                           self._scaled(color, 1 - 2 * p), -1, cv2.LINE_AA)

    def _draw_scroll_indicator(self, fx, pts, info: HudInfo) -> None:
        x = max(pts[INDEX_TIP][0], pts[MIDDLE_TIP][0]) + 45
        y = (pts[INDEX_TIP][1] + pts[MIDDLE_TIP][1]) // 2
        speed = max(-1.0, min(1.0, self._scroll_vis * 60))
        color = self.ui.scroll_bgr
        # rail
        cv2.line(fx, (x, y - 50), (x, y + 50), self._scaled(color, 0.35), 2, cv2.LINE_AA)
        # moving chevrons in the scroll direction
        if abs(speed) > 0.05:
            d = 1 if speed > 0 else -1
            phase = (time.monotonic() * 3) % 1.0
            for i in range(3):
                off = int(((i / 3 + phase) % 1.0) * 80 - 40) * d
                k = 0.4 + 0.6 * abs(speed)
                cy = y + off
                cv2.line(fx, (x - 9, cy - 6 * d), (x, cy), self._scaled(color, k), 2, cv2.LINE_AA)
                cv2.line(fx, (x + 9, cy - 6 * d), (x, cy), self._scaled(color, k), 2, cv2.LINE_AA)
        else:
            cv2.circle(fx, (x, y), 4, color, -1, cv2.LINE_AA)

    # -- static HUD --------------------------------------------------------
    def _draw_active_region(self, img, w, h, enabled):
        l, t, r, b = self.mouse.active_region
        color = self.ui.accent_on_bgr if enabled else self.ui.muted_bgr
        p1, p2 = (int(l * w), int(t * h)), (int(r * w), int(b * h))
        L = 22   # corner brackets instead of a full box: reads cleaner on video
        for (cx, cy), (dx, dy) in [((p1[0], p1[1]), (1, 1)), ((p2[0], p1[1]), (-1, 1)),
                                   ((p1[0], p2[1]), (1, -1)), ((p2[0], p2[1]), (-1, -1))]:
            cv2.line(img, (cx, cy), (cx + dx * L, cy), color, 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * L), color, 2, cv2.LINE_AA)

    def _draw_hand(self, img, fx, pts, info: HudInfo):
        cyan = self.ui.landmark_color_bgr
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], cyan, 2, cv2.LINE_AA)
            cv2.line(fx, pts[a], pts[b], self._scaled(cyan, 0.35), 3, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = 6 if i in (THUMB_TIP, INDEX_TIP) else 4
            cv2.circle(img, p, r, cyan, -1, cv2.LINE_AA)
            cv2.circle(img, p, r, (40, 30, 20), 1, cv2.LINE_AA)

        state_color = self.state_colors[info.state]
        if info.pinch_ratio is not None:
            active = info.state in (GestureState.CLICKING, GestureState.DRAGGING,
                                    GestureState.RIGHT_CLICK)
            color = state_color if active else self.ui.muted_bgr
            cv2.line(img, pts[THUMB_TIP], pts[INDEX_TIP], color, 2 if active else 1, cv2.LINE_AA)

        if info.state in (GestureState.MOVING, GestureState.CLICKING, GestureState.DRAGGING):
            # reticle: ring + ticks; shrinks while pressed
            pressed = info.state != GestureState.MOVING
            r = 12 if pressed else 18
            c = pts[INDEX_TIP]
            cv2.circle(fx, c, r, state_color, 2, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(fx, (c[0] + dx * (r + 4), c[1] + dy * (r + 4)),
                         (c[0] + dx * (r + 10), c[1] + dy * (r + 10)), state_color, 2, cv2.LINE_AA)

        # floating state tag next to the hand
        if info.state not in (GestureState.IDLE, GestureState.TRACKING):
            wx, wy = pts[0]
            self._text(img, info.state.value, (wx + 18, wy + 26), 0.55, state_color, 2)

    def _draw_status_panel(self, img, info: HudInfo):
        x, y, pw, ph = 16, 16, 300, 210
        self._panel(img, x, y, pw, ph)
        self._text(img, "DENIZ CONTROL", (x + 14, y + 28), 0.65, self.ui.landmark_color_bgr, 2)
        self._text(img, "TOUCHLESS HAND CONTROL", (x + 14, y + 48), 0.4, self.ui.muted_bgr)

        state_color = self.state_colors[info.state]
        cv2.circle(img, (x + 22, y + 74), 6, state_color, -1, cv2.LINE_AA)
        self._text(img, info.state.value, (x + 36, y + 80), 0.6, state_color, 2)

        if info.control_enabled:
            ctrl, ctrl_color = "MOUSE CONTROL: ON", self.ui.accent_on_bgr
        else:
            ctrl, ctrl_color = "MOUSE CONTROL: OFF", self.ui.accent_off_bgr
        self._text(img, ctrl, (x + 14, y + 108), 0.5, ctrl_color, 1)

        conf = f"{info.hand.score:.2f}" if info.hand else "--"
        pinch = f"{info.pinch_ratio:.2f}" if info.pinch_ratio is not None else "--"
        cursor = f"{info.cursor[0]}, {info.cursor[1]}" if info.cursor else "--"
        rows = [
            ("FPS", f"{info.fps:5.1f}"),
            ("HAND CONF", conf),
            ("PINCH", f"{pinch}  (on<{self.gesture.pinch_on_ratio:.2f})"),
            ("CURSOR", cursor),
            ("CLICKS", str(info.clicks)),
            ("VOICE", info.voice.status.upper() if info.voice is not None else "OFF"),
        ]
        for i, (k, v) in enumerate(rows):
            yy = y + 132 + i * 14
            self._text(img, k, (x + 14, yy), 0.38, self.ui.muted_bgr)
            self._text(img, v, (x + 110, yy), 0.38, self.ui.text_bgr)

    def _draw_minimal(self, img, w, h, info: HudInfo):
        """Clean mode: a small title and status dot only."""
        self._text(img, "DENIZ CONTROL", (20, 36), 0.7, self.ui.landmark_color_bgr, 2)
        dot = self.ui.accent_on_bgr if info.control_enabled else self.ui.accent_off_bgr
        cv2.circle(img, (w - 26, 28), 7, dot, -1, cv2.LINE_AA)
        if info.notice:
            self._text(img, info.notice, (20, h - 18), 0.5, self.ui.accent_off_bgr)

    def _draw_footer(self, img, w, h, info: HudInfo):
        self._panel(img, 0, h - 34, w, 34, alpha=0.8)
        talk = "  V talk " if info.voice is not None else ""
        hint = f"SPACE control  {talk} R record   H clean view   Q / ESC quit"
        if not info.mouse_enabled_allowed:
            hint = f"{talk} H clean view   Q / ESC quit   (preview only)"
        self._text(img, hint, (16, h - 12), 0.5, self.ui.muted_bgr)
        if info.notice:
            size = cv2.getTextSize(info.notice, FONT, 0.5, 1)[0]
            self._text(img, info.notice, (w - size[0] - 16, h - 12), 0.5, self.ui.accent_off_bgr)
