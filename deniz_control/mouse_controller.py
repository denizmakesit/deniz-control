"""Cursor movement, clicks, drag and scroll with smoothing and safety.

On macOS, events are posted with Quartz (installed with PyAutoGUI) rather than
PyAutoGUI's own helpers, because PyAutoGUI:
  * sends "mouse moved" instead of "mouse dragged" while a button is held, so
    drags fail in many apps, and
  * never sets the click count, so double-clicks are not recognised.
PyAutoGUI's fail-safe is still checked before every action.
"""

from __future__ import annotations

import ctypes
import logging
import math
import sys
import time
from typing import Protocol

from .config import MouseConfig

log = logging.getLogger(__name__)


def accessibility_trusted() -> bool | None:
    """Return whether this process may control the mouse on macOS.

    Uses AXIsProcessTrusted from the system ApplicationServices framework via
    ctypes (no extra dependency). Returns None if it cannot be determined.
    """
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(lib.AXIsProcessTrusted())
    except OSError:
        return None


def map_to_screen(norm_x: float, norm_y: float, screen_w: int, screen_h: int,
                  cfg: MouseConfig) -> tuple[float, float]:
    """Map a normalized camera point to screen coordinates.

    The active region is stretched to the whole screen, sensitivity scales
    movement around the region centre, and the result is clamped inside the
    screen-edge margin.
    """
    left, top, right, bottom = cfg.active_region
    u = (norm_x - left) / (right - left)
    v = (norm_y - top) / (bottom - top)
    u = 0.5 + (u - 0.5) * cfg.sensitivity
    v = 0.5 + (v - 0.5) * cfg.sensitivity
    return clamp_to_screen(u * (screen_w - 1), v * (screen_h - 1), screen_w, screen_h, cfg)


def clamp_to_screen(x: float, y: float, screen_w: int, screen_h: int,
                    cfg: MouseConfig) -> tuple[float, float]:
    m = cfg.screen_edge_margin_px
    return (min(max(x, m), screen_w - 1 - m), min(max(y, m), screen_h - 1 - m))


class ExponentialSmoother:
    """2D exponential moving average with a small deadzone."""

    def __init__(self, alpha: float, deadzone: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.deadzone = deadzone
        self.value: tuple[float, float] | None = None

    def reset(self) -> None:
        self.value = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self.value is None:
            self.value = (x, y)
            return self.value
        px, py = self.value
        # Hold still when the target is already within the deadzone; otherwise
        # take an EMA step towards it.
        if abs(x - px) < self.deadzone and abs(y - py) < self.deadzone:
            return self.value
        self.value = (px + self.alpha * (x - px), py + self.alpha * (y - py))
        return self.value


# -- OS backends ------------------------------------------------------------
class MouseBackend(Protocol):
    def size(self) -> tuple[int, int]: ...
    def position(self) -> tuple[int, int]: ...
    def fail_safe_check(self) -> None: ...
    def move(self, x: int, y: int, dragging: bool) -> None: ...
    def left(self, down: bool, x: int, y: int, click_count: int) -> None: ...
    def right_click(self, x: int, y: int) -> None: ...
    def scroll(self, pixels: int) -> None: ...
    def double_click_interval(self) -> float: ...


class MacBackend:
    """Posts Quartz events; uses PyAutoGUI for size, position and fail-safe."""

    def __init__(self) -> None:
        import pyautogui
        import Quartz
        self._gui = pyautogui
        self._q = Quartz
        pyautogui.FAILSAFE = True   # explicit: never disable the fail-safe
        pyautogui.PAUSE = 0         # default 0.1 s sleep per call would kill FPS

    def size(self) -> tuple[int, int]:
        w, h = self._gui.size()
        return int(w), int(h)

    def position(self) -> tuple[int, int]:
        p = self._gui.position()
        return int(p[0]), int(p[1])

    def fail_safe_check(self) -> None:
        self._gui.failSafeCheck()   # raises pyautogui.FailSafeException in a corner

    def _post(self, ev_type, x, y, button, click_count: int | None = None) -> None:
        q = self._q
        ev = q.CGEventCreateMouseEvent(None, ev_type, (x, y), button)
        if click_count is not None:
            q.CGEventSetIntegerValueField(ev, q.kCGMouseEventClickState, click_count)
        q.CGEventPost(q.kCGHIDEventTap, ev)

    def move(self, x: int, y: int, dragging: bool) -> None:
        q = self._q
        if dragging:
            self._post(q.kCGEventLeftMouseDragged, x, y, q.kCGMouseButtonLeft)
        else:
            self._post(q.kCGEventMouseMoved, x, y, q.kCGMouseButtonLeft)

    def left(self, down: bool, x: int, y: int, click_count: int) -> None:
        q = self._q
        ev = q.kCGEventLeftMouseDown if down else q.kCGEventLeftMouseUp
        self._post(ev, x, y, q.kCGMouseButtonLeft, click_count)

    def right_click(self, x: int, y: int) -> None:
        q = self._q
        self._post(q.kCGEventRightMouseDown, x, y, q.kCGMouseButtonRight, 1)
        self._post(q.kCGEventRightMouseUp, x, y, q.kCGMouseButtonRight, 1)

    def scroll(self, pixels: int) -> None:
        q = self._q
        # Pixel units give smooth, trackpad-like scrolling. Split big steps.
        while pixels:
            step = max(-120, min(120, pixels))
            ev = q.CGEventCreateScrollWheelEvent(None, q.kCGScrollEventUnitPixel, 1, step)
            q.CGEventPost(q.kCGHIDEventTap, ev)
            pixels -= step

    def double_click_interval(self) -> float:
        try:
            from AppKit import NSEvent
            return float(NSEvent.doubleClickInterval())
        except Exception:
            return 0.5


# -- controller --------------------------------------------------------------
class MouseController:
    """High-level mouse actions driven by gesture events.

    Safety:
      * The fail-safe is checked before every gesture action: putting the real
        cursor in a screen corner raises pyautogui.FailSafeException, which the
        app catches to disable control.
      * Our own movements are clamped away from the edges, so we never trigger
        the fail-safe ourselves.
      * release_all() always lifts a held button; the app calls it whenever
        control stops (toggle off, hand lost, fail-safe, exit, crash).
    """

    def __init__(self, cfg: MouseConfig, backend: MouseBackend | None = None) -> None:
        self.cfg = cfg
        self._os = backend or MacBackend()
        self.screen_w, self.screen_h = self._os.size()
        self._smoother = ExponentialSmoother(cfg.smoothing_alpha, cfg.deadzone_px)
        self._dbl_interval = cfg.double_click_interval_s or self._os.double_click_interval()
        self._offset = (0.0, 0.0)          # used while dragging (see anchor())
        self._last_sent: tuple[int, int] | None = None
        self._last_down: tuple[float, tuple[int, int], int] | None = None
        self._click_count = 1
        self._scroll_acc = 0.0
        self.left_is_down = False
        self.cursor: tuple[int, int] | None = None   # last position we set
        self.clicks = 0

    # -- housekeeping ------------------------------------------------------
    def reset(self) -> None:
        """Forget cursor smoothing history (call when the cursor stops being driven).

        The scroll accumulator is deliberately kept: slow scrolls add up over
        many frames while the cursor is idle.
        """
        self._smoother.reset()
        self._last_sent = None
        self._offset = (0.0, 0.0)

    def fail_safe_check(self) -> None:
        self._os.fail_safe_check()

    def release_all(self) -> None:
        """Lift the left button if we are holding it. Never raises."""
        if not self.left_is_down:
            return
        try:
            x, y = self._os.position()
            self._os.left(False, x, y, self._click_count)
        except Exception:   # noqa: BLE001 - must not fail during cleanup
            log.exception("Failed to release mouse button")
        self.left_is_down = False
        log.info("Mouse button released")

    # -- movement ----------------------------------------------------------
    def _target(self, norm_x: float, norm_y: float) -> tuple[float, float]:
        x, y = map_to_screen(norm_x, norm_y, self.screen_w, self.screen_h, self.cfg)
        return clamp_to_screen(x + self._offset[0], y + self._offset[1],
                               self.screen_w, self.screen_h, self.cfg)

    def anchor(self, norm_x: float, norm_y: float) -> None:
        """Make the current finger position map to the current cursor.

        Used when a drag starts: closing the pinch shifts the fingertip, and
        without this the cursor would jump by that amount.
        """
        if self.cursor is None:
            return
        x, y = map_to_screen(norm_x, norm_y, self.screen_w, self.screen_h, self.cfg)
        self._offset = (self.cursor[0] - x, self.cursor[1] - y)

    def clear_anchor(self) -> None:
        self._offset = (0.0, 0.0)

    def move_towards(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        sx, sy = self._smoother.update(*self._target(norm_x, norm_y))
        pos = (int(round(sx)), int(round(sy)))
        if pos != self._last_sent:        # skip redundant OS calls
            self._os.move(*pos, dragging=self.left_is_down)
            self._last_sent = pos
        self.cursor = pos
        return pos

    # -- buttons -----------------------------------------------------------
    def left_down(self) -> None:
        x, y = self._os.position()
        now = time.monotonic()
        if self._last_down is not None:
            t, (lx, ly), n = self._last_down
            near = math.hypot(x - lx, y - ly) <= self.cfg.double_click_radius_px
            self._click_count = n + 1 if (now - t <= self._dbl_interval and near) else 1
        else:
            self._click_count = 1
        self._last_down = (now, (x, y), self._click_count)
        self._os.left(True, x, y, self._click_count)
        self.left_is_down = True
        self.clicks += 1
        log.info("Left down #%d at (%d, %d)%s", self.clicks, x, y,
                 f" [x{self._click_count}]" if self._click_count > 1 else "")

    def left_up(self) -> None:
        if not self.left_is_down:
            return
        x, y = self._os.position()
        self._os.left(False, x, y, self._click_count)
        self.left_is_down = False

    def click(self) -> None:
        """A complete left click (down + up)."""
        self.left_down()
        self.left_up()

    def right_click(self) -> None:
        x, y = self._os.position()
        self._os.right_click(x, y)
        self.clicks += 1
        log.info("Right click #%d at (%d, %d)", self.clicks, x, y)

    # -- scrolling ---------------------------------------------------------
    def scroll_pixels(self, pixels: int) -> None:
        """Scroll a fixed distance (positive = content moves down / scroll up)."""
        self._os.scroll(int(pixels))

    def scroll(self, dy_norm: float) -> int:
        """Scroll by a normalized vertical hand movement. Returns pixels sent."""
        sign = -1.0 if self.cfg.scroll_invert else 1.0
        self._scroll_acc += sign * dy_norm * self.screen_h * self.cfg.scroll_speed
        pixels = int(self._scroll_acc)     # keep the fractional remainder
        if pixels:
            self._scroll_acc -= pixels
            self._os.scroll(pixels)
        return pixels
