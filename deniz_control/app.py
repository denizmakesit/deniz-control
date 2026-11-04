"""Main application loop: camera -> hand tracker -> gestures -> mouse -> HUD."""

from __future__ import annotations

import logging
import time

import cv2

from .camera import Camera, CameraError
from .config import AppConfig
from .gestures import GestureState, GestureStateMachine, extract_features
from .hand_tracker import HandTracker
from .overlay import HudInfo, Overlay
from .recorder import SessionRecorder

log = logging.getLogger(__name__)

KEY_ESC = 27
KEY_SPACE = 32


ACTIVE_STATES = (GestureState.MOVING, GestureState.CLICKING, GestureState.DRAGGING)


def act(mouse, upd, prev_state: GestureState, cfg: AppConfig) -> None:
    """Turn one gesture update into mouse actions."""
    mouse.fail_safe_check()
    f = upd.features
    S = GestureState

    if upd.state == S.DRAGGING and prev_state != S.DRAGGING and f is not None:
        mouse.anchor(*f.index_tip)        # no jump when the drag starts
    elif upd.state != S.DRAGGING and prev_state == S.DRAGGING:
        mouse.clear_anchor()

    if upd.left_down:
        mouse.left_down()
    if f is not None:
        if upd.state == S.DRAGGING:
            mouse.move_towards(*f.index_tip)
        elif upd.state == S.MOVING and not (cfg.mouse.freeze_while_pinching and upd.pinch_raw):
            mouse.move_towards(*f.index_tip)
    if upd.left_up:
        mouse.left_up()
    if upd.right_click:
        mouse.right_click()
    if upd.scroll_dy:
        mouse.scroll(upd.scroll_dy)


class FPSCounter:
    """Exponentially smoothed frames-per-second."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.fps = 0.0
        self._last: float | None = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else self.fps + self.alpha * (inst - self.fps)
        self._last = now
        return self.fps


def window_closed(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def run(cfg: AppConfig | None = None, enable_mouse: bool = True,
        max_frames: int | None = None, enable_voice: bool | None = None) -> int:
    """Run the app. enable_mouse=False gives a preview-only (tracking test) mode.

    Voice is on when enable_voice is True, or when it is None and
    cfg.voice.enabled is set. In preview-only mode voice commands are shown
    but never executed.

    Returns a process exit code.
    """
    cfg = cfg or AppConfig()
    mouse = None
    notice = ""

    if enable_mouse:
        import pyautogui
        from .mouse_controller import MouseController, accessibility_trusted

        mouse = MouseController(cfg.mouse)
        trusted = accessibility_trusted()
        if trusted is False:
            notice = "Accessibility permission missing - see README"
            log.warning("This app is not trusted for Accessibility; mouse control "
                        "will not work until you grant it (see README).")
        log.info("Screen size: %dx%d points", mouse.screen_w, mouse.screen_h)

    voice = None
    if enable_voice if enable_voice is not None else cfg.voice.enabled:
        try:
            from .voice.controller import VoiceController
            voice = VoiceController(cfg.voice, mouse)
        except Exception as exc:  # noqa: BLE001 - voice is optional
            log.error("Voice control disabled: %s", exc)

    control_enabled = False   # always start disabled
    manual_listen = False     # V key push-to-talk toggle
    rec = SessionRecorder(cfg.record)
    record_request: bool | None = None   # True = start, False = stop (from key or voice)
    machine = GestureStateMachine(cfg.gesture)
    overlay = Overlay(cfg.ui, cfg.gesture, cfg.mouse if enable_mouse else None)
    fps = FPSCounter()
    frames = 0
    prev_state = GestureState.IDLE

    cam = Camera(cfg.camera)
    try:
        cam.open()
    except CameraError as exc:
        log.error("%s", exc)
        return 1

    tracker = HandTracker(cfg.tracker)
    cv2.namedWindow(cfg.ui.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.ui.window_name, 960, 540)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                if cv2.waitKey(5) & 0xFF in (ord("q"), KEY_ESC):
                    break
                continue

            h, w = frame.shape[:2]
            hand = tracker.detect(frame)
            features = extract_features(hand.landmarks, w / h, cfg.gesture) if hand else None
            upd = machine.update(features, time.monotonic())

            # --- voice ------------------------------------------------------
            heard = None
            if voice is not None:
                voice.update(upd.state == GestureState.LISTENING or manual_listen)
                if voice.recorder.full:
                    manual_listen = False
                heard = voice.poll()
                if heard is not None and heard.command.action == "record_start":
                    record_request = True
                elif heard is not None and heard.command.action == "record_stop":
                    record_request = False
                if heard is not None and mouse is not None:
                    if heard.command.action == "enable" and not control_enabled:
                        control_enabled = True
                        mouse.reset()
                        log.info("Mouse control ENABLED (voice)")
                    elif heard.command.action == "disable" and control_enabled:
                        control_enabled = False
                        mouse.release_all()
                        mouse.reset()
                        log.info("Mouse control DISABLED (voice)")

            # --- act on the computer (only when explicitly enabled) --------
            if mouse is not None and control_enabled:
                try:
                    act(mouse, upd, prev_state, cfg)
                    if heard is not None:
                        voice.execute(heard, control_enabled)
                        heard = None
                except pyautogui.FailSafeException:
                    mouse.release_all()
                    control_enabled = False
                    mouse.reset()
                    notice = "FAIL-SAFE: control disabled (press SPACE)"
                    log.warning("PyAutoGUI fail-safe triggered; mouse control disabled")
            elif mouse is not None:
                mouse.release_all()
            if heard is not None and voice is not None:
                voice.execute(heard, False)   # only gives feedback, nothing runs
            if mouse is not None and upd.state not in ACTIVE_STATES:
                mouse.reset()   # next MOVING starts fresh instead of gliding from old position
            prev_state = upd.state

            info = HudInfo(
                fps=fps.tick(),
                state=upd.state,
                control_enabled=control_enabled,
                cursor=mouse.cursor if mouse else None,
                hand=hand,
                pinch_ratio=features.pinch_ratio if features else None,
                clicks=mouse.clicks if mouse else 0,
                notice=notice,
                mouse_enabled_allowed=mouse is not None,
                left_down=upd.left_down,
                right_click=upd.right_click,
                scroll_dy=upd.scroll_dy,
                voice=voice,
            )
            hud = overlay.draw(frame, info)

            # --- recording: write the HUD frame, then add the REC badge ----
            if record_request is True and not rec.recording:
                rec.start((w, h))
            elif record_request is False and rec.recording:
                rec.stop()
            record_request = None
            rec.add_frame(hud, frame,
                          captured_at=cam.last_frame_time - cfg.record.camera_latency_s)
            overlay.draw_record_badge(hud, rec)
            cv2.imshow(cfg.ui.window_name, hud)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), KEY_ESC):
                break
            if key in (ord("h"), ord("H")):
                overlay.clean_mode = not overlay.clean_mode
            if key in (ord("v"), ord("V")) and voice is not None:
                manual_listen = not manual_listen
            if key in (ord("r"), ord("R")):
                record_request = not rec.recording
            if key == KEY_SPACE and mouse is not None:
                control_enabled = not control_enabled
                mouse.release_all()
                mouse.reset()
                if control_enabled:
                    notice = notice if "Accessibility" in notice else ""
                log.info("Mouse control %s", "ENABLED" if control_enabled else "DISABLED")
            if window_closed(cfg.ui.window_name):
                break

            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
    except KeyboardInterrupt:
        log.info("Interrupted")
    except CameraError as exc:
        log.error("%s", exc)
        return 1
    finally:
        if mouse is not None:
            mouse.release_all()   # never leave a button held down
        if voice is not None:
            voice.close()
        if rec.recording:
            log.info("Finishing the recording before exit...")
            rec.stop(wait=True)
        rec.wait()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)   # let macOS actually close the window
        log.info("Shut down cleanly after %d frames", frames)
    return 0
