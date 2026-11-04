"""Runs parsed voice commands on macOS.

Safety: every action except "enable" requires mouse control to be ON, and
PyAutoGUI's fail-safe is checked first. Apps and URLs are opened with
`open` using an argument list (no shell), so spoken text is never executed.
"""

from __future__ import annotations

import logging
import subprocess
import time
import urllib.parse
from pathlib import Path

from ..config import VoiceConfig
from .commands import Command

log = logging.getLogger(__name__)

APP_DIRS = ("/Applications", "/System/Applications", "/System/Applications/Utilities",
            str(Path.home() / "Applications"))


def installed_apps() -> list[str]:
    names: set[str] = set()
    for d in APP_DIRS:
        p = Path(d)
        if p.is_dir():
            names.update(a.stem for a in p.glob("*.app"))
    return sorted(names)


def type_unicode(text: str, chunk: int = 16) -> None:
    """Type any text (including emoji / accented letters) via Quartz key events."""
    import Quartz
    for i in range(0, len(text), chunk):
        part = text[i:i + chunk]
        for down in (True, False):
            ev = Quartz.CGEventCreateKeyboardEvent(None, 0, down)
            Quartz.CGEventKeyboardSetUnicodeString(ev, len(part.encode("utf-16-le")) // 2, part)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.01)


def _osascript(script: str) -> None:
    subprocess.run(["osascript", "-e", script], check=False, timeout=5,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class CommandExecutor:
    """Executes Commands. Returns (ok, message) for HUD feedback."""

    def __init__(self, cfg: VoiceConfig, mouse) -> None:
        import pyautogui
        self.cfg = cfg
        self.mouse = mouse        # MouseController
        self._gui = pyautogui

    def execute(self, cmd: Command, control_enabled: bool) -> tuple[bool, str]:
        if cmd.action == "unknown":
            return False, "not understood"
        if cmd.action == "enable":
            return True, "control ON"
        if cmd.action == "disable":
            return True, "control OFF"
        if cmd.action in ("record_start", "record_stop"):
            return True, "handled by the app"
        if not control_enabled:
            return False, "control is OFF - press SPACE or say 'enable control'"

        self.mouse.fail_safe_check()     # raises pyautogui.FailSafeException
        a, arg = cmd.action, cmd.arg

        if a == "click":
            self.mouse.click()
        elif a == "double_click":
            self.mouse.click()
            self.mouse.click()           # click count 2 -> real double-click
        elif a == "right_click":
            self.mouse.right_click()
        elif a == "scroll":
            direction, factor = arg
            px = int(self.cfg.scroll_step_px * factor)
            # positive wheel values move content down (= scroll up)
            self.mouse.scroll_pixels(px if direction == "up" else -px)
        elif a == "hotkey":
            self._gui.hotkey(*arg, interval=0.02)
        elif a == "type":
            text, app = arg if isinstance(arg, tuple) else (arg, None)
            if not text:
                return False, "nothing to type"
            if app:
                if subprocess.run(["open", "-a", app], capture_output=True, timeout=10).returncode:
                    return False, f"app '{app}' not found"
                time.sleep(0.8)          # let the app come to the front
            type_unicode(str(text))
        elif a == "search":
            url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(str(arg))
            subprocess.Popen(["open", url])
        elif a == "open_url":
            url, app = arg if isinstance(arg, tuple) else (arg, None)
            subprocess.Popen(["open", "-a", app, url] if app else ["open", url])
        elif a == "open_app":
            r = subprocess.run(["open", "-a", str(arg)], capture_output=True, timeout=10)
            if r.returncode != 0:
                return False, f"app '{arg}' not found"
        elif a == "quit_app":
            safe = str(arg).replace('"', "")
            _osascript(f'tell application "{safe}" to quit')
        elif a == "volume":
            step = self.cfg.volume_step
            script = {
                "up": f"set volume output volume ((output volume of (get volume settings)) + {step})",
                "down": f"set volume output volume ((output volume of (get volume settings)) - {step})",
                "mute": "set volume with output muted",
                "unmute": "set volume without output muted",
            }[arg]
            _osascript(script)
        else:
            return False, f"unsupported action {a}"
        log.info("Voice command executed: %s", cmd)
        return True, "done"
