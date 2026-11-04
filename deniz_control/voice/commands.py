"""English voice command parser. Pure text -> Command, no side effects.

Matching order:
  1. fixed phrases ("double click", "new tab"), exact then fuzzy
  2. parametric commands ("type ...", "search for ...", "open ...", "scroll down a lot")
  3. a bare app name ("Safari") -> open it (handles "Oh Safari" mishearings)
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Command:
    action: str                  # see ACTIONS below
    arg: object = None
    label: str = ""              # short human-readable description for the HUD

    def __str__(self) -> str:
        return self.label or self.action


# action names used by the executor
ACTIONS = {
    "enable", "disable", "click", "double_click", "right_click", "scroll",
    "hotkey", "type", "search", "open_url", "open_app", "quit_app", "volume", "unknown",
    "record_start", "record_stop",
}

FILLERS_START = ("hey deniz", "deniz", "okay", "ok", "please", "can you", "could you",
                 "would you", "now", "and", "then", "uh", "um", "oh", "so", "hey")
FILLERS_END = ("please", "now", "for me", "thanks", "thank you")


def _hk(*keys: str, label: str) -> Command:
    return Command("hotkey", tuple(keys), label)


FIXED: dict[str, Command] = {}


def _add(phrases: Iterable[str], cmd: Command) -> None:
    for p in phrases:
        FIXED[p] = cmd


_add(["enable control", "control on", "turn on control", "start control", "activate control",
      "enable mouse", "mouse on"], Command("enable", label="CONTROL ON"))
_add(["disable control", "control off", "turn off control", "stop control", "deactivate control",
      "disable mouse", "mouse off", "stop"], Command("disable", label="CONTROL OFF"))
_add(["start recording", "record", "begin recording", "start the recording", "record video"],
     Command("record_start", label="RECORDING"))
_add(["stop recording", "end recording", "stop the recording", "finish recording"],
     Command("record_stop", label="STOP RECORDING"))
_add(["click", "left click", "click it", "click here", "press"], Command("click", label="CLICK"))
_add(["double click", "double-click", "double click it", "open it"],
     Command("double_click", label="DOUBLE CLICK"))
_add(["right click", "right-click", "context menu"], Command("right_click", label="RIGHT CLICK"))
_add(["new tab", "open new tab", "open a new tab"], _hk("command", "t", label="NEW TAB"))
_add(["close tab", "close the tab", "close this tab"], _hk("command", "w", label="CLOSE TAB"))
_add(["close window", "close the window"], _hk("command", "w", label="CLOSE WINDOW"))
_add(["new window"], _hk("command", "n", label="NEW WINDOW"))
_add(["next tab"], _hk("ctrl", "tab", label="NEXT TAB"))
_add(["previous tab", "last tab"], _hk("ctrl", "shift", "tab", label="PREVIOUS TAB"))
_add(["reopen tab", "reopen closed tab"], _hk("command", "shift", "t", label="REOPEN TAB"))
_add(["go back", "back"], _hk("command", "[", label="BACK"))
_add(["go forward", "forward"], _hk("command", "]", label="FORWARD"))
_add(["refresh", "reload", "refresh page", "reload page"], _hk("command", "r", label="RELOAD"))
_add(["copy", "copy that", "copy this"], _hk("command", "c", label="COPY"))
_add(["paste", "paste it", "paste that"], _hk("command", "v", label="PASTE"))
_add(["cut", "cut that"], _hk("command", "x", label="CUT"))
_add(["undo", "undo that"], _hk("command", "z", label="UNDO"))
_add(["redo"], _hk("command", "shift", "z", label="REDO"))
_add(["select all"], _hk("command", "a", label="SELECT ALL"))
_add(["save", "save it", "save file"], _hk("command", "s", label="SAVE"))
_add(["find", "find on page"], _hk("command", "f", label="FIND"))
_add(["enter", "press enter", "hit enter", "return", "press return", "send", "submit"],
     _hk("enter", label="ENTER"))
_add(["escape", "press escape", "cancel"], _hk("esc", label="ESCAPE"))
_add(["delete", "backspace", "press delete"], _hk("backspace", label="DELETE"))
_add(["delete word"], _hk("option", "backspace", label="DELETE WORD"))
_add(["space", "press space"], _hk("space", label="SPACE"))
_add(["tab", "press tab"], _hk("tab", label="TAB"))
_add(["page down"], _hk("pagedown", label="PAGE DOWN"))
_add(["page up"], _hk("pageup", label="PAGE UP"))
_add(["scroll to top", "scroll to the top", "go to top", "go to the top", "top of page"],
     _hk("command", "up", label="TOP"))
_add(["scroll to bottom", "scroll to the bottom", "go to bottom", "go to the bottom",
      "bottom of page"], _hk("command", "down", label="BOTTOM"))
_add(["zoom in"], _hk("command", "=", label="ZOOM IN"))
_add(["zoom out"], _hk("command", "-", label="ZOOM OUT"))
_add(["screenshot", "take a screenshot", "take screenshot"],
     _hk("command", "shift", "3", label="SCREENSHOT"))
_add(["spotlight", "open spotlight"], _hk("command", "space", label="SPOTLIGHT"))
_add(["switch app", "switch apps", "next app"], _hk("command", "tab", label="SWITCH APP"))
_add(["mission control", "show all windows"], _hk("ctrl", "up", label="MISSION CONTROL"))
_add(["minimize", "minimize window"], _hk("command", "m", label="MINIMIZE"))
_add(["full screen", "fullscreen", "toggle full screen"],
     _hk("ctrl", "command", "f", label="FULL SCREEN"))
_add(["volume up", "louder", "turn it up", "turn up the volume", "increase volume"],
     Command("volume", "up", "VOLUME UP"))
_add(["volume down", "quieter", "turn it down", "turn down the volume", "decrease volume"],
     Command("volume", "down", "VOLUME DOWN"))
_add(["mute", "mute sound", "mute audio"], Command("volume", "mute", "MUTE"))
_add(["unmute", "unmute sound", "unmute audio"], Command("volume", "unmute", "UNMUTE"))

# Spoken name -> macOS app name
APP_ALIASES = {
    "chrome": "Google Chrome", "google chrome": "Google Chrome",
    "vs code": "Visual Studio Code", "vscode": "Visual Studio Code", "code": "Visual Studio Code",
    "visual studio code": "Visual Studio Code",
    "settings": "System Settings", "system settings": "System Settings",
    "preferences": "System Settings", "system preferences": "System Settings",
    "app store": "App Store", "facetime": "FaceTime", "face time": "FaceTime",
    "whatsapp": "WhatsApp", "what's app": "WhatsApp",
    "itunes": "Music", "apple music": "Music", "imessage": "Messages",
    "quicktime": "QuickTime Player", "activity monitor": "Activity Monitor",
}

# Spoken site -> URL
WEBSITES = {
    "youtube": "https://www.youtube.com", "google": "https://www.google.com",
    "gmail": "https://mail.google.com", "github": "https://github.com",
    "netflix": "https://www.netflix.com", "instagram": "https://www.instagram.com",
    "twitter": "https://x.com", "x": "https://x.com", "reddit": "https://www.reddit.com",
    "wikipedia": "https://www.wikipedia.org", "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai", "linkedin": "https://www.linkedin.com",
    "maps": "https://maps.google.com", "google maps": "https://maps.google.com",
}

_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)*\.(com|org|net|io|dev|ai|app|co|edu|gov|tr|me|tv)$")

_SCROLL = re.compile(
    r"^scroll\s+(up|down)(?:\s+(a lot|lots|more|far|fast|a little|a bit|little|slowly|less))?$")
_TYPE = re.compile(r"^(type|write|dictate|enter text)\s+(.+)$")
_SEARCH = re.compile(r"^(search for|search|google|look up)\s+(.+)$")
_OPEN = re.compile(r"^(open|launch|start|go to|show)\s+(?:the\s+|up\s+)?(.+?)(?:\s+app)?$")
_QUIT = re.compile(r"^(quit|close|exit)\s+(?:the\s+)?(.+?)(?:\s+app)?$")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation (keeping dots inside domains), strip fillers."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^\w\s'.-]", " ", t)
    t = re.sub(r"\.(?=\s|$)", " ", t)          # sentence dots, not "youtube.com"
    t = re.sub(r"(?<!\w)[-'.]+|[-'.]+(?!\w)", " ", t)   # stray "-type", "'click'"
    t = re.sub(r"\s+", " ", t).strip()
    changed = True
    while changed and t:
        changed = False
        for f in FILLERS_START:
            if t == f:
                return ""
            if t.startswith(f + " "):
                t, changed = t[len(f) + 1:], True
        for f in FILLERS_END:
            if t.endswith(" " + f):
                t, changed = t[: -len(f) - 1], True
    return t


def _tail_from_raw(raw: str, keyword: str) -> str:
    """Text after `keyword` in the original transcript, keeping its casing."""
    i = raw.lower().find(keyword)
    tail = raw[i + len(keyword):] if i >= 0 else raw
    tail = tail.strip(" ,:;-\"'")
    return re.sub(r"[.!]+$", "", tail).strip()   # Whisper adds a final period


def resolve_app(name: str, installed: Iterable[str] = ()) -> str | None:
    """Best installed app for a spoken name, or None."""
    name = name.strip().lower()
    if name in APP_ALIASES:
        return APP_ALIASES[name]
    lookup = {a.lower(): a for a in installed}
    if name in lookup:
        return lookup[name]
    match = difflib.get_close_matches(name, list(lookup), n=1, cutoff=0.8)
    return lookup[match[0]] if match else None


def _fuzzy_fixed(t: str) -> Command | None:
    match = difflib.get_close_matches(t, list(FIXED), n=1, cutoff=0.86)
    return FIXED[match[0]] if match else None


@dataclass
class CommandParser:
    installed_apps: list[str] = field(default_factory=list)

    def parse(self, raw: str) -> Command:
        t = normalize(raw)
        if not t:
            return Command("unknown", label="?")

        if t in FIXED:
            return FIXED[t]

        m = _SCROLL.match(t)
        if m:
            amount = m.group(2) or ""
            factor = 2.0 if amount in ("a lot", "lots", "more", "far", "fast") else \
                0.5 if amount in ("a little", "a bit", "little", "slowly", "less") else 1.0
            return Command("scroll", (m.group(1), factor), f"SCROLL {m.group(1).upper()}")

        m = _TYPE.match(t)
        if m:
            text = _tail_from_raw(raw, m.group(1))
            text, app = self._split_target_app(text)
            if app:
                return Command("type", (text, app), f'TYPE "{text}" IN {app.upper()}')
            return Command("type", text, f'TYPE "{text}"')

        m = _SEARCH.match(t)
        if m:
            query = _tail_from_raw(raw, m.group(1))
            return Command("search", query, f'SEARCH "{query}"')

        m = _OPEN.match(t)
        if m:
            return self._open_target(m.group(2))

        m = _QUIT.match(t)
        if m:
            app = resolve_app(m.group(2), self.installed_apps)
            if app:
                return Command("quit_app", app, f"QUIT {app.upper()}")

        cmd = _fuzzy_fixed(t)
        if cmd:
            return cmd

        # A bare app or site name ("Safari") opens it.
        if t in WEBSITES or resolve_app(t, self.installed_apps):
            return self._open_target(t)

        return Command("unknown", label="?")

    def _split_target_app(self, text: str) -> tuple[str, str | None]:
        """'youtube to Safari' -> ('youtube', 'Safari') if Safari is an installed app."""
        m = re.match(r"^(.+?)\s+(?:in|into|to|on|with|using)\s+(?:the\s+)?([\w .'-]+?)(?:\s+app)?$",
                     text, re.I)
        if m:
            app = resolve_app(m.group(2), self.installed_apps)
            if app:
                return m.group(1).strip(), app
        return text, None

    def _open_target(self, target: str) -> Command:
        target = target.strip()
        site, app = self._split_target_app(target)
        if app and site != target:
            inner = self._open_target(site)
            if inner.action == "open_url":
                return Command("open_url", (inner.arg, app), f"{inner.label} IN {app.upper()}")
        if target in WEBSITES:
            return Command("open_url", WEBSITES[target], f"OPEN {target.upper()}")
        if _DOMAIN.match(target):
            return Command("open_url", "https://" + target, f"OPEN {target}")
        app = resolve_app(target, self.installed_apps)
        if app:
            return Command("open_app", app, f"OPEN {app.upper()}")
        # Unknown name: let the executor try it as-is (open -a fails safely).
        name = target.title()
        return Command("open_app", name, f"OPEN {name.upper()}")
