"""Glue between the app loop and the voice components.

Each frame the app calls update(want_listen) and then poll() for a recognised
command. Recording starts/stops on the edges of want_listen.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import VoiceConfig
from .audio import MicError, Recorder
from .commands import Command, CommandParser
from .executor import CommandExecutor, installed_apps
from .recognizer import SpeechRecognizer

log = logging.getLogger(__name__)


@dataclass
class Caption:
    """What the HUD shows after a voice command."""
    heard: str
    result: str
    ok: bool
    t: float


@dataclass
class Heard:
    text: str
    command: Command


class VoiceController:
    def __init__(self, cfg: VoiceConfig, mouse=None) -> None:
        self.cfg = cfg
        self.recorder = Recorder(cfg.sample_rate, cfg.max_record_s)
        self.recognizer = SpeechRecognizer(cfg)
        self.parser = CommandParser(installed_apps())
        self.executor = CommandExecutor(cfg, mouse) if mouse is not None else None
        self.caption: Caption | None = None
        self.levels: deque[float] = deque(maxlen=48)   # mic level history for the HUD
        self.error = ""
        self._pending = 0
        self._want_prev = False

    # -- status for the HUD ------------------------------------------------
    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.recorder.recording:
            return "listening"
        if self._pending:
            return "thinking"
        return {"loading": "loading", "error": "error"}.get(self.recognizer.status, "ready")

    @property
    def level(self) -> float:
        return self.recorder.level

    @property
    def record_seconds(self) -> float:
        return self.recorder.seconds

    def say(self, heard: str, result: str, ok: bool) -> None:
        self.caption = Caption(heard, result, ok, time.monotonic())

    # -- per-frame ---------------------------------------------------------
    def update(self, want_listen: bool) -> None:
        rising = want_listen and not self._want_prev
        self._want_prev = want_listen

        if rising and not self.recorder.recording:
            if self.recognizer.status == "error":
                self.say("", f"voice unavailable: {self.recognizer.error}", False)
            elif self.recognizer.status == "loading":
                self.say("", "voice model still loading...", False)
            else:
                try:
                    self.recorder.start()
                    self.error = ""
                except MicError as exc:
                    self.error = str(exc)
                    self.say("", "microphone unavailable - see README", False)
                    log.error("%s", exc)

        if self.recorder.recording:
            self.levels.append(self.recorder.level)
            if not want_listen or self.recorder.full:
                self._finish_recording()
        elif self.levels:
            self.levels.append(0.0)

    def _finish_recording(self) -> None:
        audio = self.recorder.stop()
        seconds = len(audio) / self.cfg.sample_rate
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        if seconds < self.cfg.min_record_s or rms < self.cfg.min_speech_rms:
            log.info("No speech (%.1fs, rms %.4f)", seconds, rms)
            self.say("", "no speech heard", False)
            return
        self._pending += 1
        self.recognizer.submit(audio)

    def poll(self) -> Heard | None:
        t = self.recognizer.poll()
        if t is None:
            return None
        self._pending = max(0, self._pending - 1)
        cmd = self.parser.parse(t.text)
        log.info('Heard "%s" -> %s (%.2fs for %.1fs audio)', t.text, cmd, t.elapsed, t.seconds)
        if not t.text:
            self.say("", "no speech heard", False)
            return None
        return Heard(t.text, cmd)

    def execute(self, heard: Heard, control_enabled: bool) -> None:
        """Run a command (raises pyautogui.FailSafeException like mouse actions)."""
        if self.executor is None:
            ok, msg = heard.command.action != "unknown", "preview only"
        else:
            ok, msg = self.executor.execute(heard.command, control_enabled)
        self.say(heard.text, str(heard.command) if ok else msg, ok)

    def close(self) -> None:
        self.recorder.close()
        self.recognizer.close()
