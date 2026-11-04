"""Microphone recording on demand (the mic is open only while listening)."""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


class MicError(RuntimeError):
    """Raised when the microphone cannot be opened."""


class Recorder:
    """Records mono float32 audio between start() and stop().

    `level` is a smoothed 0..1 loudness value for the HUD meter.
    """

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 10.0) -> None:
        import sounddevice  # PortAudio; ships with its own library on macOS
        self._sd = sounddevice
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._n = 0
        self._lock = threading.Lock()
        self.level = 0.0
        self.full = False             # True once max_seconds were captured

    @property
    def recording(self) -> bool:
        return self._stream is not None

    @property
    def seconds(self) -> float:
        return self._n / self.sample_rate

    def _callback(self, indata, frames, time_info, status) -> None:  # audio thread
        mono = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
        self.level = 0.7 * self.level + 0.3 * min(1.0, rms * 12)
        with self._lock:
            if self._n < self.max_samples:
                self._chunks.append(mono)
                self._n += len(mono)
            else:
                self.full = True

    def start(self) -> None:
        if self._stream is not None:
            return
        with self._lock:
            self._chunks, self._n, self.full = [], 0, False
        try:
            self._stream = self._sd.InputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32",
                blocksize=self.sample_rate // 50, callback=self._callback)
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - PortAudio raises various types
            self._stream = None
            raise MicError(f"Could not open the microphone: {exc}") from exc
        log.info("Listening...")

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio (may be empty)."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.level = 0.0
        with self._lock:
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
            self._chunks, self._n = [], 0
        return audio

    def close(self) -> None:
        self.stop()
