"""Offline English speech recognition with whisper.cpp (pywhispercpp, Metal GPU).

Transcription runs on a worker thread; the app submits audio and polls for
results so the video loop never blocks.
"""

from __future__ import annotations

import atexit
import gc
import logging
import queue
import re
import threading
import urllib.request
from dataclasses import dataclass

import numpy as np

from ..config import VoiceConfig

log = logging.getLogger(__name__)
# pywhispercpp logs every transcription at INFO level; keep the app log readable.
logging.getLogger("pywhispercpp").setLevel(logging.WARNING)

# Whisper emits these for silence or noise; they are not speech.
_NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")


@dataclass
class Transcript:
    text: str            # cleaned text ("" if nothing was said)
    seconds: float       # audio length
    elapsed: float       # transcription time


def ensure_model(cfg: VoiceConfig) -> None:
    if cfg.model_path.exists():
        return
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Whisper model (~470 MB) to %s - one time only", cfg.model_path)
    tmp = cfg.model_path.with_suffix(".part")
    urllib.request.urlretrieve(cfg.model_url, tmp)
    tmp.rename(cfg.model_path)


def clean_text(text: str) -> str:
    text = _NON_SPEECH.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Pad with silence: Whisper is unreliable on clips shorter than ~1 s."""
    pad = np.zeros(int(sample_rate * 0.3), np.float32)
    audio = np.concatenate([pad, audio.astype(np.float32), pad])
    min_len = int(sample_rate * 1.2)
    if len(audio) < min_len:
        audio = np.concatenate([audio, np.zeros(min_len - len(audio), np.float32)])
    return audio


class SpeechRecognizer:
    """Background Whisper worker.

    status: "loading" -> "ready" | "error"; "busy" while transcribing.
    """

    def __init__(self, cfg: VoiceConfig) -> None:
        self.cfg = cfg
        self.status = "loading"
        self.error = ""
        self._jobs: queue.Queue[np.ndarray | None] = queue.Queue()
        self._results: queue.Queue[Transcript] = queue.Queue()
        self._model = None
        self._thread = threading.Thread(target=self._run, name="whisper", daemon=True)
        self._thread.start()
        atexit.register(self._free_model)

    # -- public API (main thread) -----------------------------------------
    def submit(self, audio: np.ndarray) -> None:
        self._jobs.put(audio)

    def poll(self) -> Transcript | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def transcribe_now(self, audio: np.ndarray) -> Transcript:
        """Blocking transcription (for tests). Waits for the model to load."""
        self.submit(audio)
        return self._results.get(timeout=120)

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=10)

    def _free_model(self) -> None:
        """Free the model while Python is fully alive, then restore whisper.cpp's
        default logger, so it never calls the Python log callback during exit."""
        if self._model is None:
            return
        self._model = None
        gc.collect()
        try:
            import _pywhispercpp as pw
            pw.whisper_log_set(None)
        except Exception:  # noqa: BLE001
            pass

    # -- worker thread -----------------------------------------------------
    def _load(self) -> None:
        import time

        import _pywhispercpp as pw
        from pywhispercpp.model import Model

        ensure_model(self.cfg)
        t = time.perf_counter()
        # Silence whisper.cpp's own log spam with a no-op callback. (pywhispercpp's
        # redirect option points the WHOLE process's stderr at /dev/null while
        # loading, which swallowed the app's logs from other threads.)
        pw.whisper_log_set(lambda *args: None)
        self._model = Model(
            str(self.cfg.model_path),
            redirect_whispercpp_logs_to=False,
            print_progress=False, print_realtime=False,
            language="en", no_context=True, single_segment=True,
            n_threads=self.cfg.n_threads, initial_prompt=self.cfg.initial_prompt,
        )
        # warm-up so the first real command is fast
        self._model.transcribe(np.zeros(self.cfg.sample_rate, np.float32))
        log.info("Whisper model ready (%.1fs)", time.perf_counter() - t)

    def _run(self) -> None:
        import time
        try:
            self._load()
            self.status = "ready"
        except Exception as exc:  # noqa: BLE001
            self.status, self.error = "error", str(exc)
            log.error("Voice recognition unavailable: %s", exc)
            return

        while True:
            audio = self._jobs.get()
            if audio is None:
                self._free_model()
                return
            self.status = "busy"
            t = time.perf_counter()
            seconds = len(audio) / self.cfg.sample_rate
            try:
                segs = self._model.transcribe(prepare_audio(audio, self.cfg.sample_rate))
                text = clean_text(" ".join(s.text for s in segs))
            except Exception:  # noqa: BLE001
                log.exception("Transcription failed")
                text = ""
            self._results.put(Transcript(text, seconds, time.perf_counter() - t))
            self.status = "ready"
