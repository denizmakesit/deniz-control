"""Voice test: speech -> text -> command. NEVER executes anything.

    python -m tests.test_voice --mic 4                    # speak a command within 4 s
    python -m tests.test_voice --say "scroll down" "open Safari"   # macOS text-to-speech
"""

import argparse
import logging
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from deniz_control.config import VoiceConfig
from deniz_control.voice.commands import CommandParser
from deniz_control.voice.executor import installed_apps
from deniz_control.voice.recognizer import SpeechRecognizer


def english_voice() -> str | None:
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    for preferred in ("Samantha", "Alex", "Daniel", "Karen"):
        if any(line.startswith(preferred + " ") for line in out.splitlines()):
            return preferred
    for line in out.splitlines():
        if "en_US" in line or "en_GB" in line:
            return line.split()[0]
    return None


def synthesize(text: str, voice: str | None, sample_rate: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = Path(d) / "s.aiff", Path(d) / "s.wav"
        cmd = ["say", "-o", str(aiff)] + (["-v", voice] if voice else []) + [text]
        subprocess.run(cmd, check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}", "-c", "1",
                        str(aiff), str(wav)], check=True)
        with wave.open(str(wav)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def record(seconds: float, sample_rate: int) -> np.ndarray:
    from deniz_control.voice.audio import Recorder
    rec = Recorder(sample_rate, seconds + 1)
    print(f"Recording {seconds:.0f}s - speak a command now...")
    rec.start()
    time.sleep(seconds)
    audio = rec.stop()
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    print(f"Captured {len(audio) / sample_rate:.1f}s, loudness (RMS) {rms:.4f}")
    return audio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", type=float, metavar="SECONDS", help="record from the microphone")
    ap.add_argument("--say", nargs="+", metavar="TEXT", help="synthesize phrases with macOS 'say'")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.mic and not args.say:
        ap.error("use --mic SECONDS or --say TEXT ...")

    cfg = VoiceConfig()
    rec = SpeechRecognizer(cfg)
    parser = CommandParser(installed_apps())
    clips: list[tuple[str, np.ndarray]] = []
    if args.say:
        voice = english_voice()
        clips += [(t, synthesize(t, voice, cfg.sample_rate)) for t in args.say]
    if args.mic:
        clips.append(("(microphone)", record(args.mic, cfg.sample_rate)))

    for source, audio in clips:
        t = rec.transcribe_now(audio)
        print(f'{source:>32}  ->  "{t.text}"  ->  {parser.parse(t.text)}   ({t.elapsed:.2f}s)')
    rec.close()
    print("PASS: voice pipeline ran (nothing was executed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
