"""Session recording for videos: camera HUD + microphone + screen, then merged.

Output folder recordings/<date_time>/:
    camera.mp4        HUD camera view, constant 30 FPS (H.264, hardware encoded)
    camera_raw.mp4    optional, camera without HUD
    audio.wav         microphone, 48 kHz mono
    screen.mp4        whole screen with cursor + clicks (ffmpeg AVFoundation)
    camera_audio.mp4  camera.mp4 + audio            (needs ffmpeg)
    final.mp4         screen + camera corner + audio (needs ffmpeg + screen)
    session.json      timing offsets used for syncing

Sync: every track is placed on one timeline that starts at t0.
  * camera frames are positioned by the time the camera captured them
    (duplicated / dropped to keep a constant 30 FPS),
  * the audio start is the time of its first sample,
  * the screen start is the wall-clock time of its first frame, which ffmpeg
    reports exactly (the screen grabber takes ~0.8 s to start).
"""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import RecordConfig

log = logging.getLogger(__name__)


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


class VideoSink:
    """Writes BGR frames to an H.264 file on a background thread."""

    def __init__(self, path: Path, size: tuple[int, int], fps: int, bitrate: str,
                 ffmpeg: str | None) -> None:
        self.path = path
        self._q: queue.Queue[bytes | None] = queue.Queue(maxsize=90)
        self._proc = None
        self._cv_writer = None
        w, h = size
        if ffmpeg:
            self._proc = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                 "-c:v", "h264_videotoolbox", "-b:v", bitrate, "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:   # fallback without ffmpeg
            self._cv_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        self._size = size
        self._thread = threading.Thread(target=self._run, name="video-sink", daemon=True)
        self._thread.start()

    def add(self, frame: np.ndarray, copies: int = 1) -> None:
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        data = frame.tobytes()            # copy now: the caller keeps drawing on frame
        for _ in range(copies):
            self._q.put(data)

    def _run(self) -> None:
        while True:
            data = self._q.get()
            if data is None:
                return
            try:
                if self._proc is not None:
                    self._proc.stdin.write(data)
                else:
                    h, w = self._size[1], self._size[0]
                    self._cv_writer.write(np.frombuffer(data, np.uint8).reshape(h, w, 3))
            except (BrokenPipeError, OSError):
                log.error("Video encoder stopped unexpectedly (%s)", self.path.name)
                return

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=30)
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.wait(timeout=60)
            err = self._proc.stderr.read().decode(errors="replace").strip()
            if self._proc.returncode != 0:
                log.error("ffmpeg failed for %s: %s", self.path.name, err[-400:])
        if self._cv_writer is not None:
            self._cv_writer.release()


class AudioFileWriter:
    """Records the microphone straight to a 16-bit WAV file."""

    def __init__(self, path: Path, sample_rate: int, device: str | None = None) -> None:
        import sounddevice as sd
        self.path = path
        self.sample_rate = sample_rate
        self.first_sample_t: float | None = None   # monotonic time of the first sample
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate)
        self._thread = threading.Thread(target=self._run, name="audio-writer", daemon=True)
        self._thread.start()
        self._stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                                      blocksize=sample_rate // 50, callback=self._callback,
                                      device=device)
        log.info("Recording audio from: %s", sd.query_devices(self._stream.device)["name"])
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if self.first_sample_t is None:
            # How long ago the first sample of this buffer hit the converter (ADC).
            age = time_info.currentTime - time_info.inputBufferAdcTime
            if not 0.0 < age < 0.5:          # driver gave no usable timing
                age = frames / self.sample_rate + self._stream_latency()
            self.first_sample_t = time.monotonic() - age
        self._q.put(indata.tobytes())

    def _stream_latency(self) -> float:
        try:
            return float(self._stream.latency)
        except Exception:  # noqa: BLE001 - stream may not exist yet in the first callback
            return 0.0

    def _run(self) -> None:
        while True:
            data = self._q.get()
            if data is None:
                return
            self._wav.writeframes(data)

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._q.put(None)
            self._thread.join(timeout=10)
            self._wav.close()


class ScreenSink:
    """Records the screen with ffmpeg and reports when the first frame was taken."""

    def __init__(self, path: Path, ffmpeg: str, cfg: RecordConfig) -> None:
        self.path = path
        self.start_wall: float | None = None    # time.time() of the first frame
        self._lines: list[str] = []
        screen = f"Capture screen {(cfg.screen_display or 1) - 1}:none"
        self._proc = subprocess.Popen(
            [ffmpeg, "-y", "-hide_banner", "-nostats", "-loglevel", "info",
             "-f", "avfoundation", "-capture_cursor", "1",
             "-capture_mouse_clicks", "1" if cfg.screen_show_clicks else "0",
             "-framerate", str(cfg.fps), "-pixel_format", "nv12",
             "-use_wallclock_as_timestamps", "1", "-i", screen,
             "-c:v", "h264_videotoolbox", "-b:v", cfg.screen_bitrate, "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._reader = threading.Thread(target=self._read_log, name="screen-log", daemon=True)
        self._reader.start()

    def _read_log(self) -> None:
        for raw in self._proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            self._lines.append(line)
            if self.start_wall is None:
                m = re.search(r"start: ([\d.]+)", line)
                if m:
                    self.start_wall = float(m.group(1))

    @property
    def log_tail(self) -> str:
        return "\n".join(self._lines[-8:])

    def close(self) -> bool:
        """Stop recording; returns True if a file was written."""
        try:
            self._proc.stdin.write(b"q")       # ffmpeg's clean-stop key
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._reader.join(timeout=5)
        return self.path.exists() and self.path.stat().st_size > 0


class SessionRecorder:
    """Start/stop a recording session. status: idle | recording | saving."""

    def __init__(self, cfg: RecordConfig) -> None:
        self.cfg = cfg
        self.ffmpeg = shutil.which("ffmpeg") or (
            "/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else None)
        self.status = "idle"
        self.message = ""                  # last result for the HUD
        self.message_t = 0.0
        self.folder: Path | None = None
        self._t0 = 0.0
        self._written = 0
        self._cam: VideoSink | None = None
        self._raw: VideoSink | None = None
        self._audio: AudioFileWriter | None = None
        self._screen: ScreenSink | None = None
        self._t0_wall = 0.0
        self._finisher: threading.Thread | None = None
        self._last_disk_check = 0.0

    @property
    def recording(self) -> bool:
        return self.status == "recording"

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0 if self.recording else 0.0

    def _say(self, msg: str) -> None:
        self.message, self.message_t = msg, time.monotonic()
        log.info("Recorder: %s", msg)

    # -- start ---------------------------------------------------------------
    def start(self, frame_size: tuple[int, int]) -> bool:
        if self.status != "idle":
            self._say("still saving the last recording...")
            return False
        if free_gb(self.cfg.output_dir) < self.cfg.min_free_gb:
            self._say(f"not enough disk space (< {self.cfg.min_free_gb:.0f} GB free)")
            return False

        self.folder = self.cfg.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder.mkdir(parents=True, exist_ok=True)
        # Time zero for every track; screen and audio offsets are measured from here.
        self._t0 = time.monotonic()
        self._t0_wall = time.time()

        if self.cfg.record_screen:
            if self.ffmpeg:
                self._screen = ScreenSink(self.folder / "screen.mp4", self.ffmpeg, self.cfg)
            else:
                log.warning("Screen recording needs ffmpeg (brew install ffmpeg) - skipped")

        if self.cfg.record_audio:
            try:
                self._audio = AudioFileWriter(self.folder / "audio.wav",
                                              self.cfg.audio_sample_rate, self.cfg.audio_input)
            except Exception as exc:  # noqa: BLE001
                log.error("Audio recording unavailable: %s", exc)
                self._audio = None

        self._written = 0
        self._cam = VideoSink(self.folder / "camera.mp4", frame_size, self.cfg.fps,
                              self.cfg.camera_bitrate, self.ffmpeg)
        if self.cfg.record_raw_camera:
            self._raw = VideoSink(self.folder / "camera_raw.mp4", frame_size, self.cfg.fps,
                                  self.cfg.camera_bitrate, self.ffmpeg)
        self.status = "recording"
        self._say(f"recording to {self.folder.name}")
        return True

    # -- per frame -------------------------------------------------------------
    def add_frame(self, hud_frame: np.ndarray, raw_frame: np.ndarray | None = None,
                  captured_at: float | None = None) -> None:
        """Call once per loop iteration while recording.

        captured_at: time.monotonic() when the camera delivered this frame.
        Using it (instead of "now") keeps processing delay out of lip-sync.
        """
        if not self.recording:
            return
        now = time.monotonic()
        t = captured_at if captured_at is not None else now
        due = int((t - self._t0) * self.cfg.fps) + 1
        copies = due - self._written
        if copies <= 0:
            return                      # loop is faster than 30 FPS: skip this frame
        copies = min(copies, self.cfg.fps)   # after a long stall, don't flood the encoder
        self._cam.add(hud_frame, copies)
        if self._raw is not None and raw_frame is not None:
            self._raw.add(raw_frame, copies)
        self._written = due

        if now - self._last_disk_check > 5:
            self._last_disk_check = now
            if free_gb(self.cfg.output_dir) < self.cfg.min_free_gb / 2:
                log.warning("Disk almost full - stopping recording")
                self.stop()

    # -- stop ------------------------------------------------------------------
    def stop(self, wait: bool = False) -> None:
        if not self.recording:
            return
        self.status = "saving"
        duration = time.monotonic() - self._t0
        self._say("saving...")

        # Everything else (flushing encoders, merging) runs off the video loop.
        self._finisher = threading.Thread(target=self._finish, args=(duration,),
                                          name="recording-finish", daemon=True)
        self._finisher.start()
        if wait:
            self._finisher.join()

    def _finish(self, duration: float) -> None:
        audio_offset = 0.0
        if self._audio is not None:
            self._audio.close()
            if self._audio.first_sample_t is not None:
                audio_offset = self._audio.first_sample_t - self._t0
        for sink in (self._cam, self._raw):
            if sink is not None:
                sink.close()
        screen_ok, screen_offset, warning = False, None, ""
        if self._screen is not None:
            screen_ok = self._screen.close()
            if not screen_ok:
                log.error("Screen recording failed (Screen Recording permission?)\n%s",
                          self._screen.log_tail)
                warning = "screen recording failed - check Screen Recording permission"
            elif self._screen.start_wall is None:
                log.error("Screen start time unknown; final.mp4 skipped")
                screen_ok, warning = False, "screen start time unknown"
            else:
                screen_offset = self._screen.start_wall - self._t0_wall + self.cfg.screen_offset_s
                d = self._probe_duration(self.folder / "screen.mp4") or 0.0
                if d < duration - screen_offset - 2.0:
                    warning = f"screen recording is short ({d:.1f}s of {duration:.1f}s)"
                    log.warning(warning)

        meta = {
            "duration_s": round(duration, 3),
            "fps": self.cfg.fps,
            "audio_offset_s": round(audio_offset, 4),
            "screen_offset_s": round(screen_offset, 4) if screen_offset is not None else None,
            "warning": warning or None,
        }
        (self.folder / "session.json").write_text(json.dumps(meta, indent=2))
        self._cam = self._raw = self._audio = self._screen = None

        if self.cfg.merge and self.ffmpeg:
            self._merge(self.folder, meta, screen_ok)
        else:
            self.status = "idle"
            self._say(f"saved to recordings/{self.folder.name}")

    def _probe_duration(self, path: Path) -> float | None:
        if not self.ffmpeg:
            return None
        probe = str(Path(self.ffmpeg).with_name("ffprobe"))
        try:
            out = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration",
                                  "-of", "csv=p=0", str(path)],
                                 capture_output=True, text=True, timeout=20).stdout
            return float(out.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None

    def wait(self) -> None:
        if self._finisher is not None:
            self._finisher.join()

    def _merge(self, folder: Path, meta: dict, screen_ok: bool) -> None:
        cfg, ff = self.cfg, self.ffmpeg
        cam, wav, scr = folder / "camera.mp4", folder / "audio.wav", folder / "screen.mp4"
        has_audio = wav.exists() and wav.stat().st_size > 1000
        a_off = str(meta["audio_offset_s"])
        made = []
        try:
            if has_audio:
                a_shift = (["-itsoffset", a_off] if float(a_off) >= 0
                           else ["-ss", f"{-float(a_off):.4f}"])
                cmd = [ff, "-y", "-loglevel", "error", "-i", str(cam), *a_shift,
                       "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                       "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                       str(folder / "camera_audio.mp4")]
                if subprocess.run(cmd, capture_output=True).returncode == 0:
                    made.append("camera_audio.mp4")

            if screen_ok:
                pip_w = int(cfg.final_width * cfg.pip_width_frac) // 2 * 2
                m = cfg.pip_margin_px
                # final.mp4 starts when the screen recording really started:
                # skip that much of the camera, and line the audio up with it.
                s_off = float(meta["screen_offset_s"] or 0.0)
                a_rel = s_off - float(meta["audio_offset_s"])   # audio time at screen start
                filt = (f"[0:v]fps={cfg.fps},scale={cfg.final_width}:-2,setsar=1[s];"
                        f"[1:v]scale={pip_w}:-2[c];"
                        f"[s][c]overlay=main_w-overlay_w-{m}:main_h-overlay_h-{m}:shortest=1[v]")
                cmd = [ff, "-y", "-loglevel", "error", "-i", str(scr),
                       "-ss", f"{s_off:.3f}", "-i", str(cam)]
                if has_audio:
                    cmd += (["-ss", f"{a_rel:.3f}"] if a_rel >= 0
                            else ["-itsoffset", f"{-a_rel:.3f}"]) + ["-i", str(wav)]
                cmd += ["-filter_complex", filt, "-map", "[v]"]
                if has_audio:
                    cmd += ["-map", "2:a", "-c:a", "aac", "-b:a", "192k"]
                cmd += ["-c:v", "h264_videotoolbox", "-b:v", cfg.final_bitrate,
                        "-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart",
                        str(folder / "final.mp4")]
                r = subprocess.run(cmd, capture_output=True)
                if r.returncode == 0:
                    made.append("final.mp4")
                else:
                    log.error("final.mp4 merge failed: %s", r.stderr.decode(errors="replace")[-400:])
        finally:
            self.status = "idle"
            extra = f" ({', '.join(made)})" if made else ""
            self._say(f"saved to recordings/{folder.name}{extra}")
