"""Measures recording sync. Shows a small window for ~8 s and plays 3 beeps.

    python -m tests.test_recording_sync            # keeps nothing on disk
    python -m tests.test_recording_sync --keep     # keep the test recording

The SAME clock (in centiseconds, drawn as a 16-bit barcode) is shown in an
on-screen window and in synthetic camera frames. A 3 kHz beep plays every 2 s.
Then, in final.mp4:
  * screen barcode vs camera barcode in the same frame -> screen sync error
  * beep time in the audio vs camera barcode at that moment -> audio sync error
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from deniz_control.config import RecordConfig
from deniz_control.recorder import SessionRecorder

BITS = 16
GREEN = (0, 255, 0)
WIN = "DENIZ SYNC TEST"


def draw_code(img, value: int, x0: int, y0: int, w: int, h: int) -> None:
    """16 black/white blocks inside a green frame."""
    cv2.rectangle(img, (x0 - 6, y0 - 6), (x0 + w + 6, y0 + h + 6), GREEN, 6)
    bw = w / BITS
    for i in range(BITS):
        bit = (value >> (BITS - 1 - i)) & 1
        cv2.rectangle(img, (int(x0 + i * bw), y0), (int(x0 + (i + 1) * bw) - 1, y0 + h),
                      (255, 255, 255) if bit else (0, 0, 0), -1)


def read_code(img, x0: float, y0: float, w: float, h: float) -> int:
    bw = w / BITS
    v = 0
    for i in range(BITS):
        x, y = int(x0 + (i + 0.5) * bw), int(y0 + h / 2)
        v = (v << 1) | int(img[y - 2:y + 3, x - 2:x + 3].mean() > 128)
    return v


def find_green_box(img) -> tuple[int, int, int, int] | None:
    b, g, r = cv2.split(img.astype(np.int16))
    mask = ((g > 180) & (r < 90) & (b < 90)).astype(np.uint8)
    pts = cv2.findNonZero(mask)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    return x, y, w, h


def beep_times(wav_path: Path, freq: float = 3000.0) -> list[float]:
    with wave.open(str(wav_path)) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    win = int(sr * 0.005)
    n = len(a) // win
    t = np.arange(win) / sr
    probe = np.exp(-2j * np.pi * freq * t)
    energy = np.array([abs(np.dot(a[i * win:(i + 1) * win], probe)) for i in range(n)])
    thr = energy.max() * 0.35
    onsets, last = [], -1.0
    for i, e in enumerate(energy):
        ts = i * win / sr
        if e > thr and ts - last > 0.5:
            onsets.append(ts)
            last = ts
    return onsets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    import sounddevice as sd

    out_root = Path(tempfile.mkdtemp(prefix="deniz_sync_"))
    cfg = replace(RecordConfig(), output_dir=out_root)
    rec = SessionRecorder(cfg)
    if not rec.ffmpeg:
        print("FAIL: ffmpeg not found")
        return 1

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WIN, 40, 60)
    rec.start((1280, 720))
    t0 = rec._t0                                        # recorder timeline origin
    sr = 48000
    tone = (0.6 * np.sin(2 * np.pi * 3000 * np.arange(int(sr * 0.12)) / sr)).astype(np.float32)
    out_latency = sd.query_devices(kind="output")["default_low_output_latency"]
    beeps_planned = []
    next_beep = 2.0
    while time.monotonic() - t0 < 8.0:
        now = time.monotonic()
        t = now - t0
        cs = int(t * 100) & 0xFFFF
        scr = np.full((180, 760, 3), 30, np.uint8)
        draw_code(scr, cs, 20, 40, 720, 100)
        cv2.imshow(WIN, scr)
        cam = np.full((720, 1280, 3), 30, np.uint8)
        draw_code(cam, cs, 40, 40, 1200, 220)
        rec.add_frame(cam, None, captured_at=now)
        if t >= next_beep:
            sd.play(tone, sr)
            beeps_planned.append(time.monotonic() - t0 + out_latency)
            next_beep += 2.0
        cv2.waitKey(1)
        time.sleep(0.005)
    rec.stop(wait=True)
    cv2.destroyWindow(WIN)
    cv2.waitKey(1)

    folder = rec.folder
    print("files:", sorted(p.name for p in folder.iterdir()))
    print("session:", (folder / "session.json").read_text().replace("\n", " "))
    final = folder / "final.mp4"
    ok = True

    # 1) camera.mp4: frame k must show time k / fps
    cap = cv2.VideoCapture(str(folder / "camera.mp4"))
    errs, k = [], 0
    while True:
        good, fr = cap.read()
        if not good:
            break
        v = read_code(fr, 40, 40, 1200, 220)
        errs.append(v * 10 - k * 1000 / cfg.fps)
        k += 1
    errs = np.array(errs[5:])
    print(f"camera.mp4 timing: median {np.median(errs):+.0f} ms, max |err| {np.abs(errs).max():.0f} ms")
    ok &= abs(np.median(errs)) < 40

    if not final.exists():
        print("FAIL: final.mp4 was not created")
        return 1

    # 2) final.mp4: screen code vs camera (picture-in-picture) code per frame
    cap = cv2.VideoCapture(str(final))
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pip_w = int(cfg.final_width * cfg.pip_width_frac) // 2 * 2
    scale = pip_w / 1280
    pip_h = int(round(720 * scale / 2)) * 2
    px, py = fw - pip_w - cfg.pip_margin_px, fh - pip_h - cfg.pip_margin_px
    diffs, cam_codes, j, box = [], [], 0, None
    while True:
        good, fr = cap.read()
        if not good:
            break
        c = read_code(fr, px + 40 * scale, py + 40 * scale, 1200 * scale, 220 * scale)
        cam_codes.append(c)
        if box is None:
            # the window sits top-left; stay clear of the Dock and other green icons
            box = find_green_box(fr[: fh // 3, : fw // 2])
        if box is not None:
            # green frame outer box is 738 x 118 px around a 720 x 100 barcode, offset 9 px
            x, y, w, h = box
            s = read_code(fr, x + w * 9 / 738, y + h * 9 / 118, w * 720 / 738, h * 100 / 118)
            if abs(s - c) < 300:     # ignore undecodable frames
                diffs.append((s - c) * 10)
        j += 1
    if diffs:
        d = np.array(diffs[3:])
        print(f"final.mp4 screen vs camera: median {np.median(d):+.0f} ms "
              f"(positive = screen shows a later time), frames compared {len(d)}")
        ok &= abs(np.median(d)) < 70     # under one frame at 15 FPS; hardly visible
    else:
        print("WARN: could not find the sync window in the screen track")
        ok = False

    # 3) final.mp4: beep time in audio vs camera clock at that moment
    wav = folder / "final_audio.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(final), "-ac", "1", "-ar", "48000",
                    str(wav)], check=True)
    onsets = beep_times(wav)
    fps = cfg.fps
    for b_planned, b in zip(beeps_planned, onsets):
        idx = min(int(b * fps), len(cam_codes) - 1)
        cam_t = cam_codes[idx] * 10 / 1000 + (b * fps - idx) / fps
        err = (cam_t - b_planned) * 1000
        print(f"beep planned at {b_planned:5.2f}s; heard where the camera clock reads {cam_t:5.2f}s "
              f"-> audio/video error {err:+.0f} ms")
        ok &= abs(err) < 60   # includes speaker output latency, which is not compensated
    if len(onsets) < len(beeps_planned):
        print(f"WARN: heard {len(onsets)} of {len(beeps_planned)} beeps (volume too low?)")

    if args.keep:
        print("kept:", folder)
    else:
        shutil.rmtree(out_root, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
