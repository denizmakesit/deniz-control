"""Central configuration for DENIZ CONTROL.

Every tunable value lives here so behaviour can be adjusted without touching
the logic modules. Values are grouped into small frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0                 # 0 = built-in FaceTime camera on a MacBook
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    mirror: bool = True            # mirror so moving your hand right moves the cursor right
    reopen_after_s: float = 3.0    # no frames for this long -> try to reopen the camera
    reopen_attempts: int = 3


@dataclass(frozen=True)
class TrackerConfig:
    model_path: Path = PROJECT_ROOT / "models" / "hand_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class GestureConfig:
    # A finger counts as extended when its tip is this many times farther from
    # the wrist than its middle (PIP) joint. Rotation-independent.
    finger_extended_ratio: float = 1.15

    # Pinch = thumb-tip to index-tip distance divided by hand size
    # (wrist -> middle-finger knuckle). Two thresholds give hysteresis:
    pinch_on_ratio: float = 0.30   # below this -> pinch starts
    pinch_off_ratio: float = 0.45  # above this -> pinch released

    # Temporal confirmation: a condition must hold this long before it counts.
    pose_confirm_s: float = 0.10     # point / V-sign must be stable this long
    pinch_confirm_s: float = 0.05
    release_confirm_s: float = 0.08
    hand_lost_grace_s: float = 0.25  # tolerate brief detection dropouts

    # Short enough that two quick pinches still make a macOS double-click.
    click_cooldown_s: float = 0.20
    # Holding a pinch longer than this turns it into a drag (cursor follows).
    drag_hold_s: float = 0.35
    # EMA weight for the scroll reference point (lower = steadier scrolling).
    scroll_smoothing: float = 0.4
    # Open palm (all four fingers up) held this long starts voice listening.
    listen_confirm_s: float = 0.30


@dataclass(frozen=True)
class MouseConfig:
    # Region of the (mirrored) camera frame that maps to the full screen,
    # in normalized coords (left, top, right, bottom). A smaller region means
    # less arm movement is needed to reach the screen edges.
    active_region: tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.65)

    sensitivity: float = 1.0       # >1 amplifies movement around the region centre
    smoothing_alpha: float = 0.35  # EMA weight of the newest sample (0..1]; lower = smoother
    deadzone_px: float = 1.5       # ignore movements smaller than this (anti-jitter)

    # Keep the cursor this many points away from every screen edge. This also
    # keeps our own movements out of PyAutoGUI's fail-safe corners, so only YOU
    # (with the trackpad) can trigger the fail-safe.
    screen_edge_margin_px: int = 4

    freeze_while_pinching: bool = True  # hold the cursor still while fingers close

    # Scrolling (V-sign + vertical hand movement). Pixels scrolled per full
    # frame-height of hand travel = scroll_speed * screen height.
    scroll_speed: float = 2.5
    scroll_invert: bool = False    # False = content follows your hand, like a trackpad

    # Two left-clicks within this time and distance count as a double-click.
    # None = use the macOS system setting.
    double_click_interval_s: float | None = None
    double_click_radius_px: float = 8.0


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool = True
    # English-only Whisper model (whisper.cpp format, ~470 MB, downloaded on first run).
    model_path: Path = PROJECT_ROOT / "models" / "ggml-small.en.bin"
    model_url: str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
    n_threads: int = 4
    sample_rate: int = 16000
    max_record_s: float = 10.0       # listening stops automatically after this
    min_record_s: float = 0.3        # shorter recordings are ignored
    min_speech_rms: float = 0.008    # quieter recordings count as silence (no transcription)
    scroll_step_px: int = 500        # "scroll down" distance; "a lot" doubles, "a little" halves
    volume_step: int = 10            # percent per "volume up / down"
    feedback_s: float = 4.0          # how long the "heard ..." caption stays on screen
    # Words that bias Whisper towards the command vocabulary.
    initial_prompt: str = (
        "Voice commands: click, double click, right click, scroll up, scroll down, "
        "open Safari, type, search for, new tab, close tab, copy, paste, undo, "
        "press enter, volume up, enable control, disable control."
    )


@dataclass(frozen=True)
class RecordConfig:
    """R key / "start recording": camera HUD + microphone + screen, then merged."""
    output_dir: Path = PROJECT_ROOT / "recordings"
    fps: int = 30
    camera_bitrate: str = "12M"
    record_raw_camera: bool = False    # also save the camera without the HUD
    record_audio: bool = True
    audio_sample_rate: int = 48000
    # Microphone for recordings: None = the macOS default input, or part of a
    # device name, e.g. "Wireless Mic" (list them: python -m sounddevice).
    audio_input: str | None = None
    record_screen: bool = True         # ffmpeg screen capture (needs Screen Recording permission)
    screen_display: int | None = None  # None = main display; 2 = second display
    screen_bitrate: str = "20M"
    screen_show_clicks: bool = True
    # Sync fine-tuning (seconds). Measured with tests/test_recording_sync.py:
    # ffmpeg timestamps screen frames on arrival, ~1-2 frames after capture.
    screen_offset_s: float = -0.05
    # The webcam delivers a frame ~1-2 frames after it was exposed. Estimated
    # (cannot be measured automatically); tune with a hand clap if needed.
    camera_latency_s: float = 0.05
    merge: bool = True                 # build final.mp4 + camera_audio.mp4 with ffmpeg
    final_width: int = 1920
    final_bitrate: str = "16M"
    pip_width_frac: float = 0.28       # camera picture-in-picture width / screen width
    pip_margin_px: int = 36
    min_free_gb: float = 2.0           # refuse to start / auto-stop below this


@dataclass(frozen=True)
class UIConfig:
    window_name: str = "DENIZ CONTROL"
    landmark_color_bgr: tuple[int, int, int] = (255, 217, 101)   # #65D9FF
    accent_on_bgr: tuple[int, int, int] = (140, 230, 90)          # green
    accent_off_bgr: tuple[int, int, int] = (90, 90, 235)          # red
    text_bgr: tuple[int, int, int] = (235, 235, 235)
    muted_bgr: tuple[int, int, int] = (150, 150, 150)
    panel_bgr: tuple[int, int, int] = (22, 18, 16)
    # Camera image: out = video_gain * frame + video_lift. 1.0 / 0 = untouched.
    video_gain: float = 1.08
    video_lift: int = 6

    # Cinematic effects
    left_click_bgr: tuple[int, int, int] = (255, 240, 200)        # icy white-cyan
    right_click_bgr: tuple[int, int, int] = (210, 90, 255)        # magenta #FF5AD2
    drag_bgr: tuple[int, int, int] = (60, 190, 255)               # amber
    scroll_bgr: tuple[int, int, int] = (255, 160, 120)            # soft blue
    trail_s: float = 0.35          # fingertip trail length in seconds
    ripple_s: float = 0.45         # click ripple duration
    glow: bool = True              # additive bloom on effects (costs ~2 ms/frame)
    listening_bgr: tuple[int, int, int] = (109, 77, 255)          # coral red #FF4D6D


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
