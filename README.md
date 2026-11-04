# DENIZ CONTROL

**Touchless control for your Mac: hand gestures through the webcam, plus offline English voice commands.**

Point to move the cursor, pinch to click, make a V-sign to scroll, open your palm and talk.
Everything runs locally: hand tracking with MediaPipe, speech recognition with whisper.cpp. No audio or video leaves your Mac.
It can also record the camera, the microphone and the screen together, and merge them into a ready-to-upload video.

| ☝️ Point | 🤏 Pinch | ✌️ V-sign | ✋ Open palm |
|---|---|---|---|
| Move the cursor | Click · double-click · hold to drag | Scroll · pinch to right-click | Voice command |

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Gestures](#gestures)
- [Voice commands](#voice-commands-english)
- [Recording for videos](#recording-for-videos)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Safety](#safety)
- [Testing step by step](#testing-step-by-step)
- [Tuning](#tuning)
- [Troubleshooting](#troubleshooting)
- [How it works](#how-it-works)
- [Credits and license](#credits-and-license)

## Requirements

| | |
|---|---|
| **Mac** | Apple Silicon (M1 or newer). Developed and tested on a MacBook Air M4 with macOS 26 |
| **Python** | 3.11 (`brew install python@3.11`) |
| **Disk** | About 1 GB: ~450 MB for Python packages and ~480 MB for the models, which download on first run |
| **ffmpeg** | Optional, only needed for screen recording and merged videos (`brew install ffmpeg`) |

macOS only: the app uses AVFoundation (camera), Quartz (mouse and keyboard events) and Metal (speech model).

## Quick start

```bash
git clone https://github.com/deniz6900357/deniz-control.git
cd deniz-control

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python main.py
```

1. **First launch.** macOS asks for **Camera** access. The hand model (8 MB) and the speech model (470 MB) download automatically.
2. **Accessibility.** Grant it to your terminal (or VS Code), then restart the terminal. See [Permissions](#macos-permissions).
3. **Turn control on.** Click the preview window and press **SPACE**. Mouse control is always **off** at launch.
4. **Try it.** Raise your index finger to move the cursor, then pinch your thumb and index finger to click.
5. **Stop.** Press **SPACE** again to turn control off, or **Q** to quit.

> Don't install `opencv-python` alongside this project. MediaPipe already installs `opencv-contrib-python`, and having both breaks `cv2`.

### macOS permissions

Permissions are given to the **app that launches Python** (Terminal, iTerm, VS Code…), not to Python itself.
Find them under System Settings → Privacy & Security.

| Permission | Needed for |
|---|---|
| **Camera** | Hand tracking (asked on first launch) |
| **Accessibility** | Moving the mouse, clicking and typing. Restart the terminal after granting it |
| **Microphone** | Voice commands and recording audio (asked the first time you talk) |
| **Screen & System Audio Recording** | Only for screen recording. Restart the terminal after granting it |

## Gestures

| Gesture | State | Result |
|---|---|---|
| ☝️ Index finger up, the other three folded | `MOVING` | The cursor follows your fingertip |
| ☝️ + quick pinch (thumb to index) | `CLICKING` | Left click. Two quick pinches make a double-click |
| ☝️ + hold the pinch longer than 0.35 s, then move | `DRAGGING` | Drag and drop. Open your fingers to drop |
| ✌️ Index and middle up, move your hand up or down | `SCROLLING` | Smooth, trackpad-like scrolling |
| ✌️ + pinch | `RIGHT CLICK` | One right click |
| ✋ All four fingers up, held 0.3 s | `LISTENING` | Records a voice command until you lower your hand |
| Any other pose | `TRACKING` | Nothing happens |

How it stays accurate:

- **Pose confirmation.** Every pose must be held for 0.1 s, so passing through one by accident does nothing.
- **Size-independent pinch.** The pinch is measured relative to hand size, so it works near or far from the camera.
- **Hysteresis and cooldown.** A pinch starts below 0.30 and ends only above 0.45, with a 0.2 s cooldown. One pinch always gives exactly one click.
- **Steady clicks and drags.** The cursor freezes while you pinch, so clicks land where you aimed. When a drag starts, the cursor is anchored so it doesn't jump.
- **Native events.** Clicks, drags and scrolls are native Quartz events, so drag-and-drop and double-click work in every app.

## Voice commands (English)

**To talk:** hold up an open palm ✋ for 0.3 s, or press **V**. Coral rings and a level meter appear. Speak, then lower your hand (or press **V** again).
The HUD shows what was heard and what was done, for example `"open Safari" → OPEN SAFARI`.

| Say | Action |
|---|---|
| `enable control` / `disable control` / `stop` | Turn mouse control on or off |
| `click`, `double click`, `right click` | Click at the cursor |
| `scroll down` / `scroll up` (+ `a lot` / `a little`), `page down`, `scroll to the top` | Scroll |
| `open Safari`, `open chrome`, `launch VS Code`, or just `Spotify` | Open an app (fuzzy-matched against installed apps) |
| `open YouTube`, `go to github.com`, `open YouTube in Safari` | Open a website |
| `quit Spotify` | Quit an app |
| `type hello world`, `type hello in Notes` | Type text (any characters, no clipboard) |
| `search for weather in Istanbul`, `google …` | Google search |
| `new tab`, `close tab`, `next tab`, `reopen tab`, `go back`, `reload` | Browser |
| `copy`, `paste`, `cut`, `undo`, `redo`, `select all`, `save`, `find` | Editing |
| `enter`, `escape`, `delete`, `delete word`, `space`, `tab` | Keys |
| `volume up` / `volume down`, `mute` / `unmute` | Volume |
| `screenshot`, `spotlight`, `switch app`, `mission control`, `minimize`, `full screen`, `zoom in` / `zoom out` | System |
| `start recording` / `stop recording` | [Recording](#recording-for-videos) |

- Fillers ("hey Deniz", "please", "okay") are ignored, and near-misses ("close tap") are corrected.
- Every command except `enable control` requires mouse control to be **on**.
- The microphone is open **only while listening**, and stops by itself after 10 s.
- Recognition uses Whisper `small.en` through whisper.cpp with the Metal GPU, about 0.2 s per command on an M4.

## Recording for videos

Press **R** (or say `start recording`) to start. Press **R** again (or say `stop recording`) to stop.
A blinking `REC 00:42` badge appears in the preview but is **not** part of the recording.
Files are saved to `recordings/<date_time>/`:

| File | Content |
|---|---|
| `final.mp4` | **Ready to upload:** the screen, your camera in the bottom-right corner, and audio |
| `camera.mp4` | Camera view with the HUD, constant 30 FPS, H.264 |
| `camera_audio.mp4` | Camera view with sound |
| `screen.mp4` | Whole screen with cursor and click highlights |
| `audio.wav` | Microphone, 48 kHz |
| `session.json` | Sync offsets between the tracks |

- **Clean shots.** Press **H** for a clean HUD while filming.
- **Your screen.** The preview window is on your screen too, so move it aside if you don't want it in the screen recording.
- **Sync.** All tracks share one timeline. The camera is timed by capture time, the audio by its first sample, and the screen by the exact clock time of its first frame.
  Measured offsets are about 40–50 ms (check them yourself with `python -m tests.test_recording_sync`). Fine-tune with `RecordConfig.screen_offset_s` / `camera_latency_s`.
- **Microphone choice.** Recordings use the macOS default input. To use another mic, set `RecordConfig.audio_input = "Wireless Mic"` (part of the device name; list devices with `python -m sounddevice`).
- **Disk space.** About 100–200 MB per minute. A recording won't start with less than 2 GB free, and stops by itself below 1 GB.
- **ffmpeg** is required for `screen.mp4` and the merged files.

## Keyboard shortcuts

These work while the preview window is focused.

| Key | Action |
|---|---|
| `SPACE` | Mouse control on/off (**starts off**) |
| `V` | Start/stop listening for a voice command |
| `R` | Start/stop recording |
| `H` | Clean view for filming |
| `Q` / `ESC` | Quit |

Command-line options:

| Option | Effect |
|---|---|
| `--no-mouse` | Preview only: never touches the mouse, and voice commands are shown but not run |
| `--no-voice` | The microphone is never used |
| `--camera 1` | Use another camera |
| `-v` | Debug logging |

## Safety

- **Starts off.** Mouse control is **off** at launch.
- **Hands down, mouse free.** The mouse is only moved while you hold a control pose. Lower your hand and your trackpad works normally.
- **Fail-safe.** PyAutoGUI's fail-safe stays on: move the real cursor into any screen corner and control turns off.
  The app keeps its own movements away from the edges, so only you can trigger it.
- **No stuck buttons.** A held mouse button is always released when you open the pinch, the hand disappears, control turns off, the fail-safe fires, or the app quits (including after Ctrl+C or an error).
- **Focus.** Keyboard shortcuts need the preview window in focus. If a click moved focus elsewhere, click the preview again, use the fail-safe corner, or press Ctrl+C in the terminal.

## Testing step by step

```bash
python -m tests.test_camera                        # 1. camera only
python -m tests.test_hand_tracking                 # 2. tracking + gesture states, mouse never touched
python -m tests.test_voice --say "open Safari"     # 3. speech -> text -> command, nothing executed
python -m tests.test_voice --mic 4                 #    ... with your own voice (speak within 4 s)
python -m tests.test_recording_sync                # 4. measures recording sync (window + 3 beeps)
python -m unittest discover -s tests -p "test_*s.py" -v   # offline unit tests (no camera / mic)
```

## Tuning

All settings live in [`deniz_control/config.py`](deniz_control/config.py).

| Setting | Effect |
|---|---|
| `MouseConfig.active_region` | Part of the camera frame mapped to the full screen. Smaller means less arm movement |
| `MouseConfig.sensitivity`, `smoothing_alpha`, `deadzone_px` | Cursor speed, smoothness and jitter filter |
| `MouseConfig.scroll_speed`, `scroll_invert` | Scrolling |
| `GestureConfig.pinch_on_ratio` / `pinch_off_ratio` | Click sensitivity and hysteresis |
| `GestureConfig.drag_hold_s`, `click_cooldown_s`, `*_confirm_s` | Gesture timing |
| `VoiceConfig.model_path` / `model_url` | Use another Whisper model (e.g. `medium.en`) |
| `RecordConfig` | Bitrates, picture-in-picture size, microphone, display, sync offsets |
| `UIConfig` | Colors, glow, trail, camera brightness (`video_gain` / `video_lift`) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera` | Grant the Camera permission and close other camera apps. Try `--camera 1` |
| Cursor doesn't move / no clicks | Grant Accessibility to your terminal and **restart it**. Check that the HUD says `MOUSE CONTROL: ON` |
| SPACE does nothing | Click the preview window first (it needs keyboard focus) |
| Never reaches `MOVING` | Fold the other three fingers fully and keep your hand inside the green corner brackets |
| Accidental clicks / clicks hard to trigger | Lower or raise `pinch_on_ratio` (watch the `PINCH` value in the HUD) |
| Drags drop too early | Raise `pinch_off_ratio` (e.g. 0.5) |
| FPS around 15 | macOS lowers webcam FPS in dim light. Add light |
| `voice model still loading` | Wait a moment. On the first run the 470 MB download has to finish |
| `microphone unavailable` / `no speech heard` | Grant the Microphone permission; speak closer; lower `VoiceConfig.min_speech_rms` |
| Words misheard | Say the command alone, clearly. `python -m tests.test_voice --mic 4` shows the raw transcript |
| No `screen.mp4`, or it's black | Grant Screen Recording to your terminal and restart it. Install ffmpeg |
| `ImportError` mentioning `cv2` | `pip uninstall -y opencv-python opencv-contrib-python && pip install -r requirements.txt` |
| HUD says "Left" for your right hand | Expected: the image is mirrored before detection |

## How it works

```
Camera (thread) ─► HandTracker (MediaPipe) ─► Gesture state machine ─► Mouse (Quartz events)
                                                   │                 └─► HUD (OpenCV)  ─► Recorder (ffmpeg)
                                                   └─ open palm ─► Mic ─► Whisper (thread) ─► Command parser ─► Executor
```

| Path | Role |
|---|---|
| `deniz_control/camera.py` | Threaded, mirrored AVFoundation capture with auto-reconnect |
| `deniz_control/hand_tracker.py` | MediaPipe Tasks `HandLandmarker` (VIDEO mode) |
| `deniz_control/gestures.py` | Finger states, poses, pinch hysteresis, state machine (pure logic, unit-tested) |
| `deniz_control/mouse_controller.py` | Camera-to-screen mapping, smoothing, Quartz click/drag/scroll |
| `deniz_control/overlay.py` | Dark HUD: cyan landmarks, glowing trail, click ripples, voice meter |
| `deniz_control/voice/` | Microphone, offline Whisper, English command parser, executor |
| `deniz_control/recorder.py` | Camera + audio + screen recording, sync and merging |
| `deniz_control/config.py` | Every tunable value |
| `deniz_control/app.py`, `main.py` | Main loop and entry point |
| `tests/` | Hardware tests (camera, tracking, voice, sync) and offline unit tests |

## Credits and license

Built with [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0), [OpenCV](https://opencv.org) (Apache 2.0), [whisper.cpp](https://github.com/ggml-org/whisper.cpp) via [pywhispercpp](https://github.com/absadiki/pywhispercpp) (MIT), [PyAutoGUI](https://github.com/asweigart/pyautogui) (BSD), [sounddevice](https://github.com/spatialaudio/python-sounddevice) (MIT) and [NumPy](https://numpy.org) (BSD).
The models are downloaded from their official sources at first run and are not part of this repository: Google's hand landmarker (Apache 2.0) and OpenAI's Whisper `small.en` in ggml format (MIT).

This project is released under the [MIT License](LICENSE): free to use, modify and share.
