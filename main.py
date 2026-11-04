"""DENIZ CONTROL entry point.

    python main.py              # full app (mouse control starts DISABLED; press SPACE)
    python main.py --no-mouse   # preview + gesture states only, never touches the mouse
    python main.py --no-voice   # gestures only, microphone never used
"""

import argparse
import logging
import sys

from deniz_control.app import run


def main() -> int:
    parser = argparse.ArgumentParser(description="DENIZ CONTROL - touchless hand + voice control")
    parser.add_argument("--no-mouse", action="store_true",
                        help="preview only; never move or click the mouse")
    parser.add_argument("--no-voice", action="store_true",
                        help="disable voice commands (microphone is never opened)")
    parser.add_argument("--camera", type=int, default=None, help="camera index (default 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    from dataclasses import replace
    from deniz_control.config import AppConfig

    cfg = AppConfig()
    if args.camera is not None:
        cfg = replace(cfg, camera=replace(cfg.camera, index=args.camera))
    return run(cfg, enable_mouse=not args.no_mouse, enable_voice=False if args.no_voice else None)


if __name__ == "__main__":
    sys.exit(main())
