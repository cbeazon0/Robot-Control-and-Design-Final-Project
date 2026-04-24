"""Capture frames from the robot camera for YOLO fine-tuning.

Designed to run headlessly over SSH on the Raspberry Pi (no GUI required).
Re-uses the same robust V4L2 fallback logic as ``yolo_detector`` so it
opens the same physical camera.

Two capture modes
-----------------
* ``interval`` (default) -- auto-snap every ``--interval`` seconds.
* ``manual``             -- snap on every press of Enter from the terminal.

Saved files
-----------
Every image is prefixed with the sign label supplied via ``--label``:

    <label>_<NNNN>.jpg               (raw frame)
    <label>_<NNNN>_proc.jpg          (if --save-processed)

Allowed labels: ``Stop``, ``Turn_Around``, ``Right_Turn``, ``Left_Turn``,
``Goal``. Case-insensitive; spaces/hyphens are accepted (``"Turn Around"``
becomes ``Turn_Around``).

By default frames are written under
``~/yolo_dataset/<label>/<run_id>/``.

Usage
-----
::

    # auto-capture Stop-sign shots every 2 s
    ros2 run yolo_detector capture_dataset --label Stop

    # press Enter to snap, custom output dir
    ros2 run yolo_detector capture_dataset --label Right_Turn --mode manual \\
        --output-dir ~/yolo_dataset

    # also save the preprocessed (red/quad) frames
    ros2 run yolo_detector capture_dataset --label Goal --save-processed

    # different camera node
    ros2 run yolo_detector capture_dataset --label Left_Turn \\
        --camera-device /dev/video1

You can also use it without ``ros2 run`` -- it's just a regular Python
script::

    python3 -m yolo_detector.capture_dataset --interval 1.5

Press Ctrl-C at any time to stop. The current count is printed after every
successful capture.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time

import cv2

try:
    # Re-use the preprocessor if the user wants processed frames too.
    from .yolo_detector import preprocess_frame
except Exception:  # pragma: no cover - keep usable as a script
    preprocess_frame = None  # type: ignore[assignment]


def _open_camera(device: str) -> cv2.VideoCapture:
    """Open ``device`` using the same fallback chain as yolo_detector."""
    attempts: list[tuple[object, int, str]] = []
    if device.startswith('/dev/video'):
        attempts.append((device, cv2.CAP_V4L2, 'path+V4L2'))
        attempts.append((device, cv2.CAP_ANY, 'path+ANY'))
        try:
            idx = int(device.replace('/dev/video', ''))
            attempts.append((idx, cv2.CAP_V4L2, 'index+V4L2'))
        except ValueError:
            pass
    else:
        attempts.append((device, cv2.CAP_ANY, 'source+ANY'))

    for source, backend, label in attempts:
        print(f"[capture] opening camera '{device}' "
              f"(source={source!r}, backend={label})...", flush=True)
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            print(f"[capture] camera opened via source={source!r}.",
                  flush=True)
            return cap
        cap.release()

    raise RuntimeError(
        f"Could not open camera '{device}'. Check that the device exists, "
        "that no other process (e.g. yolo_detector) is holding it, and "
        "that your user is in the 'video' group."
    )


# Canonical label set. Keys are the lowercase normalized form; values are
# the filename-safe label (no spaces).
LABEL_CHOICES: dict[str, str] = {
    'stop': 'Stop',
    'turn_around': 'Turn_Around',
    'turnaround': 'Turn_Around',
    'right_turn': 'Right_Turn',
    'rightturn': 'Right_Turn',
    'left_turn': 'Left_Turn',
    'leftturn': 'Left_Turn',
    'goal': 'Goal',
}


def _normalize_label(raw: str) -> str:
    """Map any accepted spelling to its canonical filename form."""
    key = raw.strip().lower().replace(' ', '_').replace('-', '_')
    while '__' in key:
        key = key.replace('__', '_')
    if key not in LABEL_CHOICES:
        canonical = sorted(set(LABEL_CHOICES.values()))
        raise argparse.ArgumentTypeError(
            f"Invalid --label '{raw}'. Allowed: {canonical}"
        )
    return LABEL_CHOICES[key]


def _next_run_dir(base: str, label: str) -> str:
    """Return ``base/<label>/run_YYYYmmdd_HHMMSS``, creating it if needed."""
    stamp = _dt.datetime.now().strftime('run_%Y%m%d_%H%M%S')
    out = os.path.join(base, label, stamp)
    os.makedirs(out, exist_ok=True)
    return out


def _save_frame(out_dir: str, label: str, idx: int, frame,
                save_processed: bool,
                jpeg_quality: int) -> tuple[str, str | None]:
    """Write the raw (and optionally processed) frame. Returns paths."""
    raw_path = os.path.join(out_dir, f'{label}_{idx:04d}.jpg')
    cv2.imwrite(raw_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

    proc_path: str | None = None
    if save_processed:
        if preprocess_frame is None:
            raise RuntimeError(
                'Could not import preprocess_frame from yolo_detector. '
                'Run from inside the colcon-built workspace, or skip '
                '--save-processed.'
            )
        try:
            processed, _ = preprocess_frame(frame)
            proc_path = os.path.join(out_dir, f'{label}_{idx:04d}_proc.jpg')
            cv2.imwrite(
                proc_path, processed,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[capture] preprocess failed on frame {idx}: {exc}",
                  flush=True)
            proc_path = None
    return raw_path, proc_path


def _drain_camera(cap: cv2.VideoCapture, n: int = 3) -> None:
    """Throw away the next ``n`` frames so we capture a fresh one.

    USB/V4L2 cameras buffer a few frames; without draining, manual mode
    snaps an old frame from before the user pressed Enter.
    """
    for _ in range(max(0, n)):
        ok, _ = cap.read()
        if not ok:
            return


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Capture camera frames for YOLO fine-tuning.',
    )
    parser.add_argument(
        '--label', required=True, type=_normalize_label,
        help='Sign label for the whole run. One of: '
             'Stop, Turn_Around, Right_Turn, Left_Turn, Goal. '
             'Case-insensitive; spaces/hyphens are accepted.',
    )
    parser.add_argument(
        '--camera-device', default='/dev/video0',
        help='Camera device path or index (default: /dev/video0).',
    )
    parser.add_argument(
        '--output-dir', default=os.path.expanduser('~/yolo_dataset'),
        help='Base output directory. Frames are written to '
             '<output-dir>/<label>/run_<timestamp>/ '
             '(default: ~/yolo_dataset).',
    )
    parser.add_argument(
        '--mode', choices=('interval', 'manual'), default='interval',
        help="'interval' = auto-capture every --interval seconds. "
             "'manual'  = press Enter in the terminal to snap.",
    )
    parser.add_argument(
        '--interval', type=float, default=2.0,
        help='Seconds between captures in interval mode (default: 2.0).',
    )
    parser.add_argument(
        '--max-frames', type=int, default=0,
        help='Stop after this many frames. 0 = unlimited (default: 0).',
    )
    parser.add_argument(
        '--save-processed', action='store_true',
        help='Also save the preprocessed (red+quad mask) frame next to '
             'each raw frame.',
    )
    parser.add_argument(
        '--jpeg-quality', type=int, default=92,
        help='JPEG quality 1-100 (default: 92).',
    )
    parser.add_argument(
        '--warmup-frames', type=int, default=5,
        help='Read and discard this many frames before the first capture '
             'so auto-exposure has time to settle (default: 5).',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    out_dir = _next_run_dir(os.path.expanduser(args.output_dir), args.label)
    print(f"[capture] label={args.label}", flush=True)
    print(f"[capture] saving frames to: {out_dir}", flush=True)
    print(f"[capture] mode={args.mode}, "
          f"interval={args.interval:.2f}s, "
          f"max_frames={args.max_frames or 'unlimited'}, "
          f"save_processed={args.save_processed}",
          flush=True)

    cap = _open_camera(args.camera_device)

    # Warm up so the first frame isn't black / way underexposed.
    print(f"[capture] warming up ({args.warmup_frames} frames)...", flush=True)
    for _ in range(max(0, args.warmup_frames)):
        cap.read()

    if args.mode == 'manual':
        print('[capture] manual mode: press Enter to snap a frame, '
              'Ctrl-C to quit.', flush=True)

    count = 0
    next_tick = time.monotonic()
    try:
        while True:
            if args.max_frames and count >= args.max_frames:
                print(f"[capture] reached max_frames={args.max_frames}; "
                      "stopping.", flush=True)
                break

            if args.mode == 'manual':
                try:
                    sys.stdout.write(f"[capture] [{count + 1}] press Enter "
                                     "to snap (or 'q'+Enter to quit) > ")
                    sys.stdout.flush()
                    line = sys.stdin.readline()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if line == '':
                    # stdin closed
                    break
                if line.strip().lower() in ('q', 'quit', 'exit'):
                    break
                _drain_camera(cap, n=3)
            else:
                # interval mode: sleep until next tick
                now = time.monotonic()
                wait = next_tick - now
                if wait > 0:
                    time.sleep(wait)
                next_tick = time.monotonic() + max(0.05, args.interval)

            ok, frame = cap.read()
            if not ok or frame is None:
                print('[capture] failed to read frame, retrying...',
                      flush=True)
                time.sleep(0.1)
                continue

            count += 1
            raw_path, proc_path = _save_frame(
                out_dir=out_dir,
                label=args.label,
                idx=count,
                frame=frame,
                save_processed=args.save_processed,
                jpeg_quality=args.jpeg_quality,
            )
            extra = f"  +proc={os.path.basename(proc_path)}" if proc_path else ''
            print(f"[capture] [{count}] saved {os.path.basename(raw_path)}"
                  f"{extra}", flush=True)
    except KeyboardInterrupt:
        print()
        print('[capture] interrupted; stopping.', flush=True)
    finally:
        try:
            cap.release()
        except Exception:
            pass

    print(f"[capture] done. {count} frame(s) saved to {out_dir}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
