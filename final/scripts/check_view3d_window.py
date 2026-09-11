#!/usr/bin/env python
"""Manual check for the interactive 3D window (the dashboard's 3D button).

Opens the *real* Open3D window through the same code path the dashboard uses
(:func:`web.view3d.launch_viewer`, a detached ``python -m web.view3d`` process),
waits a few seconds, checks the child's log to prove the window was actually
created, then closes the child again.  Nothing here touches Flask.

    python scripts/check_view3d_window.py                 # 8 s, default view
    python scripts/check_view3d_window.py --view adaptive_density --seconds 15
    python scripts/check_view3d_window.py --list-views

Requires a desktop session with an OpenGL driver - the same requirement as the
offscreen PNG panels.  Exits 0 when the window opened, 1 otherwise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.render3d import VIEWS  # noqa: E402
from web.view3d import launch_viewer, viewer_log_path  # noqa: E402


def check(condition: bool, message: str) -> bool:
    print(f"[{'ok  ' if condition else 'FAIL'}] {message}")
    return condition


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Open the interactive Open3D window briefly and verify it")
    parser.add_argument("--view", default=VIEWS[0], choices=list(VIEWS))
    parser.add_argument("--seconds", type=float, default=8.0,
                        help="how long to leave the window open")
    parser.add_argument("--list-views", action="store_true")
    args = parser.parse_args(argv)

    if args.list_views:
        for name in VIEWS:
            print(name)
        return 0

    log = viewer_log_path(args.view)
    before = 0   # launch_viewer truncates the log, so the whole file is ours

    print(f"spawning the viewer window for {args.view} ({args.seconds:g} s) ...")
    result = launch_viewer(args.view)
    if not check(bool(result.get("ok")), f"launch_viewer returned ok: {result}"):
        return 1
    print(f"  pid {result['pid']}, log {result['log']}")

    time.sleep(args.seconds)

    text = ""
    if log.exists():
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(before)
            text = fh.read()

    ok = True
    ok &= check("building scene from" in text, "the child started and ran the pipeline")
    ok &= check(text.count("view ->") >= 1, "geometry was attached to the window")
    ok &= check("hotkeys:" in text,
                "the Open3D window was created (event loop reached)")
    ok &= check("ERROR:" not in text,
                "no error in the viewer log")
    if not ok:
        print("\n--- viewer log ---")
        print(text)

    subprocess.run(["taskkill", "/PID", str(result["pid"]), "/T", "/F"],
                   capture_output=True, check=False)
    print(f"closed pid {result['pid']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
