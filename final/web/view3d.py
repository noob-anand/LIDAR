#!/usr/bin/env python
"""Interactive Open3D window for the 2.5D map, launched from the dashboard.

The dashboard's own 3D panels are static PNGs rendered offscreen by
:mod:`web.render3d`.  This module opens *the same geometry* in a real Open3D
window that can be rotated, panned and zoomed with the mouse, and switches
between the four views without re-running the pipeline.

Controls
--------
    left-drag             rotate
    wheel / middle-drag   zoom
    shift + left-drag     pan
    ctrl + left-drag      roll
    1 2 3 4               adaptive-semantic / adaptive-density /
                          uniform-semantic / uniform-density
    r                     reset the camera to the configured pose
    s                     save a PNG snapshot into ``reports/``
    h                     print this control list again
    q / esc               close the window

Usage
-----
    python -m web.view3d
    python -m web.view3d --view adaptive_density
    python -m web.view3d --dataset data/fake_lidar_000000.bin

The dashboard button POSTs to ``/api/view3d``, which calls :func:`launch_viewer`
to start this module as a *detached* process: the window belongs to the desktop
session running the server, so the browser stays a browser and the Flask
process is never blocked by the Open3D event loop.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mapping import load_bin  # noqa: E402
from mapping.config import load_config  # noqa: E402
from mapping.pipeline import LiDARMapper  # noqa: E402
from mapping.preprocess import filter_points  # noqa: E402

try:  # package import: python -m web.view3d
    from .metrics import (build_uniform_grid, density_per_cell, drivability_tiers,
                          uniform_drivability_tiers)
    from .render3d import (VIEW_TITLES, VIEWS, RenderUnavailable, apply_camera,
                           camera_settings, scene_clouds)
except ImportError:  # script import: python web/view3d.py
    from metrics import (build_uniform_grid, density_per_cell, drivability_tiers,
                         uniform_drivability_tiers)
    from render3d import (VIEW_TITLES, VIEWS, RenderUnavailable, apply_camera,
                          camera_settings, scene_clouds)

DATA_PATH = ROOT / "data" / "fake_lidar_000000.bin"
CONFIG_PATH = ROOT / "configs" / "default.yaml"
SNAPSHOT_DIR = ROOT / "reports"

CONTROLS: Tuple[Tuple[str, str], ...] = (
    ("left-drag", "rotate"),
    ("wheel / middle-drag", "zoom"),
    ("shift + left-drag", "pan"),
    ("ctrl + left-drag", "roll"),
    ("1 2 3 4", "switch view (adaptive/uniform x semantic/density)"),
    ("r", "reset camera to the configured pose"),
    ("s", "save a PNG snapshot into reports/"),
    ("h", "print this control list"),
    ("q / esc", "close the window"),
)

# --------------------------------------------------------------------------- #
# scene construction (no OpenGL: safe to call headless, and testable)
# --------------------------------------------------------------------------- #
def load_config_for_viewer(config: Path = CONFIG_PATH):
    """Load the YAML config with the pipeline's own visualiser switched off."""
    cfg = load_config(str(config))
    cfg.set("viz.backend", "none")  # only this module may open a window
    return cfg


def build_scene(
    dataset: Path = DATA_PATH,
    config: Path = CONFIG_PATH,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Run the pipeline and return ``{view: (points, colours)}`` for all views.

    This is the same chain the dashboard runs (preprocess -> segment -> adaptive
    grid -> drivability/density layers -> uniform baseline), then
    :func:`web.render3d.scene_clouds`, so the live window and the PNG panels
    always show the same geometry.
    """
    cfg = load_config_for_viewer(Path(config))
    mapper = LiDARMapper(cfg)
    pc = load_bin(str(dataset))
    result = mapper.process_frame(pc)
    grid = result.grid

    n_classes = len(cfg.get("classes.names", []) or []) or 3
    pre = cfg.section("preprocess")
    xyz_f, _ = filter_points(
        pc.xyz,
        float(pre.get("min_range", 1.0)),
        float(pre.get("max_range", 100.0)),
        float(pre.get("height_min", -5.0)),
        float(pre.get("height_max", 5.0)),
        pre.get("decimation"),
    )
    if xyz_f.shape[0] != result.labels.size:  # pragma: no cover - defensive
        from mapping.preprocess import preprocess

        xyz_f = preprocess(pc.xyz, pc.intensity, cfg).xyz

    uniform = build_uniform_grid(
        xyz_f, result.labels, n_classes,
        float(cfg.get("metrics.uniform_cell_size", 0.05)),
    )
    tier, _, _ = drivability_tiers(grid, cfg)
    density = density_per_cell(grid)
    u_tier, _, _ = uniform_drivability_tiers(uniform, cfg)
    uniform["tier"] = u_tier

    return scene_clouds(grid, uniform, tier, density, cfg, n_classes)


def describe(clouds: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> str:
    """One line per view, saying how much geometry it holds."""
    lines = []
    for name in VIEWS:
        pts, _ = clouds.get(name, (np.zeros((0, 3)), np.zeros((0, 3))))
        lines.append(f"  {name:<18} {pts.shape[0]:>9,} points")
    return "\n".join(lines)


def print_controls() -> None:
    """Print the hotkey/mouse reference to the console (and the view3d log)."""
    print("\ncontrols")
    for keys, what in CONTROLS:
        print(f"  {keys:<24} {what}")


# --------------------------------------------------------------------------- #
# the interactive window
# --------------------------------------------------------------------------- #
def _make_visualizer(o3d):
    """Return a visualizer that can handle hotkeys.

    ``o3d.visualization.Visualizer`` is the plain renderer and has no
    ``register_key_callback`` (Open3D 0.19), so the key-capable subclass is used
    when it exists and the plain one is the fallback (mouse-only window).
    """
    cls = getattr(o3d.visualization, "VisualizerWithKeyCallback", None)
    if cls is None:  # pragma: no cover - very old/new Open3D builds
        print("note: this Open3D has no key-callback visualizer; "
              "the window is mouse-only (q / esc still closes it)")
        return o3d.visualization.Visualizer()
    return cls()


def run_window(
    clouds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cfg,
    start_view: str = VIEWS[0],
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> None:
    """Open the window and block in Open3D's event loop until it is closed."""
    try:
        import open3d as o3d
    except BaseException as exc:  # pragma: no cover - depends on the machine
        raise RenderUnavailable(f"open3d is not importable: {exc!r}") from exc

    cam = camera_settings(cfg)
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    state: Dict[str, object] = {
        "view": start_view if start_view in clouds else VIEWS[0],
        "pcd": None,
    }

    def title() -> str:
        name = str(state["view"])
        return (f"siih 2.5D map - {VIEW_TITLES.get(name, name)} "
                f"({clouds[name][0].shape[0]:,} points)")

    vis = _make_visualizer(o3d)
    try:
        ok = vis.create_window(window_name=title(),
                               width=int(cam["width"]),
                               height=int(cam["height"]), visible=True)
    except BaseException as exc:  # pragma: no cover - driver specific
        raise RenderUnavailable(f"Open3D window creation failed: {exc}") from exc
    if not ok:
        raise RenderUnavailable(
            "Open3D could not open a window (needs a desktop session with an "
            "OpenGL driver on the machine running the server)")
    print(f"window created: {title()}")

    opt = vis.get_render_option()
    opt.background_color = np.asarray(cam["background"], dtype=np.float64)
    opt.point_size = float(cam["point_size"])
    opt.light_on = False            # flat colours: the palette must show exactly
    opt.show_coordinate_frame = False

    def reset_camera() -> None:
        apply_camera(vis.get_view_control(), elev_deg=float(cam["elev"]),
                     azim_deg=float(cam["azim"]), zoom=float(cam["zoom"]))

    def show(name: str) -> None:
        """Swap the geometry in place; the camera keeps the user's pose."""
        state["view"] = name
        pts, cols = clouds[name]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(cols, dtype=np.float64), 0.0, 1.0))
        old = state["pcd"]
        if old is not None:
            vis.remove_geometry(old, reset_bounding_box=False)
        vis.add_geometry(pcd, reset_bounding_box=old is None)
        state["pcd"] = pcd
        vis.poll_events()
        vis.update_renderer()
        if hasattr(vis, "update_window_title"):
            vis.update_window_title(title())
        print(f"view -> {name} ({pts.shape[0]:,} points)")

    def snapshot() -> None:
        path = snapshot_dir / (
            f"view3d_{time.strftime('%Y%m%d_%H%M%S')}_{state['view']}.png")
        vis.capture_screen_image(str(path), do_render=True)
        print(f"saved {path}")

    register = getattr(vis, "register_key_callback", None)
    if callable(register):
        for idx, name in enumerate(VIEWS, start=1):
            register(ord(str(idx)), lambda _v, n=name: show(n))
        for key in ("R", "r"):
            register(ord(key), lambda _v: reset_camera())
        for key in ("S", "s"):
            register(ord(key), lambda _v: snapshot())
        for key in ("H", "h"):
            register(ord(key), lambda _v: print_controls())
    else:  # pragma: no cover - plain Visualizer fallback
        print("note: this window has no hotkeys; drag with the mouse instead")

    show(str(state["view"]))
    reset_camera()
    print(f"\n{title()}\nhotkeys: 1-4 switch view, r resets the camera, "
          "s snapshots, q quits")
    try:
        vis.run()        # blocking event loop: the key callbacks fire in here
    finally:
        vis.destroy_window()


# --------------------------------------------------------------------------- #
# detached launcher (used by the dashboard's "Open interactive 3D" button)
# --------------------------------------------------------------------------- #
def viewer_log_path(view: Optional[str] = None) -> Path:
    """Where a spawned viewer writes its stdout/stderr.

    One log per view, so two windows (e.g. adaptive and uniform) open at the
    same time do not overwrite each other's output.  The file is truncated on
    every launch: it always describes the *current* window, which is what makes
    it useful when a window fails to appear.
    """
    name = "siih_view3d.log" if not view else f"siih_view3d_{view}.log"
    return Path(tempfile.gettempdir()) / name


def _launch_result(
    view: str,
    *,
    pid: Optional[int] = None,
    log: Optional[Path] = None,
    cmd: Optional[list] = None,
    already_open: bool = False,
) -> Dict[str, object]:
    """Uniform success payload for :func:`launch_viewer` (JSON-friendly)."""
    out: Dict[str, object] = {
        "ok": True,
        "view": view,
        "title": VIEW_TITLES.get(view, view),
        "views": list(VIEWS),
        "snapshots": str(SNAPSHOT_DIR),
        "already_open": bool(already_open),
    }
    if pid is not None:
        out["pid"] = int(pid)
    if log is not None:
        out["log"] = str(log)
    if cmd is not None:
        out["command"] = " ".join(cmd)
    return out


def launch_viewer(
    view: str = VIEWS[0],
    dataset: Path = DATA_PATH,
    config: Path = CONFIG_PATH,
    log_path: Optional[Path] = None,
    registry: Optional[Dict[str, subprocess.Popen]] = None,
) -> Dict[str, object]:
    """Start :mod:`web.view3d` as a detached process and report how it went.

    Returns a JSON-friendly dict; ``ok`` is ``False`` with an ``error`` string
    when the request cannot be served at all (unknown view, missing file, spawn
    failure).  The child is detached, so it neither blocks the Flask worker nor
    dies when the server stops - it owns its Open3D window until closed.

    Pass ``registry`` (a ``{view: Popen}`` dict, e.g. owned by the caller) to
    remember spawned windows: a viewer that is still running for ``view`` is
    reported as ``already_open`` instead of opening a second identical window,
    and entries whose process has exited are pruned.
    """
    if view not in VIEWS:
        return {"ok": False, "error": f"unknown view {view!r}",
                "views": list(VIEWS)}
    if registry is not None:
        for name in [k for k, p in registry.items() if p.poll() is not None]:
            del registry[name]              # that window has been closed
        running = registry.get(view)
        if running is not None:
            return _launch_result(view, pid=int(running.pid), already_open=True)
    dataset, config = Path(dataset), Path(config)
    if not dataset.exists():
        return {"ok": False, "error": f"dataset not found: {dataset}"}
    if not config.exists():
        return {"ok": False, "error": f"config not found: {config}"}

    log = Path(log_path) if log_path else viewer_log_path(view)
    cmd = [sys.executable, "-m", "web.view3d",
           "--view", view,
           "--dataset", str(dataset),
           "--config", str(config),
           "--snapshot-dir", str(SNAPSHOT_DIR)]

    kwargs: Dict[str, object] = {}
    if os.name == "nt":     # own process group, no console: survives Ctrl+C
        flags = getattr(subprocess, "DETACHED_PROCESS", 0)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    try:
        # "wb": one log describes one window, so a stale ERROR from a previous
        # launch can never be mistaken for a failure of this one.
        with open(log, "wb") as fh:
            fh.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} "
                     f"{' '.join(cmd)}\n".encode("utf-8"))
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                    stdout=fh, stderr=subprocess.STDOUT, **kwargs)
    except BaseException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "log": str(log)}
    if registry is not None:
        registry[view] = proc

    return _launch_result(view, pid=int(proc.pid), log=log, cmd=cmd)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive Open3D window for the adaptive 2.5D map")
    parser.add_argument("--dataset", default=str(DATA_PATH),
                        help="point cloud file (KITTI/Velodyne .bin)")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="YAML config file")
    parser.add_argument("--view", default=VIEWS[0], choices=list(VIEWS),
                        help="view to open first")
    parser.add_argument("--snapshot-dir", default=str(SNAPSHOT_DIR),
                        help="directory for the 's' PNG snapshots")
    parser.add_argument("--list-views", action="store_true",
                        help="print the available view names and exit")
    parser.add_argument("--check", action="store_true",
                        help="build the geometry, report it and exit (no window)")
    args = parser.parse_args(argv)

    # The dashboard redirects this process' stdout into a log file, so make it
    # line-buffered: the log has to be useful *while* the window is open.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except BaseException:  # pragma: no cover - exotic/old stdout objects
        pass

    if args.list_views:
        for name in VIEWS:
            print(f"{name:<18} {VIEW_TITLES.get(name, '')}")
        return 0

    print(f"building scene from {Path(args.dataset).name} ...")
    t0 = time.perf_counter()
    try:
        clouds = build_scene(Path(args.dataset), Path(args.config))
    except BaseException as exc:
        print(f"ERROR: could not build the scene: {type(exc).__name__}: {exc}")
        return 2
    print(f"built in {(time.perf_counter() - t0) * 1000.0:.0f} ms")
    print(describe(clouds))

    if args.check:
        empty = [n for n in VIEWS if clouds[n][0].shape[0] == 0]
        if empty:
            print(f"ERROR: empty views: {', '.join(empty)}")
            return 1
        print(f"all {len(VIEWS)} views ready (--check: no window opened)")
        return 0

    print_controls()
    try:
        run_window(clouds, load_config_for_viewer(Path(args.config)),
                   start_view=args.view, snapshot_dir=Path(args.snapshot_dir))
    except RenderUnavailable as exc:
        print(f"ERROR: {exc}")
        return 3
    except BaseException as exc:  # pragma: no cover - driver specific
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 4
    print("window closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
