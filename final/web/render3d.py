"""Offscreen 3D renderers (Open3D) for the dashboard.

Two colour schemes, both drawn with the same deterministic camera so the
adaptive and the uniform map can be compared side by side:

``semantic``
    * walkable ground  -> three colours, one per drivability tier
    * static obstacles -> ramp of the cell's height above the ground plane
    * dynamic objects  -> one high-contrast colour
``density``
    * every occupied cell -> gradient of the grid's sampling density
      (cells per m^2).  This is the foveation itself: the near band is
      extremely dense, the far band is coarse.

Geometry: a 2.5D cell is drawn at its *mean* height when it is walkable terrain
and at its *max* height when it is an obstacle (or dynamic), so structure rises
out of the ground instead of being flattened into a carpet.  Obstacle cells are
additionally extruded downwards with a short column of points so the volume
reads in 3D.

Open3D's ``OffscreenRenderer`` requires EGL, which does not exist on Windows, so
these renderers drive a *hidden* GLFW window via ``o3d.visualization`` and grab
the framebuffer with ``capture_screen_image``.  That needs a desktop session; if
it is missing the functions raise :class:`RenderUnavailable` and the dashboard
prints the reason instead of the panel.
"""
from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mapping import CLASS_DRIVABLE, CLASS_DYNAMIC, CLASS_STATIC  # noqa: E402

try:  # package import: python -m web.app
    from .metrics import density_extent, density_to_t
except ImportError:  # script import: python web.py
    from metrics import density_extent, density_to_t

try:
    import open3d as o3d
    _IMPORT_ERROR: Optional[BaseException] = None
except BaseException as _exc:  # pragma: no cover - depends on the machine
    o3d = None
    _IMPORT_ERROR = _exc


class RenderUnavailable(RuntimeError):
    """No OpenGL/desktop context available for the offscreen 3D render."""


# --------------------------------------------------------------------------- #
# colour helpers
# --------------------------------------------------------------------------- #
def hex_to_rgb(value: str) -> np.ndarray:
    """``"#00e676"`` / ``"0e6"`` -> ``array([0.0, 0.902, 0.463])``."""
    s = str(value).lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"not a hex colour: {value!r}")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def ramp(stops: Sequence[str], t: np.ndarray) -> np.ndarray:
    """Sample a piecewise-linear multi-stop colour ramp at ``t`` in ``[0, 1]``."""
    cols = [hex_to_rgb(c) for c in stops]
    if not cols:
        cols = [hex_to_rgb("#ffffff")]
    stops_rgb = np.asarray(cols, dtype=np.float64)
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    if len(stops_rgb) == 1:
        return np.repeat(stops_rgb, t.size, axis=0)
    pos = np.linspace(0.0, 1.0, len(stops_rgb))
    out = np.empty((t.size, 3), dtype=np.float64)
    for ch in range(3):
        out[:, ch] = np.interp(t, pos, stops_rgb[:, ch])
    return out


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #
#: The four 3D views, in the order they are rendered into the dashboard.
#: ``web/view3d.py`` reuses exactly this list for the interactive window, so a
#: view name always means the same geometry on both paths.
VIEWS: Tuple[str, ...] = (
    "adaptive_semantic",
    "adaptive_density",
    "uniform_semantic",
    "uniform_density",
)

#: Human-readable label per view (window titles, dashboard buttons).
VIEW_TITLES: Dict[str, str] = {
    "adaptive_semantic": "adaptive - semantic",
    "adaptive_density": "adaptive - density",
    "uniform_semantic": "uniform - semantic",
    "uniform_density": "uniform - density",
}


def camera_settings(cfg) -> Dict[str, object]:
    """Render options shared by the offscreen panels and the live 3D window."""
    r3 = cfg.section("render3d")
    return {
        "width": int(r3.get("width", 1280)),
        "height": int(r3.get("height", 720)),
        "point_size": float(r3.get("point_size", 2.5)),
        "max_range": float(r3.get("max_range", 60.0)),
        "zoom": float(r3.get("zoom", 0.42)),
        "elev": float(r3.get("elev_deg", 30.0)),
        "azim": float(r3.get("azim_deg", -110.0)),
        "background": np.asarray(
            r3.get("background", [0.043, 0.051, 0.063]), dtype=np.float64),
    }


def apply_camera(ctl, *, elev_deg: float, azim_deg: float, zoom: float) -> None:
    """Point an Open3D view control at the map with the configured pose.

    Shared by the offscreen panels and the live window so the interactive view
    opens exactly on the framing the dashboard shows.
    """
    e = np.radians(float(elev_deg))
    a = np.radians(float(azim_deg))
    ctl.set_front([float(np.cos(e) * np.cos(a)),
                   float(np.cos(e) * np.sin(a)),
                   float(-np.sin(e))])
    ctl.set_up([0.0, 0.0, 1.0])
    ctl.set_zoom(float(zoom))


def _palette(cfg) -> Dict[str, object]:
    r3 = cfg.section("render3d")
    return {
        "tiers": [hex_to_rgb(c) for c in r3.get(
            "ground_tier_colors", ["#00e676", "#d4e157", "#2e7d32"])],
        "obstacles": list(r3.get("obstacle_ramp", ["#1f6feb", "#8957e5", "#db61a2"])),
        "dynamic": hex_to_rgb(r3.get("dynamic_color", "#ff8c00")),
        "unknown": hex_to_rgb(r3.get("unknown_color", "#484f58")),
        "density": list(r3.get("density_ramp",
                                ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"])),
        "max_height": float(r3.get("obstacle_max_height", 3.0)),
        "extrude": bool(r3.get("extrude_obstacles", True)),
        "min_extrude": float(r3.get("min_extrude", 0.30)),
        "max_range": float(r3.get("max_range", 60.0)),
        "zoom": float(r3.get("zoom", 0.42)),
        "elev": float(r3.get("elev_deg", 30.0)),
        "azim": float(r3.get("azim_deg", -110.0)),
    }


# --------------------------------------------------------------------------- #
# point-cloud construction
# --------------------------------------------------------------------------- #
def semantic_cloud(
    x: np.ndarray,
    y: np.ndarray,
    cls: np.ndarray,
    mean_z: np.ndarray,
    max_z: np.ndarray,
    tier: np.ndarray,
    cfg,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(points Nx3, colours Nx3)`` for the semantic 3D view.

    Ground cells sit at ``mean_z`` and are coloured by drivability tier;
    obstacle / dynamic cells sit at ``max_z`` and carry a height-ramped colour.
    Obstacle and dynamic cells are also extruded downwards so the volume shows.
    """
    pal = _palette(cfg)
    n = int(cls.size)
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    z = mean_z.astype(np.float64).copy()
    obj = cls != CLASS_DRIVABLE
    z[obj] = max_z[obj].astype(np.float64)

    colors = np.zeros((n, 3), dtype=np.float64)
    tiers = pal["tiers"]

    ground = cls == CLASS_DRIVABLE
    for t in range(len(tiers)):
        colors[ground & (tier == t)] = tiers[t]
    known_tier = np.isin(tier, np.arange(len(tiers)))
    colors[ground & ~known_tier] = tiers[0]

    static = cls == CLASS_STATIC
    if np.any(static):
        h = np.clip(max_z[static].astype(np.float64), 0.0, pal["max_height"])
        colors[static] = ramp(pal["obstacles"], h / max(pal["max_height"], 1e-9))

    dyn = np.zeros(n, dtype=bool)
    if n_classes > CLASS_DYNAMIC:
        dyn = cls == CLASS_DYNAMIC
        colors[dyn] = pal["dynamic"]

    colors[cls >= n_classes] = pal["unknown"]

    base = np.stack([x.astype(np.float64), y.astype(np.float64), z], axis=1)

    if not pal["extrude"]:
        return base, colors

    stems = obj | dyn
    if not np.any(stems):
        return base, colors

    idx = np.nonzero(stems)[0]
    top = max_z[idx].astype(np.float64)
    bot = np.minimum(mean_z[idx].astype(np.float64), top - pal["min_extrude"])
    span = np.maximum(top - bot, pal["min_extrude"])

    steps = np.clip(np.rint(span / 0.10).astype(np.int64), 2, 40)
    total = int(steps.sum())
    if total == 0:
        return base, colors

    rep = np.repeat(np.arange(idx.size), steps)
    starts = np.concatenate([[0], np.cumsum(steps)[:-1]])
    within = np.arange(total) - np.repeat(starts, steps)
    frac = within / np.maximum(np.repeat(steps - 1, steps), 1).astype(np.float64)

    stem_xyz = np.empty((total, 3), dtype=np.float64)
    stem_xyz[:, 0] = x[idx][rep].astype(np.float64)
    stem_xyz[:, 1] = y[idx][rep].astype(np.float64)
    stem_xyz[:, 2] = np.repeat(bot, steps) + frac * np.repeat(span, steps)

    # vertical shading: bright at the top of the structure, darker at the base
    shade = 0.35 + 0.65 * frac
    stem_rgb = colors[idx][rep] * shade[:, None]

    return (np.vstack([base, stem_xyz]),
            np.vstack([colors, np.clip(stem_rgb, 0.0, 1.0)]))


def density_cloud(
    x: np.ndarray,
    y: np.ndarray,
    cls: np.ndarray,
    mean_z: np.ndarray,
    max_z: np.ndarray,
    density: np.ndarray,
    cfg,
    dens_min: float,
    dens_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the density-gradient view: colour encodes cells per m^2.

    ``density`` spans two orders of magnitude (4 cells/m^2 in the outer 50 cm
    band up to 400 cells/m^2 in the inner 5 cm band), so the colour coordinate
    is ``log10``-scaled over the *absolute* ``[dens_min, dens_max]`` extent of
    the configured bands (see :func:`web.metrics.density_to_t`).  Geometry is the
    same 2.5D surface as the semantic view.
    """
    pal = _palette(cfg)
    n = int(density.size)
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    z = mean_z.astype(np.float64).copy()
    obj = cls != CLASS_DRIVABLE
    z[obj] = max_z[obj].astype(np.float64)

    colors = ramp(pal["density"], density_to_t(density, dens_min, dens_max))
    points = np.stack([x.astype(np.float64), y.astype(np.float64), z], axis=1)
    return points, colors


# --------------------------------------------------------------------------- #
# hidden-window renderer
# --------------------------------------------------------------------------- #
class SceneRenderer:
    """A reusable *hidden* GLFW window for offscreen point-cloud captures.

    ``open3d.visualization.rendering.OffscreenRenderer`` needs EGL, which is
    unavailable on Windows, so we instead create an invisible window and grab
    the framebuffer.  One window is reused for every panel of a request.
    """

    def __init__(self, cfg):
        if o3d is None:  # pragma: no cover - depends on the machine
            raise RenderUnavailable(f"open3d is not importable: {_IMPORT_ERROR!r}")
        cam = camera_settings(cfg)
        self.width = int(cam["width"])
        self.height = int(cam["height"])
        self.point_size = float(cam["point_size"])
        self.max_range = float(cam["max_range"])
        self.zoom = float(cam["zoom"])
        self.elev = float(cam["elev"])
        self.azim = float(cam["azim"])
        self.background = np.asarray(cam["background"], dtype=np.float64)
        self._vis = None
        self._current = None

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "SceneRenderer":
        vis = o3d.visualization.Visualizer()
        try:
            ok = vis.create_window(window_name="siih-offscreen",
                                   width=self.width, height=self.height,
                                   visible=False)
        except BaseException as exc:  # pragma: no cover
            raise RenderUnavailable(
                f"Open3D window creation failed: {exc}") from exc
        if not ok:
            raise RenderUnavailable(
                "Open3D could not create a rendering window (needs a desktop "
                "session with an OpenGL driver)")
        self._vis = vis
        opt = vis.get_render_option()
        opt.background_color = self.background
        opt.point_size = self.point_size
        opt.light_on = False            # flat colours: palette must show exactly
        opt.show_coordinate_frame = False
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            finally:
                self._vis = None
                self._current = None

    # -- rendering ---------------------------------------------------------
    def render(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        *,
        elev_deg: Optional[float] = None,
        azim_deg: Optional[float] = None,
        zoom: Optional[float] = None,
    ) -> str:
        """Render one cloud with a fixed camera and return a base64 PNG."""
        if self._vis is None:
            raise RenderUnavailable(
                "renderer is not open (use it as a context manager)")
        if points.shape[0] == 0:
            raise RenderUnavailable("nothing to render (empty map)")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

        if self._current is not None:
            self._vis.remove_geometry(self._current, reset_bounding_box=False)
        self._vis.add_geometry(pcd, reset_bounding_box=True)
        self._current = pcd

        ctl = self._vis.get_view_control()
        apply_camera(
            ctl,
            elev_deg=self.elev if elev_deg is None else elev_deg,
            azim_deg=self.azim if azim_deg is None else azim_deg,
            zoom=self.zoom if zoom is None else zoom,
        )

        self._vis.poll_events()
        self._vis.update_renderer()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "view.png"
            self._vis.capture_screen_image(str(path), do_render=True)
            if not path.exists():  # pragma: no cover
                raise RenderUnavailable("Open3D produced no image")
            data = path.read_bytes()
        return base64.b64encode(data).decode("ascii")


def scene_clouds(
    adaptive,
    uniform: Dict[str, np.ndarray],
    adaptive_tier: np.ndarray,
    adaptive_density: np.ndarray,
    cfg,
    n_classes: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build ``{view: (points, colours)}`` for every name in :data:`VIEWS`.

    Shared by the offscreen PNG panels (:func:`render_panels`) and the
    interactive window (:mod:`web.view3d`), so both draw identical geometry.
    Pure numpy - no OpenGL needed, so it is safe to call without a display.
    """
    occ = adaptive.occupancy
    ax_, ay_, _ = adaptive.occupied_centers()

    # Crop the *view* (the metrics above stay whole-frame):
    #  * radius crop - a handful of far-range returns would otherwise dominate
    #    the auto-fitted camera and shrink the informative map to a speck;
    #  * percentile crop - removes the sparse strays so the dense part of the
    #    map sits centred instead of being pushed into a corner.
    max_range = float(cfg.get("render3d.max_range", 60.0))
    in_range = np.hypot(ax_, ay_) <= max_range
    quant = float(cfg.get("render3d.view_crop_quantile", 0.5))
    if quant > 0.0 and np.any(in_range):
        lo, hi = quant, 100.0 - quant
        xr = np.percentile(ax_[in_range], [lo, hi])
        yr = np.percentile(ay_[in_range], [lo, hi])
        in_range &= ((ax_ >= xr[0]) & (ax_ <= xr[1])
                     & (ay_ >= yr[0]) & (ay_ <= yr[1]))
    keep = in_range
    ax_, ay_ = ax_[keep], ay_[keep]

    a_cls = adaptive.pred_class[occ][keep]
    a_mean = adaptive.mean_height[occ][keep]
    a_max = adaptive.max_z[occ][keep]
    a_tier = adaptive_tier[occ][keep]
    a_dens = adaptive_density[occ][keep]

    u_keep = np.hypot(uniform["x"], uniform["y"]) <= max_range
    if quant > 0.0 and np.any(u_keep):
        lo, hi = quant, 100.0 - quant
        xr = np.percentile(uniform["x"][u_keep], [lo, hi])
        yr = np.percentile(uniform["y"][u_keep], [lo, hi])
        u_keep &= ((uniform["x"] >= xr[0]) & (uniform["x"] <= xr[1])
                   & (uniform["y"] >= yr[0]) & (uniform["y"] <= yr[1]))
    u_x = uniform["x"][u_keep]
    u_y = uniform["y"][u_keep]
    u_cls = uniform["pred"][u_keep]
    u_mean = uniform["mean_z"][u_keep]
    u_max = uniform["max_z"][u_keep]
    u_tier = uniform["tier"][u_keep]
    u_dens = np.full(u_cls.size, float(uniform["density_per_m2"]))

    # colour gradients are anchored to the configured band extent, not to the
    # crop, so the same colour always means the same sampling density
    dens_min, dens_max = density_extent(adaptive.bands)

    return {
        "adaptive_semantic": semantic_cloud(
            ax_, ay_, a_cls, a_mean, a_max, a_tier, cfg, n_classes),
        "adaptive_density": density_cloud(
            ax_, ay_, a_cls, a_mean, a_max, a_dens, cfg, dens_min, dens_max),
        "uniform_semantic": semantic_cloud(
            u_x, u_y, u_cls, u_mean, u_max, u_tier, cfg, n_classes),
        "uniform_density": density_cloud(
            u_x, u_y, u_cls, u_mean, u_max, u_dens, cfg, dens_min, dens_max),
    }


def render_panels(
    adaptive,
    uniform: Dict[str, np.ndarray],
    adaptive_tier: np.ndarray,
    adaptive_density: np.ndarray,
    cfg,
    n_classes: int,
) -> Tuple[Dict[str, str], Optional[str]]:
    """Render the four 3D panels inside one hidden window.

    Returns ``(images, error)``; ``images`` maps ``adaptive_semantic`` /
    ``adaptive_density`` / ``uniform_semantic`` / ``uniform_density`` to base64
    PNGs.  When OpenGL is unavailable ``images`` is empty and ``error`` carries
    the reason so the dashboard can state it instead of showing a blank panel.
    The interactive window (:mod:`web.view3d`) draws the same geometry from
    :func:`scene_clouds`, live instead of into a PNG.
    """
    try:
        clouds = scene_clouds(adaptive, uniform, adaptive_tier,
                              adaptive_density, cfg, n_classes)
    except BaseException as exc:  # pragma: no cover - geometry is pure numpy
        return {}, f"{type(exc).__name__}: {exc}"

    images: Dict[str, str] = {}
    try:
        with SceneRenderer(cfg) as scene:
            for name in VIEWS:
                points, colors = clouds[name]
                images[name] = scene.render(points, colors)
    except RenderUnavailable as exc:
        return {}, str(exc)
    except BaseException as exc:  # pragma: no cover - GPU/driver specific
        return {}, f"{type(exc).__name__}: {exc}"

    return images, None
