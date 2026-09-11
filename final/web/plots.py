"""Base64 PNG renderers for the dashboard panels.

All figures use the non-interactive Agg backend and return a base64 string that
can be embedded directly in an ``<img src="data:image/png;base64,...">`` tag, so
the server needs no static-file plumbing.
"""
from __future__ import annotations

import base64
import io
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

try:  # package import: python -m web.app
    from .metrics import density_extent, density_to_t, human_bytes
except ImportError:  # script import: python web/app.py
    from metrics import density_extent, density_to_t, human_bytes

BG = "#0f1216"
PANEL = "#161b22"
FG = "#e6edf3"
MUTED = "#8b949e"
EDGE = "#30363d"
GUIDE = "#3d444d"


def palette(cfg) -> Tuple[List[Tuple[float, float, float]], Tuple[float, float, float]]:
    """Return (per-class RGB in 0..1, unknown RGB) from ``classes.colors``."""
    colors = cfg.get("classes.colors", {}) or {}
    names = list(cfg.get("classes.names", []) or [])
    out: List[Tuple[float, float, float]] = []
    for name in names:
        rgb = colors.get(name, [200, 200, 200])
        out.append(tuple(float(c) / 255.0 for c in rgb))
    unknown = colors.get("unknown", [50, 50, 50])
    return out, tuple(float(c) / 255.0 for c in unknown)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style_map(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(EDGE)
    ax.set_title(title, color=FG, fontsize=11, pad=10)


def _legend(ax, names: Sequence[str], pal, unknown) -> None:
    handles = [Patch(facecolor=pal[i], edgecolor=EDGE, label=str(n))
               for i, n in enumerate(names) if i < len(pal)]
    handles.append(Patch(facecolor=unknown, edgecolor=EDGE, label="unknown"))
    leg = ax.legend(handles=handles, loc="upper right", fontsize=7,
                    facecolor=PANEL, framealpha=0.85)
    leg.get_frame().set_edgecolor(EDGE)
    for text in leg.get_texts():
        text.set_color(FG)


def _class_colors(cls, conf, counts, pal, unknown):
    """Blend class colour with confidence x log-count so weak cells look dim."""
    colors = np.empty((len(cls), 3), dtype=np.float64)
    n_cls = len(pal)
    for i in range(len(cls)):
        c = int(cls[i])
        base = np.asarray(pal[c] if 0 <= c < n_cls else unknown, dtype=np.float64)
        gain = 0.30 + 0.70 * float(conf[i]) * min(1.0, np.log1p(float(counts[i])) / 3.0)
        colors[i] = base * gain
    return colors


# --------------------------------------------------------------------------- #
# map panels
# --------------------------------------------------------------------------- #
def render_adaptive(grid, cfg) -> str:
    """Adaptive variable-resolution 2.5D map (marker size == cell size)."""
    pal, unknown = palette(cfg)
    names = list(cfg.get("classes.names", []) or [])

    xs, ys, band_ids = grid.occupied_centers()
    occ = grid.occupancy
    cls = grid.pred_class[occ]
    conf = grid.confidence[occ]
    counts = grid.count[occ]

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=110)
    fig.patch.set_facecolor(BG)

    if xs.size:
        colors = _class_colors(cls, conf, counts, pal, unknown)
        cell = np.asarray([b.cell for b in grid.bands], dtype=np.float64)
        sizes = 1.0 + (cell[band_ids] * 42.0) ** 2
        ax.scatter(xs, ys, c=colors, s=sizes, marker="s",
                   linewidths=0, rasterized=True)

    if cfg.get("viz.show_band_boundaries", True):
        for band in grid.bands[1:]:
            ax.add_patch(Circle((0.0, 0.0), band.rmin, fill=False,
                                ec=GUIDE, lw=0.7, ls=":"))

    _style_map(ax, "Adaptive variable-resolution 2.5D map")
    _legend(ax, names, pal, unknown)
    return _fig_to_base64(fig)


def render_uniform(uniform: Dict[str, np.ndarray], cfg) -> str:
    """Uniform-cell-size 2.5D comparison map over the same footprint."""
    pal, unknown = palette(cfg)
    names = list(cfg.get("classes.names", []) or [])

    cls = uniform["pred"]
    conf = uniform["confidence"]
    counts = uniform["counts"]

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=110)
    fig.patch.set_facecolor(BG)

    if len(cls):
        colors = _class_colors(cls, conf, counts, pal, unknown)
        ax.scatter(uniform["x"], uniform["y"], c=colors, s=3.0, marker="s",
                   linewidths=0, rasterized=True)

    cell_cm = uniform["cell"] * 100.0
    _style_map(ax, f"Uniform 2.5D map ({cell_cm:.0f} cm cells, same footprint)")
    _legend(ax, names, pal, unknown)
    return _fig_to_base64(fig)


def _ramp_colors(stops: Sequence[str], t: np.ndarray) -> np.ndarray:
    """Piecewise-linear multi-stop ramp -> ``(N, 3)`` RGB in 0..1."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("siih", list(stops))
    return cmap(np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0))[:, :3]


def render_drivability(grid, tier: np.ndarray, cfg) -> str:
    """Top-down drivability layer: three walkable tiers + obstacle classes."""
    pal, unknown = palette(cfg)
    names = list(cfg.get("classes.names", []) or [])
    tier_colors = [c for c in (cfg.get("render3d.ground_tier_colors", []) or [])
                   or ["#00e676", "#d4e157", "#2e7d32"]]
    tier_names = (list(cfg.get("drivability.tier_names", []) or [])
                  or ["firm", "moderate", "rough"])

    xs, ys, _ = grid.occupied_centers()
    occ = grid.occupancy
    cls = grid.pred_class[occ]
    t = tier[occ]

    obj_color = pal[1] if len(pal) > 1 else (0.4, 0.4, 0.4)

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=110)
    fig.patch.set_facecolor(BG)

    ground = cls == 0
    if xs.size:
        for i, hexc in enumerate(tier_colors):
            m = ground & (t == i)
            if np.any(m):
                ax.scatter(xs[m], ys[m], c=[hexc], s=2.2, marker="s",
                           linewidths=0, rasterized=True)
        m_other = ground & ~np.isin(t, np.arange(len(tier_colors)))
        if np.any(m_other):
            ax.scatter(xs[m_other], ys[m_other], c=[tier_colors[0]], s=2.2,
                       marker="s", linewidths=0, rasterized=True)
        if np.any(~ground):
            ax.scatter(xs[~ground], ys[~ground], c=[obj_color], s=2.2,
                       marker="s", linewidths=0, rasterized=True)

    if cfg.get("viz.show_band_boundaries", True):
        for band in grid.bands[1:]:
            ax.add_patch(Circle((0.0, 0.0), band.rmin, fill=False,
                                ec=GUIDE, lw=0.7, ls=":"))

    handles = [Patch(facecolor=c, edgecolor=EDGE, label=f"{tier_names[i]} ground")
               for i, c in enumerate(tier_colors)]
    handles.append(Patch(facecolor=obj_color, edgecolor=EDGE, label="static obstacle"))
    leg = ax.legend(handles=handles, loc="upper right", fontsize=7,
                    facecolor=PANEL, framealpha=0.85)
    leg.get_frame().set_edgecolor(EDGE)
    for text in leg.get_texts():
        text.set_color(FG)

    _style_map(ax, "Drivability layer - 3 walkable-ground tiers")
    return _fig_to_base64(fig)


def render_density(grid, density: np.ndarray, cfg) -> str:
    """Top-down map coloured by sampling density (cells per m^2).

    This is the foveation made visible: the colour gradient collapses from the
    dense 5 cm inner band to the coarse 50 cm outer band.
    """
    stops = list(cfg.get("render3d.density_ramp", []) or []) or [
        "#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"]

    xs, ys, _ = grid.occupied_centers()
    occ = grid.occupancy
    d = np.maximum(density[occ], 1e-9)
    dens_min, dens_max = density_extent(grid.bands)

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=110)
    fig.patch.set_facecolor(BG)

    if xs.size:
        ax.scatter(xs, ys, c=_ramp_colors(stops, density_to_t(d, dens_min, dens_max)),
                   s=1.6, marker="s", linewidths=0, rasterized=True)

    if cfg.get("viz.show_band_boundaries", True):
        for band in grid.bands[1:]:
            ax.add_patch(Circle((0.0, 0.0), band.rmin, fill=False,
                                ec=GUIDE, lw=0.7, ls=":"))

    # one legend swatch per configured band, placed at its true position on the
    # log-density gradient so the swatch colour matches the map exactly
    handles = []
    for band in grid.bands:
        band_density = 1.0 / (band.cell * band.cell)
        pos = float(density_to_t(np.array([band_density]), dens_min, dens_max)[0])
        c = _ramp_colors(stops, np.array([pos]))[0]
        handles.append(Patch(facecolor=c, edgecolor=EDGE,
                             label=f"{band.cell * 100:.1f} cm cell "
                                   f"({band_density:,.0f}/m2)"))
    leg = ax.legend(handles=handles, loc="upper right", fontsize=7,
                    title="cell size  (colour = log density)", facecolor=PANEL,
                    framealpha=0.85)
    leg.get_frame().set_edgecolor(EDGE)
    leg.get_title().set_color(FG)
    leg.get_title().set_fontsize(7)
    for text in leg.get_texts():
        text.set_color(FG)

    _style_map(ax, "Rendering density - cells per m2 (foveation)")
    return _fig_to_base64(fig)


# --------------------------------------------------------------------------- #
# bar-chart panels
# --------------------------------------------------------------------------- #
def _style_chart(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=FG, fontsize=11, pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(EDGE)
    ax.grid(axis="x", color="#21262d", lw=0.6)
    ax.set_axisbelow(True)


def render_timing(metrics: Dict[str, float]) -> str:
    """Per-stage latency breakdown + end-to-end FPS."""
    stages = [
        ("load", metrics.get("load_ms", 0.0), "#58a6ff"),
        ("preprocess", metrics.get("preprocess_ms", 0.0), "#3fb950"),
        ("segment", metrics.get("segmentation_ms", 0.0), "#d29922"),
        ("project", metrics.get("projection_ms", 0.0), "#a371f7"),
        ("temporal", metrics.get("temporal_ms", 0.0), "#f778ba"),
    ]
    names = [s[0] for s in stages]
    values = [max(0.0, float(s[1])) for s in stages]
    colors = [s[2] for s in stages]

    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=110)
    fig.patch.set_facecolor(BG)
    _style_chart(ax, f"Stage latency - {metrics.get('total_ms', 0.0):.1f} ms total "
                     f"({metrics.get('fps', 0.0):.1f} FPS)")
    bars = ax.bar(names, values, color=colors, width=0.62)
    top = max(values + [1e-6]) * 1.18
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + top * 0.02,
                f"{value:.1f}", ha="center", va="bottom", color=FG, fontsize=8)
    ax.set_ylim(0.0, top)
    ax.set_ylabel("latency (ms)", color=FG, fontsize=9)
    ax.grid(axis="y", color="#21262d", lw=0.6)
    ax.grid(axis="x", visible=False)
    return _fig_to_base64(fig)


def render_memory(memory: Dict[str, float]) -> str:
    """Memory footprint: variable resolution vs uniform baselines (log scale)."""
    cell_cm = memory["cell_size"] * 100.0
    rows = [
        ("Variable-res 2.5D\n(this pipeline, allocated)", memory["variable_bytes"], "#3fb950"),
        ("Variable-res 2.5D\n(occupied cells only)", memory["variable_bytes_sparse"], "#2ea043"),
        ("Uniform 2.5D\n(same footprint)", memory["uniform25d_bytes"], "#58a6ff"),
        ("Uniform 3D voxel\n(same footprint)", memory["uniform3d_bytes"], "#f85149"),
    ]
    labels = [r[0] for r in rows]
    values = [max(1.0, float(r[1])) for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=110)
    fig.patch.set_facecolor(BG)
    _style_chart(ax, f"Memory footprint vs uniform {cell_cm:.0f} cm baselines\n"
                     f'{memory["reduction_vs_uniform3d_pct"]:.1f}% smaller than the '
                     f'uniform 3D voxel map')
    xpos = np.arange(len(rows))
    ax.bar(xpos, values, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, color=FG, fontsize=8)
    for x, value in zip(xpos, values):
        ax.text(x, value * 1.15, human_bytes(value), ha="center", va="bottom",
                color=FG, fontsize=8)
    ax.set_ylim(1.0, max(values) * 6.0)
    ax.set_ylabel("bytes (log scale)", color=FG, fontsize=8)
    ax.grid(axis="y", color="#21262d", lw=0.6)
    ax.grid(axis="x", visible=False)
    return _fig_to_base64(fig)


def render_comparison(comp: Dict[str, object]) -> str:
    """Adaptive vs dense uniform 5 cm, on the same frame - the head-to-head proof.

    The per-axis measured magnitudes on a log axis, so the ~24x cell-count and
    ~5.9x memory gaps are read directly.  Absolute values are printed as text so
    the chart stays readable even when a bar bottoms out on the log axis.
    """
    rows = list(comp["rows"])  # type: ignore[index]
    labels = [r["label"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=110)
    fig.patch.set_facecolor(BG)

    _style_chart(ax, f'Adaptive grid vs uniform {comp["cell_cm"]:.0f} cm lattice - '
                     f'absolute values')
    xpos = np.arange(len(rows), dtype=np.float64)
    width = 0.36
    a_vals = np.array([max(float(r["adaptive"]), 1e-6) for r in rows], dtype=np.float64)
    u_vals = np.array([max(float(r["uniform"]), 1e-6) for r in rows], dtype=np.float64)
    ax.bar(xpos - width / 2.0, a_vals, width=width, color="#3fb950",
           label="adaptive (this pipeline)")
    ax.bar(xpos + width / 2.0, u_vals, width=width, color="#58a6ff",
           label=f'uniform {comp["cell_cm"]:.0f} cm')
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, color=FG, fontsize=8, rotation=15, ha="right")
    for x, row in zip(xpos, rows):
        ax.text(x - width / 2.0, max(float(row["adaptive"]), 1e-6) * 1.25,
                row["adaptive_text"], ha="center", va="bottom", color=FG, fontsize=7.5)
        ax.text(x + width / 2.0, max(float(row["uniform"]), 1e-6) * 1.25,
                row["uniform_text"], ha="center", va="bottom", color=MUTED, fontsize=7.5)
    ax.set_ylim(min(a_vals.min(), u_vals.min()) * 0.5,
                max(a_vals.max(), u_vals.max()) * 60.0)
    ax.set_ylabel("measured value (log scale)", color=FG, fontsize=8)
    ax.grid(axis="y", color="#21262d", lw=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=EDGE,
              labelcolor=FG, fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    return _fig_to_base64(fig)

