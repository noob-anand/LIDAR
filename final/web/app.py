#!/usr/bin/env python
"""Flask dashboard: Adaptive Variable-Resolution 2.5D LiDAR Mapping.

Serves a single-page HTML dashboard containing

* the adaptive (foveated) 2.5D map,
* a uniform-cell-size 2.5D map over the same footprint for comparison,
* per-stage latency and memory-footprint panels,
* metric tables (per-band occupancy/confidence, semantic mix, memory model).

Usage
-----
    python web/app.py                  # http://127.0.0.1:5000
    python web/app.py --port 8080
    python -m web.app --host 0.0.0.0

Routes
------
``/``            the HTML dashboard (``?refresh=1`` re-runs the pipeline).
``/api/data``    the same payload as JSON.
``/api/view3d``  spawn the interactive Open3D window on the server host.
``/healthz``     liveness probe.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from flask import Flask, jsonify, render_template_string, request

from mapping import CLASS_DRIVABLE, CLASS_DYNAMIC, CLASS_STATIC, load_bin
from mapping.config import load_config
from mapping.pipeline import LiDARMapper
from mapping.preprocess import filter_points

try:  # package import: python -m web.app
    from .metrics import (band_statistics, build_uniform_grid, class_breakdown,
                          comparison_summary, density_per_cell, detect_objects,
                          drivability_summary, drivability_tiers, human_bytes,
                          measure_comparison, memory_comparison,
                          uniform_drivability_tiers)
    from .plots import (render_adaptive, render_comparison, render_density,
                        render_drivability, render_memory, render_timing,
                        render_uniform)
    from .render3d import VIEW_TITLES, VIEWS, render_panels
    from .view3d import (CONTROLS, SNAPSHOT_DIR, launch_viewer,
                         viewer_log_path)
except ImportError:  # script import: python web/app.py
    from metrics import (band_statistics, build_uniform_grid, class_breakdown,
                         comparison_summary, density_per_cell, detect_objects,
                         drivability_summary, drivability_tiers, human_bytes,
                         measure_comparison, memory_comparison,
                         uniform_drivability_tiers)
    from plots import (render_adaptive, render_comparison, render_density,
                       render_drivability, render_memory, render_timing,
                       render_uniform)
    from render3d import VIEW_TITLES, VIEWS, render_panels
    from view3d import (CONTROLS, SNAPSHOT_DIR, launch_viewer,
                        viewer_log_path)

DATA_PATH = ROOT / "data" / "fake_lidar_000000.bin"
CONFIG_PATH = ROOT / "configs" / "default.yaml"
CACHE_TTL = float(os.environ.get("SIIH_CACHE_TTL", "30"))

app = Flask(__name__)

_LOCK = threading.Lock()
_VIEWER_LOCK = threading.Lock()
_STATE = {"mapper": None, "payload": None, "computed_at": 0.0}
#: Open3D windows spawned by ``/api/view3d``: view name -> Popen.  Kept so a
#: second click reports the running window instead of opening a duplicate.
_VIEWERS: dict = {}


def load_dashboard_config():
    """Load ``configs/default.yaml`` with the GUI backend switched off."""
    cfg = load_config(str(CONFIG_PATH))
    cfg.set("viz.backend", "none")  # the web server must never open a window
    return cfg


def _filtered_points(pc, cfg, labels):
    """Reproduce the pipeline's filtered point set without re-running RANSAC.

    ``filter_points`` is exactly the range/height clip used by
    :func:`mapping.preprocess.preprocess`, so the returned array lines up
    one-to-one with ``result.labels``.  Both the filtered points and their
    indices into the original cloud are returned, so any per-point array
    (labels, dynamic flags) can be aligned to the filtered frame.
    """
    pre = cfg.section("preprocess")
    xyz_f, src_idx = filter_points(
        pc.xyz,
        float(pre.get("min_range", 1.0)),
        float(pre.get("max_range", 100.0)),
        float(pre.get("height_min", -5.0)),
        float(pre.get("height_max", 5.0)),
        pre.get("decimation"),
    )
    if xyz_f.shape[0] != labels.size:
        # Defensive fallback: deterministic re-preprocess keeps labels aligned.
        from mapping.preprocess import preprocess

        xyz_f = preprocess(pc.xyz, pc.intensity, cfg).xyz
        src_idx = np.arange(xyz_f.shape[0], dtype=np.int64)
    return xyz_f, src_idx


def run_pipeline(force: bool = False) -> dict:
    """Run the full pipeline and assemble the dashboard payload.

    The result is cached for ``CACHE_TTL`` seconds so reloading the page does
    not re-run the pipeline; pass ``force=True`` (``?refresh=1``) to recompute.
    """
    with _LOCK:
        age = time.time() - _STATE["computed_at"]
        if not force and _STATE["payload"] is not None and age < CACHE_TTL:
            return _STATE["payload"]

        cfg = load_dashboard_config()
        if _STATE["mapper"] is None:
            _STATE["mapper"] = LiDARMapper(cfg)
        mapper = _STATE["mapper"]
        mapper.cfg.set("viz.backend", "none")

        t_begin = time.perf_counter()
        t0 = time.perf_counter()
        pc = load_bin(str(DATA_PATH))
        load_ms = (time.perf_counter() - t0) * 1000.0

        result = mapper.process_frame(pc)
        total_ms = (time.perf_counter() - t_begin) * 1000.0

        grid = result.grid
        class_names = list(cfg.get("classes.names", []) or [])
        n_classes = len(class_names) or 3

        # Uniform baseline over the *same* filtered points and labels.
        xyz_f, src_idx = _filtered_points(pc, cfg, result.labels)
        uniform = build_uniform_grid(
            xyz_f, result.labels, n_classes,
            float(cfg.get("metrics.uniform_cell_size", 0.05)),
        )

        # --- drivability layer + rendering density ------------------------- #
        tier, slope, rough = drivability_tiers(grid, cfg)
        density = density_per_cell(grid)
        u_tier, u_slope, u_rough = uniform_drivability_tiers(uniform, cfg)
        uniform["tier"] = u_tier

        drivable_mask = grid.occupancy & (grid.pred_class == CLASS_DRIVABLE)
        drivability = {
            "tiers": drivability_summary(tier, cfg, class_names),
            "drivable_cells": int(drivable_mask.sum()),
            "obstacle_cells": int((grid.occupancy & (grid.pred_class == CLASS_STATIC)).sum()),
            "mean_slope": float(slope[drivable_mask].mean()) if drivable_mask.any() else 0.0,
            "max_slope": float(slope[drivable_mask].max()) if drivable_mask.any() else 0.0,
            "mean_roughness": float(rough[drivable_mask].mean()) if drivable_mask.any() else 0.0,
            "slope_threshold": float(cfg.get("drivability.slope_threshold", 0.35)),
            "roughness_threshold": float(cfg.get("drivability.roughness_threshold", 0.08)),
            "uniform_tiers": [
                {"tier": t,
                 "name": (list(cfg.get("drivability.tier_names", []) or [])
                          or ["firm", "moderate", "rough"])[t],
                 "count": int(np.count_nonzero(u_tier == t)),
                 "pct": (100.0 * np.count_nonzero(u_tier == t)
                         / max(1, int(np.count_nonzero(u_tier >= 0))))}
                for t in range(3)
            ],
        }

        # --- memory + latency metrics ------------------------------------- #
        xs_ad, ys_ad, _ = grid.occupied_centers()
        memory = memory_comparison(grid, cfg, xs_ad, ys_ad)
        memory["uniform_cells_hit"] = int(uniform["n_cells"])
        memory["uniform_cells_hit_bytes"] = (
            int(uniform["n_cells"]) * memory["uniform_bytes_per_cell"]
        )

        # --- measured adaptive-vs-uniform-5cm head-to-head ----------------- #
        # Builds the dense uniform 5 cm lattice with the real grid engine and
        # times projection + fusion for both maps over this same frame.  The
        # impact on the reported end-to-end latency is subtracted below so the
        # dashboard's own numbers are not inflated by the experiment.
        measure_t0 = time.perf_counter()
        measured = measure_comparison(
            grid, cfg,
            xyz_f,
            pc.intensity[src_idx],
            result.labels,
            result.dynamic_flag,
            uniform, memory, n_classes,
            float(result.projection_time * 1000.0),
        )
        comparison = comparison_summary(
            memory,
            float(measured["adaptive"]["ms"]),
            float(measured["uniform"]["ms"]),
            int(uniform["n_cells"]),
            measured=measured,
        )
        # do not charge the measurement experiment to the pipeline's own latency
        measurement_ms = (time.perf_counter() - measure_t0) * 1000.0

        temporal_state = result.temporal_state
        cell_sizes = [float(b.cell) for b in grid.bands]
        total_ms = (time.perf_counter() - t_begin) * 1000.0 - measurement_ms
        metrics = {
            "load_ms": load_ms,
            "preprocess_ms": result.preprocessing_time * 1000.0,
            "segmentation_ms": result.segmentation_time * 1000.0,
            "projection_ms": result.projection_time * 1000.0,
            "temporal_ms": result.temporal_time * 1000.0,
            "measurement_ms": measurement_ms,
            "total_ms": total_ms,
            "fps": (1000.0 / total_ms) if total_ms > 0 else 0.0,
            "points_raw": int(pc.n),
            "points_filtered": int(xyz_f.shape[0]),
            "points_projected": int(grid.count.sum()),
            "cells_total": int(grid.total_cells),
            "cells_per_point": float(grid.total_cells / max(1, int(xyz_f.shape[0]))),
            "cells_occupied": int(grid.occupancy.sum()),
            "cells_uniform": int(uniform["n_cells"]),
            "cells_dynamic": int(temporal_state.dynamic.sum()) if temporal_state is not None else 0,
            "bands": len(grid.bands),
            "classes": n_classes,
            "frame_index": int(mapper.frame_idx),
            "cell_sizes": cell_sizes,
            "band_edges": [float(b.rmin) for b in grid.bands] + [float(grid.bands[-1].rmax)],
            "foveation_ratio": (cell_sizes[-1] / cell_sizes[0]) if cell_sizes and cell_sizes[0] else 0.0,
            "density_min": float(density[density > 0].min()) if np.any(density > 0) else 0.0,
            "density_max": float(density.max()) if density.size else 0.0,
            "uniform_density": float(uniform["density_per_m2"]),
        }

        bands = band_statistics(grid, cfg)
        classes = class_breakdown(grid.class_hist.sum(axis=0), class_names)
        uniform_classes = class_breakdown(uniform["hist"].sum(axis=0), class_names)

        # --- discrete objects (grouped non-ground occupied cells) ---------- #
        # Nearest-first blobs, capped at ``objects.max_objects`` so the table
        # stays a panel rather than a dump.  IDs are 1..N by distance.
        objects = detect_objects(grid, cfg, class_names, ground_class=CLASS_DRIVABLE)
        objects["rows"] = [
            {
                "id": int(item["id"]),
                "class_name": str(item["class_name"]),
                "text": (f'{item["range_m"]:.1f} m '
                         f'({item["size_x_m"]:.1f} x {item["size_y_m"]:.1f} m, '
                         f'{item["height_m"]:.2f} m tall)'),
                "position": f'({item["x"]:.1f}, {item["y"]:.1f}, {item["z"]:.1f})',
                "cells_text": f'{int(item["cells"]):,}',
                "confidence_text": f'{float(item["confidence"]):.2f}',
            }
            for item in objects["items"]
        ]

        images = {
            "adaptive": render_adaptive(grid, cfg),
            "uniform": render_uniform(uniform, cfg),
            "drivability": render_drivability(grid, tier, cfg),
            "density2d": render_density(grid, density, cfg),
            "timing": render_timing(metrics),
            "memory": render_memory(memory),
            "comparison": render_comparison(comparison),
        }

        # --- Open3D 3D panels (one hidden window for all four) ------------- #
        t0 = time.perf_counter()
        render3d_images, render3d_error = render_panels(
            grid, uniform, tier, density, cfg, n_classes)
        metrics["render3d_ms"] = (time.perf_counter() - t0) * 1000.0
        metrics["render3d_ok"] = bool(render3d_images)
        images.update(render3d_images)

        cell_cm = memory["cell_size"] * 100.0
        tables = {
            "latency": [
                {"label": "Load point cloud", "value": f"{metrics['load_ms']:.2f} ms", "note": ""},
                {"label": "Preprocess (range/height filter, polar, RANSAC ground)",
                 "value": f"{metrics['preprocess_ms']:.2f} ms", "note": ""},
                {"label": "Semantic segmentation", "value": f"{metrics['segmentation_ms']:.2f} ms",
                 "note": "heuristic height/slope/intensity head"},
                {"label": "Adaptive projection", "value": f"{metrics['projection_ms']:.2f} ms",
                 "note": "exact 3D -> 2.5D, no double counting"},
                {"label": "Temporal fusion", "value": f"{metrics['temporal_ms']:.2f} ms",
                 "note": "log-odds occupancy + height/intensity EMA"},
                {"label": "End-to-end frame", "value": f"{metrics['total_ms']:.2f} ms", "note": ""},
                {"label": "Throughput", "value": f"{metrics['fps']:.1f} FPS", "note": ""},
            ],
            "memory": [
                {"label": "Variable-resolution 2.5D (allocated)",
                 "value": human_bytes(memory["variable_bytes"]),
                 "note": f'{memory["variable_cells"]:,} cells x {memory["variable_bytes_per_cell"]} B'},
                {"label": "Variable-resolution 2.5D (occupied only)",
                 "value": human_bytes(memory["variable_bytes_sparse"]),
                 "note": (f'{memory["variable_occupied_cells"]:,} occupied cells '
                          f'x {memory["variable_bytes_per_cell"]} B')},
                {"label": "Variable-resolution 2.5D (measured footprint)",
                 "value": human_bytes(memory["variable_bytes_measured"]),
                 "note": "sum of every per-cell numpy array in the grid engine"},
                {"label": f"Uniform 2.5D @ {cell_cm:.0f} cm (dense, same footprint)",
                 "value": human_bytes(memory["uniform25d_bytes"]),
                 "note": (f'{memory["uniform_cells"]:,} cells '
                          f'({memory["uniform_nx"]} x {memory["uniform_ny"]}) '
                          f'x {memory["uniform_bytes_per_cell"]} B')},
                {"label": f"Uniform 3D voxel @ {cell_cm:.0f} cm (dense, same footprint)",
                 "value": human_bytes(memory["uniform3d_bytes"]),
                 "note": (f'{memory["uniform3d_voxels"]:,} voxels '
                          f'x {memory["uniform3d_bytes_per_voxel"]} B')},
                {"label": "Uniform cells actually hit by the frame",
                 "value": human_bytes(memory["uniform_cells_hit_bytes"]),
                 "note": f'{memory["uniform_cells_hit"]:,} cells x {memory["uniform_bytes_per_cell"]} B'},
            ],
            "drivability": [
                {"label": f'{t["name"]} ground (tier {t["tier"]})',
                 "value": f'{t["count"]:,} cells',
                 "note": f'{t["pct"]:.1f}% of walkable ground'}
                for t in drivability["tiers"]
            ] + [
                {"label": "Walkable ground (all tiers)",
                 "value": f'{drivability["drivable_cells"]:,} cells',
                 "note": "semantic class = drivable"},
                {"label": "Static obstacle cells",
                 "value": f'{drivability["obstacle_cells"]:,} cells',
                 "note": "semantic class = static obstacle"},
                {"label": "Mean / max slope on walkable ground",
                 "value": f'{drivability["mean_slope"]:.3f} / {drivability["max_slope"]:.3f}',
                 "note": f'threshold {drivability["slope_threshold"]:.2f} (rise/run)'},
                {"label": "Mean roughness on walkable ground",
                 "value": f'{drivability["mean_roughness"]:.4f} m',
                 "note": f'threshold {drivability["roughness_threshold"]:.2f} m (sqrt of height variance)'},
            ],
            "density": [
                {"label": f'Band {b["band"]} ({b["rmin"]:.0f}-{b["rmax"]:.0f} m, {b["cell"] * 100:.1f} cm cells)',
                 "value": f'{b["density_per_m2"]:,.1f} cells/m2',
                 "note": f'actual scan return rate {b["points_per_m2"]:.1f} points/m2'}
                for b in bands
            ] + [
                {"label": "Uniform baseline (5 cm everywhere)",
                 "value": f'{metrics["uniform_density"]:,.1f} cells/m2',
                 "note": "constant sampling density of the uniform map"},
                {"label": "Density contrast (inner vs outer band)",
                 "value": f'{metrics["density_max"] / max(metrics["density_min"], 1e-9):.0f}x',
                 "note": "this ratio is what the density gradient shows"},
                {"label": "Open3D render time (4 panels)",
                 "value": f'{metrics["render3d_ms"]:.1f} ms' if metrics["render3d_ok"] else "n/a",
                 "note": render3d_error or "offscreen OpenGL, 1280x720 per panel"},
            ],
            "comparison": [
                {"label": "Measured per-stage split (adaptive)",
                 "value": f'{comparison["measured"]["adaptive"]["projection_ms"]:.1f} + '
                          f'{comparison["measured"]["adaptive"]["fusion_ms"]:.1f} = '
                          f'{comparison["adaptive"]["ms"]:.1f} ms'
                 if comparison["measured"] else "n/a",
                 "note": f'projection + fusion, median of '
                         f'{comparison["measured"]["repeats"]} runs' if comparison["measured"] else ""},
                {"label": "Measured per-stage split (uniform 5 cm lattice)",
                 "value": f'{comparison["measured"]["uniform"]["projection_ms"]:.1f} + '
                          f'{comparison["measured"]["uniform"]["fusion_ms"]:.1f} = '
                          f'{comparison["uniform"]["ms"]:.1f} ms'
                 if comparison["measured"] else "n/a",
                 "note": f'projection + fusion on {comparison["uniform"]["cells"]:,} cells, '
                         f'median of {comparison["measured"]["repeats"]} runs'
                 if comparison["measured"] else ""},
            ],
        }

        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": DATA_PATH.name,
            "config": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "metrics": metrics,
            "memory": memory,
            "comparison": comparison,
            "bands": bands,
            "classes": classes,
            "uniform_classes": uniform_classes,
            "drivability": drivability,
            "objects": objects,
            "render3d_error": render3d_error,
            "images": images,
            "tables": tables,
            # interactive viewer metadata (drives the 3D buttons in the template)
            "views": list(VIEWS),
            "view_titles": {name: VIEW_TITLES.get(name, name) for name in VIEWS},
            "viewer": {
                "log": str(viewer_log_path()),
                "snapshots": str(SNAPSHOT_DIR),
                "controls": [{"keys": k, "action": a} for k, a in CONTROLS],
            },
        }
        _STATE["payload"] = payload
        _STATE["computed_at"] = time.time()
        return payload


TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adaptive Variable-Resolution 2.5D LiDAR Mapping</title>
<style>
  :root {
    --bg:#ffffff; --panel:#f7f8fa; --panel2:#eef1f5; --edge:#d8dde3;
    --fg:#1f2328; --muted:#5b6570; --accent:#1a7f37; --blue:#0969da;
    --amber:#9a6700; --red:#cf222e;
  }
  * { box-sizing: border-box; }
  body {
    margin:0; padding:24px 28px 48px; background:var(--bg); color:var(--fg);
    font-family:"Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size:14px; line-height:1.5;
  }
  h1 { font-size:22px; margin:0 0 6px; letter-spacing:.2px; }
  h2 { font-size:15px; margin:0 0 12px; font-weight:600; }
  .sub { color:var(--muted); max-width:1100px; margin:0 0 14px; }
  .meta { display:flex; flex-wrap:wrap; gap:14px; align-items:center;
          color:var(--muted); font-size:12.5px; margin-bottom:20px; }
  .meta b { color:var(--fg); font-weight:600; }
  a.btn, button.btn { color:var(--fg); text-decoration:none; border:1px solid var(--edge);
          background:var(--panel2); padding:5px 12px; font-size:12.5px;
          font-family:inherit; line-height:1.4; cursor:pointer; }
  a.btn:hover, button.btn:hover { border-color:var(--blue); color:var(--blue); }
  .views { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 4px; }
  .vstatus { display:block; }
  .vstatus.ok { color:var(--accent); }
  .vstatus.err { color:var(--red); }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
           gap:12px; margin-bottom:22px; }
  .card { background:var(--panel); border:1px solid var(--edge);
           padding:12px 14px; }
  .card .k { color:var(--muted); font-size:11.5px; text-transform:uppercase; letter-spacing:.6px; }
  .card .v { font-size:20px; font-weight:600; margin-top:4px; }
  .card .n { color:var(--muted); font-size:11.5px; margin-top:2px; }
  .good { color:var(--accent); } .blue { color:var(--blue); } .amber { color:var(--amber); }
  .panels { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
            gap:18px; margin-bottom:22px; }
  .panels.wide { grid-template-columns:1fr; }
  .panel { background:var(--panel); border:1px solid var(--edge);
           padding:14px; }
  .panel img { width:100%; height:auto; display:block;  }
  .panel .cap { color:var(--muted); font-size:12px; margin-top:10px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--edge); }
  th { color:var(--muted); font-weight:600; text-transform:uppercase;
       font-size:11px; letter-spacing:.5px; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  tr:last-child td { border-bottom:none; }
  .note { color:var(--muted); font-size:11.5px; }
  footer { color:var(--muted); font-size:12px; margin-top:26px;
           border-top:1px solid var(--edge); padding-top:14px; }
</style>
</head>
<body>
<header>
  <h1>Adaptive Variable-Resolution 2.5D LiDAR Mapping</h1>
  <p class="sub">
    A foveated polar 2.5D elevation map with semantic layers. Cell size grows from
    {{ (p.metrics.cell_sizes[0]*100)|round(1) }} cm near the sensor to
    {{ (p.metrics.cell_sizes[-1]*100)|round(1) }} cm at
    {{ p.metrics.band_edges[-1] }} m (a {{ p.metrics.foveation_ratio|round(1) }}x resolution
    falloff), keeping safety-critical detail close by while the far field stays cheap.
    Every panel is compared against a uniform {{ (p.memory.cell_size*100)|round(0)|int }} cm
    baseline over the same sensor footprint.
  </p>
  <div class="meta">
    <span>dataset <b>{{ p.dataset }}</b></span>
    <span>config <b>{{ p.config }}</b></span>
    <span>generated <b>{{ p.generated_at }}</b></span>
    <span>frame <b>#{{ p.metrics.frame_index }}</b></span>
    <a class="btn" href="/?refresh=1">Re-run pipeline</a>
    <a class="btn" href="/api/data">JSON API</a>
    <button class="btn" data-view3d="{{ p.views[0] }}">Open interactive 3D</button>
    <span class="note vstatus" data-view3d-status>3D opens a native Open3D window on the server machine.</span>
  </div>
</header>

<section class="cards">
  <div class="card">
    <div class="k">End-to-end latency</div>
    <div class="v blue">{{ p.metrics.total_ms|round(1) }} ms</div>
    <div class="n">{{ p.metrics.fps|round(1) }} FPS on this frame</div>
  </div>
  <div class="card">
    <div class="k">Memory vs uniform 3D</div>
    <div class="v good">{{ p.memory.reduction_vs_uniform3d_pct|round(1) }}%</div>
    <div class="n">smaller than a {{ (p.memory.cell_size*100)|round(0)|int }} cm voxel map</div>
  </div>
  <div class="card">
    <div class="k">Points projected</div>
    <div class="v">{{ '{:,}'.format(p.metrics.points_projected) }}</div>
    <div class="n">of {{ '{:,}'.format(p.metrics.points_raw) }} raw returns</div>
  </div>
  <div class="card">
    <div class="k">Occupied cells</div>
    <div class="v">{{ '{:,}'.format(p.metrics.cells_occupied) }}</div>
    <div class="n">adaptive grid, {{ p.metrics.bands }} radial bands</div>
  </div>
  <div class="card">
    <div class="k">Uniform cells</div>
    <div class="v amber">{{ '{:,}'.format(p.metrics.cells_uniform) }}</div>
    <div class="n">{{ (p.memory.cell_size*100)|round(0)|int }} cm grid, same footprint</div>
  </div>
  <div class="card">
    <div class="k">Foveation ratio</div>
    <div class="v">{{ p.metrics.foveation_ratio|round(1) }}x</div>
    <div class="n">{{ (p.metrics.cell_sizes[0]*100)|round(1) }} cm inner /
      {{ (p.metrics.cell_sizes[-1]*100)|round(1) }} cm outer</div>
  </div>
</section>

<section class="panels">
</section>
<section class="panels">
  <div class="panel">
    <h2>Drivability layer - 3 walkable-ground tiers</h2>
    <img src="data:image/png;base64,{{ p.images.drivability }}" alt="Drivability layer">
    <p class="cap">
      Walkable ground is split into
      {% for t in p.drivability.tiers %}<b>{{ t.name }}</b>
      ({{ '{:,}'.format(t.count) }} cells, {{ t.pct|round(1) }}% of walkable ground){{ ", " if not loop.last else "" }}{% endfor %}
      by counting how many of the slope and roughness criteria each cell violates. Mean slope on
      walkable ground is {{ p.drivability.mean_slope|round(3) }} against a
      {{ p.drivability.slope_threshold|round(2) }} threshold, and mean roughness is
      {{ p.drivability.mean_roughness|round(4) }} m against a
      {{ p.drivability.roughness_threshold|round(2) }} m threshold. Static obstacles
      ({{ '{:,}'.format(p.drivability.obstacle_cells) }} cells) are drawn in the obstacle colour
      instead of a tier colour.
    </p>
  </div>
  <div class="panel">
    <h2>Sampling density - the foveation made visible</h2>
    <img src="data:image/png;base64,{{ p.images.density2d }}" alt="Sampling density">
    <p class="cap">
      Colour is the sampling density in cells per m<sup>2</sup> on a log ramp, so the drop from
      the {{ (p.metrics.cell_sizes[0]*100)|round(1) }} cm inner band to the
      {{ (p.metrics.cell_sizes[-1]*100)|round(1) }} cm outer band reads as one continuous
      gradient rather than four discrete steps. Measured density spans
      {{ p.metrics.density_min|round(1) }} to {{ p.metrics.density_max|round(1) }}
      cells/m<sup>2</sup>, against {{ p.metrics.uniform_density|round(1) }} cells/m<sup>2</sup>
      for the uniform baseline - {{ p.metrics.foveation_ratio|round(1) }}x coarser in the far
      field, which is what the memory saving buys.
    </p>
  </div>
</section>
<section class="panels">
  <div class="panel">
    <h2>3D model - interactive Open3D window</h2>
    <p class="cap">
      The four preview panels further down are static offscreen renders. A picture cannot be
      rotated, so the interactive version is served as a native window instead: these buttons
      start <code>python -m web.view3d</code> on the machine running the server and open a real
      Open3D window there, showing the same geometry the offscreen renderer used for those panels
      (both call <code>web.render3d.scene_clouds()</code>). The browser stays a browser; the
      window is native to the host running this dashboard.
    </p>
    <div class="views">
      {% for name in p.views %}
      <button class="btn" data-view3d="{{ name }}">{{ p.view_titles[name] }}</button>
      {% endfor %}
    </div>
    <p class="note vstatus" data-view3d-status>
      {% if p.render3d_error %}
      Offscreen 3D rendering failed here ({{ p.render3d_error }}), so the 3D preview panels
      below are missing - the interactive window is the way to inspect the 3D map.
      {% else %}
      Each click opens a detached process, so the dashboard keeps responding. Press <b>s</b> in
      the window to save a PNG snapshot into <code>{{ p.viewer.snapshots }}</code>; the window's
      log is <code>{{ p.viewer.log }}</code>.
      {% endif %}
    </p>
  </div>

</section>
{% if p.views and p.views[0] in p.images %}
<section class="panels">
  {% for name in p.views %}
  <div class="panel">
    <h2>3D preview - {{ p.view_titles[name] }}</h2>
    <img src="data:image/png;base64,{{ p.images[name] }}" alt="3D preview {{ p.view_titles[name] }}">
    <p class="cap">
      {% if name.endswith('semantic') %}
      Same geometry as the panel above, raised into 3D: walkable ground sits at its mean height
      with one colour per drivability tier, obstacles and dynamic cells sit at their max height
      and are extruded downwards so the volume reads, so structure rises out of the ground
      instead of being flattened into a carpet.
      {% else %}
      Same 3D geometry as the semantic preview, coloured by sampling density instead of semantic
      class, so the near-band detail and the coarse far field are visible at a glance.
      {% endif %}
      Rendered offscreen at {{ p.metrics.render3d_ms|round(0)|int }} ms for all four panels with
      the shared <code>scene_clouds()</code> camera. Open the interactive window above to rotate
      this exact geometry.
    </p>
  </div>
  {% endfor %}
</section>
{% endif %}
<section class="panels">
  <div class="panel">
    <h2>Latency breakdown</h2>
    <img src="data:image/png;base64,{{ p.images.timing }}" alt="Stage latency">
    <p class="cap">
      Per-stage wall-clock cost measured around each pipeline stage on this frame: load,
      preprocessing, segmentation, adaptive projection and temporal fusion.
    </p>
  </div>
  <div class="panel">
    <h2>Memory footprint comparison</h2>
    <img src="data:image/png;base64,{{ p.images.memory }}" alt="Memory footprint">
    <p class="cap">
      Bytes on a log scale. The adaptive grid needs
      {{ p.memory.reduction_vs_uniform3d_pct|round(1) }}% less memory than the dense uniform 3D
      voxel map and {{ p.memory.reduction_vs_uniform25d_pct|round(1) }}% less than the dense
      uniform 2.5D map over the same footprint.
    </p>
  </div>
</section>

<section class="panels wide">
  <div class="panel">
    <h2>Adaptive vs uniform {{ p.comparison.cell_cm|round(0)|int }} cm - measured head to head</h2>
    <img src="data:image/png;base64,{{ p.images.comparison }}" alt="Adaptive versus uniform comparison">
    <p class="cap">
      Every number here is measured on this frame by building a dense single-band
      uniform {{ p.comparison.cell_cm|round(0)|int }} cm lattice with the same projection engine
      the adaptive map uses, then timing projection and temporal fusion for both maps
      (median of {{ p.comparison.measured.repeats }} runs). It is the same frame, the same machine
      and the same code paths - nothing is modelled or extrapolated.
      The adaptive grid wins the axes that decide whether the map fits and keeps up:
      <strong>{{ p.comparison.measured.cells_ratio|round(1) }}&times; fewer cells</strong>
      ({{ '{:,}'.format(p.comparison.adaptive.cells) }} vs
      {{ '{:,}'.format(p.comparison.uniform.cells) }}),
      <strong>{{ p.comparison.speedup|round(1) }}&times; faster per frame</strong>
      ({{ p.comparison.adaptive.ms|round(1) }} ms vs {{ p.comparison.uniform.ms|round(1) }} ms, i.e.
      {{ p.comparison.adaptive.fps|round(1) }} vs {{ p.comparison.uniform.fps|round(1) }} FPS) and
      <strong>{{ p.comparison.measured.memory_ratio|round(1) }}&times; less resident memory</strong>.
      The dense lattice is slower because temporal fusion visits every allocated cell every
      frame - {{ '{:,}'.format(p.comparison.uniform.cells) }} cells - while the adaptive grid only
      carries {{ '{:,}'.format(p.comparison.adaptive.cells) }}.
      The one axis the uniform lattice wins is occupied-cell-only storage
      ({{ p.comparison.rows[2].uniform_text }} vs {{ p.comparison.rows[2].adaptive_text }}),
      because an empty cell is free to store - it is still paid for in latency above. That
      trade-off, spelled out rather than hidden, is the point of the panel.
    </p>
    <table>
      <tr>
        <th>Axis</th><th class="num">Adaptive</th><th class="num">Uniform</th>
        <th class="num">Advantage</th><th>Winner</th>
      </tr>
      {% for row in p.comparison.rows %}
      <tr>
        <td>{{ row.label }}</td>
        <td class="num">{{ row.adaptive_text }}</td>
        <td class="num">{{ row.uniform_text }}</td>
        <td class="num">{{ row.advantage|round(2) }}&times;</td>
        <td>{{ row.winner }}</td>
      </tr>
      {% endfor %}
    </table>
    {% if p.comparison.measured %}
    <table style="margin-top:14px">
      <tr><th>Measured per-stage split</th><th>Projection + fusion</th><th>Detail</th></tr>
      {% for row in p.tables.comparison %}
      <tr>
        <td>{{ row.label }}</td>
        <td class="num">{{ row.value }}</td>
        <td class="note">{{ row.note }}</td>
      </tr>
      {% endfor %}
    </table>
    {% endif %}
  </div>
</section>
 <section class="panels wide">
   <div class="panel">
     <h2>Discrete objects - non-ground cells grouped into blobs</h2>
     <p class="cap">
       Occupied cells whose semantic class is not drivable are pooled onto a
       {{ p.objects.cell_size }} m Cartesian lattice and merged with 8-connectivity, so a wall, a
       pole or a parked car becomes one row instead of hundreds of cells. Blobs are kept when they
       cover at least {{ p.objects.min_cells }} cells, reach {{ p.objects.min_height }} m above the
       ground plane and sit within {{ p.objects.max_range|round(0)|int }} m. Rows are ordered by
       distance from the sensor and the nearest {{ p.objects.max_objects }} are shown.
     </p>
     <p class="cap">
       {% if p.objects.count %}
       {{ p.objects.count }} object{{ "s" if p.objects.count != 1 else "" }} found out of
       {{ '{:,}'.format(p.objects.cells_considered) }} candidate cells.
       {% else %}
       No object blob met the thresholds on this frame.
       {% endif %}
     </p>
     {% if p.objects.count %}
     <table>
       <tr>
         <th class="num">#</th><th>Class</th><th class="num">Range</th>
         <th class="num">Centroid (x, y, z) (m)</th><th class="num">Cells</th>
         <th class="num">Confidence</th>
       </tr>
       {% for o in p.objects.rows %}
       <tr>
         <td class="num">{{ o.id }}</td>
         <td>{{ o.class_name }}</td>
         <td class="num">{{ o.text }}</td>
         <td class="num">{{ o.position }}</td>
         <td class="num">{{ o.cells_text }}</td>
         <td class="num">{{ o.confidence_text }}</td>
       </tr>
       {% endfor %}
     </table>
     {% endif %}
   </div>
 </section>
 <section class="panels">
   <div class="panel">
     <h2>Adaptive bands (foveation schedule)</h2>
     <table>
       <tr>
         <th>Band</th><th>Range (m)</th><th class="num">Cell (cm)</th>
         <th class="num">Grid</th><th class="num">Occupied</th>
         <th class="num">Fill</th><th class="num">Confidence</th>
       </tr>
       {% for b in p.bands %}
       <tr>
         <td>{{ b.band }}</td>
         <td>{{ b.rmin|round(1) }} - {{ b.rmax|round(1) }}</td>
         <td class="num">{{ (b.cell*100)|round(1) }}</td>
         <td class="num">{{ b.n_r }} x {{ b.n_theta }}</td>
         <td class="num">{{ '{:,}'.format(b.occupied) }}</td>
         <td class="num">{{ b.occupancy_pct|round(1) }}%</td>
         <td class="num">{{ b.mean_confidence|round(2) }}</td>
       </tr>
       {% endfor %}
     </table>
     <p class="note">
       Cell count rises linearly with radius inside a band, then drops when the next band coarsens
       the sampling - the trade that makes a 100 m range affordable.
     </p>
   </div>
 </section>
 <section class="panels">
   <div class="panel">
     <h2>Memory model</h2>
    <table>
      <tr><th>Representation</th><th class="num">Footprint</th><th>Derivation</th></tr>
      {% for row in p.tables.memory %}
      <tr>
        <td>{{ row.label }}</td>
        <td class="num">{{ row.value }}</td>
        <td class="note">{{ row.note }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  <div class="panel">
    <h2>Semantic mix</h2>
    <table>
      <tr>
        <th>Class</th><th class="num">Adaptive points</th><th class="num">Share</th>
        <th class="num">Uniform points</th>
      </tr>
      {% for c in p.classes %}
      <tr>
        <td>{{ c.name }}</td>
        <td class="num">{{ '{:,}'.format(c.count) }}</td>
        <td class="num">{{ c.pct|round(1) }}%</td>
        <td class="num">{{ '{:,}'.format(p.uniform_classes[loop.index0].count) }}</td>
      </tr>
      {% endfor %}
    </table>
    <p class="note">
      Temporal fusion flags {{ '{:,}'.format(p.metrics.cells_dynamic) }} cells as dynamic. A single
      scan cannot see motion, so those come from the log-odds occupancy and height/intensity EMA
      update rather than from the point-level labels.
    </p>
  </div>
</section>

<footer>
  {{ '{:,}'.format(p.metrics.points_raw) }} raw returns &rarr;
  {{ '{:,}'.format(p.metrics.points_filtered) }} after range/height filtering &rarr;
  {{ '{:,}'.format(p.metrics.points_projected) }} projected into
  {{ '{:,}'.format(p.metrics.cells_total) }} adaptive cells
  ({{ p.metrics.cells_per_point|round(2) }} cells per point).
  Panels are rendered server-side with matplotlib (Agg) and embedded as base64 PNG; the 3D map is
  explored in a native Open3D window started by <em>Open interactive 3D</em>, not inside the
  browser. Results are cached for {{ ttl }} s; use <em>Re-run pipeline</em> to recompute.
</footer>

<script>
(function () {
  function setStatus(text, kind) {
    document.querySelectorAll('[data-view3d-status]').forEach(function (el) {
      el.textContent = text;
      el.classList.remove('ok', 'err');
      if (kind) { el.classList.add(kind); }
    });
  }

  async function openViewer(view, label) {
    setStatus('starting the Open3D window for "' + label + '" ...', null);
    try {
      var response = await fetch('/api/view3d?view=' + encodeURIComponent(view),
                                {method: 'POST'});
      var data = await response.json();
      if (!data.ok) {
        setStatus('could not open the window: ' + (data.error || response.status) +
                  (data.log ? '  (log: ' + data.log + ')' : ''), 'err');
        return;
      }
      setStatus((data.already_open ? 'window already open' : 'window opened') +
                ' on ' + data.title + ' (pid ' + data.pid + ') - drag to rotate, wheel to zoom,'
                + ' 1-4 to switch view, r to reset, s to save a snapshot, q to quit.'
                + ' log: ' + data.log, 'ok');
    } catch (err) {
      setStatus('request failed: ' + err, 'err');
    }
  }

  document.querySelectorAll('[data-view3d]').forEach(function (button) {
    button.addEventListener('click', function (event) {
      event.preventDefault();
      openViewer(button.getAttribute('data-view3d'), button.textContent.trim() || '3D');
    });
  });
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
def _wants_refresh() -> bool:
    return request.args.get("refresh", "").lower() in ("1", "true", "yes")


@app.route("/")
def index():
    """Single-page dashboard."""
    payload = run_pipeline(force=_wants_refresh())
    return render_template_string(TEMPLATE, p=payload, ttl=int(CACHE_TTL))


@app.route("/api/data")
def api_data():
    """Machine-readable payload (metrics + base64 PNG panels)."""
    return jsonify(run_pipeline(force=_wants_refresh()))


@app.route("/api/view3d", methods=["GET", "POST"])
def api_view3d():
    """Open the interactive Open3D window on the machine running the server.

    The dashboard button POSTs here; GET works too so the endpoint can be
    triggered from the address bar or curl.  The window is a separate
    *detached* process (:mod:`web.view3d`), so this request returns immediately
    and the Flask worker is never blocked by the Open3D event loop.  Nothing is
    streamed to the browser - a native GLFW window cannot live inside a web page
    - the JSON response just reports whether the window started, its pid and
    where to find its log.
    """
    view = request.args.get("view", VIEWS[0])
    with _VIEWER_LOCK:
        result = launch_viewer(view, dataset=DATA_PATH, config=CONFIG_PATH,
                               registry=_VIEWERS)
    result["dataset"] = DATA_PATH.name
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/healthz")
def healthz():
    """Liveness probe."""
    return jsonify({"status": "ok", "dataset": DATA_PATH.name,
                    "config": CONFIG_PATH.name})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Dashboard for the adaptive variable-resolution 2.5D LiDAR map"
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=5000, help="bind port")
    parser.add_argument("--debug", action="store_true", help="enable Flask debug mode")
    parser.add_argument("--export", metavar="PATH", default=None,
                        help="render the dashboard to a standalone HTML file and exit")
    args = parser.parse_args(argv)

    if args.export:
        payload = run_pipeline(force=True)
        with app.app_context():  # render_template_string needs an app context
            html = render_template_string(TEMPLATE, p=payload, ttl=int(CACHE_TTL))
        out = Path(args.export)
        if out.parent != Path(""):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"Wrote {out} ({len(html):,} chars, "
              f"{payload['metrics']['total_ms']:.1f} ms end-to-end)")
        return 0

    print(f"Dashboard: http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())









