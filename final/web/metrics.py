"""Metric helpers for the dashboard.

Pure-numpy utilities that turn the pipeline output into the numbers the
dashboard reports: per-band statistics, semantic breakdowns, uniform-grid
baselines and the memory-comparison table required by the problem statement
("demonstrating a significant reduction in memory usage compared to a uniform
high-resolution 3D map").

No plotting and no Flask here so the numbers can be unit-tested directly.
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def human_bytes(n: float) -> str:
    """Format a byte count with a binary unit suffix (``1.50 MB``)."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(round(value))} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


# --------------------------------------------------------------------------- #
# uniform 2.5D baseline
# --------------------------------------------------------------------------- #
def build_uniform_grid(
    xyz: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    cell: float,
) -> Dict[str, np.ndarray]:
    """Bin labelled points onto a *uniform* Cartesian 2.5D grid.

    Every point contributes to exactly one cell, mirroring the "no double
    counting / no drop" guarantee of the adaptive grid engine so that the two
    maps can be compared fairly.  Points labelled ``-1`` are collected in an
    extra ``unknown`` bucket, so ``pred == n_classes`` means *unknown*.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)

    x0 = float(np.floor(xyz[:, 0].min() / cell) * cell)
    y0 = float(np.floor(xyz[:, 1].min() / cell) * cell)
    ix = np.floor((xyz[:, 0] - x0) / cell).astype(np.int64)
    iy = np.floor((xyz[:, 1] - y0) / cell).astype(np.int64)
    n_rows = int(iy.max()) + 1
    flat = ix * n_rows + iy

    keys, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    cols = keys // n_rows
    rows = keys % n_rows

    hist = np.zeros((keys.size, n_classes + 1), dtype=np.int64)
    known = (lab >= 0) & (lab < n_classes)
    if np.any(known):
        np.add.at(hist, (inv[known], lab[known]), 1)
    if np.any(~known):
        np.add.at(hist[:, n_classes], inv[~known], 1)

    # height columns, mirroring the adaptive grid's 2.5D attributes so both
    # maps can be rendered / classified the same way.
    z = xyz[:, 2]
    sum_z = np.zeros(keys.size, dtype=np.float64)
    sum_z2 = np.zeros(keys.size, dtype=np.float64)
    max_z = np.full(keys.size, -np.inf, dtype=np.float64)
    np.add.at(sum_z, inv, z)
    np.add.at(sum_z2, inv, z * z)
    np.maximum.at(max_z, inv, z)
    mean_z = sum_z / counts
    var_z = np.clip(sum_z2 / counts - mean_z * mean_z, 0.0, None)

    return {
        "x": x0 + (cols + 0.5) * cell,
        "y": y0 + (rows + 0.5) * cell,
        "col": cols.astype(np.int64),
        "row": rows.astype(np.int64),
        "counts": counts,
        "hist": hist,
        "pred": np.argmax(hist, axis=1).astype(np.int64),
        "confidence": hist.max(axis=1).astype(np.float64) / counts,
        "sum_z": sum_z,
        "sum_z2": sum_z2,
        "max_z": max_z,
        "mean_z": mean_z,
        "var_z": var_z,
        "density_per_m2": 1.0 / (cell * cell),
        "n_cells": int(keys.size),
        "n_cols": int(cols.max()) + 1,
        "n_rows": n_rows,
        "cell": float(cell),
    }


# --------------------------------------------------------------------------- #
# adaptive (variable-resolution) grid statistics
# --------------------------------------------------------------------------- #
def band_statistics(grid, cfg) -> List[Dict[str, float]]:
    """Per-band occupancy / confidence table (one row per radial band)."""
    occ = grid.occupancy
    conf = grid.confidence
    rows: List[Dict[str, float]] = []
    for band in grid.bands:
        sl = slice(band.offset, band.offset + band.num_cells)
        occ_n = int(occ[sl].sum())
        points = int(grid.count[sl].sum())
        ring_area = float(np.pi * (band.rmax * band.rmax - band.rmin * band.rmin))
        rows.append(
            {
                "band": band.id + 1,
                "rmin": float(band.rmin),
                "rmax": float(band.rmax),
                "cell": float(band.cell),
                "n_r": int(band.n_r),
                "n_theta": int(band.n_theta),
                "cells": int(band.num_cells),
                "occupied": occ_n,
                "occupancy_pct": (100.0 * occ_n / band.num_cells) if band.num_cells else 0.0,
                "points": points,
                "mean_confidence": float(conf[sl][occ[sl]].mean()) if occ_n else 0.0,
                "dynamic_points": int(grid.dyn_count[sl].sum()),
                # rendering density of the band (the foveation) and how much real
                # scan data actually lands per square metre of that annulus
                "density_per_m2": 1.0 / (band.cell * band.cell),
                "points_per_m2": (points / ring_area) if ring_area > 0.0 else 0.0,
            }
        )
    return rows


def class_breakdown(
    hist_sum: Sequence[int],
    names: Sequence[str],
    unknown_name: str = "unknown",
) -> List[Dict[str, float]]:
    """Turn a summed semantic histogram into ``name / count / pct`` rows."""
    total = int(np.sum(hist_sum))
    out: List[Dict[str, float]] = []
    for i, name in enumerate(list(names) + [unknown_name]):
        count = int(hist_sum[i]) if i < len(hist_sum) else 0
        out.append(
            {
                "name": name,
                "count": count,
                "pct": (100.0 * count / total) if total else 0.0,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# memory model (variable resolution vs uniform baselines)
# --------------------------------------------------------------------------- #
_GRID_ARRAYS = ("count", "sum_z", "sum_z2", "max_z", "max_intensity",
                "class_hist", "dyn_count")


def variable_grid_footprint_bytes(grid) -> int:
    """Measured numpy footprint of the adaptive grid's per-cell arrays."""
    return int(sum(getattr(grid, name).nbytes for name in _GRID_ARRAYS))


def dense_cell_count(x, y, cell: float, pad_cells: int = 0) -> Dict[str, float]:
    """Number of uniform cells needed to densely tile the observed footprint."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0:
        return {"width": 0.0, "height": 0.0, "area": 0.0,
                "nx": 0, "ny": 0, "cells": 0}
    pad = cell * (1 + 2 * pad_cells)
    width = float(x.max() - x.min()) + pad
    height = float(y.max() - y.min()) + pad
    nx = max(1, int(np.ceil(width / cell)))
    ny = max(1, int(np.ceil(height / cell)))
    return {"width": width, "height": height, "area": float(width * height),
            "nx": nx, "ny": ny, "cells": nx * ny}


def memory_comparison(grid, cfg, xs, ys) -> Dict[str, float]:
    """Build the variable-vs-uniform memory table.

    All baselines are evaluated over the *same* sensor footprint so the
    comparison is apples-to-apples.  Per-cell byte costs come from
    ``metrics.*`` in ``configs/default.yaml``.
    """
    b_var = int(cfg.get("metrics.bytes_per_variable_cell", 32))
    b_25d = int(cfg.get("metrics.bytes_per_uniform25d_cell", 8))
    b_3d = int(cfg.get("metrics.bytes_per_uniform3d_voxel", 2))
    cell = float(cfg.get("metrics.uniform_cell_size", 0.05))
    span = float(cfg.get("metrics.uniform_height_span", 25.0))

    dense = dense_cell_count(xs, ys, cell)
    nz = max(1, int(np.ceil(span / cell)))

    var_cells = int(grid.total_cells)
    var_occupied = int(grid.occupancy.sum())
    var_bytes = var_cells * b_var
    var_measured = variable_grid_footprint_bytes(grid)
    u25_bytes = dense["cells"] * b_25d
    u3_bytes = dense["cells"] * nz * b_3d

    def reduction(base: float, other: float) -> float:
        return 100.0 * (1.0 - base / other) if other else 0.0

    return {
        "cell_size": cell,
        "height_span": span,
        "nz": nz,
        "variable_cells": var_cells,
        "variable_occupied_cells": var_occupied,
        "variable_bytes": var_bytes,
        "variable_bytes_sparse": var_occupied * b_var,
        "variable_bytes_measured": var_measured,
        "variable_bytes_per_cell": b_var,
        "uniform_cells": dense["cells"],
        "uniform_nx": dense["nx"],
        "uniform_ny": dense["ny"],
        "uniform_width": dense["width"],
        "uniform_height": dense["height"],
        "uniform_bytes_per_cell": b_25d,
        "uniform25d_bytes": u25_bytes,
        "uniform3d_voxels": dense["cells"] * nz,
        "uniform3d_bytes_per_voxel": b_3d,
        "uniform3d_bytes": u3_bytes,
        "reduction_vs_uniform25d_pct": reduction(var_bytes, u25_bytes),
        "reduction_vs_uniform3d_pct": reduction(var_bytes, u3_bytes),
        "reduction_measured_vs_uniform3d_pct": reduction(var_measured, u3_bytes),
    }


# --------------------------------------------------------------------------- #
# adaptive vs uniform 5 cm: measured head-to-head (the problem-statement proof)
# --------------------------------------------------------------------------- #
def _fmt_bytes(n: float) -> str:
    """MiB with two decimals, for the comparison labels."""
    return f"{float(n) / (1024.0 * 1024.0):.1f} MiB"


def comparison_summary(
    memory: Dict[str, float],
    adaptive_ms: float,
    uniform_ms: float,
    uniform_hit_cells: int,
    measured: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Assemble the adaptive-vs-uniform-5cm head-to-head for the proof panel.

    ``adaptive_ms`` and ``uniform_ms`` are *measured* per-frame costs on the same
    frame and the same machine (see :func:`measure_comparison`): the adaptive
    figure is this pipeline's projection plus temporal fusion, the uniform figure
    is the dense 5 cm 2.5D lattice doing those same two steps.  When ``measured``
    is supplied the cell-count and resident-memory rows are taken from the lattice
    that was actually built (real numpy array sizes); otherwise they fall back to
    the modelled byte-per-cell estimate from :func:`memory_comparison`.  No timing
    is ever modelled.

    Two storage rows are reported on purpose.  ``allocated`` is the state a map
    must keep alive *between* frames, which is what the variable-resolution grid
    shares across the whole polar domain.  ``occupied`` counts only the cells the
    current frame actually touched, where a dense lattice is trivially smaller
    because an empty cell costs nothing to store - but it still has to be
    *visited* every frame, and that bill lands in the latency row instead.
    """
    cell_cm = float(memory["cell_size"]) * 100.0
    a_cells = int(memory["variable_cells"])
    a_occ = int(memory["variable_occupied_cells"])
    a_bytes = float(memory["variable_bytes"])
    a_sparse = float(memory["variable_bytes_sparse"])
    u_cells = int(memory["uniform_cells"])
    u_hit = int(uniform_hit_cells)
    u_bytes = float(memory["uniform25d_bytes"])
    u_hit_bytes = float(u_hit * memory["uniform_bytes_per_cell"])

    # When the head-to-head experiment ran, the cell-count and resident-memory
    # rows are driven by the lattice that was *actually built* (same code path,
    # real numpy array sizes) instead of the modelled byte-per-cell estimate, so
    # this panel and its caption agree to the last digit.
    if measured:
        a_cells = int(measured["adaptive"]["cells"])
        u_cells = int(measured["uniform"]["cells"])
        a_bytes = float(measured["adaptive"]["bytes_measured"])
        u_bytes = float(measured["uniform"]["bytes_measured"])
        a_bytes_measured = a_bytes
        u_bytes_measured = u_bytes
    else:
        a_bytes_measured = float(memory["variable_bytes_measured"])
        u_bytes_measured = u_bytes

    adaptive_ms = float(adaptive_ms)
    uniform_ms = float(uniform_ms)
    a_fps = 1000.0 / adaptive_ms if adaptive_ms > 0.0 else 0.0
    u_fps = 1000.0 / uniform_ms if uniform_ms > 0.0 else 0.0

    def ratio(baseline: float, other: float) -> float:
        return (baseline / other) if other > 0.0 else 0.0

    # `advantage` is how many times better the adaptive map is on that axis, so
    # it is unit-free and >= 1.0 always means "adaptive wins".  It flips the
    # numerator for metrics where a *higher* value is the better one (FPS).
    def advantage(lower_is_better: bool, adaptive: float, uniform: float) -> float:
        if adaptive <= 0.0 or uniform <= 0.0:
            return 0.0
        return (uniform / adaptive) if lower_is_better else (adaptive / uniform)

    rows = [
        {"key": "cells",
         "label": "Grid cells (allocated)",
         "unit": "cells",
         "adaptive": float(a_cells),
         "uniform": float(u_cells),
         "adaptive_text": f"{a_cells:,}",
         "uniform_text": f"{u_cells:,}",
         "advantage": advantage(True, float(a_cells), float(u_cells)),
         "note": "state kept alive between frames"},
        {"key": "memory",
         "label": "Resident map memory",
         "unit": "MiB",
         "adaptive": a_bytes / (1024.0 * 1024.0),
         "uniform": u_bytes / (1024.0 * 1024.0),
         "adaptive_text": _fmt_bytes(a_bytes),
         "uniform_text": _fmt_bytes(u_bytes),
         "advantage": advantage(True, a_bytes, u_bytes),
         "note": "allocated per-cell arrays, measured" if measured
                     else "allocated per-cell arrays"},
        {"key": "memory_occupied",
         "label": "Occupied-cell storage",
         "unit": "MiB",
         "adaptive": a_sparse / (1024.0 * 1024.0),
         "uniform": u_hit_bytes / (1024.0 * 1024.0),
         "adaptive_text": _fmt_bytes(a_sparse),
         "uniform_text": _fmt_bytes(u_hit_bytes),
         "advantage": advantage(True, a_sparse, u_hit_bytes),
         "note": "cells hit this frame only - the one axis the lattice wins"},
        {"key": "latency",
         "label": "Projection + fusion per frame",
         "unit": "ms",
         "adaptive": adaptive_ms,
         "uniform": uniform_ms,
         "adaptive_text": f"{adaptive_ms:.1f} ms",
         "uniform_text": f"{uniform_ms:.1f} ms",
         "advantage": advantage(True, adaptive_ms, uniform_ms),
         "note": "measured wall clock, same frame"},
        {"key": "fps",
         "label": "Throughput",
         "unit": "FPS",
         "adaptive": a_fps,
         "uniform": u_fps,
         "adaptive_text": f"{a_fps:.1f} FPS",
         "uniform_text": f"{u_fps:.1f} FPS",
         "advantage": advantage(False, a_fps, u_fps),
         "note": "1000 / (projection + fusion latency)"},
    ]
    for row in rows:
        if row["advantage"] > 1.02:
            row["winner"] = "adaptive"
        elif row["advantage"] < 0.98:
            row["winner"] = "uniform"
        else:
            row["winner"] = "tie"

    return {
        "cell_size": float(memory["cell_size"]),
        "cell_cm": cell_cm,
        "rows": rows,
        "adaptive": {"cells": a_cells, "cells_occupied": a_occ, "bytes": a_bytes,
                     "bytes_occupied": a_sparse, "ms": adaptive_ms, "fps": a_fps},
        "uniform": {"cells": u_cells, "cells_occupied": u_hit, "bytes": u_bytes,
                    "bytes_occupied": u_hit_bytes, "ms": uniform_ms, "fps": u_fps},
        "measured": {
            "adaptive": {"cells": int(measured["adaptive"]["cells"]), "bytes": a_bytes_measured,
                         "projection_ms": float(measured["adaptive"]["projection_ms"]),
                         "fusion_ms": float(measured["adaptive"]["fusion_ms"])},
            "uniform": {"cells": int(measured["uniform"]["cells"]), "bytes": u_bytes_measured,
                        "projection_ms": float(measured["uniform"]["projection_ms"]),
                        "fusion_ms": float(measured["uniform"]["fusion_ms"])},
            "repeats": int(measured["repeats"]),
            "memory_ratio": (u_bytes_measured / a_bytes_measured) if a_bytes_measured else 0.0,
            "cells_ratio": (float(measured["uniform"]["cells"]) / measured["adaptive"]["cells"])
            if measured["adaptive"]["cells"] else 0.0,
        } if measured else None,
        "ratios": {
            "cells": ratio(u_cells, a_cells),
            "memory": ratio(u_bytes, a_bytes),
            "memory_occupied": ratio(u_hit_bytes, a_sparse),
            "latency": ratio(uniform_ms, adaptive_ms),
            "fps": ratio(a_fps, u_fps),
        },
        "speedup": ratio(uniform_ms, adaptive_ms),
        "memory_reduction_pct": 100.0 * (1.0 - a_bytes / u_bytes) if u_bytes else 0.0,
        "uniform_hit_cells": u_hit,
    }


# --------------------------------------------------------------------------- #
# measured adaptive-vs-uniform 5 cm head-to-head (projection + fusion + memory)
# --------------------------------------------------------------------------- #
def uniform_lattice_footprint_bytes(grid) -> int:
    """Measured numpy footprint of a built grid's per-cell arrays.

    Works for any :class:`mapping.grid_engine.VariableGrid`, adaptive or the
    single-band uniform 5 cm lattice, which is what makes the memory row in
    :func:`measure_comparison` a real measurement of the dense map rather than a
    cell-count-times-byte-size estimate.
    """
    return int(sum(getattr(grid, name).nbytes for name in _GRID_ARRAYS))


def _median_ms(fn, repeats: int) -> Tuple[float, object]:
    """Run ``fn`` ``repeats`` times, returning ``(median milliseconds, last result)``.

    The median rather than the mean, so a single scheduler hiccup cannot distort
    the reported per-frame cost.
    """
    times: List[float] = []
    out = None
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), out


def measure_comparison(
    grid,
    cfg,
    xyz: np.ndarray,
    intensity: np.ndarray,
    labels: np.ndarray,
    dynamic: np.ndarray,
    uniform: Dict[str, np.ndarray],
    memory: Dict[str, float],
    n_classes: int,
    adaptive_projection_ms: float,
    repeats: int = 3,
) -> Dict[str, object]:
    """Measure the adaptive grid against a *dense uniform 5 cm* map, head to head.

    Both maps are built and fused with the **same code paths** the live pipeline
    uses - :func:`mapping.grid_engine.build_grid` and
    :func:`mapping.temporal.update_temporal` - over the **same frame**, and
    everything here is wall-clock measured on this machine.  Nothing is modelled
    or extrapolated; nothing is invented.

    The uniform baseline is a single uniform-resolution polar band of the
    configured cell size (default 5 cm) spanning the whole sensor footprint, so
    it is projected exactly like the adaptive map rather than onto a Cartesian
    lattice.  Fusion is measured as the steady per-frame update over an
    *already-allocated* state, because that is what the adaptive grid pays every
    frame; the dense map's one-off, ~80 ms ``init_temporal_state`` is therefore
    deliberately excluded and the comparison is, if anything, generous to it.

    Returns adaptive/uniform cell counts, byte footprints, per-frame projection
    and fusion latencies, throughput and size ratios.  The single-band 12.6 M
    cell grid is freed before returning so the dashboard's resident memory is not
    permanently inflated by the measurement.
    """
    from mapping.grid_engine import build_grid as _build_grid
    from mapping.preprocess import compute_polar, filter_points
    from mapping.temporal import init_temporal_state, update_temporal

    cell = float(cfg.get("metrics.uniform_cell_size", 0.05))
    pre = cfg.section("preprocess")
    xyz_f, _ = filter_points(
        np.asarray(xyz, dtype=np.float32),
        float(pre.get("min_range", 1.0)),
        float(pre.get("max_range", 100.0)),
        float(pre.get("height_min", -5.0)),
        float(pre.get("height_max", 5.0)),
        pre.get("decimation"),
    )
    r, theta, _ = compute_polar(xyz_f)
    rmax = float(r.max()) + cell if r.size else float(cfg.get("preprocess.max_range", 100.0))
    lattice_spec = [{"rmin": 0.0, "rmax": rmax, "cell": cell}]
    inten_f = np.asarray(intensity, dtype=np.float32)
    lab_f = np.asarray(labels, dtype=np.int64)
    dyn_f = np.asarray(dynamic, dtype=bool)

    # --- adaptive: projection (measured by the live pipeline) + fusion --------
    adaptive_proj_ms = float(adaptive_projection_ms)
    adaptive_state = init_temporal_state(grid.total_cells, n_classes, cfg)
    adaptive_fusion_ms, _ = _median_ms(
        lambda: update_temporal(adaptive_state, grid.occupancy, grid.mean_height,
                                grid.max_intensity, grid.class_hist, 0, cfg),
        repeats,
    )

    # --- dense uniform 5 cm: build once, then measure steady-state stages -----
    dense_proj_ms, dense = _median_ms(
        lambda: _build_grid(r, theta, xyz_f, inten_f, lab_f, dyn_f, lattice_spec, n_classes),
        repeats,
    )
    dense_state = init_temporal_state(dense.total_cells, n_classes, cfg)
    dense_fusion_ms, _ = _median_ms(
        lambda: update_temporal(dense_state, dense.occupancy, dense.mean_height,
                                dense.max_intensity, dense.class_hist, 0, cfg),
        repeats,
    )
    dense_cells = int(dense.total_cells)
    dense_bytes_measured = uniform_lattice_footprint_bytes(dense)

    # release the 12.6 M-cell lattice + its temporal state before returning
    del dense, dense_state

    adaptive_ms = adaptive_proj_ms + adaptive_fusion_ms
    uniform_ms = dense_proj_ms + dense_fusion_ms
    return {
        "cell_size": cell,
        "cell_cm": cell * 100.0,
        "adaptive": {
            "cells": int(grid.total_cells),
            "cells_occupied": int(grid.occupancy.sum()),
            "bytes": float(memory["variable_bytes"]),
            "bytes_measured": float(memory["variable_bytes_measured"]),
            "projection_ms": adaptive_proj_ms,
            "fusion_ms": adaptive_fusion_ms,
            "ms": adaptive_ms,
            "fps": (1000.0 / adaptive_ms) if adaptive_ms > 0.0 else 0.0,
        },
        "uniform": {
            "cells": dense_cells,
            "cells_hit": int(uniform["n_cells"]),
            "bytes": float(memory["uniform25d_bytes"]),
            "bytes_measured": float(dense_bytes_measured),
            "projection_ms": float(dense_proj_ms),
            "fusion_ms": float(dense_fusion_ms),
            "ms": uniform_ms,
            "fps": (1000.0 / uniform_ms) if uniform_ms > 0.0 else 0.0,
        },
        "repeats": int(repeats),
    }


# --------------------------------------------------------------------------- #
# drivability layer (roadmap Phase 4, step 17)
# --------------------------------------------------------------------------- #
def cell_slope(grid) -> np.ndarray:
    """Local terrain slope per cell, from the 2.5D height field.

    The finite difference is taken between *neighbouring cells of the same
    band* (radial and tangential, tangentially wrapping at +/-pi) and divided
    by the cell size, which is the arc length of a tangential step by
    construction (``dtheta = cell / r_mean``).  The larger of the two gradients
    wins.  Cells with no occupied neighbour keep slope ``0``.
    """
    h = grid.mean_height.astype(np.float64)
    occ = grid.occupancy
    slope = np.zeros(grid.total_cells, dtype=np.float64)

    for b in grid.bands:
        sl = slice(b.offset, b.offset + b.num_cells)
        hb = h[sl].reshape(b.n_r, b.n_theta)
        ob = occ[sl].reshape(b.n_r, b.n_theta)

        s = np.zeros_like(hb)

        # radial neighbours: rows r and r+1
        dr = np.abs(np.diff(hb, axis=0)) / b.cell
        pair = ob[:-1] & ob[1:]
        s[:-1] = np.where(pair, np.maximum(s[:-1], dr), s[:-1])
        s[1:] = np.where(pair, np.maximum(s[1:], dr), s[1:])

        # tangential neighbours: cols c and c+1 (wrap around the circle)
        hw = np.concatenate([hb, hb[:, :1]], axis=1)
        dt = np.abs(np.diff(hw, axis=1)) / b.cell
        pair_t = ob & np.roll(ob, -1, axis=1)
        s = np.where(pair_t, np.maximum(s, dt), s)

        slope[sl] = s.ravel()

    return slope


def drivability_tiers(grid, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split drivable cells into three walkability tiers.

    A cell violates the *slope* criterion when ``cell_slope > slope_threshold``
    and the *roughness* criterion when ``sqrt(height_variance) >
    roughness_threshold``; the tier is the number of violated criteria, so

    * tier 0 -> meets both (firm, smooth ground),
    * tier 1 -> violates one,
    * tier 2 -> violates both (rough / steep ground).

    Returns ``(tier, slope, roughness)``; ``tier`` is ``-1`` for unoccupied
    cells so the caller can mask it.
    """
    slope_thr = float(cfg.get("drivability.slope_threshold", 0.35))
    rough_thr = float(cfg.get("drivability.roughness_threshold", 0.08))

    slope = cell_slope(grid)
    rough = np.sqrt(np.maximum(grid.height_variance.astype(np.float64), 0.0))
    occ = grid.occupancy

    violations = ((slope > slope_thr).astype(np.int64)
                  + (rough > rough_thr).astype(np.int64))
    tier = np.full(grid.total_cells, -1, dtype=np.int64)
    tier[occ] = violations[occ]
    return tier, slope, rough


def drivability_summary(tier: np.ndarray, cfg, names: Sequence[str]) -> List[Dict[str, float]]:
    """Per-tier cell counts + share of the walkable ground."""
    labels = list(cfg.get("drivability.tier_names", []) or []) or ["firm", "moderate", "rough"]
    occupied = int(np.count_nonzero(tier >= 0))
    rows: List[Dict[str, float]] = []
    for t in range(len(labels)):
        count = int(np.count_nonzero(tier == t))
        rows.append({
            "tier": t,
            "name": labels[t] if t < len(labels) else f"tier {t}",
            "count": count,
            "pct": (100.0 * count / occupied) if occupied else 0.0,
        })
    return rows


# --------------------------------------------------------------------------- #
# rendering density (the foveation itself)
# --------------------------------------------------------------------------- #
def density_per_cell(grid) -> np.ndarray:
    """Grid samples per square metre for every cell of the adaptive grid.

    Each band is uniform, so the sampling density is constant inside a band and
    equals ``1 / cell^2``: 400 cells/m^2 for the 5 cm inner band, 4 cells/m^2
    for the 50 cm outer band.  This is the quantity the adaptive map renders as
    a colour gradient.
    """
    density = np.zeros(grid.total_cells, dtype=np.float64)
    for b in grid.bands:
        density[b.offset:b.offset + b.num_cells] = 1.0 / (b.cell * b.cell)
    return density


def density_extent(bands) -> Tuple[float, float]:
    """``(min, max)`` cells per m^2 over the configured bands.

    Used as the *absolute* range of the density colour gradient: 400 cells/m^2
    for a 5 cm band, 4 cells/m^2 for a 50 cm band.  Anchoring the ramp to the
    configuration instead of to the data means a colour always means the same
    density, whatever is in view.
    """
    dens = np.array([1.0 / (float(b.cell) ** 2) for b in bands], dtype=np.float64)
    if dens.size == 0:
        return 1.0, 1.0
    return float(dens.min()), float(dens.max())


def density_to_t(density: np.ndarray, dens_min: float, dens_max: float) -> np.ndarray:
    """Map cells/m^2 to the ``[0, 1]`` coordinate of the density gradient.

    ``log10``-scaled, because the range spans two orders of magnitude.
    """
    d = np.maximum(np.asarray(density, dtype=np.float64), 1e-9)
    lo = float(np.log10(max(dens_min, 1e-9)))
    hi = float(np.log10(max(dens_max, 1e-9)))
    if hi <= lo:
        return np.zeros(d.shape, dtype=np.float64)
    return np.clip((np.log10(d) - lo) / (hi - lo), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# drivability on the uniform baseline (so the two maps are classified alike)
# --------------------------------------------------------------------------- #
def _label_components(cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """8-connected component labels for integer ``(col, row)`` cell coords.

    A tiny union-find over the occupied coarse cells, kept in pure
    numpy/Python so this module stays dependency-free (the project does not
    use scipy anywhere).  Returns one label per input cell, ``0..k-1``.
    """
    n = int(cols.size)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    c = (cols - cols.min()).astype(np.int64)
    r = (rows - rows.min()).astype(np.int64)
    width = int(r.max()) + 1
    keys = c * width + r

    uniq = np.unique(keys)
    node_of = {int(k): i for i, k in enumerate(uniq)}
    parent = np.arange(uniq.size, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for k in uniq.tolist():
        node = node_of[k]
        cc, rr = divmod(int(k), width)
        for dc, dr in ((0, 1), (1, -1), (1, 0), (1, 1)):
            nb = node_of.get((cc + dc) * width + (rr + dr))
            if nb is not None:
                ra, rb = find(node), find(nb)
                if ra != rb:
                    parent[rb] = ra

    roots = np.array([find(node_of[int(k)]) for k in keys], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def detect_objects(
    grid,
    cfg,
    class_names: Sequence[str] | None = None,
    ground_class: int = 0,
) -> Dict[str, object]:
    """Group non-ground occupied cells into discrete objects for the dashboard.

    Cells are pooled onto a coarse Cartesian lattice (``objects.cell_size``,
    metres) and merged with 8-connectivity, so a wall, a pole or a parked car
    becomes a single row instead of hundreds of individual cells.  Every blob
    reports its centroid in the sensor frame (the viewer/ego origin sits at
    ``(0, 0, 0)``), its extent and its distance from that origin.

    Coordinates are metres in the same frame as the point cloud: ``+x``
    forward, ``+y`` left, ``+z`` up.  ``range_m`` is the ground-plane
    (horizontal) distance and ``distance_m`` the full 3D distance from the
    viewer, both measured to the blob centroid.
    """
    names = list(class_names or [])
    cell_size = float(cfg.get("objects.cell_size", 0.5))
    min_cells = int(cfg.get("objects.min_cells", 5))
    min_height = float(cfg.get("objects.min_height", 0.30))
    max_range = float(cfg.get("objects.max_range", 100.0))
    max_objects = int(cfg.get("objects.max_objects", 200))

    summary = {
        "enabled": bool(cfg.get("objects.enabled", True)),
        "cell_size": cell_size,
        "min_cells": min_cells,
        "min_height": min_height,
        "max_range": max_range,
        "count": 0,
        "cells_considered": 0,
        "items": [],
    }
    if not summary["enabled"]:
        return summary

    xs, ys = grid.cell_centers()
    occ = np.asarray(grid.occupancy, dtype=bool)
    if not np.any(occ):
        return summary

    band_of = np.empty(grid.total_cells, dtype=np.int64)
    for b in grid.bands:
        band_of[b.offset:b.offset + b.num_cells] = b.id

    idx = np.nonzero(occ)[0]
    px = xs[idx].astype(np.float64)
    py = ys[idx].astype(np.float64)
    pz = grid.mean_height[idx].astype(np.float64)
    top = np.asarray(grid.max_z, dtype=np.float64)[idx]
    cls = grid.pred_class[idx].astype(np.int64)
    conf = np.asarray(grid.confidence, dtype=np.float64)[idx]
    band = band_of[idx]

    rng = np.sqrt(px * px + py * py)
    keep = (cls != ground_class) & np.isfinite(rng) & (rng <= max_range)
    if min_height > 0.0:
        keep &= top >= min_height
    if not np.any(keep):
        return summary

    px, py, pz, top, cls, conf, rng, band = (a[keep] for a in
                                             (px, py, pz, top, cls, conf, rng, band))
    summary["cells_considered"] = int(px.size)

    cols = np.floor(px / cell_size).astype(np.int64)
    rows = np.floor(py / cell_size).astype(np.int64)
    labels = _label_components(cols, rows)

    items: List[Dict[str, object]] = []
    for label in range(int(labels.max()) + 1 if labels.size else 0):
        sel = labels == label
        counts = int(np.count_nonzero(sel))
        if counts < min_cells:
            continue
        cx = float(px[sel].mean())
        cy = float(py[sel].mean())
        cz = float(pz[sel].mean())
        spread_x = float(px[sel].max() - px[sel].min())
        spread_y = float(py[sel].max() - py[sel].min())
        top_z = float(top[sel].max())
        base_z = float(top[sel].min())
        vote = np.bincount(cls[sel], minlength=max(1, len(names)))
        class_id = int(np.argmax(vote))
        class_name = names[class_id] if 0 <= class_id < len(names) else f"class {class_id}"
        # band = coarsest band the blob reaches (its far edge)
        blob_band = int(band[sel].max())
        range_m = float(np.hypot(cx, cy))
        items.append({
            "id": 0,  # filled in after sorting
            "class_id": class_id,
            "class_name": class_name,
            "x": cx,
            "y": cy,
            "z": cz,
            "range_m": range_m,
            "distance_m": float(np.sqrt(cx * cx + cy * cy + cz * cz)),
            "cells": counts,
            "size_x_m": spread_x,
            "size_y_m": spread_y,
            "height_m": max(0.0, top_z),
            "base_m": base_z,
            "band": blob_band,
            "confidence": float(conf[sel].mean()),
        })

    items.sort(key=lambda item: item["distance_m"])
    items = items[:max_objects]
    for i, item in enumerate(items, start=1):
        item["id"] = i
    summary["count"] = len(items)
    summary["items"] = items
    return summary


def uniform_drivability_tiers(
    uniform: Dict[str, np.ndarray], cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drivability tiers for the uniform 2.5D grid.

    On a uniform Cartesian lattice both neighbours of a cell are exactly one
    ``cell`` away, so the finite difference is a plain ``|dh| / cell`` in the
    x and y directions.  Classification then uses the identical slope /
    roughness thresholds as the adaptive map.
    """
    cell = float(uniform["cell"])
    n_cols = int(uniform["n_cols"])
    n_rows = int(uniform["n_rows"])

    h = np.full((n_cols, n_rows), np.nan, dtype=np.float64)
    h[uniform["col"], uniform["row"]] = uniform["mean_z"]
    occ = ~np.isnan(h)

    slope = np.zeros_like(h)

    # x direction
    dx = np.abs(np.diff(h, axis=0)) / cell
    pair = occ[:-1] & occ[1:]
    dx = np.nan_to_num(dx, nan=0.0)
    slope[:-1] = np.where(pair, np.maximum(slope[:-1], dx), slope[:-1])
    slope[1:] = np.where(pair, np.maximum(slope[1:], dx), slope[1:])

    # y direction
    dy = np.abs(np.diff(h, axis=1)) / cell
    pair = occ[:, :-1] & occ[:, 1:]
    dy = np.nan_to_num(dy, nan=0.0)
    slope[:, :-1] = np.where(pair, np.maximum(slope[:, :-1], dy), slope[:, :-1])
    slope[:, 1:] = np.where(pair, np.maximum(slope[:, 1:], dy), slope[:, 1:])

    cell_sl = slope[uniform["col"], uniform["row"]]
    rough = np.sqrt(np.maximum(uniform["var_z"], 0.0))

    slope_thr = float(cfg.get("drivability.slope_threshold", 0.35))
    rough_thr = float(cfg.get("drivability.roughness_threshold", 0.08))
    violations = ((cell_sl > slope_thr).astype(np.int64)
                  + (rough > rough_thr).astype(np.int64))
    return violations, cell_sl, rough

