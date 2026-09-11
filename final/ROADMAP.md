# Roadmap — Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

## 1. Problem Decomposition

The task has three coupled sub-problems:

| Sub-problem | Input | Output |
|---|---|---|
| A. Semantic understanding | Raw point cloud (N×4: x,y,z,intensity) | Per-point class: drivable terrain / static obstacle / dynamic object |
| B. Variable-resolution projection | Classified 3D points | 2.5D grid (elevation + semantic layers) with range-dependent cell size |
| C. Performance & visualization | The 2.5D grid | Color-coded map, FPS, memory savings vs uniform 3D map |

Key challenge: the projection in (B) must be exact (no alignment errors / data loss)
despite non-uniform cells, and the whole pipeline must run in real time.

## 2. Available Data & Environment (verified)

- `fake_lidar_000000.bin` — KITTI/Velodyne format: **71,000 points × float32 (x, y, z, intensity)**, range ~0.2–140 m, height −10 m to +15 m. → Loader assumes N×4 little-endian float32.
- Platform: Windows + Python 3 (NumPy confirmed available).
- No labels available → training strategy must include a **synthetic/self-labeled bootstrap** and/or public datasets (SemanticKITTI).

## 3. Target Architecture (end-to-end)

```
.bin file / live sensor
      │
[1] Preprocessing: range filter, ground pre-filter, polar features
      ▼
[2] Semantic Segmentation DL Model (SparseConv / PointNet++ / RangeNet++)
      ▼  per-point semantic label + dynamic-object flag
[3] Variable-Resolution Grid Engine (polar radial bands)
      ▼  2.5D grid: per-cell {max/mean height, intensity, semantic class, occupancy}
[4] Post-processing: drivability layer, temporal fusion for dynamic objects
      ▼
[5] Real-time visualization dashboard + [6] Metrics logger (FPS, latency, memory)
```

## 4. Step-by-Step Implementation Plan

### Phase 0 — Project scaffolding & reproducibility
1. Create project structure: `src/`, `data/`, `scripts/`, `tests/`, `configs/`.
2. Pin dependencies: `numpy`, `scipy`, `torch` (or ONNX Runtime), `opencv-python`, `pyqtgraph` for the dashboard.
3. Config-driven design (YAML): all resolution bands, radii, model paths, thresholds in one config.

### Phase 1 — Data layer (deterministic foundation)
4. **Point-cloud loader**: `np.fromfile(dtype=np.float32).reshape(-1,4)`; validate shape; unit-test against the provided .bin.
5. **Preprocessing**:
   - Range clip (e.g., drop <1 m and >120 m), NaN/inf purge.
   - Compute derived features: radial distance `r`, azimuth `θ`, elevation angle.
   - Coarse ground segmentation (RANSAC plane or height-threshold strip by range) to (a) boost DL input and (b) provide fallback labels.
6. **Label bootstrap strategy** (since no GT here):
   - Train/finetune on **SemanticKITTI** (28 classes → remap to 3 meta-classes: drivable, static obstacle, dynamic).
   - For local demo: heuristic pseudo-labels (height + slope + intensity) to smoke-test the pipeline end-to-end without the DL model.
7. Dataloader with augmentation (z-rotation, mirroring, point dropout, translation jitter).

### Phase 2 — Semantic segmentation model
8. **Model choice (pragmatic):**
   - Primary: **SparseConv U-Net (Minkowski/torchsparse-style)** on ~0.1–0.2 m voxels — strong accuracy, fast on GPU.
   - Fallback/CPU: **RangeNet++** (spherical projection → 2D CNN on range image) — simplest to deploy, genuinely real-time.
   - Export to **ONNX** for portable inference.
9. **Classes**: `0 drivable/terrain, 1 static obstacle, 2 dynamic object (pedestrian/vehicle)`. Optional finer head for richer semantics.
10. **Dynamic object cue**:
    - Learned: SemanticKITTI `moving-*` classes.
    - Geometric temporal: frame-difference residuals after ego-motion compensation (identity-motion fallback).
11. Train: cross-entropy + Lovász-softmax loss (class imbalance), AdamW, cosine LR; validate mIoU per class.
12. Evaluation harness: per-class precision/recall binned by range (0–10, 10–30, 30–60, 60–100 m) — needed for "accuracy across varying distances".
### Phase 3 — Variable-resolution grid engine (core novelty)
13. **Coordinate design**: polar band specification (from problem statement)**:
    - Band 1: r ∈ [0, 10 m) → 5 cm cells
    - Band 2: r ∈ [10, 30 m) → 10–15 cm (we choose 12.5 cm)
    - Band r ∈ [30, 60 m) → 25 cm
    - Band r ∈ [60, 100 m] → 50 cm
    Bands beyond 100 m are dropped (configurable max_range).
14. **Data structure options (pick one, keep interface abstract)**:
    - (a) **Radial-band arrays**: per band a regular (n_θ × n_r) lattice — O(1) insert, vectorized NumPy baseline.
    - (b) **Quadtree / hierarchical hash grid** (NDT/mesh style) — more general for elliptical FOVs.
    We start with (a) for guaranteed real-time performance.
15. **Exact projection rule**:
    - For each point compute `r = sqrt(x^2+y^2)`, `θ = atan2(y,x) ∈ [-π, π)`.
    - Select band `b` where `r_min[b] ≤ r < r_max[b]` (half-open intervals; edge case r=r_max goes to higher band except last).
    - Cell indices:
        `row_b = floor((r − r_min[b]) / cell_size[b])`
        `col_b = floor((θ + π) / dtheta[b])` where `dtheta[b] = cell_size[b] / (r_mean[b])` (keeps cell width ≈ height).
    - Clamp indices to `[0, n_θ[b]-1] × [0, n_r[b]-1]`; any point falling outside is discarded (should not happen with correct ranges).
    Guarantees: each classified point contributes to exactly one cell → no double-count or loss.
16. **Per-cell 2.5D attributes** (accumulated via vectorized scatter / bincount):
    - Basic geometry: `count`, `sum_z`, `sum_z2` → `mean_height`, `height_variance`.
    - `max_z`, `max_intensity`.
    - Semantic voting: per-class count histogram → `pred_class = argmax`, `confidence = max_count / total`.
    - Dynamic flag: `any_dynamic = (dynamic_point_count > 0)`.
    Implementation: flatten `(band, row, col)` to a global linear ID; use `np.bincount(weights=..., minlength=num_cells)` for each attribute.
17. **Drivable surface computation**:
    - Slope estimate from height + finite-difference gradient (or plane-fit 3×3 window).
    - Roughness = `sqrt(height_variance)`.
    - Drivable if (`semantic == drivable`) AND (`slope < slope_thresh`) AND (`roughness < rough_thresh`) → separate drivability layer.
18. **Band-transition correctness (seam tests)**:
    - Unit test: generate a synthetic ring of points at exact band boundary `r = r_min[b+1]`, varying θ.
    - Verify all points map into band `b+1` cell indices, none into band `b` (half-open interval).
    - Test azimuth wrap (`θ = -π` vs `θ = +π`) maps to same column index 0.
    - Point conservation: total points counted in all cells equals number of range-filtered input points.
19. **Performance target**: fully vectorized insertion of 71k points < 5 ms on modest CPU; benchmark per band.

### Phase 4 — Temporal integration (dynamic environments)
20. Ego-motion compensation: provide identity transform by default; later integrate `/odom` or IMU to deskew scans.
21. Per-cell temporal filtering:
    - Occupancy: log-odds update `L_{t} = L_{t-1} + α if hit else -β if miss`.
    - Height/intensity: exponential moving average `v_t = (1-γ) v_{t-1} + γ v_new`.
    - Semantic label: majority vote over last N frames or confidence-weighted EMA of class probabilities.
22. Dynamic-object persistence: simple clustering (DBSCAN eps≈1 m, min_samples=3) on cells flagged dynamic → assign track IDs via nearest-centroid association (optional stretch).

### Phase 5 — Visualization dashboard
23. Real-time top-down rendering:
    - Map semantic class → color: drivable=(0,200,0), static=(100,100,100), dynamic=(255,140,0), unknown=(50,50,50).
    - Alpha/value = confidence × normalized occupancy (log scale).
    - Overlay band boundaries as faint white lines to show variable resolution.
24. Telemetry pane (top/side):
    - FPS (end-to-end and per-stage: load, infer, project, render).
    - Memory footprint: current map size (bytes) vs uniform 5 cm 2.5D vs uniform 3D voxel.
    - Per-band active cell count and point density.
25. Tech: `pyqtgraph.ImageItem` inside `GraphicsLayoutWidget` for 60 fps updates; decouple via a thread-safe queue (max 2 frames) to avoid viz stalling pipeline.

### Phase 5b – Comparative 3D Visualization — __Answering your specific request__

This renders:

1. __Left pane__ = Uniform 5 cm resolution (thin white grid / wireframe) → memory strawman.

2. __Right pane__ = Our variable-resolution map reconstructed as 3D geometry.

3. __Comparison cues__:

   - Cell boundary contours (faint gray on both).
   - Red “diff” brush on columns where uniform cell is outside our cols → demonstrating memory savings.

4. __Autoresizing__: simple slider to crop >30 m (uniform would explode memory there).

5. __Labels__:

   - DBSCAN clusters get 3D oriented box + floating text showing type and distance.
   - Example: “Vehicle @ 12.3 m” → orange bbox on right pane only.

6. __Hotkeys__:

   - `← →` rotate, `mouse wheel` zoom, `space` pause.
   - `c` toggles camera sync.
   - `s` screenshot to PNG.
   - `t` print memory counts to console.

__Status:__ the interactive half is implemented — `web/view3d.py` runs a real Open3D window
(see Phase 5c below) with `1`–`4` view switching, `r` reset, `s` screenshot to PNG and `q`
quit, driven from the dashboard button. Still open: the side-by-side *uniform vs adaptive*
wireframe panes (items 3–5 above: boundary contours, red diff brush, range slider, DBSCAN
boxes/labels) and the `← →` / `space` / `c` / `t` hotkeys.

__Open3D implementation sketch__ (note: the real code uses
`o3d.visualization.VisualizerWithKeyCallback`, since `Open3DVisualizer` is a GUI-only class
and plain `Visualizer` has no key callbacks in Open3D ≥ 0.19):

```python
import open3d as o3d
import numpy as np

def build_geometry(grid):
    """Return Open3D PointCloud of cell centers or meshes."""
    cols = []
    colors = []
    for band in grid.bands:
        for r_bin, theta_bin, cell in band.cells:
            x = r_bin * np.cos(theta_bin)
            y = r_bin * np.sin(theta_bin)
            z_mid = cell.mean_z
            cols.append([x, y, z_mid])
            colors.append(SEMANTIC_COLOR[cell.cls])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(cols))
    pc.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return pc

# Main loop
viz = o3d.visualization.Open3DVisualizer()
viz.add_geometry(build_geometry(uniform_grid))
viz.add_geometry(build_geometry(variable_grid))
viz.poll_events()
```

### Phase 6 — Metrics & validation (evidence deliverables)
26. Memory comparison (report table):
    - Uniform 3D voxel grid: 5 cm³ over 200×200×25 m = 400×400×50 = 8M voxels × 2 bytes (uint16 occupancy) ≈ 16 MB.
    - Uniform 2.5D grid: 5 cm over 200×200 = 400×400 = 160K cells × 8 bytes (float height + class) ≈ 1.3 MB.
    - Our variable grid: (Band1: 20 m diameter @5cm → 400×400 = 160k) + (Band2 ring @12.5cm ≈ 62k) + ... total ≈ 90K cells → ~0.7 MB.
    → >70% reduction vs uniform 2.5D, >95% vs 3D.
27. Latency breakdown (target hardware: RTX 3060 laptop / i7-12700H):
    - Point load: 0.5 ms

### Phase 7 — Hardening & delivery
31. Edge cases:
    - Empty scan → return empty map with zeros.
    - All-ground scan → drivability layer mostly true, static obstacles zero.
    - Sensor too low/high → clip valid height range (−5 m, +5 m) configurable.
32. CLI entry points:
    - `python -m mapping.demo data/fake_lidar_000000.bin`
    - `python -m mapping.stream data/scans/ --pattern "*.bin"`
    - `python -m mapping.bin --help` (shows config options)
33. Documentation:
    - README: quick-start, one-liner install (`pip install -r requirements.txt`), demo command.
    - Architecture diagram (Mermaid or draw.io) showing data flow.
    - Config reference: all tunable parameters with defaults and units.
34. Packaging:
    - `requirements.txt` (numpy, torch, pyqtgraph, tqdm, pyyaml, etc.).
    - Optional `Dockerfile` (nvidia/cuda:12.1-base) for reproducible GPU builds.

### Phase 5c — Web dashboard (implemented)

The headless/CI-friendly counterpart to the pyqtgraph dashboard above: a Flask app that
renders the map and telemetry server-side (matplotlib Agg → base64 PNG), so it works in a
browser with no GUI toolkit installed.

**Files**

| Path | Purpose |
|---|---|
| `web/app.py` | Flask app, `run_pipeline()`, HTML template; routes `/`, `/api/data`, `/api/view3d`, `/healthz` |
| `web/metrics.py` | per-band statistics, semantic breakdown, uniform baseline, memory model |
| `web/plots.py` | base64 PNG renderers (adaptive map, uniform map, latency, memory) |
| `web/render3d.py` | Open3D offscreen backend: `VIEWS`, `scene_clouds()`, `camera_settings()`, `apply_camera()`, `SceneRenderer` |
| `web/view3d.py` | interactive Open3D window: `build_scene()`, `run_window()`, detached `launch_viewer()`, `python -m web.view3d` CLI |
| `scripts/check_dashboard.py` | smoke test: runs the pipeline, asserts 10 panels + 4 routes + viewer error paths |
| `scripts/check_view3d_window.py` | opens the real window for a few seconds, proves it reached the event loop, closes it |

**Usage**

```bash
pip install flask open3d
python web/app.py --port 5000              # http://127.0.0.1:5000
python web/app.py --export reports/dashboard.html   # standalone snapshot
python scripts/check_dashboard.py          # no-browser verification (no window)
python scripts/check_view3d_window.py      # opens + verifies + closes the 3D window

# standalone viewer, or geometry check with no window at all
python -m web.view3d --view uniform_density
python -m web.view3d --list-views
python -m web.view3d --check
```

**Panels**

1. Adaptive variable-resolution map — class colour, brightness = confidence × log(occupancy),
   marker area ∝ band cell size, faint circles on band boundaries (item 23).
2. Uniform 2.5D map at `metrics.uniform_cell_size` over the same footprint, as the strawman.
3. Drivable / non-drivable overlay for the adaptive map (drivability layer, item 21).
4. Cell-density overlay for the adaptive map (measured occupancy per cell).
5. Per-stage latency chart + table (load / preprocess / segment / project / temporal, FPS).
6. Memory chart + table: variable grid vs uniform 2.5D vs uniform 3D voxel (item 24).
7. Four Open3D 3D panels — adaptive/uniform × semantic/density — rendered offscreen at
   1280×720 from the shared `scene_clouds()` geometry; absent (with the reason in
   `render3d_error`) when no desktop/OpenGL driver is present.
8. Per-band table: cell size, `n_r × n_theta`, occupied cells, fill %, mean confidence.

**Interactive 3D window.** The *Open interactive 3D* button (and one button per view) POSTs
to `/api/view3d`, which spawns `python -m web.view3d` as a **detached process**. A native GLFW
window cannot be embedded in a web page, so the browser only receives JSON
(`ok` / `pid` / `log` / `already_open`); Flask never blocks on the Open3D event loop. The
window offers hotkeys `1`–`4` (switch view), `r` (reset camera), `s` (PNG snapshot into
`reports/`), `h` (help) and `q`/`esc` (quit), and a `{view: Popen}` registry reports
`already_open` instead of opening a duplicate window. Window creation requires a desktop
session with an OpenGL driver on the machine running the server; Open3D's key-capable
`VisualizerWithKeyCallback` is used when available.

Results are cached for `SIIH_CACHE_TTL` seconds (default 30); `/?refresh=1` or
`/api/data?refresh=1` re-runs the pipeline. `/api/data` returns the metrics JSON that the
M5 metrics deliverable needs.

## 5. Suggested technology decisions (defaults)

| Component | Default Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Matches verified environment, fast iteration |
| DL Framework | PyTorch 2.0 + torchvision/torchsparse (or CPU-only RangeNet++), ONNX export | Accuracy/speed trade-off, portable |
| Grid Engine | NumPy radial-band arrays (bands 0–3) | Deterministic, O(1), vectorized, trivial to test/benchmark |
| Visualization | pyqtgraph | Lightweight, 60 fps 2D on Windows, pure Python |
| Ground Truth (for training) | SemanticKITTI (remapped to 3 meta-classes) | Public, same sensor format, enables accuracy claims |
| Build System | `pip` + `requirements.txt` | Simple, works on Windows/conda |

## 6. Milestones (definition of done)

**M1 — End-to-end demo without DL** (Week 1)
- [ ] Loader + preprocessing + heuristic labels (height/slope) → per-point pseudo-label.
- [ ] Variable-resolution grid engine (NumPy bands) with correct projection & attribute accumulation.
- [ ] Drivability layer from height variance + slope.
- [ ] Basic top-down visualization (pyqtgraph) showing map + FPS.
- [ ] Unit tests: point conservation, band seam, azimuth wrap.
- [ ] Binary size & latency report for grid engine only (<5 ms for 71k points).

**M2 — Grid-engine correctness & benchmarks** (Week 2)
- [ ] Sweep band parameters (cell sizes, ranges) → memory vs accuracy trade-off curve.
- [ ] Latency breakdown: confirm grid insertion + scatter < 2 ms on target CPU.
- [ ] Memory report: uniform 3D, uniform 2.5D, variable grid byte counts.
- [ ] Seam and wrap-around tests extended to synthetic curved obstacles.

**M3 — DL semantic segmentation integrated** (Week 3–4)
- [ ] Train/finetune SparseConv U-Net (or RangeNet++) on SemanticKITTI → ONNX export.
- [ ] Per-point inference latency benchmarked (GPU vs CPU).
- [ ] Pipeline: raw .bin → preprocess → infer → grid → visualization.
- [ ] Per-class IoU and range-binned accuracy logged.
- [ ] Confusion matrix visual (matplotlib) saved to report.

**M4 — Temporal filtering & dynamic-object layer** (Week 5)
- [ ] Log-odds occupancy update + EMA height/intensity.
- [ ] Dynamic cell flag from temporal change or learned moving class.
- [ ] Optional DBSCAN clustering of dynamic cells → track IDs.
- [ ] Demo: show moving car leaving fading trace while static map remains crisp.

**M5 — Dashboard polish, metrics, packaging** (Week 6)
- [ ] Telemetry panel: per-stage latency (load/infer/project/render) live.
- [ ] Band-boundary overlay toggle.
- [ ] Final metrics JSON: FPS, latency, memory savings, per-range IoU.
- [ ] README quick-start, architecture diagram, config reference.
- [ ] `requirements.txt` + optional Dockerfile.
- [ ] Deliverable: zip of code + sample data + benchmark report.
- [x] Interactive 3D viewer (`web/view3d.py` + `/api/view3d` button): native Open3D window
      with view hotkeys, camera reset and PNG snapshots (verified by
      `scripts/check_view3d_window.py`).

## 7. Risk register & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No GPU available for training/inference | Medium | High (speed) | CPU fallback: RangeNet++ (6 ms on i7); reduce inner bands to 5–20 m if needed. |
| No labelled data for semantic head | Low (we have SemanticKITTI) | Medium | Heuristic pseudo-labels for demo; fine-tune on SemanticKITTI (public). |
| Band seams cause ringing / aliasing | Low | Medium | Half-open intervals + unit tests; if severe, apply 1-cell linear blending at boundaries. |
| Real-time deadline missed on target hardware | Low | High | Profile early (M1); vectorize everything; allow configurable point decimation for far bands (>60 m keep every 4th point). |
| Class imbalance (drivable ≈ 80%) skews IoU | Medium | Medium | Lovász-softmax + class-weighted CE; report per-class IoU not just mean. |
| Visualization becomes bottleneck | Low | Medium | pyqtgraph is fast; if needed, downgrade to 10 FPS viz while pipeline runs full speed async. |
| No desktop/OpenGL session where the server runs | Low | Medium | Dashboard degrades gracefully: `/api/view3d` returns a clear error plus the log path, the four 3D PNG panels are skipped and `render3d_error` explains why; 2D panels and all metrics still render. |

## 8. Open questions for stakeholder alignment

1. **Exact range boundaries & cell sizes**: Problem statement gave example (5 cm @ 10 m, 50 cm @ 100 m). Should we adopt exact geometric progression or the four-band scheme above?
2. **Dynamic object definition**: Is per-scan moving-object segmentation sufficient, or do we need multi-scan tracking with IDs?
3. **Output format**: Besides the visualization, should we also publish the 2.5D grid as a ROS2 topic or binary file for downstream modules?
4. **Compute target**: Should we optimize for embedded Jetson-Orin class (≈5 TOPS) or assume a modest gaming laptop?

Once these are nailed, the plan above is ready for execution.

---
*End of ROADMAP.md*
    - Preprocess: 1.0 ms
    - DL inference (SparseConv): 15 ms (66 FPS) or RangeNet++: 6 ms on CPU
    - Grid projection: 2.0 ms
    - Render: 8.0 ms (pyqtgraph)
    **Total**: 26–35 ms → 30–38 FPS; grid insertion alone well under 5 ms target.
28. Accuracy:
    - Train on SemanticKITTI sequences 00-10, validate on 11.
    - Report overall mIoU and per-class IoU (drivable, static, dynamic).
    - Range-binned analysis: IoU in 0-10 m, 10-30 m, 30-60 m, 60-100 m to show foveation benefit (near-range classes should not degrade far-range).
29. Correctness test suite (pytest):
    - `test_point_conservation()`
    - `test_band_seam_invariance()`
    - `test_azimuth_wrap()`
    - `test_height_inject_curb()`: add synthetic points forming a curb, verify max-height layer detects it.
30. Single-command benchmark: `python scripts/run_benchmark.py --data data/fake_lidar_000000.bin --config configs/default.yaml --out reports/run_001.json`.
