"""Temporary helper: dump camera candidates to PNG so they can be eyeballed.

It goes through :func:`web.render3d.scene_clouds`, so the probe frames exactly
the geometry the dashboard panels and the interactive window (:mod:`web.view3d`)
show.
"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapping import load_bin  # noqa: E402
from mapping.config import load_config  # noqa: E402
from mapping.grid_engine import build_grid  # noqa: E402
from mapping.labels import segment_heuristic  # noqa: E402
from mapping.preprocess import preprocess  # noqa: E402
from web.metrics import (build_uniform_grid, density_per_cell,  # noqa: E402
                         drivability_tiers, uniform_drivability_tiers)
from web.render3d import SceneRenderer, scene_clouds  # noqa: E402

cfg = load_config("configs/default.yaml")
cfg.set("viz.backend", "none")
pc = load_bin("data/fake_lidar_000000.bin")
pre = preprocess(pc.xyz, pc.intensity, cfg)
lab, dyn = segment_heuristic(pre, cfg)
grid = build_grid(pre.r, pre.theta, pre.xyz, pre.intensity, lab, dyn,
                  cfg.get("bands"), 3)

n_classes = len(cfg.get("classes.names", []) or []) or 3
tier, _, _ = drivability_tiers(grid, cfg)
density = density_per_cell(grid)
uniform = build_uniform_grid(pre.xyz, lab, n_classes,
                             float(cfg.get("metrics.uniform_cell_size", 0.05)))
u_tier, _, _ = uniform_drivability_tiers(uniform, cfg)
uniform["tier"] = u_tier

# one shared code path with the dashboard panels and the interactive window
clouds = scene_clouds(grid, uniform, tier, density, cfg, n_classes)
for name, (pts, _) in clouds.items():
    print(f"{name:<18} {pts.shape[0]:>8,} points")

s_pts, s_cols = clouds["adaptive_semantic"]
d_pts, d_cols = clouds["adaptive_density"]

out = Path("reports")
out.mkdir(parents=True, exist_ok=True)

cams = [(30.0, -110.0, 0.40), (30.0, -110.0, 0.30), (30.0, -110.0, 0.24),
        (45.0, -110.0, 0.30), (20.0, -110.0, 0.30)]
with SceneRenderer(cfg) as scene:
    for e, a, z in cams:
        b64 = scene.render(s_pts, s_cols, elev_deg=e, azim_deg=a, zoom=z)
        (out / f"_cam3_sem_e{e:.0f}_a{a:.0f}_z{z:.2f}.png").write_bytes(
            base64.b64decode(b64))
    b64 = scene.render(d_pts, d_cols, elev_deg=30.0, azim_deg=-110.0, zoom=0.30)
    (out / "_cam3_dens_e30_z0.30.png").write_bytes(base64.b64decode(b64))
print("wrote", len(cams) + 1, "probe images to", out)
