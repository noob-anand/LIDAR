"""Temporary helper: sweep the Open3D camera to pick good dashboard defaults."""
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.image as mpimg  # noqa: E402

from mapping import load_bin  # noqa: E402
from mapping.config import load_config  # noqa: E402
from mapping.grid_engine import build_grid  # noqa: E402
from mapping.labels import segment_heuristic  # noqa: E402
from mapping.preprocess import preprocess  # noqa: E402
from web.metrics import density_per_cell, drivability_tiers  # noqa: E402
from web.render3d import SceneRenderer, semantic_cloud  # noqa: E402

cfg = load_config("configs/default.yaml")
cfg.set("viz.backend", "none")
pc = load_bin("data/fake_lidar_000000.bin")
pre = preprocess(pc.xyz, pc.intensity, cfg)
lab, dyn = segment_heuristic(pre, cfg)
grid = build_grid(pre.r, pre.theta, pre.xyz, pre.intensity, lab, dyn,
                  cfg.get("bands"), 3)

occ = grid.occupancy
x, y, _ = grid.occupied_centers()
cls = grid.pred_class[occ]
tier, _, _ = drivability_tiers(grid, cfg)
tier = tier[occ]
pts, cols = semantic_cloud(x, y, cls, grid.mean_height[occ], grid.max_z[occ],
                           tier, cfg, 3)
print("cloud points", pts.shape)

bg = np.array(cfg.get("render3d.background", [0.043, 0.051, 0.063])) * 255.0

with SceneRenderer(cfg) as scene:
    for zoom in (0.50, 0.40, 0.32, 0.25, 0.20, 0.15):
        b64 = scene.render(pts, cols, zoom=zoom)
        img = mpimg.imread(io.BytesIO(__import__("base64").b64decode(b64)))
        rgb = img[..., :3] * 255.0
        nonbg = float((np.abs(rgb - bg).max(axis=2) > 6.0).mean())
        uniq = np.unique((rgb).astype(int).reshape(-1, 3), axis=0).shape[0]
        print(f"zoom={zoom:<5} coverage={nonbg * 100:6.2f}%  distinct_colours={uniq}")
