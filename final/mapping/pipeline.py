"""LiDAR mapping pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import CLASS_DRIVABLE, CLASS_STATIC, CLASS_DYNAMIC
from .config import Config
from .grid_engine import VariableGrid, build_grid
from .labels import segment_heuristic
from .loader import PointCloud, load_bin
from .preprocess import Preprocessed, preprocess
from .segmenter import Segmenter, get_segmenter
from .temporal import TemporalState, init_temporal_state, update_temporal
from .viz import Visualizer


@dataclass
class FrameResult:
    grid: VariableGrid
    labels: np.ndarray
    dynamic_flag: np.ndarray
    temporal_state: TemporalState
    preprocessing_time: float
    segmentation_time: float
    projection_time: float
    temporal_time: float
    total_time: float


class LiDARMapper:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config.from_yaml()
        self.segmenter: Segmenter = get_segmenter(self.cfg)
        self.temp_state: Optional[TemporalState] = None
        self.frame_idx = 0
        self.viz: Optional[Visualizer] = None
        if self.cfg.get("viz.backend", "matplotlib") != "none":
            self.viz = Visualizer(self.cfg)

    def reset(self):
        self.temp_state = None
        self.frame_idx = 0
        if self.viz:
            self.viz.close()
            self.viz = Visualizer(self.cfg)

    def process_frame(self, points: PointCloud) -> FrameResult:
        t0 = time.perf_counter()
        pre = preprocess(points.xyz, points.intensity, self.cfg)
        t1 = time.perf_counter()
        prep = t1 - t0

        t0 = time.perf_counter()
        labels, dyn = self.segmenter.segment(pre, self.cfg)
        t1 = time.perf_counter()
        seg = t1 - t0

        t0 = time.perf_counter()
        bands = self.cfg.get("bands", [])
        grid = build_grid(pre.r, pre.theta, pre.xyz, pre.intensity, labels, dyn, bands, 3)
        t1 = time.perf_counter()
        proj = t1 - t0

        t0 = time.perf_counter()
        if self.cfg.get("temporal.enabled", True):
            if self.temp_state is None:
                self.temp_state = init_temporal_state(grid.total_cells, 3, self.cfg)
            occ = grid.occupancy
            h = grid.mean_height
            i = grid.max_intensity
            ch = grid.class_hist
            self.temp_state = update_temporal(self.temp_state, occ, h, i, ch, self.frame_idx, self.cfg)
        else:
            self.temp_state = None
        t1 = time.perf_counter()
        temp = t1 - t0

        if self.viz:
            tel = {
                "FPS": 0.0,
                "Load": 0.0,
                "Prep": prep * 1000,
                "Seg": seg * 1000,
                "Proj": proj * 1000,
                "Temp": temp * 1000,
                "Points": len(points),
                "Cells": int(grid.occupancy.sum()),
                "DynamicCells": int(self.temp_state.dynamic.sum()) if self.temp_state else 0,
            }
            self.viz.update(grid, tel)

        self.frame_idx += 1
        total = time.perf_counter() - t0
        return FrameResult(grid, labels, dyn, self.temp_state, prep, seg, proj, temp, total)

    def process_file(self, path: Optional[str] = None) -> FrameResult:
        if path is None:
            path = self.cfg.get("data.path", "data/fake_lidar_000000.bin")
        pc = load_bin(path)
        return self.process_frame(pc)

    def close(self):
        if self.viz:
            self.viz.close()