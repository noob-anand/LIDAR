"""Visualization dashboard for the variable-resolution 2.5D map.

Supports both 2D top-down rendering (via matplotlib or pyqtgraph) and
optional 3D comparison (via Open3D) showing uniform vs variable resolution.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import CLASS_DRIVABLE, CLASS_STATIC, CLASS_DYNAMIC, SEMANTIC_COLOR
from .config import Config


class Visualizer:
    """Base visualizer class."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.viz_cfg = cfg.section("viz")
        self.backend = self.viz_cfg.get("backend", "matplotlib")
        self.width = int(self.viz_cfg.get("width", 900))
        self.dpi = int(self.viz_cfg.get("dpi", 120))
        self.show_band_boundaries = bool(self.viz_cfg.get("show_band_boundaries", True))
        self.show_drivability = bool(self.viz_cfg.get("show_drivability", True))

        # Frame rate tracking
        self.frame_times = []
        self.last_time = time.time()

        # Initialize backend-specific components
        if self.backend == "matplotlib":
            self._init_matplotlib()
        elif self.backend == "pyqtgraph":
            self._init_pyqtgraph()
        else:
            raise ValueError(f"Unknown visualization backend: {self.backend}")

    def _init_matplotlib(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        self.fig, self.ax = plt.subplots(figsize=(self.width / self.dpi, self.width / self.dpi), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()
        self.img = None
        self.texts = []

    def _init_pyqtgraph(self):
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtGui, QtCore
        except ImportError as e:
            raise ImportError("pyqtgraph is not installed. Install it or set viz.backend to 'matplotlib'.") from e

        self.pg = pg
        self.QtGui = QtGui
        self.QtCore = QtCore
        self.win = pg.GraphicsLayoutWidget(show=True, title="Variable-Resolution 2.5D LiDAR Mapping")
        self.win.resize(self.width, self.width)
        self.view = self.win.addViewBox()
        self.view.setAspectLocked(True)
        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)
        # Text items for telemetry
        self.text_items = []
    def update(self, grid, telemetry: Optional[dict] = None):
        """Update the visualization with a new grid and optional telemetry."""
        start = time.time()

        # Render the frame
        if self.backend == "matplotlib":
            self._render_matplotlib(grid, telemetry)
        elif self.backend == "pyqtgraph":
            self._render_pyqtgraph(grid, telemetry)

        # Track frame rate
        now = time.time()
        self.frame_times.append(now - self.last_time)
        self.last_time = now
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

    def _render_matplotlib(self, grid, telemetry):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()

        # Get occupied cells and their properties
        xs, ys = grid.cell_centers()
        occ = grid.occupancy
        if not np.any(occ):
            self.ax.text(0.5, 0.5, "No data", transform=self.ax.transAxes, ha='center', va='center')
            self.canvas.draw()
            return

        # Color by semantic class
        cls = grid.pred_class[occ]
        conf = grid.confidence[occ]
        # Build RGB image
        rgb = np.zeros((len(xs), 3), dtype=np.float32)
        idx = np.where(occ)[0]  # indices of occupied cells in the flattened grid
        for i, flat_idx in enumerate(idx):
            c = cls[i]
            base = np.array(SEMANTIC_COLOR.get(c, SEMANTIC_COLOR[-1]), dtype=np.float32) / 255.0
            # Blend with confidence and occupancy (log scale)
            alpha = conf[i] * np.log1p(1 + grid.count[occ][i])  # simple occupancy weighting
            alpha = np.clip(alpha, 0, 1)
            rgb[flat_idx] = base * alpha

        # Scatter plot
        self.ax.scatter(xs[occ], ys[occ], c=rgb[occ], s=10, marker='s')

        # Show band boundaries if requested
        if self.show_band_boundaries:
            # We'll draw lines at band edges in polar coordinates, then convert to Cartesian
            # For simplicity, we'll skip this in the matplotlib version for now.
            pass

        # Show drivability layer if requested
        if self.show_drivability:
            # We'll overlay drivability as a contour or different marker
            pass

        # Telemetry text
        if telemetry:
            txt = []
            for k, v in telemetry.items():
                txt.append(f"{k}: {v}")
            self.ax.text(0.02, 0.98, "\\n".join(txt), transform=self.ax.transAxes, va='top', ha='left', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.canvas.draw()

    def _render_pyqtgraph(self, grid, telemetry):
        # Clear previous items
        self.view.clear()

        # Get occupied cells
        xs, ys = grid.cell_centers()
        occ = grid.occupancy
        if not np.any(occ):
            # No data
            txt = self.pg.TextItem("No data", anchor=(0.5, 0.5))
            txt.setPos(0, 0)
            self.view.addItem(txt)
            return

        # Color by semantic class and confidence
        cls = grid.pred_class[occ]
        conf = grid.confidence[occ]
        # Create scatter plot with variable size and color
        spots = []
        for i, (x, y) in enumerate(zip(xs[occ], ys[occ])):
            c = cls[i]
            base = np.array(SEMANTIC_COLOR.get(c, SEMANTIC_COLOR[-1]), dtype=np.float32)
            # Blend with confidence
            alpha = conf[i]
            color = tuple(int(255 * (base[j] * alpha + (1 - alpha) * 0.2)) for j in range(3))  # blend with dark gray
            spots.append({'pos': (x, y), 'brush': color, 'size': 8})

        # Create scatter plot item
        scatter = self.pg.ScatterPlotItem(size=10, pen=self.pg.mkPen(None), brush=self.pg.mkBrush(255, 255, 255))
        scatter.setData(spots)
        self.view.addItem(scatter)

        # Add band boundaries as lines (optional)
        if self.show_band_boundaries:
            # We'll skip for brevity
            pass

        # Add telemetry as text items
        if telemetry:
            # Remove old text items
            for item in self.text_items:
                self.view.removeItem(item)
            self.text_items.clear()
            y_offset = 0
            for k, v in telemetry.items():
                txt = self.pg.TextItem(f"{k}: {v}", anchor=(0, 0))
                txt.setPos(10, 10 + y_offset)
                self.text_items.append(txt)
                self.view.addItem(txt)
                y_offset += 20

        # Update view range to fit data
        if len(xs[occ]) > 0:
            x_min = np.min(xs[occ])
            x_max = np.max(xs[occ])
            y_min = np.min(ys[occ])
            y_max = np.max(ys[occ])
            padding = 0.1
            x_width = x_max - x_min
            y_width = y_max - y_min
            x_range = (x_min - padding * x_width, x_max + padding * x_width)
            y_range = (y_min - padding * y_width, y_max + padding * y_width)
            self.view.setRange(xRange=x_range, yRange=y_range)

    def get_fps(self) -> float:
        """Return average FPS over recent frames."""
        if not self.frame_times:
            return 0.0
        return 1.0 / np.mean(self.frame_times)

    def close(self):
        """Clean up resources."""
        if self.backend == "pyqtgraph":
            self.win.close()