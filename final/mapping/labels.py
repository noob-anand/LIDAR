"""Feature-based semantic labelling (heuristic pseudo-label bootstrap).

The roadmap (Phase 1, item 6) explicitly builds a heuristic labeler from
*height + slope + intensity* so the whole pipeline can be smoke-tested end to
end before a learned DL head is available.  This module implements that
deterministic classifier and exposes the shared :class:`Segmenter` interface
used by the (optional) DL backend in :mod:`mapping.rangeseg`.

Class ids (see config ``classes.names``):
    0 = drivable terrain
    1 = static obstacle
    2 = dynamic object  (set temporally in :mod:`mapping.temporal`)
"""
from __future__ import annotations

import numpy as np

from . import CLASS_DRIVABLE, CLASS_STATIC, CLASS_DYNAMIC
from .config import Config
from .preprocess import Preprocessed


class Segmenter:
    """Abstract per-point semantic segmenter interface."""

    def segment(self, pre: Preprocessed, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(labels, dynamic_flag)`` arrays of length ``pre.xyz.shape[0]``."""
        raise NotImplementedError


class HeuristicSegmenter(Segmenter):
    """Deterministic height/slope/intensity rule-based segmenter.

    - A point within ``ground_tolerance`` of the fitted ground plane is
      *drivable terrain* (class 0).
    - A point above the ground plane is a *static obstacle* (class 1).
    - Points with near-zero intensity are treated as *unknown* noise and are
      passed through with a low-confidence marker (they still occupy a cell).
    """

    def __init__(self, cfg: Config):
        lab = cfg.section("labels")
        self.ground_tol = float(lab.get("ground_tolerance", 0.30))
        self.object_min = float(lab.get("object_min_height", 0.30))
        self.use_intensity = bool(lab.get("use_intensity", True))
        self.intensity_floor = float(lab.get("intensity_floor", 0.02))

    def segment(self, pre: Preprocessed, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
        n = pre.xyz.shape[0]
        labels = np.full(n, CLASS_DRIVABLE, dtype=np.int64)
        dynamic = np.zeros(n, dtype=bool)

        # signed distance above ground plane (positive == above ground)
        normal, offset = pre.ground_plane
        z_above = pre.xyz @ normal + offset  # n·x + d ; at plane this is 0

        # ground / drivable terrain
        on_ground = z_above <= self.ground_tol
        labels[~on_ground] = CLASS_STATIC

        # near-zero intensity -> unknown (kept but flagged via intensity)
        if self.use_intensity:
            dark = pre.intensity < self.intensity_floor
            labels[dark] = CLASS_STATIC  # still occupies space; treated static

        # dynamic objects are not identifiable from a single static scan;
        # the temporal stage flips cells to CLASS_DYNAMIC when motion appears.
        return labels, dynamic


def segment_heuristic(pre: Preprocessed, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning ``(labels, dynamic_flag)``."""
    return HeuristicSegmenter(cfg).segment(pre, cfg)