"""Semantic segmentation interface and factory.

The roadmap specifies a deep learning model (SparseConv U-Net or RangeNet++)
with a heuristic fallback for bootstrapping. This module provides a unified
:class:`Segmenter` interface and a factory that returns the appropriate
implementation based on configuration.
"""
from __future__ import annotations

from . import CLASS_DRIVABLE, CLASS_STATIC, CLASS_DYNAMIC
from .config import Config
from .labels import HeuristicSegmenter, segment_heuristic
from .loader import PointCloud
from .preprocess import Preprocessed


class Segmenter:
    """Abstract per-point semantic segmenter interface."""

    def segment(self, pre: Preprocessed, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(labels, dynamic_flag)`` arrays of length ``pre.xyz.shape[0]``."""
        raise NotImplementedError


def get_segmenter(cfg: Config) -> Segmenter:
    """Factory returning a segmenter instance based on config.

    The config key ``model.backend`` selects the implementation:
        - "heuristic": always use the rule-based segmenter (default)
        - "rangenet": placeholder for a learned model (not implemented)

    Returns:
        A Segmenter instance.
    """
    backend = cfg.get("model.backend", "heuristic")
    if backend == "heuristic":
        return HeuristicSegmenter(cfg)
    elif backend == "rangenet":
        # Placeholder: in a full implementation we would load a TorchScript
        # or ONNX model here. For now we fall back to heuristic.
        return HeuristicSegmenter(cfg)
    else:
        raise ValueError(f"Unknown model backend: {backend}")