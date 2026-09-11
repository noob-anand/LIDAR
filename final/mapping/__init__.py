"""Adaptive Variable-Resolution 2.5D LiDAR Mapping.

A deep-learning-informed pipeline that converts raw LiDAR point clouds into a
foveated (range-dependent resolution) 2.5D elevation map with semantic layers:
drivable terrain, static obstacles and dynamic objects.
"""
from __future__ import annotations

__version__ = "1.0.0"

# Semantic class ids (see config/classes.names)
CLASS_DRIVABLE = 0
CLASS_STATIC = 1
CLASS_DYNAMIC = 2

# RGB semantics (0-255)
SEMANTIC_COLOR = {
    CLASS_DRIVABLE: (0, 200, 0),
    CLASS_STATIC: (100, 100, 100),
    CLASS_DYNAMIC: (255, 140, 0),
    -1: (50, 50, 50),  # unknown
}

# Public submodules and classes
from .config import Config
from .loader import PointCloud, load_bin
from .preprocess import preprocess, Preprocessed
from .labels import HeuristicSegmenter, segment_heuristic
from .segmenter import Segmenter, get_segmenter
from .grid_engine import VariableGrid, build_grid
from .temporal import TemporalState, init_temporal_state, update_temporal
from .viz import Visualizer
from .pipeline import LiDARMapper, FrameResult

__all__ = [
    "__version__",
    "CLASS_DRIVABLE",
    "CLASS_STATIC",
    "CLASS_DYNAMIC",
    "SEMANTIC_COLOR",
    "Config",
    "PointCloud",
    "load_bin",
    "Preprocessed",
    "preprocess",
    "HeuristicSegmenter",
    "segment_heuristic",
    "Segmenter",
    "get_segmenter",
    "VariableGrid",
    "build_grid",
    "TemporalState",
    "init_temporal_state",
    "update_temporal",
    "Visualizer",
    "LiDARMapper",
    "FrameResult",
]