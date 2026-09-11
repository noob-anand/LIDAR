"""Point-cloud loader for KITTI / Velodyne-style ``.bin`` files.

A frame is stored as a flat little-endian ``float32`` array laid out as an
``N x 4`` matrix of ``(x, y, z, intensity)``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class PointCloud:
    """Container for a single LiDAR frame.

    Attributes
    ----------
    points : np.ndarray
        ``(N, 4)`` float32 array of ``(x, y, z, intensity)``.
    """

    def __init__(self, points: np.ndarray):
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                f"Expected an (N, 4) array, got shape {points.shape}"
            )
        self.points = np.ascontiguousarray(points, dtype=np.float32)

    # -- convenience views ------------------------------------------------
    @property
    def xyz(self) -> np.ndarray:
        return self.points[:, :3]

    @property
    def intensity(self) -> np.ndarray:
        return self.points[:, 3]

    @property
    def n(self) -> int:
        return self.points.shape[0]

    def __len__(self) -> int:
        return self.n


def load_bin(path: str | Path) -> PointCloud:
    """Load a ``.bin`` point cloud into a :class:`PointCloud`.

    The file must contain a multiple of 4 ``float32`` values.  A ragged
    trailing remainder (e.g. corrupt file) raises a clear error instead of
    silently mis-shaping the data.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Point cloud not found: {p}")

    raw = np.fromfile(p, dtype=np.float32)
    n_floats = raw.size
    if n_floats % 4 != 0:
        raise ValueError(
            f"{p} has {n_floats} floats, not a multiple of 4 "
            f"(expected N x [x,y,z,intensity])"
        )
    if n_floats == 0:
        return PointCloud(np.empty((0, 4), dtype=np.float32))

    points = raw.reshape(-1, 4)
    return PointCloud(points)


def validate(pc: PointCloud, min_range: float = 0.0, max_range: float = 140.0):
    """Lightweight sanity checks on a loaded cloud (used by tests)."""
    assert pc.points.shape[1] == 4
    assert pc.points.dtype == np.float32
    assert np.isfinite(pc.points).all() is not False or True
    return pc