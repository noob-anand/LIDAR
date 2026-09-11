"""Preprocessing: range/height filtering, polar features, ground segmentation.

The output is the *foundation* for both the DL head (richer input) and the
heuristic pseudo-labels (fallback labels) described in the roadmap Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Preprocessed:
    """Filtered frame plus derived features and a fitted ground plane."""

    xyz: np.ndarray          # (M, 3) float32 filtered points
    intensity: np.ndarray    # (M,) float32
    r: np.ndarray            # (M,) radial distance [m]
    theta: np.ndarray        # (M,) azimuth in [-pi, pi)
    elevation: np.ndarray    # (M,) elevation angle [rad]
    ground_plane: tuple[np.ndarray, float]   # (normal (3,), offset d) n.x + d = 0
    ground_inliers: np.ndarray  # (M,) bool within RANSAC distance threshold
    src_index: np.ndarray    # (M,) index into the original cloud


def filter_points(
    xyz: np.ndarray,
    min_range: float,
    max_range: float,
    height_min: float,
    height_max: float,
    decimation: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Range/height clip + NaN/inf purge (+ optional far-range decimation)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=1)

    r2 = xyz[:, 0] * xyz[:, 0] + xyz[:, 1] * xyz[:, 1]
    r = np.sqrt(r2)
    in_range = (r >= min_range) & (r <= max_range)
    in_height = (xyz[:, 2] >= height_min) & (xyz[:, 2] <= height_max)

    keep = finite & in_range & in_height

    if decimation and decimation.get("enabled", False):
        beyond = float(decimation.get("beyond", 60.0))
        step = int(decimation.get("keep_every", 1))
        if step > 1:
            far = r > beyond
            far_positions = np.nonzero(far)[0]
            if far_positions.size:
                within = far_positions - far_positions[0]
                decimate_keep = np.zeros_like(far, dtype=bool)
                decimate_keep[far_positions[within % step == 0]] = True
                keep = keep & (~far | decimate_keep)

    idx = np.nonzero(keep)[0]
    return xyz[idx], idx


def compute_polar(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(r, theta, elevation)`` for an ``(M, 3)`` array."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    elevation = np.arctan2(z, r)
    return r, theta, elevation


def _plane_from_points(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a plane (normal, offset) to >=3 points via SVD."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def ransac_ground(
    xyz: np.ndarray,
    max_iterations: int,
    distance_threshold: float,
    candidate_ratio: float,
    min_inliers: int,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit the dominant ground plane using RANSAC.

    Returns ``(normal, offset, inlier_mask)``.  Falls back to a horizontal
    plane ``z = 0`` on failure (documented edge case: all-ground / empty scan).
    """
    n = xyz.shape[0]
    if n < 3:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return normal, 0.0, np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    n_cand = max(3, int(np.ceil(n * candidate_ratio)))
    order = np.argsort(xyz[:, 2])
    cand_idx = order[:n_cand]

    best_inliers = None
    best_count = -1
    best_plane = (np.array([0.0, 0.0, 1.0]), 0.0)

    for _ in range(max_iterations):
        sample = cand_idx[rng.choice(n_cand, size=3, replace=False)]
        pts = xyz[sample]
        if np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
            continue
        normal, offset = _plane_from_points(pts)
        d = np.abs(xyz @ normal + offset)
        inliers = d < distance_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_plane = (normal, offset)

    if best_inliers is None or best_count < min_inliers:
        normal = np.array([0.0, 0.0, 1.0])
        offset = 0.0
        best_inliers = np.abs(xyz[:, 2]) < distance_threshold
        return normal, offset, best_inliers

    return best_plane[0], best_plane[1], best_inliers


def preprocess(
    xyz: np.ndarray,
    intensity: np.ndarray,
    cfg: Config,
) -> Preprocessed:
    """Full preprocessing pipeline driven by config."""
    pre = cfg.section("preprocess")
    xyz_f, src_idx = filter_points(
        xyz,
        float(pre.get("min_range", 1.0)),
        float(pre.get("max_range", 100.0)),
        float(pre.get("height_min", -5.0)),
        float(pre.get("height_max", 5.0)),
        pre.get("decimation"),
    )
    int_f = np.asarray(intensity, dtype=np.float32)[src_idx]

    r, theta, elevation = compute_polar(xyz_f)

    ground = cfg.section("ground")
    normal, offset, inliers = ransac_ground(
        xyz_f,
        int(ground.get("max_iterations", 100)),
        float(ground.get("distance_threshold", 0.18)),
        float(ground.get("candidate_ratio", 0.05)),
        int(ground.get("min_inliers", 500)),
    )

    return Preprocessed(
        xyz=xyz_f,
        intensity=int_f,
        r=r,
        theta=theta,
        elevation=elevation,
        ground_plane=(normal, offset),
        ground_inliers=inliers,
        src_index=src_idx,
    )