"""Temporal integration for dynamic environments.

Implements log-odds occupancy update, exponential moving average for height/intensity,
and dynamic-cell flagging based on temporal change (height/intensity deviation) or
learned moving class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CLASS_DYNAMIC
from .config import Config


@dataclass
class TemporalState:
    """Per-cell temporal state."""
    log_odds: np.ndarray          # log-odds of occupancy
    height_ema: np.ndarray        # exponential moving average of height
    intensity_ema: np.ndarray     # exponential moving average of intensity
    class_prob_ema: np.ndarray    # EMA of class probabilities (n_classes,)
    dynamic: np.ndarray           # boolean flag: True if cell is dynamic
    last_update: np.ndarray       # frame index of last update (for missing updates)


def init_temporal_state(num_cells: int, n_classes: int = 3, cfg: Config | None = None) -> TemporalState:
    """Initialize temporal state for a grid with ``num_cells`` cells."""
    temp = cfg.section("temporal") if cfg else None
    log_odds_init = float(temp.get("log_odds_init", 0.0)) if temp else 0.0
    height_init = float(temp.get("height_init", 0.0)) if temp else 0.0
    intensity_init = float(temp.get("intensity_init", 0.0)) if temp else 0.0
    class_prob_init = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
    return TemporalState(
        log_odds=np.full(num_cells, log_odds_init, dtype=np.float32),
        height_ema=np.full(num_cells, height_init, dtype=np.float32),
        intensity_ema=np.full(num_cells, intensity_init, dtype=np.float32),
        class_prob_ema=np.tile(class_prob_init, (num_cells, 1)),
        dynamic=np.zeros(num_cells, dtype=bool),
        last_update=np.full(num_cells, -1, dtype=np.int64),
    )


def update_temporal(
    state: TemporalState,
    occupancy: np.ndarray,
    height: np.ndarray,
    intensity: np.ndarray,
    class_hist: np.ndarray,  # (num_cells, n_classes) integer counts
    frame_idx: int,
    cfg: Config,
) -> TemporalState:
    """Update temporal state with new observations.

    Args:
        state: current temporal state (modified in-place and also returned).
        occupancy: boolean array of which cells were hit in this frame.
        height: mean height per cell (from grid) for hits; ignored for misses.
        intensity: mean intensity per cell (from grid) for hits; ignored for misses.
        class_hist: histogram of class counts per cell (from grid) for this frame.
        frame_idx: current frame index.
        cfg: configuration.

    Returns:
        Updated TemporalState (same object as input).
    """
    temp = cfg.section("temporal")
    alpha = float(temp.get("log_odds_hit", 0.85))
    beta = float(temp.get("log_odds_miss", 0.40))
    gamma = float(temp.get("ema_gamma", 0.7))
    clamp_min = float(temp.get("log_odds_min", -4.0))
    clamp_max = float(temp.get("log_odds_max", 4.0))
    height_delta_thresh = float(temp.get("dynamic_height_delta", 0.5))
    dynamic_class_thresh = float(temp.get("dynamic_class_thresh", 0.6))

    # --- log-odds occupancy update ---
    hit = occupancy
    miss = ~occupancy
    state.log_odds[hit] += alpha
    state.log_odds[miss] -= beta
    np.clip(state.log_odds, clamp_min, clamp_max, out=state.log_odds)

    # --- exponential moving average for height and intensity (only update on hits) ---
    if np.any(hit):
        state.height_ema[hit] = gamma * height[hit] + (1 - gamma) * state.height_ema[hit]
        state.intensity_ema[hit] = gamma * intensity[hit] + (1 - gamma) * state.intensity_ema[hit]

    # --- class probability EMA (normalize histogram to get probabilities) ---
    eps = 1e-6
    total = class_hist.sum(axis=1, keepdims=True) + eps
    class_prob = class_hist / total  # (num_cells, n_classes)
    state.class_prob_ema = gamma * class_prob + (1 - gamma) * state.class_prob_ema
    # Renormalize to ensure sum to 1 (optional, but keeps it valid)
    state.class_prob_ema /= state.class_prob_ema.sum(axis=1, keepdims=True)

    # --- dynamic flag computation ---
    # Reset dynamic flag for all cells (we will recompute for hit cells)
    state.dynamic.fill(False)
    if np.any(hit):
        # Height-based dynamic: significant deviation from EMA
        height_diff = np.abs(height[hit] - state.height_ema[hit])
        dynamic_by_height = height_diff > height_delta_thresh
        # Class-based dynamic: probability of dynamic class above threshold
        dynamic_by_class = state.class_prob_ema[hit, CLASS_DYNAMIC] > dynamic_class_thresh
        # Combine: cell is dynamic if either condition is met
        state.dynamic[hit] = dynamic_by_height | dynamic_by_class
    # For miss cells, we leave the dynamic flag as False (so dynamic cells must be hit each frame to be considered dynamic).
    # Alternatively, we could store the dynamic flag and let it decay with a timeout.
    # We'll leave it as is for now.

    # --- update last_update timestamp ---
    state.last_update[occupancy] = frame_idx

    return state