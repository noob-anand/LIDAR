#!/usr/bin/env python
"""Demo script for the variable-resolution 2.5D LiDAR mapping pipeline."""

import argparse
import sys
from pathlib import Path

import numpy as np

from mapping.config import load_config
from mapping.pipeline import LiDARMapper


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the adaptive variable-resolution 2.5D LiDAR mapping pipeline"
    )
    parser.add_argument(
        "--pc",
        type=str,
        default="data/fake_lidar_000000.bin",
        help="Path to the point cloud file (KITTI .bin format)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization (useful for benchmarking)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset temporal state before processing",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Override viz backend if disabled
    if args.no_viz:
        cfg.set("viz.backend", "none")

    # Create mapper
    mapper = LiDARMapper(cfg)
    if args.reset:
        mapper.reset()

    # Process frame
    print(f"Processing {args.pc}...")
    result = mapper.process_file(args.pc)
    print(f"Finished in {result.total_time:.3f} s")
    print(f"  Points: {len(result.labels)}")
    print(f"  Grid cells occupied: {result.grid.occupancy.sum()}")
    if result.temporal_state is not None:
        print(f"  Dynamic cells (temporal): {result.temporal_state.dynamic.sum()}")
    else:
        print("  Temporal integration disabled")

    # If visualization is enabled, the pipeline already showed the plot.
    # For matplotlib non-blocking, we need to keep the script alive.
    if not args.no_viz:
        if cfg.get("viz.backend", "matplotlib") == "matplotlib":
            # Keep the window open until user closes it
            import matplotlib.pyplot as plt
            plt.show(block=True)
        else:  # pyqtgraph
            # Start the Qt event loop
            import pyqtgraph as pg
            pg.exec()

    mapper.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())