#!/usr/bin/env python
"""Benchmark the LiDAR mapping pipeline."""

import time
import statistics
from pathlib import Path

import numpy as np

from mapping.config import load_config
from mapping.pipeline import LiDARMapper


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark LiDAR mapping pipeline")
    parser.add_argument(
        "--pc",
        type=str,
        default="data/fake_lidar_000000.bin",
        help="Path to point cloud file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of times to run the pipeline (excluding first warm-up)",
    )
    parser.add_argument(
        "--warm-up",
        type=int,
        default=10,
        help="Number of warm-up runs",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_viz:
        cfg.set("viz.backend", "none")

    mapper = LiDARMapper(cfg)

    # Warm-up
    print(f"Warming up with {args.warm_up} runs...")
    for i in range(args.warm_up):
        mapper.process_file(args.pc)
        if i == 0:
            mapper.reset()  # after first run, keep temporal state for subsequent runs

    # Timed runs
    print(f"Running {args.num_runs} iterations...")
    times = []
    for i in range(args.num_runs):
        start = time.perf_counter()
        result = mapper.process_file(args.pc)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if i % 10 == 0:
            print(f"  Iteration {i}: {elapsed:.3f} s")

    # Statistics
    times_arr = np.array(times)
    print("\n=== Benchmark Results ===")
    print(f"Runs: {len(times)}")
    print(f"Mean: {times_arr.mean()*1000:.2f} ms")
    print(f"Median: {np.median(times_arr)*1000:.2f} ms")
    print(f"Std: {times_arr.std()*1000:.2f} ms")
    print(f"Min: {times_arr.min()*1000:.2f} ms")
    print(f"Max: {times_arr.max()*1000:.2f} ms")
    print(f"FPS: {1.0/times_arr.mean():.2f}")

    mapper.close()


if __name__ == "__main__":
    main()