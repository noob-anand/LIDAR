"""Variable-resolution (foveated) 2.5D grid engine — the core novelty.

Projects classified 3D points into a polar, range-dependent-resolution grid
where cell size grows with distance from the sensor:

    Band 1  r in [ 0, 10) m  ->  5.0 cm cells
    Band 2  r in [10, 30) m  -> 12.5 cm cells
    Band 3  r in [30, 60) m  -> 25.0 cm cells
    Band 4  r in [60,100] m  -> 50.0 cm cells

Per-cell 2.5D attributes (accumulated via vectorized ``np.bincount`` /
``np.add.at``): ``count``, ``mean_height``, ``height_variance``,
``max_height``, ``max_intensity``, semantic class histogram -> ``pred_class`` +
``confidence``, and a ``dynamic`` flag.

Exact projection guarantees each classified point contributes to *exactly one*
cell — no double counting and no drop within the configured range.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Band:
    """A single radial band of uniform cell size."""

    id: int
    rmin: float
    rmax: float
    cell: float
    n_r: int
    n_theta: int
    offset: int  # first global linear id of this band
    n_classes: int = 3

    @property
    def r_mean(self) -> float:
        return 0.5 * (self.rmin + self.rmax)

    @property
    def dtheta(self) -> float:
        # keeps angular cell width ~ radial cell size at the mean radius
        return self.cell / self.r_mean

    @property
    def num_cells(self) -> int:
        return self.n_r * self.n_theta

    @classmethod
    def build(cls, band_id: int, rmin: float, rmax: float, cell: float,
              offset: int, n_classes: int = 3) -> "Band":
        n_r = int(np.ceil((rmax - rmin) / cell))
        r_mean = 0.5 * (rmin + rmax)
        dtheta = cell / r_mean
        n_theta = int(np.floor(2.0 * np.pi / dtheta))
        return cls(id=band_id, rmin=rmin, rmax=rmax, cell=cell,
                   n_r=n_r, n_theta=n_theta, offset=offset,
                   n_classes=n_classes)


def build_bands(band_specs: list[dict[str, float]],
                n_classes: int = 3) -> list[Band]:
    """Construct band objects + global offsets from a list of specs."""
    bands: list[Band] = []
    offset = 0
    for i, spec in enumerate(band_specs):
        b = Band.build(band_id=i,
                       rmin=float(spec["rmin"]),
                       rmax=float(spec["rmax"]),
                       cell=float(spec["cell"]),
                       offset=offset,
                       n_classes=n_classes)
        bands.append(b)
        offset += b.num_cells
    return bands


def assign_cells(r: np.ndarray, theta: np.ndarray, bands: list[Band]):
    """Map every point to its ``(band, row, col, global id)`` using half-open
    intervals ``[rmin, rmax)``.

    Points outside every band (``r >= max_range`` / ``r < first rmin``) get id
    ``-1`` and are dropped by the caller.
    """
    n = r.size
    flat = np.full(n, -1, dtype=np.int64)
    band_id = np.full(n, -1, dtype=np.int64)
    rows = np.full(n, -1, dtype=np.int64)
    cols = np.full(n, -1, dtype=np.int64)

    for b in bands:
        mask = (r >= b.rmin) & (r < b.rmax)
        if not np.any(mask):
            continue
        ii = np.nonzero(mask)[0]
        rr = (r[ii] - b.rmin) / b.cell
        row = np.floor(rr).astype(np.int64)
        np.clip(row, 0, b.n_r - 1, out=row)

        tw = (theta[ii] + np.pi) % (2.0 * np.pi)
        col = np.floor(tw / b.dtheta).astype(np.int64)
        np.clip(col, 0, b.n_theta - 1, out=col)

        flat[ii] = b.offset + row * b.n_theta + col
        band_id[ii] = b.id
        rows[ii] = row
        cols[ii] = col

    return flat, band_id, rows, cols
class VariableGrid:
    """Accumulated variable-resolution 2.5D map.

    Attributes (all flattened over every band, length ``total_cells``):
        count, sum_z, sum_z2, max_z, max_intensity,
        class_hist (total_cells x n_classes), dyn_count
    """

    def __init__(self, bands: list[Band], n_classes: int = 3):
        self.bands = bands
        self.n_classes = n_classes
        self.total_cells = sum(b.num_cells for b in bands)

        self.count = np.zeros(self.total_cells, dtype=np.int64)
        self.sum_z = np.zeros(self.total_cells, dtype=np.float64)
        self.sum_z2 = np.zeros(self.total_cells, dtype=np.float64)
        self.max_z = np.full(self.total_cells, -np.inf, dtype=np.float32)
        self.max_intensity = np.full(self.total_cells, -np.inf, dtype=np.float32)
        self.class_hist = np.zeros((self.total_cells, n_classes), dtype=np.int64)
        self.dyn_count = np.zeros(self.total_cells, dtype=np.int64)

    # -- projection --------------------------------------------------------
    def clear(self):
        self.count[:] = 0
        self.sum_z[:] = 0.0
        self.sum_z2[:] = 0.0
        self.max_z[:] = -np.inf
        self.max_intensity[:] = -np.inf
        self.class_hist[:] = 0
        self.dyn_count[:] = 0

    def insert(self, r, theta, xyz, intensity, labels, dynamic):
        """Vectorized point insertion.  Returns ``flat`` ids for kept points."""
        flat, band_id, rows, cols = assign_cells(r, theta, self.bands)
        valid = flat >= 0
        f = flat[valid]

        z = xyz[:, 2]
        np.add.at(self.count, f, 1)
        np.add.at(self.sum_z, f, z[valid])
        np.add.at(self.sum_z2, f, z[valid] * z[valid])
        np.maximum.at(self.max_z, f, z[valid].astype(np.float32))
        np.maximum.at(self.max_intensity, f, intensity[valid].astype(np.float32))

        lab = labels[valid].astype(np.int64)
        # NOTE: `self.class_hist[idx, c] += 1` would be a *buffered* fancy-index
        # assignment and silently drop every repeat of the same cell id, which
        # collapses the semantic histogram to one vote per cell.  `np.add.at`
        # is the unbuffered accumulate we need here.
        for c in range(self.n_classes):
            hit = lab == c
            if np.any(hit):
                np.add.at(self.class_hist[:, c], f[hit], 1)
        dyn = dynamic[valid]
        if np.any(dyn):
            np.add.at(self.dyn_count, f[dyn], 1)

        return flat, valid

    # -- derived per-cell attributes --------------------------------------
    @property
    def occupancy(self) -> np.ndarray:
        return self.count > 0

    @property
    def mean_height(self) -> np.ndarray:
        out = np.zeros(self.total_cells, dtype=np.float32)
        nz = self.count > 0
        out[nz] = self.sum_z[nz] / self.count[nz]
        return out

    @property
    def height_variance(self) -> np.ndarray:
        out = np.zeros(self.total_cells, dtype=np.float32)
        nz = self.count > 0
        mean = self.sum_z[nz] / self.count[nz]
        var = self.sum_z2[nz] / self.count[nz] - mean * mean
        out[nz] = np.clip(var, 0.0, None)
        return out

    @property
    def pred_class(self) -> np.ndarray:
        return np.argmax(self.class_hist, axis=1).astype(np.int64)

    @property
    def confidence(self) -> np.ndarray:
        total = self.count.astype(np.float64)
        out = np.zeros(self.total_cells, dtype=np.float32)
        nz = total > 0
        out[nz] = self.class_hist[nz].max(axis=1) / total[nz]
        return out

    @property
    def dynamic(self) -> np.ndarray:
        return self.dyn_count > 0

    # -- geometry helpers --------------------------------------------------
    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return Cartesian (x, y) centers of every cell (all bands)."""
        xs = np.zeros(self.total_cells, dtype=np.float32)
        ys = np.zeros(self.total_cells, dtype=np.float32)
        for b in self.bands:
            rows = np.arange(b.n_r)
            cols = np.arange(b.n_theta)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            r = b.rmin + (rr + 0.5) * b.cell
            th = -np.pi + (cc + 0.5) * b.dtheta
            idx = b.offset + (rr * b.n_theta + cc).ravel()
            xs[idx] = (r * np.cos(th)).ravel()
            ys[idx] = (r * np.sin(th)).ravel()
        return xs, ys

    def occupied_centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, y, band_id) for occupied cells only."""
        xs, ys = self.cell_centers()
        occ = self.occupancy
        bid = np.empty(self.total_cells, dtype=np.int64)
        for b in self.bands:
            bid[b.offset:b.offset + b.num_cells] = b.id
        return xs[occ], ys[occ], bid[occ]

    def __len__(self) -> int:
        return int(self.occupancy.sum())

    def __repr__(self) -> str:  # pragma: no cover
        return (f"VariableGrid(bands={len(self.bands)}, "
                f"cells={self.total_cells}, occupied={len(self)}, "
                f"dynamic={int(self.dynamic.sum())}")


def build_grid(
    r, theta, xyz, intensity, labels, dynamic, band_specs, n_classes=3
) -> VariableGrid:
    """Convenience factory: build bands and insert in one call."""
    bands = build_bands(band_specs, n_classes=n_classes)
    grid = VariableGrid(bands, n_classes=n_classes)
    if r.size:
        grid.insert(r, theta, xyz, intensity, labels, dynamic)
    return grid