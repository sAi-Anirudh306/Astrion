"""Deterministic regular-grid control of already verified correspondences.

No fitting, filtering by geometric thresholds, or image warping occurs here.
Coordinates use (x=column, y=row); shapes use (height, width).
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _shape(shape: tuple[int, int]) -> None:
    if (not isinstance(shape, (tuple, list, np.ndarray)) or len(shape) != 2 or
            any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0 for n in shape)):
        raise ValueError("Shape must contain two positive integer dimensions")


def _points(points: np.ndarray) -> np.ndarray:
    xy = np.asarray(points)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.dtype.kind not in "fiu" or not np.isfinite(xy).all():
        raise ValueError("Points must be finite real N x 2 coordinates")
    return xy.astype(float, copy=True)


@dataclass(frozen=True)
class GridAssignment:
    """One row, column and row-major flattened cell index per input point."""
    rows: np.ndarray
    columns: np.ndarray
    cells: np.ndarray


def assign_grid_cells(points: np.ndarray, image_shape: tuple[int, int],
                      grid_shape: tuple[int, int] = (4, 4)) -> GridAssignment:
    """Assign equal half-open cells over [0,width) x [0,height).

    Internal boundaries belong to the cell to the right/below. Last integer
    pixel (width-1,height-1) is valid, as are fractional coordinates below the
    final boundary. x==width/y==height, negative and nonfinite values raise.
    No clipping/wrapping. Searching explicit edges avoids rounding a value just
    below the final boundary into an out-of-range cell.
    """
    _shape(image_shape)
    _shape(grid_shape)
    xy = _points(points)
    if (xy < 0).any() or (xy >= np.asarray(image_shape[::-1])).any():
        raise ValueError("Points must lie in [0,width) x [0,height)")
    rows = np.searchsorted(np.linspace(0, image_shape[0], grid_shape[0]+1)[1:-1], xy[:, 1], side="right")
    columns = np.searchsorted(np.linspace(0, image_shape[1], grid_shape[1]+1)[1:-1], xy[:, 0], side="right")
    return GridAssignment(rows, columns, rows*grid_shape[1]+columns)


def distribution_statistics(points: np.ndarray, image_shape: tuple[int, int],
                             grid_shape: tuple[int, int] = (4, 4)) -> dict:
    """M9-compatible grid counts and entropy; std is population (ddof=0).

    Entropy describes count concentration, not uniform coverage or geometric
    observability. Empty entropy/occupied-cell statistics are None; occupancy
    and maximum count are zero. A one-cell grid has undefined normalization
    and is rejected, preserving the established evaluation contract.
    """
    assignment = assign_grid_cells(points, image_shape, grid_shape)
    total_cells = int(np.prod(grid_shape))
    if total_cells <= 1:
        raise ValueError("Entropy requires at least two grid cells")
    counts = np.bincount(assignment.cells, minlength=total_cells).reshape(grid_shape)
    occupied = counts[counts > 0]
    n = len(assignment.cells)
    probabilities = occupied/n if n else np.array([])
    entropy = float(np.clip(-np.sum(probabilities*np.log(probabilities))/np.log(total_cells), 0, 1)) if n else None
    return dict(grid_shape=list(grid_shape), counts=counts.tolist(), occupied_cells=len(occupied),
                total_cells=total_cells, occupancy_percentage=100*len(occupied)/total_cells,
                maximum_cell_count=int(counts.max()),
                minimum_occupied_count=int(occupied.min()) if occupied.size else None,
                mean_occupied_count=float(occupied.mean()) if occupied.size else None,
                std_occupied_count=float(occupied.std()) if occupied.size else None,
                normalized_entropy=entropy,
                entropy_definition="-sum(p*ln(p))/ln(total cells); grid-scale count concentration, not proof of uniformity")


def spatial_spread(points: np.ndarray, image_shape: tuple[int, int]) -> dict:
    """Bounding-box extent / image dimensions; correspondence footprint, not image coverage.

    Extent is max-min of point centers (no added pixel width). A singleton has
    zero extent; empty inputs return None, since no bounding box exists.
    """
    assign_grid_cells(points, image_shape)
    xy = _points(points)
    lower, upper = (xy.min(axis=0), xy.max(axis=0)) if len(xy) else (None, None)
    return dict(bounding_box_min_xy=lower.tolist() if lower is not None else None,
                bounding_box_max_xy=upper.tolist() if upper is not None else None,
                width_fraction=float((upper[0]-lower[0])/image_shape[1]) if len(xy) else None,
                height_fraction=float((upper[1]-lower[1])/image_shape[0]) if len(xy) else None,
                meaning="Correspondence spatial footprint; not registered image coverage")


@dataclass(frozen=True)
class SpatialSelection:
    """Independent selected arrays in INPUT order, with indices into the input set."""
    source: np.ndarray
    destination: np.ndarray
    original_indices: np.ndarray
    cell_rows: np.ndarray
    cell_columns: np.ndarray
    cell_indices: np.ndarray
    confidence: np.ndarray | None
    residuals: np.ndarray | None


def _scores(values: np.ndarray | None, length: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if (array.shape != (length,) or array.dtype.kind not in "fiu" or
            not np.isfinite(array).all() or (array < 0).any()):
        raise ValueError(f"{name} must be an aligned finite nonnegative vector")
    if name == "Confidence" and (array > 1).any():
        raise ValueError("Confidence must be in [0,1]")
    return array.astype(float, copy=True)


def _select_partition_indices(group_ids: np.ndarray, max_per_region: int,
                              max_matches: int | None, conf: np.ndarray | None,
                              errors: np.ndarray | None) -> np.ndarray:
    """Shared ranked round-robin allocator for validated grid/leaf memberships.

    Regions are visited by ascending ID; output is returned in input order.
    Callers validate scores and memberships. Zero budgets select nothing.
    """
    for name, value in (("max_per_region", max_per_region), ("max_matches", max_matches)):
        if value is None and name == "max_matches":
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    keys = [np.arange(len(group_ids))]
    if conf is not None:
        keys.append(-conf)
    if errors is not None:
        keys.append(errors)
    ranking = np.lexsort(tuple(keys))
    queues = [ranking[group_ids[ranking] == cell][:max_per_region]
              for cell in np.unique(group_ids)]
    limit = len(group_ids) if max_matches is None else max_matches
    chosen = []
    for level in range(max((len(queue) for queue in queues), default=0)):
        for queue in queues:
            if len(chosen) >= limit:
                break
            if level < len(queue):
                chosen.append(int(queue[level]))
        if len(chosen) >= limit:
            break
    return np.sort(np.asarray(chosen, dtype=int))


def select_spatially_balanced_matches(
    source: np.ndarray, destination: np.ndarray, image_shape: tuple[int, int],
    grid_shape: tuple[int, int] = (4, 4), max_per_cell: int = 10,
    max_matches: int | None = None, confidence: np.ndarray | None = None,
    residuals: np.ndarray | None = None, source_shape: tuple[int, int] | None = None,
) -> SpatialSelection:
    """Select within each reference cell by residual ASC, confidence DESC, index ASC.

    Missing score keys are omitted; without scores, original order wins.
    Confidence is a [0,1] score; residuals must be nonnegative, in common units.
    Source coordinates must be finite; source bounds are checked if source_shape
    is supplied (source and reference grids need not have identical dimensions).

    Ranked per-cell queues are capped, then visited round-robin in ascending
    row-major cell ID, skipping exhausted queues, until the global limit is met.
    Thus all occupied cells get one point before any gets a second when budget
    permits. A partial final round favors earlier cell IDs; if budget is smaller
    than occupancy, retaining every occupied cell is impossible. No composite
    score, random sampling or implicit model estimation is used. Zero limits
    explicitly select nothing. Return order is original INPUT order, not ranking
    or round-robin order; original_indices exposes exact input membership.
    """
    for name, value in (("max_per_cell", max_per_cell), ("max_matches", max_matches)):
        if value is None and name == "max_matches":
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    a, b = _points(source), _points(destination)
    if a.shape != b.shape:
        raise ValueError("Source and destination arrays must have matching lengths")
    assignment = assign_grid_cells(b, image_shape, grid_shape)
    if source_shape is not None:
        assign_grid_cells(a, source_shape)
    conf = _scores(confidence, len(a), "Confidence")
    errors = _scores(residuals, len(a), "Residuals")
    selected = _select_partition_indices(assignment.cells, max_per_cell, max_matches, conf, errors)
    return SpatialSelection(a[selected], b[selected], selected,
        assignment.rows[selected], assignment.columns[selected], assignment.cells[selected],
        conf[selected] if conf is not None else None, errors[selected] if errors is not None else None)
