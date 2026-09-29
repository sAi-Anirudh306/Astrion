"""Adaptive partitioning and selection of already verified correspondences.

Reference coordinates are (x,y). Bounds are (xmin,ymin,xmax,ymax), half-open.
Split order and depth-first node IDs are TL, TR, BL, BR. Splits on an exact
midpoint go right/below. x==width/y==height are rejected; last integer pixel
and fractional coordinates below the outer boundary are valid. No clamping,
deduplication, geometric estimation or movement of points occurs.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from src.spatial.spatial_distribution import (
    _points, _scores, _select_partition_indices, assign_grid_cells)


@dataclass(frozen=True)
class QuadtreeNode:
    """Immutable spatial node; indices refer to the original destination array."""
    node_id: int
    bounds: tuple[float, float, float, float]
    depth: int
    original_indices: tuple[int, ...]
    children: tuple["QuadtreeNode", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(frozen=True)
class Quadtree:
    """Root plus an independent read-only coordinate snapshot and build settings."""
    root: QuadtreeNode
    destination: np.ndarray
    image_shape: tuple[int, int]
    capacity: int
    max_depth: int
    min_cell_size: tuple[float, float]

    @property
    def nodes(self) -> tuple[QuadtreeNode, ...]:
        """Depth-first preorder; includes internal nodes and empty leaves."""
        stack, nodes = [self.root], []
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(reversed(node.children))
        return tuple(nodes)

    @property
    def leaves(self) -> tuple[QuadtreeNode, ...]:
        return tuple(node for node in self.nodes if node.is_leaf)


def build_quadtree(destination: np.ndarray, image_shape: tuple[int, int],
                   capacity: int = 10, max_depth: int = 4,
                   min_cell_size: tuple[float, float] = (1., 1.)) -> Quadtree:
    """Split when count > capacity, depth < max_depth, and all children fit.

    image_shape is (height,width); min_cell_size is (minimum height,width) in
    pixels, applies to prospective children, and must be positive. Midpoints
    are floating point, so odd image sizes need no rounding. All four children
    are retained, including empty ones. Identical points follow one branch and
    stop at a depth/size limit; capacity is a subdivision trigger, NOT a promise
    that all terminal leaves have at most that many points. Root depth is zero.
    Maximum supported depth is 64 to bound recursion and numerical precision.
    """
    for name, value, minimum in (("capacity", capacity, 1), ("max_depth", max_depth, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max_depth > 64:
        raise ValueError("Maximum supported tree depth is 64")
    if not isinstance(min_cell_size, (tuple, list)) or len(min_cell_size) != 2:
        raise ValueError("Minimum cell size must be (height,width)")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or
           not np.isfinite(v) or v <= 0 for v in min_cell_size):
        raise ValueError("Minimum cell dimensions must be finite and positive")
    xy = _points(destination)
    assign_grid_cells(xy, image_shape)  # Reuse M11 coordinate and extent validation.
    counter = 0

    def build(bounds: tuple, depth: int, indices: np.ndarray) -> QuadtreeNode:
        nonlocal counter
        node_id = counter
        counter += 1
        x0, y0, x1, y1 = bounds
        xm, ym = x0+(x1-x0)/2, y0+(y1-y0)/2
        possible = (x0 < xm < x1 and y0 < ym < y1 and
                    min(xm-x0, x1-xm) >= min_cell_size[1] and
                    min(ym-y0, y1-ym) >= min_cell_size[0])
        children = ()
        if len(indices) > capacity and depth < max_depth and possible:
            points = xy[indices]
            quadrant = (points[:, 0] >= xm).astype(int)+2*(points[:, 1] >= ym)
            child_bounds = ((x0, y0, xm, ym), (xm, y0, x1, ym),
                            (x0, ym, xm, y1), (xm, ym, x1, y1))
            children = tuple(build(rect, depth+1, indices[quadrant == i])
                             for i, rect in enumerate(child_bounds))
        return QuadtreeNode(node_id, bounds, depth, tuple(int(i) for i in indices), children)

    root = build((0., 0., float(image_shape[1]), float(image_shape[0])), 0, np.arange(len(xy)))
    xy.flags.writeable = False
    return Quadtree(root, xy, tuple(int(n) for n in image_shape), int(capacity),
                    int(max_depth), tuple(float(v) for v in min_cell_size))


def tree_statistics(tree: Quadtree) -> dict:
    """Population depth/count summaries; empty occupied statistics are None.

    Empty input is a single unoccupied root leaf. Depth extrema include empty
    leaves; mean depth and point-count statistics use occupied leaves only.
    No unequal-area adaptive-leaf entropy is computed.
    """
    nodes, leaves = tree.nodes, tree.leaves
    occupied = [leaf for leaf in leaves if leaf.original_indices]
    counts = np.array([len(leaf.original_indices) for leaf in occupied])
    deepest = max(node.depth for node in nodes)
    return dict(total_node_count=len(nodes), leaf_count=len(leaves), occupied_leaf_count=len(occupied),
        maximum_reached_depth=deepest, minimum_leaf_depth=min(leaf.depth for leaf in leaves),
        mean_occupied_leaf_depth=float(np.mean([leaf.depth for leaf in occupied])) if occupied else None,
        maximum_points_per_occupied_leaf=int(counts.max()) if counts.size else None,
        minimum_points_per_occupied_leaf=int(counts.min()) if counts.size else None,
        mean_points_per_occupied_leaf=float(counts.mean()) if counts.size else None,
        std_points_per_occupied_leaf=float(counts.std()) if counts.size else None,
        occupied_leaves_by_depth=[dict(depth=d, count=sum(leaf.depth == d for leaf in occupied))
                                  for d in range(deepest+1)],
        deepest_occupied_regions=[dict(leaf_id=leaf.node_id, bounds=list(leaf.bounds))
                                  for leaf in occupied if leaf.depth == deepest])


@dataclass(frozen=True)
class QuadtreeSelection:
    """Independent selected arrays in original input order; leaf IDs are node IDs."""
    source: np.ndarray
    destination: np.ndarray
    original_indices: np.ndarray
    leaf_ids: np.ndarray
    confidence: np.ndarray | None
    residuals: np.ndarray | None


def select_quadtree_matches(tree: Quadtree, source: np.ndarray,
                            max_per_leaf: int = 1, max_matches: int | None = None,
                            confidence: np.ndarray | None = None, residuals: np.ndarray | None = None,
                            source_shape: tuple[int, int] | None = None) -> QuadtreeSelection:
    """Rank by residual ASC, confidence DESC, original index ASC (omit absent keys).

    Source rows must correspond to the tree's destination rows. Optional source
    dimensions enforce source bounds. Per-leaf selection cap is independent of
    subdivision capacity. Optional global budget uses the shared M11 round-robin
    allocator, visiting occupied leaves by increasing depth-first node ID.
    Exhausted leaves are skipped; a partial round favors earlier IDs. If budget
    is below occupancy, not all leaves can be retained. Zero budgets select none.
    Final output is input order; no scores, coordinates or model are changed.
    """
    if not isinstance(tree, Quadtree):
        raise TypeError("Expected a built Quadtree")
    a = _points(source)
    if a.shape != tree.destination.shape:
        raise ValueError("Source must align with tree destination points")
    if source_shape is not None:
        assign_grid_cells(a, source_shape)
    conf = _scores(confidence, len(a), "Confidence")
    errors = _scores(residuals, len(a), "Residuals")
    membership = np.empty(len(a), dtype=int)
    for leaf in tree.leaves:
        membership[list(leaf.original_indices)] = leaf.node_id
    selected = _select_partition_indices(membership, max_per_leaf, max_matches, conf, errors)
    return QuadtreeSelection(a[selected], tree.destination[selected], selected, membership[selected],
                             conf[selected] if conf is not None else None,
                             errors[selected] if errors is not None else None)
