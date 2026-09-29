"""Scientific visualization copies and composable Matplotlib axes functions.

Pixel coordinates are (x=column, y=row), zero based; no resizing or fitting is
performed. Display normalization is independent P1/P99 clipping, not calibrated
radiometry. Masked pixels are shown in lavender, never as black terrain.
Callers own figure creation, saving and closing; no global style/backend changes.
"""
from numbers import Integral, Real
from typing import Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import matplotlib as mpl

from src.evaluation.metrics import reprojection_residuals, spatial_distribution
from src.geometry.registration import transform_points
from src.spatial.quadtree import Quadtree


def _binary(mask: np.ndarray, shape: tuple) -> np.ndarray:
    values = np.asarray(mask)
    if values.shape != shape or not np.isin(values, [0, 1, 255]).all():
        raise ValueError("Mask must be binary and match the expected shape")
    return values.astype(bool)


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_scalar(value: float, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _points(points: np.ndarray, shape: tuple | None = None) -> np.ndarray:
    values = np.asarray(points)
    if (values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()):
        raise ValueError("Points must be finite real N x 2 coordinates")
    if shape is not None and ((values < 0).any() or (values >= np.array(shape[::-1])).any()):
        raise ValueError("Points must lie inside their image grid")
    return values.astype(float, copy=True)


def display_image(image: np.ndarray, valid_mask: np.ndarray | None = None,
                  nodata: float | None = None,
                  percentiles: tuple[float, float] = (1., 99.)) -> np.ma.MaskedArray:
    """Return an independent masked float64 display copy in [0,1].

    Nonfinite values, explicit nodata, existing array masks and invalid-mask
    pixels are excluded from percentile estimation. Valid zeros remain valid.
    Constant/all-invalid images are supported; collapsed percentiles fall back
    to the valid min/max. Scientific input values/masks are never modified.
    """
    values = np.asarray(np.ma.getdata(image))
    if values.ndim != 2 or not values.size or values.dtype.kind not in "fiu":
        raise ValueError("Image must be a nonempty real numeric 2D array")
    if len(percentiles) != 2:
        raise ValueError("Expected two percentile bounds")
    for bound in percentiles:
        _finite_scalar(bound, "Percentile")
    if not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    valid = np.isfinite(values) & ~np.ma.getmaskarray(image)
    if valid_mask is not None:
        valid &= _binary(valid_mask, values.shape)
    if nodata is not None:
        if not np.isscalar(nodata) or not isinstance(nodata, Real):
            raise ValueError("Nodata must be a real scalar")
        valid &= values != nodata
    output = np.zeros(values.shape, dtype=float)
    if valid.any():
        samples = values[valid].astype(float)
        low, high = np.percentile(samples, percentiles)
        if high <= low:
            low, high = samples.min(), samples.max()
        if high > low:
            output[valid] = np.clip((samples-low)/(high-low), 0, 1)
    return np.ma.array(output, mask=~valid, copy=True)


def show_display(ax: Axes, display: np.ma.MaskedArray, title: str,
                 cmap: str = "gray") -> mpl.image.AxesImage:
    """Render an already normalized display copy without renormalizing it."""
    color_map = mpl.colormaps[cmap].with_extremes(bad="#d7cfe3")
    artist = ax.imshow(display, cmap=color_map, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("Column (pixels)", fontsize=8)
    ax.set_ylabel("Row (pixels)", fontsize=8)
    ax.tick_params(labelsize=8)
    return artist


def plot_image(ax: Axes, image: np.ndarray, valid_mask: np.ndarray | None = None,
               title: str = "Display-normalized image") -> mpl.image.AxesImage:
    """Display one scientific input using a separate percentile-normalized copy."""
    return show_display(ax, display_image(image, valid_mask), title)


def plot_keypoints(ax: Axes, image: np.ndarray, points: np.ndarray,
                   valid_mask: np.ndarray | None = None, scales: np.ndarray | None = None,
                   orientations: np.ndarray | None = None,
                   title: str = "Detected keypoints") -> None:
    """Draw unchanged positions, optional diameters in pixels and angles in degrees.

    Angles increase clockwise in image coordinates (x right, y down), matching
    OpenCV's convention. Unknown orientations should be omitted, not passed as -1.
    Without scales, orientation indicators have a five-pixel display length.
    """
    xy = _points(points, np.shape(image))
    attributes = []
    for name, values in (("scales", scales), ("orientations", orientations)):
        if values is None:
            attributes.append(None)
            continue
        array = np.asarray(values)
        if array.shape != (len(xy),) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite vector aligned with keypoints")
        if name == "scales" and (array <= 0).any():
            raise ValueError("Keypoint diameters must be positive")
        if name == "orientations" and ((array < 0).any() or (array >= 360).any()):
            raise ValueError("Keypoint angles must lie in [0,360)")
        attributes.append(array)
    diameters, angles = attributes
    plot_image(ax, image, valid_mask, f"{title} ({len(xy)})")
    ax.scatter(xy[:, 0], xy[:, 1], s=12, facecolors="none", edgecolors="#00d5c7", linewidths=.7)
    if diameters is not None:
        for point, diameter in zip(xy, diameters):
            ax.add_patch(Circle(point, diameter/2, fill=False, color="#00d5c7", linewidth=.6))
    if angles is not None:
        lengths = diameters/2 if diameters is not None else np.full(len(xy), 5.)
        radians = np.deg2rad(angles)
        ax.quiver(xy[:, 0], xy[:, 1], lengths*np.cos(radians), lengths*np.sin(radians),
                  angles="xy", scale_units="xy", scale=1, color="#ffbd59")


def select_match_indices(count: int, max_matches: int | None = 150) -> np.ndarray:
    """Evenly spaced original-order indices for DISPLAY ONLY; no RNG or ranking."""
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0:
        raise ValueError("Match count must be a nonnegative integer")
    if max_matches is not None:
        _positive_integer(max_matches, "Maximum displayed matches")
    if max_matches is None or count <= max_matches:
        return np.arange(count, dtype=int)
    return np.linspace(0, count-1, max_matches, dtype=int)


def plot_matches(ax: Axes, source_image: np.ndarray, reference_image: np.ndarray,
                 source: np.ndarray, destination: np.ndarray,
                 source_mask: np.ndarray | None = None, reference_mask: np.ndarray | None = None,
                 confidence: np.ndarray | None = None, inlier_mask: np.ndarray | None = None,
                 max_matches: int | None = 150, title: str = "Correspondences") -> np.ndarray:
    """Side-by-side match display; return selected original indices for provenance.

    Inlier colors take priority over confidence when both are supplied. Full-set
    counts remain in the annotation regardless of display subsampling. The
    reference x offset only places its panel; scientific coordinates stay intact.
    """
    a, b = display_image(source_image, source_mask), display_image(reference_image, reference_mask)
    src, dst = _points(source, a.shape), _points(destination, b.shape)
    if src.shape != dst.shape:
        raise ValueError("Source and destination points must be aligned")
    keep = _binary(inlier_mask, (len(src),)) if inlier_mask is not None else None
    scores = None
    if confidence is not None:
        scores = np.asarray(confidence)
        if (scores.shape != (len(src),) or scores.dtype.kind not in "fiu" or
                not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()):
            raise ValueError("Confidence must be an aligned finite vector in [0,1]")
    indices = select_match_indices(len(src), max_matches)
    canvas = np.ma.masked_all((max(a.shape[0], b.shape[0]), a.shape[1]+b.shape[1]))
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:] = b
    subtitle = f"{len(src)} {'candidates' if keep is not None else 'pairs'}; {len(indices)} displayed"
    if keep is not None:
        ratio = f"{100*keep.mean():.2f}%" if keep.size else "undefined ratio"
        subtitle += f"; {int(keep.sum())} inliers ({ratio})"
    show_display(ax, canvas, title+"\n"+subtitle)
    shifted = dst[indices]+[a.shape[1], 0]
    segments = np.stack((src[indices], shifted), axis=1)
    if keep is not None:
        colors = np.where(keep[indices], "#00d5c7", "#ff725e")
    elif scores is not None:
        colors = mpl.colormaps["viridis"](scores[indices])
    else:
        colors = "#00d5c7"
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=.65, alpha=.7))
    ax.scatter(src[indices, 0], src[indices, 1], s=5, c=colors)
    ax.scatter(shifted[:, 0], shifted[:, 1], s=5, c=colors)
    if keep is not None:
        ax.legend(handles=[Line2D([], [], color="#00d5c7", label="Inlier"),
                           Line2D([], [], color="#ff725e", label="Outlier")], fontsize=8, loc="lower right")
    elif scores is not None:
        ax.figure.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), ax=ax,
                           label="Saved confidence", fraction=.025)
    return indices


def _common_displays(moving: np.ndarray, reference: np.ndarray,
                     moving_mask: np.ndarray | None, reference_mask: np.ndarray | None) -> tuple:
    a, b = display_image(moving, moving_mask), display_image(reference, reference_mask)
    if a.shape != b.shape:
        raise ValueError("Registered and reference images must have identical dimensions")
    common = ~(np.ma.getmaskarray(a) | np.ma.getmaskarray(b))
    # Comparison normalization excludes everything outside the compared support.
    return display_image(moving, common), display_image(reference, common)


def overlay_image(moving: np.ndarray, reference: np.ndarray,
                  moving_mask: np.ndarray | None = None, reference_mask: np.ndarray | None = None,
                  alpha: float = .5) -> np.ma.MaskedArray:
    """Weighted display overlay on common valid pixels only; alpha weights moving."""
    _finite_scalar(alpha, "Alpha")
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must lie in [0,1]")
    a, b = _common_displays(moving, reference, moving_mask, reference_mask)
    return alpha*a+(1-alpha)*b


def checkerboard_image(moving: np.ndarray, reference: np.ndarray,
                       moving_mask: np.ndarray | None = None, reference_mask: np.ndarray | None = None,
                       block_size: int = 32) -> np.ma.MaskedArray:
    """Alternate display blocks on common coverage; top-left block uses moving."""
    _positive_integer(block_size, "Checkerboard block size")
    a, b = _common_displays(moving, reference, moving_mask, reference_mask)
    rows, cols = np.indices(a.shape)
    use_moving = (rows//block_size+cols//block_size) % 2 == 0
    return np.ma.array(np.where(use_moving, a.data, b.data), mask=np.ma.getmaskarray(a), copy=True)


def normalized_difference(moving: np.ndarray, reference: np.ndarray,
                          moving_mask: np.ndarray | None = None,
                          reference_mask: np.ndarray | None = None) -> np.ma.MaskedArray:
    """Absolute independently normalized difference; diagnostic, not radiometric error."""
    a, b = _common_displays(moving, reference, moving_mask, reference_mask)
    return np.ma.abs(a-b)


def plot_registration(axes: np.ndarray, moving: np.ndarray, reference: np.ndarray,
                       registered: np.ndarray, moving_mask: np.ndarray,
                       reference_mask: np.ndarray, registered_mask: np.ndarray) -> None:
    """Populate four axes: before, reference, registered and common-valid overlay."""
    panels = np.asarray(axes, dtype=object).ravel()
    if len(panels) != 4:
        raise ValueError("Registration comparison requires four axes")
    for ax, image, mask, title in zip(panels[:3], (moving, reference, registered),
            (moving_mask, reference_mask, registered_mask),
            ("A | Moving before", "B | Fixed reference", "C | Registered moving")):
        plot_image(ax, image, mask, title)
    show_display(panels[3], overlay_image(registered, reference, registered_mask, reference_mask),
                 "D | 50% overlay / common valid coverage")


def plot_residuals(ax: Axes, reference: np.ndarray, source: np.ndarray,
                   destination: np.ndarray, matrix: np.ndarray,
                   reference_mask: np.ndarray | None = None, vectors: bool = False,
                   arrow_scale: float = 1.) -> np.ndarray:
    """Show fit residual magnitude or predicted->matched arrows on reference.

    Arrow displacement alone is multiplied by arrow_scale for visibility; color
    always encodes the unscaled Euclidean residual. All supplied pairs are used.
    """
    _finite_scalar(arrow_scale, "Arrow magnification")
    if arrow_scale <= 0:
        raise ValueError("Arrow magnification must be positive")
    src, dst = _points(source), _points(destination, np.shape(reference))
    residuals = reprojection_residuals(src, dst, matrix)
    predicted = transform_points(src, matrix)
    title = (f"Residual vectors / arrows {arrow_scale:g}x (display only)" if vectors
             else "Verified feature residual magnitudes")
    plot_image(ax, reference, reference_mask, title)
    norm = Normalize(0, max(float(residuals.max()), 1e-12) if len(residuals) else 1.)
    if vectors:
        delta = (dst-predicted)*arrow_scale
        ax.quiver(predicted[:, 0], predicted[:, 1], delta[:, 0], delta[:, 1], residuals,
                  angles="xy", scale_units="xy", scale=1, cmap="plasma", norm=norm, width=.004)
    else:
        ax.scatter(dst[:, 0], dst[:, 1], c=residuals, s=13, cmap="plasma", norm=norm)
    ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax,
                       label="Feature reprojection residual (pixels)", fraction=.045)
    return residuals


def plot_spatial_distribution(ax: Axes, reference: np.ndarray, points: np.ndarray,
                               reference_mask: np.ndarray | None = None,
                               grid_shape: tuple[int, int] = (4, 4)) -> dict:
    """Visualize all supplied inliers and count-based entropy; no spatial filtering."""
    stats = spatial_distribution(points, np.shape(reference), grid_shape)
    xy = _points(points, np.shape(reference))
    entropy = "undefined" if stats['normalized_entropy'] is None else f"{stats['normalized_entropy']:.6f}"
    plot_image(ax, reference, reference_mask,
        f"{stats['occupied_cells']}/{stats['total_cells']} occupied ({stats['occupancy_percentage']:.2f}%)\nNormalized entropy {entropy} / count concentration")
    ax.scatter(xy[:, 0], xy[:, 1], s=5, c="#00d5c7", alpha=.6)
    h, w = np.shape(reference)
    rows, cols = grid_shape
    for x in np.linspace(0, w, cols+1):
        ax.axvline(x, color="#ffbd59", linewidth=.8)
    for y in np.linspace(0, h, rows+1):
        ax.axhline(y, color="#ffbd59", linewidth=.8)
    for row in range(rows):
        for col in range(cols):
            ax.text((col+.5)*w/cols, (row+.5)*h/rows, str(stats['counts'][row][col]),
                    ha="center", va="center", color="white", bbox=dict(facecolor="#15283a", alpha=.85, edgecolor="none"))
    ax.set_xlim(-.5, w-.5)
    ax.set_ylim(h-.5, -.5)
    return stats


def plot_metrics_panel(ax: Axes, metrics: Mapping[str, str],
                        title: str = "Evaluation", footnote: str =
                        "Feature fit residuals are not independently verified\nabsolute geographic accuracy.") -> None:
    """Compact reusable label/value panel; values and scientific labels are caller-owned."""
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=12)
    n = len(metrics)
    for index, (label, value) in enumerate(metrics.items()):
        y = .94-index*(.73/max(n-1, 1))
        ax.text(.02, y, label, transform=ax.transAxes, fontsize=10, color="#34495e", va="center")
        ax.text(.98, y, str(value), transform=ax.transAxes, fontsize=11, fontweight="bold",
                color="#123b51", ha="right", va="center")
    ax.text(.02, .04, footnote, transform=ax.transAxes, fontsize=9, color="#66464a", va="bottom")


def plot_quadtree(ax: Axes, reference: np.ndarray, tree: Quadtree,
                  selected_indices: np.ndarray, reference_mask: np.ndarray | None = None,
                  title: str = "Quadtree correspondence selection") -> None:
    """Plot every input, selected points and depth-colored leaf boundaries.

    Selected indices refer to the tree input. Empty leaves remain visible; no
    coordinates are moved/subsampled. Boundary color denotes depth, not quality.
    """
    if np.shape(reference) != tree.image_shape:
        raise ValueError("Reference dimensions must match the quadtree")
    selected = np.asarray(selected_indices)
    if (selected.ndim != 1 or selected.dtype.kind not in 'iu' or
            (selected < 0).any() or (selected >= len(tree.destination)).any() or
            len(np.unique(selected)) != len(selected)):
        raise ValueError("Selected indices must be a unique valid integer vector")
    plot_image(ax, reference, reference_mask, title)
    deepest = max(node.depth for node in tree.nodes)
    norm = Normalize(0, max(deepest, 1))
    for leaf in tree.leaves:
        x0, y0, x1, y1 = leaf.bounds
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
            edgecolor=mpl.colormaps['plasma'](norm(leaf.depth)), linewidth=.65, alpha=.85))
    xy, kept = tree.destination, tree.destination[selected]
    ax.scatter(xy[:, 0], xy[:, 1], c='#aaaaaa', s=8, alpha=.7, label=f'Input ({len(xy)})')
    ax.scatter(kept[:, 0], kept[:, 1], c='#00edcf', s=24, edgecolors='#123b51', linewidths=.4,
               label=f'Selected ({len(selected)})')
    ax.legend(loc='lower left', fontsize=8)
    ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax,
                       ticks=range(deepest+1), label='Leaf depth', fraction=.035)
    ax.set_xlim(-.5, tree.image_shape[1]-.5)
    ax.set_ylim(tree.image_shape[0]-.5, -.5)
