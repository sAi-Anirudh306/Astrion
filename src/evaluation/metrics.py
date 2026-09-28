"""Registration diagnostics, not external geographic accuracy measurements.

Coordinates are zero-based (x=column, y=row). Undefined statistics return None
for strict JSON serialization; malformed inputs raise ValueError. Inputs are
never modified. Standard deviations are population values (ddof=0).
"""
import numpy as np

from src.geometry.registration import transform_points


def _real(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(np.float64)


def _mask(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape or not np.isin(array, [0, 1, 255]).all():
        raise ValueError("Mask must be binary and match the expected shape")
    return array.astype(bool)


def _shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2 or any(isinstance(n, (bool, np.bool_)) or
            not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
        raise ValueError("Shape must contain two positive integers")


def correspondence_statistics(inlier_mask: np.ndarray) -> dict:
    """Count saved candidate classifications without refitting/reclassifying."""
    mask = np.asarray(inlier_mask)
    if mask.ndim != 1:
        raise ValueError("Inlier mask must be one-dimensional")
    mask = _mask(mask, mask.shape)
    count, inliers = mask.size, int(mask.sum())
    return dict(candidate_count=count, inlier_count=inliers,
                outlier_count=count-inliers, inlier_ratio=inliers/count if count else None)


def reprojection_residuals(moving: np.ndarray, reference: np.ndarray,
                           matrix: np.ndarray) -> np.ndarray:
    """Forward Euclidean affine residuals for aligned N x 2 correspondence pairs."""
    a, b = _real(moving, "Moving points"), _real(reference, "Reference points")
    if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape:
        raise ValueError("Correspondences must be aligned N x 2 arrays")
    return np.linalg.norm(transform_points(a, matrix)-b, axis=1)


def _residuals(values: np.ndarray) -> np.ndarray:
    array = _real(values, "Residuals")
    if array.ndim != 1 or (array < 0).any():
        raise ValueError("Residuals must be a nonnegative vector")
    return array


def pixel_to_map_distance(residuals: np.ndarray, metres_per_pixel: float) -> np.ndarray:
    """Nominal isotropic map-grid distances, not ground-truth geolocation errors."""
    scale = np.asarray(metres_per_pixel)
    if scale.ndim or scale.dtype.kind not in "fiu" or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Map scale must be a finite positive scalar")
    return _residuals(residuals)*float(scale)


def residual_statistics(residuals: np.ndarray) -> dict:
    """RMSE, population standard deviation and linearly interpolated percentiles."""
    r = _residuals(residuals)
    functions = dict(rmse=lambda x: np.sqrt(np.mean(x*x)), mean=np.mean,
                     median=np.median, minimum=np.min, maximum=np.max, std=np.std,
                     p90=lambda x: np.percentile(x, 90), p95=lambda x: np.percentile(x, 95))
    return {name: float(function(r)) if r.size else None for name, function in functions.items()}


def residual_thresholds(residuals: np.ndarray, metres_per_pixel: float,
                        thresholds: tuple[float, ...] = (.5, 1., 2., 3.)) -> list[dict]:
    """Inclusive threshold counts; denominator is supplied residual count."""
    r, limits = _residuals(residuals), _residuals(thresholds)
    distances = pixel_to_map_distance(limits, metres_per_pixel)
    return [dict(pixels=float(t), nominal_map_metres=float(m), count=int((r <= t).sum()),
                 percentage=float(100*np.mean(r <= t)) if r.size else None)
            for t, m in zip(limits, distances)]


def spatial_distribution(points: np.ndarray, image_shape: tuple[int, int],
                         grid_shape: tuple[int, int] = (4, 4)) -> dict:
    """Bin reference coordinates in equal cells over [0,width) x [0,height).

    Entropy -sum(p log p)/log(all cells) measures concentration of match counts
    on this grid, not proof of uniformity, independence or geographic accuracy.
    Empty distributions have undefined entropy; a one-cell grid is rejected.
    """
    _shape(image_shape)
    _shape(grid_shape)
    if np.prod(grid_shape) <= 1:
        raise ValueError("Entropy requires at least two grid cells")
    xy = _real(points, "Reference points")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Reference points must be N x 2")
    if (xy < 0).any() or (xy >= np.array(image_shape[::-1])).any():
        raise ValueError("Reference points must lie inside the image grid")
    indices = np.floor(xy/np.array(image_shape[::-1])*np.array(grid_shape[::-1])).astype(int)
    counts = np.zeros(grid_shape, dtype=int)
    np.add.at(counts, (indices[:, 1], indices[:, 0]), 1)
    occupied = counts[counts > 0]
    probabilities = occupied/len(xy) if len(xy) else np.array([])
    entropy = float(np.clip(-np.sum(probabilities*np.log(probabilities))/np.log(counts.size), 0, 1)) if len(xy) else None
    return dict(grid_shape=list(grid_shape), counts=counts.tolist(), occupied_cells=len(occupied),
                total_cells=counts.size, occupancy_percentage=100*len(occupied)/counts.size,
                maximum_cell_count=int(counts.max()),
                minimum_occupied_count=int(occupied.min()) if occupied.size else None,
                mean_occupied_count=float(occupied.mean()) if occupied.size else None,
                std_occupied_count=float(occupied.std()) if occupied.size else None,
                normalized_entropy=entropy,
                entropy_definition="-sum(p*ln(p))/ln(total cells); grid-scale count concentration, not proof of uniformity")


def coverage_statistics(registered_mask: np.ndarray, reference_mask: np.ndarray) -> dict:
    """Area counts from explicit valid/nodata masks; zero intensity can be valid."""
    shape = np.asarray(registered_mask).shape
    _shape(shape)
    a, b = _mask(registered_mask, shape), _mask(reference_mask, shape)
    common = int((a & b).sum())
    return dict(total_output_pixels=a.size, valid_registered_pixels=int(a.sum()),
                valid_reference_pixels=int(b.sum()), common_valid_pixels=common,
                registered_coverage_percentage=float(100*a.mean()),
                common_output_percentage=100*common/a.size,
                common_reference_percentage=100*common/int(b.sum()) if b.any() else None)


def affine_diagnostics(matrix: np.ndarray) -> dict:
    """SVD principal stretches and signed determinant; no rotation/shear shortcut.

    Singular matrices are reported with null condition number and a flag.
    Scale factors describe a pixel-to-pixel transform, not physical calibration.
    """
    affine = _real(matrix, "Affine transform")
    if affine.shape != (2, 3):
        raise ValueError("Affine transform must be 2 x 3")
    linear = affine[:, :2]
    singular = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    return dict(determinant=determinant, singular_values=singular.tolist(),
                principal_scale_factors=singular.tolist(), area_scale_factor=abs(determinant),
                condition_number=float(singular[0]/singular[1]) if singular[1] > 0 else None,
                singular=bool(singular[1] == 0))


def image_similarity(moving: np.ndarray, reference: np.ndarray,
                     moving_mask: np.ndarray, reference_mask: np.ndarray) -> dict:
    """Cross-image similarity diagnostics only, on finite common-valid pixels.

    Independently clip/stretch P1/P99 on the evaluated common support, following
    the Milestone 8 preview approach but retaining float64 without quantization.
    Pearson correlation is undefined for fewer than two pixels or a constant
    normalized image. Nodata must be excluded by the supplied masks.
    SSIM is deliberately omitted: full-image/window averages across irregular
    nodata coverage are inappropriate, and no masked-window SSIM is implemented.
    """
    a, b = np.asarray(moving), np.asarray(reference)
    _shape(a.shape)
    if b.shape != a.shape or any(x.dtype.kind not in "fiu" for x in (a, b)):
        raise ValueError("Images must be equally shaped real 2D arrays")
    common = _mask(moving_mask, a.shape) & _mask(reference_mask, a.shape)
    common &= np.isfinite(a) & np.isfinite(b)
    count = int(common.sum())
    normalized, bounds = [], []
    for image in (a, b):
        values = image[common].astype(np.float64)
        low, high = np.percentile(values, [1, 99]) if count else (0., 0.)
        normalized.append(np.clip((values-low)/max(float(high-low), 1e-12), 0, 1))
        bounds.append([float(low), float(high)] if count else None)
    defined = count >= 2 and all(np.ptp(x) > 0 for x in normalized)
    correlation = float(np.clip(np.corrcoef(*normalized)[0, 1], -1, 1)) if defined else None
    return dict(label="cross-image similarity diagnostics", common_valid_pixels=count,
                correlation=correlation, correlation_undefined_reason=None if defined else "Insufficient pixels or constant normalized image",
                normalization="Independent P1/P99 clipping on finite common support; float64, no quantization",
                percentile_bounds_moving_reference=bounds, ssim=None,
                ssim_reason="Omitted: irregular nodata requires fully valid local windows; masked-window SSIM is not implemented. Cross-sensor radiometry also limits interpretation.")
