"""Registration diagnostics, not external geographic accuracy measurements.

Coordinates are zero-based (x=column, y=row). Undefined statistics return None
for strict JSON serialization; malformed inputs raise ValueError. Inputs are
never modified. Standard deviations are population values (ddof=0).
"""
import numpy as np

from src.geometry.registration import transform_points
from src.spatial.spatial_distribution import distribution_statistics


def localization_overlap(predicted_bounds, comparison_bounds, metres_per_pixel: float,
                         *, same_observation: bool) -> dict:
    """Compare WAC edge-coordinate rectangles, without inventing geographic truth.

    Center displacement of differently sized ROIs is descriptive, not positional
    error. Even a same-observation crop is only approximate evaluation evidence.
    Unrelated observations explicitly have no localization-error measurement.
    """
    rectangles = _real([predicted_bounds, comparison_bounds], 'ROI bounds')
    if rectangles.shape != (2, 4) or np.any(rectangles[:, 2:] <= rectangles[:, :2]):
        raise ValueError('Expected nonempty [left,top,right,bottom] rectangles')
    if not isinstance(same_observation, bool):
        raise ValueError('same_observation must be boolean')
    pixel_to_map_distance(np.zeros(1), metres_per_pixel)
    a, b = rectangles
    overlap = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    area_a, area_b = np.prod(a[2:]-a[:2]), np.prod(b[2:]-b[:2])
    delta = (a[:2]+a[2:]-b[:2]-b[2:])/2
    displacement = float(np.linalg.norm(delta))
    return dict(iou=intersection/float(area_a+area_b-intersection),
        comparison_area_covered=intersection/float(area_b),
        contains_comparison=bool(np.all(a[:2] <= b[:2]) and np.all(a[2:] >= b[2:])),
        center_delta_pixels=delta.tolist(), center_displacement_pixels=displacement,
        center_displacement_map_km=displacement*float(metres_per_pixel)/1000,
        same_observation=same_observation, localization_error_pixels=None, localization_error_km=None,
        interpretation=('ROI overlap only; no independent control points or absolute error available'
                        if same_observation else 'Unrelated observation: not localization ground truth'))


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
    return distribution_statistics(points, image_shape, grid_shape)


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
