"""Six-parameter affine USAC/MAGSAC scoring with sigma-consensus local optimization.

Requires OpenCV's working estimateAffine2D(pts1, pts2, UsacParams) overload.
No fallback to RANSAC or homography is performed. Threshold is the USAC support /
termination threshold, also used for explicit final ASTRION classification; it
is not an independently exposed upper sigma bound. Backend internal refinement
is retained, with no additional project-level fitting.
"""
from dataclasses import asdict, dataclass
from time import perf_counter

import cv2
import numpy as np

from src.geometry.ransac import RANSACConfig, _points, _spans_plane
from src.evaluation.metrics import reprojection_residuals


@dataclass(frozen=True)
class MAGSACConfig:
    reprojection_threshold: float = 3.0
    confidence: float = 0.99
    max_iterations: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        RANSACConfig(self.reprojection_threshold, self.max_iterations,
                     self.confidence, 0, self.seed)


@dataclass(frozen=True)
class MAGSACResult:
    """Original input order is preserved in all masks/errors and inlier indices."""
    matrix: np.ndarray
    inlier_mask: np.ndarray
    backend_inlier_mask: np.ndarray
    inlier_indices: np.ndarray
    errors: np.ndarray
    config: MAGSACConfig
    metadata: dict
    runtime_seconds: float


def verify_affine(source_points: np.ndarray, destination_points: np.ndarray,
                  config: MAGSACConfig | None = None) -> MAGSACResult:
    """Estimate forward source -> reference affine; reject invalid/failed fits.

    Identical coordinate pairs are deduplicated for fitting (as in RANSAC), then
    mapped back to every original index. Serial USAC uses a local backend seed.
    Final support is residual <= threshold against the returned matrix, without
    refitting that classification. Backend support is returned separately.
    """
    started = perf_counter()
    config = MAGSACConfig() if config is None else config
    if not isinstance(config, MAGSACConfig):
        raise TypeError("Expected MAGSACConfig")
    a, b = _points(source_points, "source"), _points(destination_points, "destination")
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Affine estimation needs equal lengths and at least three pairs")
    pairs, inverse = np.unique(np.column_stack((a, b)), axis=0, return_inverse=True)
    fa, fb = pairs[:, :2].astype(np.float32), pairs[:, 2:].astype(np.float32)
    if not _spans_plane(fa.astype(float)) or not _spans_plane(fb.astype(float)):
        raise ValueError("Degenerate or collinear affine point configuration")
    try:
        params = cv2.UsacParams()
        params.score = cv2.SCORE_METHOD_MAGSAC
        params.loMethod = cv2.LOCAL_OPTIM_SIGMA
        params.sampler = cv2.SAMPLING_UNIFORM
        params.isParallel = False
        params.randomGeneratorState = int(config.seed)
        params.threshold = float(config.reprojection_threshold)
        params.confidence = float(config.confidence)
        params.maxIterations = int(config.max_iterations)
        matrix, backend_mask = cv2.estimateAffine2D(fa, fb, params)
    except (AttributeError, cv2.error) as exc:
        raise RuntimeError("OpenCV affine USAC/MAGSAC backend unavailable or failed; no fallback") from exc
    if matrix is None or backend_mask is None:
        raise RuntimeError("Affine MAGSAC failed to estimate a model")
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (2, 3) or not np.isfinite(matrix).all()
            or np.linalg.cond(matrix[:, :2]) > 1e12):
        raise RuntimeError("Affine MAGSAC returned an invalid or singular model")
    raw = np.asarray(backend_mask).reshape(-1)
    if len(raw) != len(pairs) or not np.isin(raw, [0, 1]).all():
        raise RuntimeError("Affine MAGSAC returned malformed backend support")
    errors = reprojection_residuals(a, b, matrix)
    keep = errors <= config.reprojection_threshold
    if not _spans_plane(a[keep]) or not _spans_plane(b[keep]):
        raise RuntimeError("Affine MAGSAC final support is insufficient or degenerate")
    metadata = dict(backend="cv2.estimateAffine2D / UsacParams", opencv_version=cv2.__version__,
        model="affine_2d_six_parameter", score="SCORE_METHOD_MAGSAC",
        local_optimization="LOCAL_OPTIM_SIGMA", parameters=asdict(config),
        backend_parameters={name: getattr(params, name) for name in dir(params) if not name.startswith('_')},
        unique_fit_pairs=len(pairs), final_support="Euclidean forward residual <= threshold",
        threshold_semantics="USAC support/termination threshold; internal maximum sigma not exposed",
        refinement="Backend internal optimization/polishing only; no external refit")
    return MAGSACResult(matrix.copy(), keep, raw.astype(bool)[inverse], np.flatnonzero(keep),
                        errors, config, metadata, perf_counter()-started)


def affine_backend_capability() -> dict:
    """Exercise the actual overload on a full affine synthetic model with outliers."""
    rng = np.random.default_rng(10)
    a = rng.uniform(0, 100, (60, 2))
    truth = np.array([[1.1, .12, 8], [-.08, .95, 12]])
    b = a @ truth[:, :2].T + truth[:, 2]
    b[-15:] += 100
    try:
        result = verify_affine(a, b)
    except RuntimeError as exc:
        return dict(opencv_version=cv2.__version__, supported=False, reason=str(exc))
    difference = float(np.max(np.abs(result.matrix-truth)))
    return dict(opencv_version=cv2.__version__, supported=difference < .01,
                model="six_parameter_affine", probe_max_matrix_error=difference,
                probe_inlier_count=int(result.inlier_mask.sum()), metadata=result.metadata)
