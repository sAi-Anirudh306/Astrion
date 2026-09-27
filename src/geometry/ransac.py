"""Affine RANSAC verification of ratio-test correspondences (no image warp)."""
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from src.matching.ratio_test import RatioMatch, RatioTestResult


@dataclass(frozen=True)
class RANSACConfig:
    """Pixel threshold; local seeded ordering avoids changing global RNG state."""
    reprojection_threshold: float = 3.0
    max_iterations: int = 2000
    confidence: float = 0.99
    refinement_iterations: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        for name, lower in (("max_iterations", 1), ("refinement_iterations", 0), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= 2147483647:
                raise ValueError(f"{name} must be an integer in [{lower}, 2147483647]")
        for name in ("reprojection_threshold", "confidence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.confidence >= 1:
            raise ValueError("confidence must be less than 1")


@dataclass(frozen=True)
class RANSACResult:
    """Matrix maps source (x,y,1) to destination (x,y); mask follows accepted order.

    Final inliers have forward Euclidean residual <= configured pixel threshold
    against the refined model. This reclassification may differ from OpenCV's
    pre-refinement mask. No further fitting is done after reclassification.
    """
    matrix: NDArray[np.float64]
    inlier_mask: NDArray[np.bool_]
    inliers: tuple[RatioMatch, ...]
    outliers: tuple[RatioMatch, ...]
    errors: NDArray[np.float64]
    inlier_error_statistics: dict[str, float]
    config: RANSACConfig
    runtime_seconds: float
    model: str = "affine_2d"

    @property
    def inlier_count(self) -> int:
        return len(self.inliers)

    @property
    def outlier_count(self) -> int:
        return len(self.outliers)

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / len(self.errors)


def _points(points: np.ndarray, name: str) -> NDArray[np.float64]:
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must be an (N, 2) coordinate array")
    if points.dtype.kind not in "fiu" or not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite real coordinates")
    if np.any(np.abs(points.astype(np.float64)) > 1e7):
        raise ValueError("Coordinates exceed supported pixel range (+/- 1e7)")
    return points.astype(np.float64, copy=True)


def _spans_plane(points: np.ndarray) -> bool:
    """Reject duplicate-only, collinear and numerically near-collinear support."""
    if len(points) < 3:
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return bool(singular[0] > 0 and singular[1] > 1e-6 * singular[0])


def verify_affine(
    matches: RatioTestResult, source_points: np.ndarray, destination_points: np.ndarray,
    config: RANSACConfig | None = None,
) -> RANSACResult:
    """Estimate a six-parameter affine model with cv2.estimateAffine2D/RANSAC.

    Coordinates can come directly from SIFTResult.points or RootSIFTResult.points.
    Invalid/insufficient/degenerate inputs raise ValueError; failed estimators
    raise RuntimeError, never return a fabricated identity model. Identical
    coordinate pairs are deduplicated for fitting, then mapped back to every
    original correspondence. Conflicting pairs are left for RANSAC to resolve.
    Local seeded input permutation provides repeatable runs on the same OpenCV
    build; exact cross-version reproducibility is not promised. No global RNG
    state or source arrays are modified. Refinement is affine least-squares,
    not subpixel image refinement.
    """
    started = perf_counter()
    config = RANSACConfig() if config is None else config
    if not isinstance(config, RANSACConfig) or not isinstance(matches, RatioTestResult):
        raise TypeError("Expected RANSACConfig and RatioTestResult")
    source, destination = _points(source_points, "source"), _points(destination_points, "destination")
    if not isinstance(matches.accepted, tuple):
        raise ValueError("accepted correspondences must be a tuple")
    indices = []
    for match in matches.accepted:
        if not isinstance(match, RatioMatch):
            raise ValueError("Expected RatioMatch correspondences")
        for index, length in ((match.query_index, len(source)), (match.train_index, len(destination))):
            if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < length:
                raise ValueError("Malformed correspondence index")
        indices.append((match.query_index, match.train_index))
    if len(indices) < 3:
        raise ValueError("Affine estimation needs at least three correspondences")
    qi, ti = np.asarray(indices).T
    a, b = source[qi], destination[ti]
    pairs = np.unique(np.column_stack((a, b)), axis=0)
    # OpenCV's estimator uses float32 internally; validate at that precision too.
    fit_a, fit_b = pairs[:, :2].astype(np.float32), pairs[:, 2:].astype(np.float32)
    if not _spans_plane(fit_a.astype(np.float64)) or not _spans_plane(fit_b.astype(np.float64)):
        raise ValueError("Degenerate or collinear affine point configuration")
    order = np.random.default_rng(config.seed).permutation(len(pairs))
    try:
        matrix, mask = cv2.estimateAffine2D(
            fit_a[order], fit_b[order], method=cv2.RANSAC,
            ransacReprojThreshold=float(config.reprojection_threshold),
            maxIters=int(config.max_iterations), confidence=float(config.confidence),
            refineIters=int(config.refinement_iterations))
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV affine RANSAC failed: {exc}") from exc
    if matrix is None or mask is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise RuntimeError("Affine RANSAC did not produce a finite model")
    if np.linalg.matrix_rank(matrix[:, :2]) < 2:
        raise RuntimeError("Affine RANSAC produced a singular model")
    errors = np.linalg.norm(a @ matrix[:, :2].T + matrix[:, 2] - b, axis=1)
    inlier_mask = errors <= config.reprojection_threshold
    if not _spans_plane(a[inlier_mask]) or not _spans_plane(b[inlier_mask]):
        raise RuntimeError("Affine RANSAC has insufficient nondegenerate inlier support")
    residuals = errors[inlier_mask]
    statistics = {"min": float(residuals.min()), "median": float(np.median(residuals)),
                  "mean": float(residuals.mean()), "max": float(residuals.max())}
    return RANSACResult(matrix.copy(), inlier_mask,
        tuple(m for m, keep in zip(matches.accepted, inlier_mask) if keep),
        tuple(m for m, keep in zip(matches.accepted, inlier_mask) if not keep),
        errors, statistics, config, perf_counter() - started)
