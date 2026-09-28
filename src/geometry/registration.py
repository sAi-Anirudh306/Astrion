"""Scientific affine resampling in zero-based (x=column, y=row) pixels."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    """Independent float32 image and boolean registered coverage mask."""
    image: np.ndarray
    valid_mask: np.ndarray


def _matrix(matrix: np.ndarray) -> np.ndarray:
    affine = np.asarray(matrix, dtype=np.float64)
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        raise ValueError("Affine transform must be a finite 2x3 matrix")
    if np.linalg.cond(affine[:, :2]) > 1e12:
        raise ValueError("Affine linear component is singular or numerically unstable")
    return affine


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply the forward source-to-destination affine to finite N x 2 points."""
    affine = _matrix(matrix)
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Points must be finite N x 2 coordinates")
    return xy @ affine[:, :2].T + affine[:, 2]


def warp_affine(moving: np.ndarray, matrix: np.ndarray,
                output_shape: tuple[int, int], interpolation: str = "bilinear",
                nodata: float = np.nan,
                validity_mask: np.ndarray | None = None) -> RegistrationResult:
    """Warp scientific float32 counts using a FORWARD source -> destination matrix.

    output_shape is (height, width). OpenCV internally inverts the forward
    matrix: WARP_INVERSE_MAP is deliberately absent. Mask membership uses
    nearest-neighbor sampling only. Bilinear intensity contributions from
    nodata are excluded and remaining weights renormalized, avoiding artificial
    dark borders. The separate weight-support image is not a coverage mask.
    Coverage outside the nearest-sampled source mask remains nodata.
    No input arrays are modified. This performs no geometric estimation.
    """
    affine = _matrix(matrix)
    if not isinstance(moving, np.ndarray) or moving.dtype != np.float32:
        raise TypeError("Moving scientific image must be a float32 array")
    if moving.ndim != 2 or not moving.size:
        raise ValueError("Moving image must be a nonempty 2D array")
    if (len(output_shape) != 2 or any(isinstance(n, (bool, np.bool_)) or
            not isinstance(n, (int, np.integer)) or n <= 0 for n in output_shape)):
        raise ValueError("Output shape must contain positive integer height and width")
    if interpolation not in ("bilinear", "nearest"):
        raise ValueError("Interpolation must be bilinear or nearest")
    if not np.isscalar(nodata) or np.isinf(nodata) or (np.isfinite(nodata) and abs(nodata) > np.finfo(np.float32).max):
        raise ValueError("Nodata must be NaN or a finite float32 value")
    valid = np.isfinite(moving) & (moving != nodata)
    if validity_mask is not None:
        mask = np.asarray(validity_mask)
        if mask.shape != moving.shape or not np.isin(mask, [0, 1, 255]).all():
            raise ValueError("Validity mask must be binary and match image dimensions")
        valid &= mask.astype(bool)
    size = (int(output_shape[1]), int(output_shape[0]))
    mode = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
    coverage = cv2.warpAffine(valid.astype(np.uint8), affine, size,
                              flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
    support = cv2.warpAffine(valid.astype(np.float32), affine, size, flags=mode, borderValue=0)
    values = cv2.warpAffine(np.where(valid, moving, 0), affine, size, flags=mode, borderValue=0)
    coverage &= support > 0
    result = np.full(output_shape, nodata, dtype=np.float32)
    np.divide(values, support, out=result, where=coverage)
    if not np.isfinite(result[coverage]).all():
        raise ValueError("Intensity interpolation overflowed in valid coverage")
    return RegistrationResult(result, coverage)
