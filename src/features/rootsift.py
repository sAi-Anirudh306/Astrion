"""RootSIFT transformation of existing SIFT descriptors; no detection logic."""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from .sift import SIFTConfig, SIFTResult


def rootsift_descriptors(descriptors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return sqrt(d / sum(d)) independently for each nonnegative SIFT row.

    Requires finite float32 (N, 128) input, including N=0. Zero rows stay
    zero. Float64 working arithmetic safely sums even maximum float32 values
    and normalizes subnormal nonzero rows without an epsilon that would bias
    their normalization. Square root precedes the float32 cast. Nonzero rows
    therefore have unit L2 norm to float32 precision. No extra L2 step occurs.
    Read-only/noncontiguous inputs are supported and never modified.
    """
    if not isinstance(descriptors, np.ndarray):
        raise TypeError("descriptors must be a NumPy array")
    if descriptors.ndim != 2 or descriptors.shape[1] != 128:
        raise ValueError("SIFT descriptors must have shape (N, 128)")
    if descriptors.dtype != np.float32:
        raise TypeError("SIFT descriptors must have dtype float32")
    if not np.isfinite(descriptors).all():
        raise ValueError("SIFT descriptors must contain only finite values")
    if np.any(descriptors < 0):
        raise ValueError("SIFT descriptors must be nonnegative")
    working = descriptors.astype(np.float64, copy=True)
    totals = working.sum(axis=1, keepdims=True)
    np.divide(working, totals, out=working, where=totals > 0)
    np.sqrt(working, out=working)
    # Subnormal output components may round during the required float32 cast.
    # This is expected precision loss, not a failed normalization.
    with np.errstate(under="ignore"):
        return working.astype(np.float32)


@dataclass(frozen=True)
class RootSIFTResult:
    """RootSIFT rows aligned to the unchanged SIFT keypoint order.

    KeyPoint objects are shared with the source result, never mutated here.
    runtime_seconds measures validation/transformation only, excluding SIFT.
    """

    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: NDArray[np.float32]
    sift_config: SIFTConfig
    image_shape: tuple[int, int]
    opencv_version: str
    runtime_seconds: float
    descriptor_type: str = "RootSIFT"
    transformation: str = "row L1 normalization followed by element-wise square root"
    zero_policy: str = "zero rows remain zero; no additive epsilon"

    @property
    def points(self) -> NDArray[np.float32]:
        """Return (N, 2) input-image coordinates aligned with descriptor rows."""
        return np.asarray([point.pt for point in self.keypoints], dtype=np.float32).reshape(-1, 2)


def transform_sift(result: SIFTResult) -> RootSIFTResult:
    """Transform an existing SIFT result without redetection or source edits."""
    started = perf_counter()
    if not isinstance(result, SIFTResult):
        raise TypeError("result must be a SIFTResult")
    descriptors = rootsift_descriptors(result.descriptors)
    if len(result.keypoints) != descriptors.shape[0]:
        raise ValueError("SIFT keypoint and descriptor counts must agree")
    return RootSIFTResult(result.keypoints, descriptors, result.config,
                          result.image_shape, result.opencv_version,
                          perf_counter() - started)
