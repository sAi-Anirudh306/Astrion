"""Physical-resolution downsampling of derived processing images."""
from dataclasses import dataclass
from numbers import Real

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScaleResult:
    """Rounded grid dimensions can give slightly different x/y resolutions."""
    data: NDArray[np.float32]
    nominal_factor: float
    actual_factors_xy: tuple[float, float]
    effective_resolution_xy: tuple[float, float]


def normalize_resolution(image: NDArray[np.float32], source_resolution: float,
                         target_resolution: float) -> ScaleResult:
    """Area-average float32 [0,1] pixels to a coarser nominal physical grid.

    Assumes nominal square source pixels; this is not map reprojection or
    correction for spatially varying ground sampling distance. Dimensions use
    nearest integer rounding. No upsampling or source mutation is allowed.
    Pixel-center coordinates map back as (p + .5) / actual_factor - .5.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise TypeError("image must be a float32 NumPy array")
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be nonempty and two-dimensional")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image must be finite and in [0,1]")
    for value in (source_resolution, target_resolution):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError("resolutions must be finite positive numbers")
    if target_resolution < source_resolution:
        raise ValueError("target resolution must not require upsampling")
    factor = float(source_resolution / target_resolution)
    height, width = image.shape
    out_h, out_w = (int(np.floor(n * factor + .5)) for n in (height, width))
    if min(out_h, out_w) < 1:
        raise ValueError("target resolution would produce an empty dimension")
    output = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    np.clip(output, 0, 1, out=output)
    factors = (out_w / width, out_h / height)
    return ScaleResult(output, factor, factors,
                       tuple(float(source_resolution / f) for f in factors))
