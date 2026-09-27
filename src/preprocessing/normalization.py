"""Intensity normalization for derived grayscale processing arrays."""

from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray


def validate_image(image: NDArray[Any], *, unit_range: bool = False) -> None:
    """Reject empty, non-grayscale, unsupported or nonfinite pixel arrays.

    Floating inputs must already be in [0, 1]. Unsigned 8/16-bit inputs
    retain their full sensor storage range until normalization.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a nonempty 2-D grayscale array")
    if not ((image.dtype.kind == "u" and image.dtype.itemsize in (1, 2))
            or (image.dtype.kind == "f" and image.dtype.itemsize in (4, 8))):
        raise TypeError("image dtype must be uint8, uint16, float32 or float64")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    if unit_range or image.dtype.kind == "f":
        if image.min() < 0 or image.max() > 1:
            raise ValueError("processing image must be in [0, 1]")


def validate_number(name: str, value: float, lower: float, upper: float) -> None:
    """Validate a finite scalar parameter with inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")


def normalize_intensity(
    image: NDArray[Any], lower_percentile: float = 1.0,
    upper_percentile: float = 99.0, min_span: float = 0.01,
) -> NDArray[np.float32]:
    """Return an independent float32 array in [0, 1] using percentile clipping.

    Bounds use all pixels and NumPy's linear percentile interpolation. The
    denominator is floored at min_span times the dtype's full range (1 for
    float inputs), limiting amplification of low contrast/noise. Constants
    map to zero. Collapsed percentile bounds fall back to min/max to avoid
    erasing sparse structure. No pixels are assumed to be nodata; callers
    must supply a valid image region. This is not radiometric calibration.
    """
    validate_image(image)
    validate_number("lower_percentile", lower_percentile, 0, 100)
    validate_number("upper_percentile", upper_percentile, 0, 100)
    validate_number("min_span", min_span, np.finfo(np.float32).eps, 1)
    if lower_percentile >= upper_percentile:
        raise ValueError("lower_percentile must be less than upper_percentile")
    low, high = np.percentile(image, [lower_percentile, upper_percentile])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    full_range = float(np.iinfo(image.dtype).max) if image.dtype.kind == "u" else 1.0
    output = image.astype(np.float32, copy=True)
    output -= np.float32(low)
    np.clip(output, 0, np.float32(high - low), out=output)
    output /= np.float32(max(high - low, min_span * full_range))
    np.clip(output, 0, 1, out=output)
    return output
