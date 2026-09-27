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
    denominator is max(high-low, min_span*max(1, abs(low), abs(high))). The
    unit floor retains an absolute gain bound for normalized float inputs.
    Thus min_span limits contrast relative to observed signal,
    never relative to unused container capacity. This assumes a meaningful
    intensity zero; it is not invariant to additive offsets or a noise model.
    Constants, integer images spanning at most one count, and float images
    spanning at most epsilon*max(1, abs(min), abs(max)) map to zero. The
    one-count guard conservatively avoids stretching quantization toggles;
    it can suppress real binary structure. Collapsed percentile bounds fall back to min/max to avoid
    erasing sparse structure. No pixels are assumed to be nodata; callers
    must supply a valid image region. This is not radiometric calibration.
    """
    validate_image(image)
    validate_number("lower_percentile", lower_percentile, 0, 100)
    validate_number("upper_percentile", upper_percentile, 0, 100)
    validate_number("min_span", min_span, np.finfo(np.float32).eps, 1)
    if lower_percentile >= upper_percentile:
        raise ValueError("lower_percentile must be less than upper_percentile")
    minimum, maximum = float(image.min()), float(image.max())
    epsilon = float(np.finfo(np.float32).eps)
    constant_tolerance = 1.0 if image.dtype.kind == "u" else epsilon * max(1.0, abs(minimum), abs(maximum))
    if maximum - minimum <= constant_tolerance:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(image, [lower_percentile, upper_percentile])
    if high <= low:
        low, high = minimum, maximum
    denominator = max(high - low, min_span * max(1.0, abs(low), abs(high)))
    output = image.astype(np.float32, copy=True)
    output -= np.float32(low)
    np.clip(output, 0, np.float32(high - low), out=output)
    output /= np.float32(denominator)
    np.clip(output, 0, 1, out=output)
    return output
