"""Optional small-scale smoothing of normalized processing arrays."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .normalization import validate_image, validate_number


def denoise(image: NDArray[np.float32], sigma: float = 0.0) -> NDArray[np.float32]:
    """Return a copy with optional Gaussian smoothing (sigma in pixels).

    Zero disables smoothing. Sigma is capped at 1 pixel to discourage heavy
    blur. Even mild smoothing can attenuate narrow features; enable only
    when appropriate to the noise. Reflected borders avoid zero-padding edges.
    """
    validate_image(image, unit_range=True)
    validate_number("denoise sigma", sigma, 0, 1)
    source = image.astype(np.float32, copy=True)
    if sigma == 0:
        return source
    output = cv2.GaussianBlur(source, (0, 0), sigmaX=float(sigma),
                              borderType=cv2.BORDER_REFLECT_101)
    return np.clip(output, 0, 1, out=output)
