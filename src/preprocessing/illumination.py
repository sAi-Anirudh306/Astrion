"""Optional broad-scale additive illumination normalization."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .normalization import validate_image, validate_number


def normalize_illumination(
    image: NDArray[np.float32], sigma: float = 16.0, strength: float = 0.0,
) -> NDArray[np.float32]:
    """Subtract strength * (Gaussian background - global mean), then clip.

    Sigma is in pixels. Strength zero disables correction; values up to 0.5
    allow partial background removal without full high-pass filtering. This
    targets broad additive brightness variation, not physical photometric
    correction or shadow recovery. Choose sigma larger than structures of
    interest; inappropriate scales may alter crater profiles or cause halos.
    Constants remain constant. No local variance division or sharpening occurs.
    """
    validate_image(image, unit_range=True)
    validate_number("illumination sigma", sigma, 1, 256)
    validate_number("illumination strength", strength, 0, 0.5)
    output = image.astype(np.float32, copy=True)
    if strength == 0:
        return output
    background = cv2.GaussianBlur(output, (0, 0), sigmaX=float(sigma),
                                 borderType=cv2.BORDER_REFLECT_101)
    background -= np.float32(output.mean(dtype=np.float64))
    background *= np.float32(strength)
    output -= background
    return np.clip(output, 0, 1, out=output)
