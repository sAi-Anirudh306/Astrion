"""Configurable preprocessing, separate from ingestion and visualization."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.ingestion.image_loader import ImageData
from .denoising import denoise
from .illumination import normalize_illumination
from .normalization import normalize_intensity, validate_number


@dataclass(frozen=True)
class PreprocessingConfig:
    """Conservative defaults; spatial filter scales are measured in pixels.

    min_span is a fraction of the observed percentile-bound intensity scale,
    not the container dtype range; see normalize_intensity for flatness guards.
    """

    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    min_span: float = 0.01
    denoise_sigma: float = 0.0
    illumination_sigma: float = 16.0
    illumination_strength: float = 0.0

    def __post_init__(self) -> None:
        validate_number("lower_percentile", self.lower_percentile, 0, 100)
        validate_number("upper_percentile", self.upper_percentile, 0, 100)
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError("lower_percentile must be less than upper_percentile")
        validate_number("min_span", self.min_span, np.finfo(np.float32).eps, 1)
        validate_number("denoise sigma", self.denoise_sigma, 0, 1)
        validate_number("illumination sigma", self.illumination_sigma, 1, 256)
        validate_number("illumination strength", self.illumination_strength, 0, 0.5)


@dataclass(frozen=True)
class PreprocessingResult:
    """Derived pixels and reproducible operation settings; no source mutation."""

    data: NDArray[np.float32]
    config: PreprocessingConfig
    operations: tuple[str, ...]
    input_shape: tuple[int, ...]
    input_dtype: str
    input_range: tuple[float, float]
    output_range: tuple[float, float]
    runtime_seconds: float


def preprocess_image(
    image: NDArray[Any] | ImageData, config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Normalize, optionally denoise, then optionally correct illumination.

    Accepts ingestion ImageData or a grayscale array. Output retains the
    full input dimensions and is an independent float32 processing image in
    [0, 1], not scientific counts or a uint8 visualization. No file I/O occurs.
    """
    started = perf_counter()
    if config is None:
        config = PreprocessingConfig()
    if not isinstance(config, PreprocessingConfig):
        raise TypeError("config must be a PreprocessingConfig")
    source = image.data if isinstance(image, ImageData) else image
    output = normalize_intensity(source, config.lower_percentile,
                                 config.upper_percentile, config.min_span)
    operations = ["percentile_normalization"]
    if config.denoise_sigma > 0:
        output = denoise(output, config.denoise_sigma)
        operations.append("gaussian_denoising")
    if config.illumination_strength > 0:
        output = normalize_illumination(output, config.illumination_sigma,
                                        config.illumination_strength)
        operations.append("additive_illumination_normalization")
    return PreprocessingResult(
        output, config, tuple(operations), source.shape, str(source.dtype),
        (float(source.min()), float(source.max())),
        (float(output.min()), float(output.max())), perf_counter() - started,
    )
