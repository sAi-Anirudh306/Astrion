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
    original_shape: tuple[int, int] | None = None
    source_resolution: float | None = None
    target_resolution: float | None = None

    @property
    def forward_matrix(self) -> NDArray[np.float64]:
        """Pixel-center mapping, original -> scaled; homogeneous 3x3 matrix."""
        fx, fy = self.actual_factors_xy
        return np.array([[fx, 0, (fx-1)/2], [0, fy, (fy-1)/2], [0, 0, 1.]])

    @property
    def inverse_matrix(self) -> NDArray[np.float64]:
        return np.linalg.inv(self.forward_matrix)

    def original_to_scaled(self, points: np.ndarray, *, pixel_footprint: bool = False) -> np.ndarray:
        """Map (x,y) centers with (p+.5)*actual_factor-.5; never clip.

        Default input bounds are [0,width-1] x [0,height-1]. A first pixel
        center can map to a negative fractional coordinate after downsampling.
        Use pixel_footprint=True for its inverse: explicit extended bounds
        [-.5,width-.5] x [-.5,height-.5] represent the full pixel footprint.
        These are continuous coordinates, not array indices.
        """
        if self.original_shape is None:
            raise ValueError('Original shape missing: construct level with normalize_resolution')
        p = _coordinate_points(points, self.original_shape, pixel_footprint)
        return (p+.5)*self.actual_factors_xy-.5

    def scaled_to_original(self, points: np.ndarray, *, pixel_footprint: bool = False) -> np.ndarray:
        """Inverse pixel-center mapping; see original_to_scaled for boundary rules."""
        p = _coordinate_points(points, self.data.shape, pixel_footprint)
        return (p+.5)/self.actual_factors_xy-.5

    def metadata(self) -> dict:
        return dict(original_shape=self.original_shape, scaled_shape=self.data.shape,
                    source_gsd=self.source_resolution, target_gsd=self.target_resolution,
                    effective_gsd_xy=self.effective_resolution_xy, factors_xy=self.actual_factors_xy,
                    original_to_scaled_matrix=self.forward_matrix.tolist(),
                    scaled_to_original_matrix=self.inverse_matrix.tolist(),
                    coordinate_convention='Pixel centers (x,y); (p+0.5)*actual_factor-0.5')


def _coordinate_points(points: np.ndarray, shape: tuple[int, int], footprint: bool) -> np.ndarray:
    if not isinstance(footprint, bool):
        raise TypeError('pixel_footprint must be boolean')
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 2 or p.dtype.kind not in 'fiu' or not np.isfinite(p).all():
        raise ValueError('Points must be finite real (N,2) coordinates')
    lower = -.5 if footprint else 0.
    upper = np.array(shape[::-1])-(.5 if footprint else 1.)
    if np.any(p < lower) or np.any(p > upper):
        raise ValueError('Points outside the selected image coordinate domain')
    return p.astype(np.float64, copy=True)


@dataclass(frozen=True)
class ScalePyramid:
    """Levels in first-request order, independently sampled from the original.

    Requests rounding to the same (height,width) share the first level; the
    request_to_level mapping preserves every requested GSD without duplicate
    images or feature extraction. No recursive resampling or merging of matches.
    """
    levels: tuple[ScaleResult, ...]
    requested_resolutions: tuple[float, ...]
    request_to_level: tuple[int, ...]


def build_scale_pyramid(image: NDArray[np.float32], source_resolution: float,
                        target_resolutions: tuple[float, ...] | list[float]) -> ScalePyramid:
    """Construct GSD levels without upsampling; retain float32 [0,1] contract."""
    if not isinstance(target_resolutions, (list, tuple)) or not target_resolutions:
        raise ValueError('target_resolutions must be a nonempty list or tuple')
    levels, lookup, mapping = [], {}, []
    for target in target_resolutions:
        level = normalize_resolution(image, source_resolution, target)
        shape = level.data.shape
        if shape not in lookup:
            lookup[shape] = len(levels)
            levels.append(level)
        mapping.append(lookup[shape])
    return ScalePyramid(tuple(levels), tuple(float(t) for t in target_resolutions), tuple(mapping))


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
                       tuple(float(source_resolution / f) for f in factors),
                       image.shape, float(source_resolution), float(target_resolution))
