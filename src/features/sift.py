"""OpenCV SIFT detection and description on normalized processing images."""

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SIFTConfig:
    """Baseline OpenCV parameters; nfeatures=0 retains all detected features.

    contrastThreshold is passed directly (OpenCV divides it by nOctaveLayers).
    A larger edgeThreshold rejects fewer edge-like responses. No tuning or
    automatic contrast adjustment is performed by this wrapper.
    """

    nfeatures: int = 0
    nOctaveLayers: int = 3
    contrastThreshold: float = 0.04
    edgeThreshold: float = 10.0
    sigma: float = 1.6
    enable_precise_upscale: bool = False

    def __post_init__(self) -> None:
        for name, minimum in (("nfeatures", 0), ("nOctaveLayers", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= 2147483647:
                raise ValueError(f"{name} must be an integer >= {minimum} within int32 range")
            object.__setattr__(self, name, int(value))
        for name in ("contrastThreshold", "edgeThreshold", "sigma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.enable_precise_upscale, bool):
            raise ValueError("enable_precise_upscale must be bool")


@dataclass(frozen=True)
class SIFTResult:
    """Aligned keypoints and descriptor rows in the supplied image coordinates.

    KeyPoint.pt is (x, y), with origin at the input's top-left. Size, angle,
    response and octave remain available on each OpenCV KeyPoint. Empty
    detections have descriptors shaped (0, 128), never None. Descriptors are
    unmodified OpenCV float32 SIFT descriptors, not RootSIFT or unit vectors.
    Runtime includes validation, uint8 preparation and detectAndCompute.
    """

    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: NDArray[np.float32]
    config: SIFTConfig
    image_shape: tuple[int, int]
    runtime_seconds: float
    opencv_version: str
    preparation: str = "round(255 * image) to uint8; no contrast rescaling"

    @property
    def points(self) -> NDArray[np.float32]:
        """Return an (N, 2) coordinate array aligned with descriptor rows."""
        return np.asarray([point.pt for point in self.keypoints], dtype=np.float32).reshape(-1, 2)


def prepare_sift_image(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Validate float32 [0, 1] grayscale input and return independent uint8.

    Rounding quantizes to 256 levels without stretching weak contrast. Raw
    uint16 and browse uint8 must first pass through preprocessing. Read-only
    and noncontiguous arrays are accepted; the source is never modified.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("SIFT input must be a NumPy array")
    if image.ndim != 2 or image.size == 0:
        raise ValueError("SIFT input must be a nonempty 2-D grayscale image")
    if image.dtype != np.float32:
        raise TypeError("SIFT input must be float32 preprocessing output in [0, 1]")
    if not np.isfinite(image).all():
        raise ValueError("SIFT input must contain only finite values")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("SIFT input must be in [0, 1]")
    scaled = np.multiply(image, np.float32(255))
    np.rint(scaled, out=scaled)
    return np.ascontiguousarray(scaled, dtype=np.uint8)


def extract_sift(image: NDArray[np.float32], config: SIFTConfig | None = None) -> SIFTResult:
    """Detect keypoints and compute 128-component SIFT descriptors.

    Images with a dimension below 8 pixels return an empty result under an
    explicit wrapper policy: they offer insufficient spatial support for this
    baseline. Constant/featureless images also return empty results. OpenCV
    failures raise RuntimeError rather than masquerading as empty detections.
    No resizing, cropping, file writes or global RNG/thread changes occur.
    """
    started = perf_counter()
    if config is None:
        config = SIFTConfig()
    if not isinstance(config, SIFTConfig):
        raise TypeError("config must be a SIFTConfig")
    prepared = prepare_sift_image(image)
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV installation does not provide SIFT_create")
    try:
        detector = cv2.SIFT_create(**asdict(config))
        if min(image.shape) < 8 or prepared.min() == prepared.max():
            keypoints, descriptors = (), None
        else:
            keypoints, descriptors = detector.detectAndCompute(prepared, None)
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV SIFT failed for image shape {image.shape}: {exc}") from exc
    keypoints = tuple(keypoints)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    if descriptors.shape != (len(keypoints), 128) or descriptors.dtype != np.float32:
        raise RuntimeError("OpenCV returned inconsistent SIFT keypoints/descriptors")
    return SIFTResult(keypoints, descriptors, config, image.shape,
                      perf_counter() - started, cv2.__version__)
