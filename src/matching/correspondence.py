"""Per-level classical correspondence generation; no merging or geometric fitting."""
from dataclasses import dataclass, replace
from time import perf_counter

import cv2
import numpy as np

from src.features.sift import extract_sift, SIFTConfig
from src.features.rootsift import transform_sift, RootSIFTResult
from src.preprocessing.scale_normalization import ScalePyramid, ScaleResult
from src.matching.descriptor_matching import match_descriptors, MatchingResult
from src.matching.ratio_test import filter_ratio, RatioTestResult


@dataclass(frozen=True)
class ScaleCorrespondences:
    """Candidate identity is (level_index, query_index, train_index).

    KNN and ratio indices address unchanged feature rows. accepted_indices
    indexes ratio.accepted, exposing any validity-mask exclusions. All original
    points remain available through source_features and source_points_original.
    Reference coordinates always use the supplied reference image grid.
    """
    level_index: int
    level: ScaleResult
    source_features: RootSIFTResult
    reference_features: RootSIFTResult
    source_points_original: np.ndarray
    knn: MatchingResult
    ratio: RatioTestResult
    accepted_indices: np.ndarray
    valid_matches: RatioTestResult
    runtime_seconds: float

    @property
    def source(self) -> np.ndarray:
        return self.source_points_original[[m.query_index for m in self.valid_matches.accepted]]

    @property
    def destination(self) -> np.ndarray:
        return self.reference_features.points[[m.train_index for m in self.valid_matches.accepted]]


def _mask(mask: np.ndarray | None, shape: tuple) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    if not isinstance(mask, np.ndarray) or mask.shape != shape or mask.dtype != bool:
        raise ValueError('Validity mask must be boolean with image dimensions')
    return mask


def _membership(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Require all four neighboring pixel centers valid; no coordinate movement."""
    p = points.astype(float)
    valid = (p >= 0).all(axis=1) & (p <= np.array(mask.shape[::-1])-1).all(axis=1)
    indices = np.flatnonzero(valid)
    x, y = np.floor(p[valid]).astype(int).T
    xx, yy = np.minimum(x+1, mask.shape[1]-1), np.minimum(y+1, mask.shape[0]-1)
    valid[indices] &= mask[y, x] & mask[y, xx] & mask[yy, x] & mask[yy, xx]
    return valid


def match_scale_pyramid(pyramid: ScalePyramid, reference: np.ndarray,
                        sift_config: SIFTConfig | None = None, ratio_threshold: float = .75,
                        source_mask: np.ndarray | None = None,
                        reference_mask: np.ndarray | None = None) -> tuple[ScaleCorrespondences, ...]:
    """Reuse SIFT -> RootSIFT -> L2 KNN -> Lowe ratio independently per level.

    Reference features are computed once. No hidden scale ranking or merged
    support: duplicates across levels are retained in separate results.
    Optional masks reject ratio matches whose endpoints lack four valid pixel
    neighbors. Downsampled source mask requires full area support (within 1e-6).
    Invalid pixels must be finite in supplied processing images; masks do not
    prevent descriptors near boundaries from seeing the caller's fill values.
    """
    if not isinstance(pyramid, ScalePyramid) or not pyramid.levels:
        raise ValueError('Expected a nonempty ScalePyramid')
    shape = pyramid.levels[0].original_shape
    if shape is None or any(level.original_shape != shape for level in pyramid.levels):
        raise ValueError('Levels must refer to one known original grid')
    sm, rm = _mask(source_mask, shape), _mask(reference_mask, reference.shape)
    reference_features = transform_sift(extract_sift(reference, sift_config))
    reference_valid = _membership(reference_features.points, rm)
    records = []
    for index, level in enumerate(pyramid.levels):
        started = perf_counter()
        features = transform_sift(extract_sift(level.data, sift_config))
        original = level.scaled_to_original(features.points, pixel_footprint=True)
        knn = match_descriptors(features.descriptors, reference_features.descriptors, k=2)
        ratio = filter_ratio(knn, threshold=ratio_threshold)
        resized_mask = cv2.resize(sm.astype(np.float32), level.data.shape[::-1], interpolation=cv2.INTER_AREA) >= 1-1e-6
        source_valid = _membership(features.points, resized_mask)
        keep = np.array([source_valid[m.query_index] and reference_valid[m.train_index]
                         for m in ratio.accepted], dtype=bool)
        accepted = tuple(m for m, valid in zip(ratio.accepted, keep) if valid)
        filtered = replace(ratio, accepted=accepted, rejected_count=ratio.query_count-len(accepted))
        records.append(ScaleCorrespondences(index, level, features, reference_features, original,
            knn, ratio, np.flatnonzero(keep), filtered, perf_counter()-started))
    return tuple(records)
