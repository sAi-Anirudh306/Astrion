"""Unfiltered Euclidean k-nearest descriptor candidates."""

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MatchCandidate:
    """Indices address original descriptor rows and corresponding keypoints."""

    query_index: int
    train_index: int
    distance: float
    neighbor_rank: int  # One-based: 1 is nearest, 2 is second-nearest.


@dataclass(frozen=True)
class MatchingResult:
    """One neighbor tuple per query, including empty tuples for empty train.

    Candidates are not filtered or geometrically verified. Distances are L2,
    not squared L2. Runtime includes validation and result construction.
    """

    neighbors: tuple[tuple[MatchCandidate, ...], ...]
    query_count: int
    train_count: int
    descriptor_dimension: int
    k: int
    backend: str
    runtime_seconds: float
    metric: str = "L2"
    cross_check: bool = False


def _validate(descriptors: NDArray[np.float32], name: str) -> None:
    """Validate arrays even when either side contains no descriptors."""
    if not isinstance(descriptors, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if descriptors.ndim != 2 or descriptors.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, D) with D > 0")
    if descriptors.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if not np.isfinite(descriptors).all():
        raise ValueError(f"{name} must contain only finite values")


def match_descriptors(
    query: NDArray[np.float32], train: NDArray[np.float32], k: int = 2,
) -> MatchingResult:
    """Return up to k nearest train rows for each query, without filtering.

    Uses OpenCV BFMatcher(NORM_L2, crossCheck=False). Finite values outside
    a conservative float32 arithmetic envelope use direct float64 differences
    (one query at a time), avoiding overflow, underflow and cancellation from
    the squared-norm identity. No descriptor normalization occurs here.

    Returned neighbors sort by (distance, train index). At a tied k-boundary,
    OpenCV chooses the retained subset; repeatability is expected within the
    same build, not guaranteed across builds. Float64 fallback breaks all ties
    by train index. Empty query returns (); empty train returns () per query.
    Inputs may be read-only or noncontiguous and are never modified.
    """
    started = perf_counter()
    _validate(query, "query")
    _validate(train, "train")
    if query.shape[1] != train.shape[1]:
        raise ValueError("query/train descriptor dimensionality mismatch")
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    count = min(k, len(train))
    rows: list[tuple[MatchCandidate, ...]] = []
    backend = "OpenCV BFMatcher"
    if len(query) == 0 or count == 0:
        rows = [() for _ in range(len(query))]
    else:
        # Bound squared differences by float32 max with substantial headroom.
        safe_max = np.sqrt(np.finfo(np.float32).max / (16 * query.shape[1]))
        safe_min = np.sqrt(np.finfo(np.float32).tiny)
        extreme = False
        for array in (query, train):
            absolute = np.abs(array)
            extreme |= bool(np.any(absolute > safe_max) or np.any((absolute > 0) & (absolute < safe_min)))
        if extreme:
            backend = "NumPy float64 brute force"
            train64 = train.astype(np.float64)
            for qi, descriptor in enumerate(query):
                delta = train64 - descriptor.astype(np.float64)
                distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
                indices = np.argsort(distances, kind="stable")[:count]
                rows.append(tuple(MatchCandidate(qi, int(ti), float(distances[ti]), rank)
                                  for rank, ti in enumerate(indices, 1)))
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = matcher.knnMatch(np.ascontiguousarray(query), np.ascontiguousarray(train), k=count)
            for qi, neighbors in enumerate(matches):
                ordered = sorted(neighbors, key=lambda match: (match.distance, match.trainIdx))
                rows.append(tuple(MatchCandidate(qi, match.trainIdx, float(match.distance), rank)
                                  for rank, match in enumerate(ordered, 1)))
            if len(rows) != len(query) or any(len(row) != count for row in rows):
                raise RuntimeError("OpenCV returned incomplete nearest-neighbor results")
    return MatchingResult(tuple(rows), len(query), len(train), query.shape[1], k,
                          backend, perf_counter() - started)
