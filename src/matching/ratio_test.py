"""Lowe ratio filtering of existing L2 nearest-neighbor groups."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from time import perf_counter

from .descriptor_matching import MatchCandidate, MatchingResult


@dataclass(frozen=True)
class RatioMatch:
    """Accepted descriptor indices remain aligned to source keypoint rows."""

    query_index: int
    train_index: int
    nearest_distance: float
    second_distance: float
    ratio: float


@dataclass(frozen=True)
class RatioTestResult:
    """Descriptor-only decisions; no geometric information is used."""

    accepted: tuple[RatioMatch, ...]
    query_count: int
    rejected_count: int
    missing_second_count: int
    zero_second_count: int
    threshold: float
    runtime_seconds: float


def _integer(value: object, minimum: int) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum


def filter_ratio(matches: MatchingResult, threshold: float = 0.75) -> RatioTestResult:
    """Accept nearest candidates only when d1/d2 < threshold (strict).

    Threshold must be finite and strictly between 0 and 1; 0.75 is a baseline,
    not a validated optimum for lunar imagery. Missing second neighbors and
    d2=0 (including 0/0 ambiguity) are rejected without division. d1=0,d2>0
    is accepted. Malformed groups, nonfinite/negative distances, invalid
    indices/ranks and unsorted neighbors raise ValueError, never get repaired.
    All supplied neighbors are validated; only the first two determine the
    decision. Empty groups are legitimate rejections. Inputs are not mutated.
    """
    started = perf_counter()
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and strictly between 0 and 1")
    if not isinstance(matches, MatchingResult):
        raise TypeError("matches must be a MatchingResult")
    if (not _integer(matches.query_count, 0) or not _integer(matches.train_count, 0)
            or not _integer(matches.k, 1) or not _integer(matches.descriptor_dimension, 1)
            or matches.metric != "L2" or matches.cross_check):
        raise ValueError("Invalid matcher metadata; requires unfiltered L2 neighbors")
    if not isinstance(matches.neighbors, tuple) or len(matches.neighbors) != matches.query_count:
        raise ValueError("Expected one neighbor tuple per query")
    accepted = []
    missing = zero = 0
    for qi, group in enumerate(matches.neighbors):
        if not isinstance(group, tuple) or len(group) > min(matches.k, matches.train_count):
            raise ValueError(f"Malformed candidate group for query {qi}")
        seen = set()
        previous = -1.0
        for rank, candidate in enumerate(group, 1):
            if not isinstance(candidate, MatchCandidate):
                raise ValueError("Groups must contain MatchCandidate objects")
            if (not _integer(candidate.query_index, 0) or candidate.query_index != qi
                    or not _integer(candidate.train_index, 0) or candidate.train_index >= matches.train_count
                    or candidate.train_index in seen or not _integer(candidate.neighbor_rank, 1)
                    or candidate.neighbor_rank != rank):
                raise ValueError(f"Invalid candidate index/rank for query {qi}")
            distance = candidate.distance
            if isinstance(distance, bool) or not isinstance(distance, Real) or not isfinite(distance) or distance < previous or distance < 0:
                raise ValueError("Distances must be finite, nonnegative and sorted")
            previous = distance
            seen.add(candidate.train_index)
        if len(group) < 2:
            missing += 1
            continue
        first, second = group[:2]
        if second.distance == 0:
            zero += 1
            continue
        ratio = float(first.distance) / float(second.distance)
        if ratio < threshold:
            accepted.append(RatioMatch(qi, first.train_index, float(first.distance),
                                       float(second.distance), ratio))
    return RatioTestResult(tuple(accepted), matches.query_count,
                           matches.query_count - len(accepted), missing, zero,
                           float(threshold), perf_counter() - started)
