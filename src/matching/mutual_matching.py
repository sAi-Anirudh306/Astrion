"""Reciprocal L2 nearest neighbors, separate from the unfiltered baseline."""
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
from .descriptor_matching import MatchCandidate, MatchingResult, match_descriptors
from .ratio_test import RatioTestResult, filter_ratio


@dataclass(frozen=True)
class MutualMatchingResult:
    """Original descriptor indices, forward distances, and reusable KNN data.

    Output is ordered by query index. Ties use the existing matcher's policy;
    repeatability is expected within the same OpenCV build. Reciprocity is
    descriptor consistency, not geometric verification.
    """
    accepted: tuple[MatchCandidate, ...]
    forward: MatchingResult
    reverse: MatchingResult
    runtime_seconds: float


def mutual_nearest_neighbors(query: np.ndarray, train: np.ndarray) -> MutualMatchingResult:
    """Keep A[i]->B[j] exactly when B[j]->A[i]; never mutate descriptors.

    Reuses the validated L2 matcher including empty/extreme-value handling.
    Forward k=2 also supplies the later ratio diagnostic; reverse k=1 suffices
    for reciprocity. A single train descriptor can still yield one mutual pair.
    """
    started = perf_counter()
    forward = match_descriptors(query, train, k=2)
    reverse = match_descriptors(train, query, k=1)
    accepted = tuple(group[0] for group in forward.neighbors if group and
                     reverse.neighbors[group[0].train_index] and
                     reverse.neighbors[group[0].train_index][0].train_index == group[0].query_index)
    return MutualMatchingResult(accepted, forward, reverse, perf_counter()-started)


def filter_mutual_ratio(matches: MutualMatchingResult, threshold: float = .75) -> RatioTestResult:
    """Apply forward Lowe ratio, then reciprocal consistency (no reverse ratio).

    Original ratio metadata is retained; rejected_count includes candidates
    rejected by either rule. No source results are modified.
    """
    started = perf_counter()
    if not isinstance(matches, MutualMatchingResult):
        raise TypeError("matches must be MutualMatchingResult")
    ratio = filter_ratio(matches.forward, threshold)
    reciprocal = {(m.query_index, m.train_index) for m in matches.accepted}
    accepted = tuple(m for m in ratio.accepted if (m.query_index, m.train_index) in reciprocal)
    return replace(ratio, accepted=accepted, rejected_count=ratio.query_count-len(accepted),
                   runtime_seconds=perf_counter()-started)
