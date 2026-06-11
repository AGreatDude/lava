"""CSNN analysis statistics helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActivityStats:
    """Activity statistics for a feature tensor.

    Parameters
    ----------
    sparsity : float
        Sparsity of the feature tensor.
    active_unit_percent : float
        Percentage of active units.
    quiet : bool
        True if all units are quiet (no spikes).
    """

    sparsity: float
    active_unit_percent: float
    quiet: bool


@dataclass(frozen=True)
class CoherenceStats:
    """Coherence statistics for a weight tensor.

    Parameters
    ----------
    mean_weights : float
        Mean of the weights.
    count : int
        Number of filters or weights.
    minimum : float, optional
        Minimum weight value.
    q1 : float, optional
        First quartile of weight values.
    q2 : float, optional
        Second quartile (median) of weight values.
    q3 : float, optional
        Third quartile of weight values.
    maximum : float, optional
        Maximum weight value.
    """

    mean_weights: float
    count: int
    minimum: float | None
    q1: float | None
    q2: float | None
    q3: float | None
    maximum: float | None


def activity_stats(sample: np.ndarray) -> ActivityStats:
    """Compute the simulator Activity metrics for one feature tensor."""

    data = np.asarray(sample, dtype=np.float32).reshape(-1)
    size = int(data.size)
    if size == 0:
        raise ValueError("activity sample must not be empty")

    l1 = float(np.abs(data).sum())
    l2 = float(np.square(data).sum())
    active = int(np.count_nonzero(data > 0.0))
    tmp = 1 if l2 == 0.0 else int(max(1.0, l1 / np.sqrt(l2)))
    if size <= 1:
        sparsity = 0.0
    else:
        sparsity = (np.sqrt(size) - tmp) / (np.sqrt(size) - 1.0)
    return ActivityStats(
        sparsity=float(sparsity),
        active_unit_percent=float(active / size * 100.0),
        quiet=active == 0,
    )


def coherence_stats(weights: np.ndarray) -> CoherenceStats:
    """Compute simulator-style pairwise filter cosine coherence."""

    kernel = np.asarray(weights, dtype=np.float32)
    if kernel.ndim != 4:
        raise ValueError(f"weights must be 4D, got shape {kernel.shape}")
    filters = int(kernel.shape[3])
    mean_weights = float(kernel.mean())
    if filters < 2:
        return CoherenceStats(mean_weights, 0, None, None, None, None, None)

    flat = kernel.reshape(-1, filters).T.astype(np.float64, copy=False)
    norms = np.linalg.norm(flat, axis=1)
    values: list[float] = []
    eps = np.finfo(np.float32).eps
    for i in range(filters):
        for j in range(i + 1, filters):
            denom = eps + norms[i] * norms[j]
            values.append(float(np.dot(flat[i], flat[j]) / denom))
    values.sort()

    count = len(values)
    return CoherenceStats(
        mean_weights=mean_weights,
        count=count,
        minimum=values[0],
        q1=values[min(count - 1, count // 4)],
        q2=values[min(count - 1, (count * 2) // 4)],
        q3=values[min(count - 1, (count * 3) // 4)],
        maximum=values[-1],
    )


def finite_spike_count(spike_times: np.ndarray) -> int:
    """Return the number of finite timestamps in spike_times.

    Parameters
    ----------
    spike_times : np.ndarray
        Array containing spike timestamps.

    Returns
    -------
    int
        Number of finite entries in the array.
    """
    return int(np.isfinite(spike_times).sum())


def activity_to_vector(stats: ActivityStats) -> np.ndarray:
    """Convert ActivityStats to a 1D float vector.

    Parameters
    ----------
    stats : ActivityStats
        Activity statistics to convert.

    Returns
    -------
    np.ndarray
        1D float array representing the activity stats.
    """
    return np.array(
        [stats.sparsity, stats.active_unit_percent, float(stats.quiet)],
        dtype=np.float32,
    )


def activity_from_vector(values: np.ndarray) -> ActivityStats:
    """Reconstruct ActivityStats from a 1D float vector.

    Parameters
    ----------
    values : np.ndarray
        1D float array to reconstruct from.

    Returns
    -------
    ActivityStats
        Reconstructed activity statistics.
    """
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return ActivityStats(
        sparsity=float(flat[0]),
        active_unit_percent=float(flat[1]),
        quiet=bool(flat[2] > 0.5),
    )


def coherence_to_vector(stats: CoherenceStats) -> np.ndarray:
    """Convert CoherenceStats to a 1D float vector.

    Parameters
    ----------
    stats : CoherenceStats
        Coherence statistics to convert.

    Returns
    -------
    np.ndarray
        1D float array representing the coherence stats.
    """
    values = [
        stats.mean_weights,
        float(stats.count),
        stats.minimum,
        stats.q1,
        stats.q2,
        stats.q3,
        stats.maximum,
    ]
    return np.array(
        [np.nan if value is None else value for value in values],
        dtype=np.float32,
    )


def coherence_from_vector(values: np.ndarray) -> CoherenceStats | None:
    """Reconstruct CoherenceStats from a 1D float vector.

    Parameters
    ----------
    values : np.ndarray
        1D float array to reconstruct from.

    Returns
    -------
    CoherenceStats or None
        Reconstructed coherence statistics, or None if invalid.
    """
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size != 7 or not np.isfinite(flat[1]) or flat[1] <= 0:
        return None
    return CoherenceStats(
        mean_weights=float(flat[0]),
        count=int(flat[1]),
        minimum=float(flat[2]),
        q1=float(flat[3]),
        q2=float(flat[4]),
        q3=float(flat[5]),
        maximum=float(flat[6]),
    )


__all__ = [
    "ActivityStats",
    "CoherenceStats",
    "activity_from_vector",
    "activity_stats",
    "activity_to_vector",
    "coherence_from_vector",
    "coherence_stats",
    "coherence_to_vector",
    "finite_spike_count",
]
