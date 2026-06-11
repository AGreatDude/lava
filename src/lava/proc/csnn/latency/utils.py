"""Latency coding helpers for CSNN timestamp tensors."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


INFINITE_TIME = np.inf


class SpikeEvent:
    """Finite spike event extracted from a spike-time tensor."""

    __slots__ = ("time", "index")

    def __init__(self, time: float, index: Tuple[int, ...]) -> None:
        self.time = float(time)
        self.index = tuple(index)

    def __lt__(self, other: "SpikeEvent") -> bool:
        return (self.time, self.index) < (other.time, other.index)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpikeEvent):
            return NotImplemented
        return self.time == other.time and self.index == other.index

    def __repr__(self) -> str:
        return f"SpikeEvent(time={self.time!r}, index={self.index!r})"


def latency_code(
        values: np.ndarray,
        *,
        max_timestamp: Optional[float] = None,
        dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert normalized intensities to CSNN first-spike timestamps."""

    dtype = np.dtype(dtype)
    scalar_type = dtype.type
    arr = np.asarray(values, dtype=dtype)
    timestamps = np.maximum(scalar_type(0.0), scalar_type(1.0) - arr)
    no_spike = timestamps == scalar_type(1.0)
    if max_timestamp is not None and max_timestamp > 0:
        no_spike = np.logical_or(no_spike, timestamps > max_timestamp)
    return np.where(no_spike, INFINITE_TIME, timestamps).astype(dtype)


def finite_spike_events(spike_times: np.ndarray) -> list[SpikeEvent]:
    """Return finite timestamp entries sorted by time, then coordinates."""

    times = np.asarray(spike_times)
    finite_indices = np.argwhere(np.isfinite(times))
    events = [
        SpikeEvent(float(times[tuple(index)]), tuple(int(i) for i in index))
        for index in finite_indices
    ]
    events.sort()
    return events


def iter_finite_spike_events(spike_times: np.ndarray) -> Iterator[SpikeEvent]:
    """Yield finite timestamp entries sorted by time, then coordinates."""

    yield from finite_spike_events(spike_times)


def spike_times_to_raster(
        spike_times: np.ndarray,
        *,
        time_steps: int,
        dtype: np.dtype = np.bool_) -> np.ndarray:
    """Convert timestamp tensors to a diagnostic boolean raster."""

    if time_steps <= 0:
        raise ValueError("time_steps must be positive")

    times = np.asarray(spike_times, dtype=np.float64)
    raster = np.zeros(times.shape + (time_steps,), dtype=dtype)
    finite = np.isfinite(times)
    if not np.any(finite):
        return raster

    steps = np.rint(times[finite] * (time_steps - 1)).astype(np.int64)
    valid = np.logical_and(steps >= 0, steps < time_steps)
    coords = np.argwhere(finite)[valid]
    steps = steps[valid]
    for coord, step in zip(coords, steps):
        raster[tuple(coord) + (int(step),)] = True
    return raster


def raster_to_first_spike_times(
        raster: np.ndarray,
        *,
        dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert a diagnostic raster back to normalized first-spike times."""

    data = np.asarray(raster)
    if data.ndim == 0:
        raise ValueError("raster must have at least one time dimension")

    dtype = np.dtype(dtype)
    scalar_type = dtype.type
    time_steps = data.shape[-1]
    scale = 1.0 if time_steps <= 1 else float(time_steps - 1)

    has_spike = np.any(data, axis=-1)
    first_step = np.argmax(data, axis=-1).astype(dtype)
    times = first_step / scalar_type(scale)
    return np.where(has_spike, times, INFINITE_TIME).astype(dtype)


__all__ = [
    "INFINITE_TIME",
    "SpikeEvent",
    "finite_spike_events",
    "iter_finite_spike_events",
    "latency_code",
    "raster_to_first_spike_times",
    "spike_times_to_raster",
]
