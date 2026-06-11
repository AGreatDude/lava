"""Objective-output utilities for CSNN feature tensors."""

from __future__ import annotations

import numpy as np


T_OBJ = 0.75

def time_objective_output(
        spike_times: np.ndarray,
        *,
        t_obj: float = T_OBJ,
        dtype=np.float32) -> np.ndarray:
    """Convert timestamps with csnn-simulator ``TimeObjectiveOutput`` logic."""

    data = np.asarray(spike_times, dtype=dtype)
    finite = np.isfinite(data)
    out = np.zeros(data.shape, dtype=dtype)
    converted = 1.0 - (data[finite] - t_obj) / (1.0 - t_obj)
    out[finite] = np.clip(converted, 0.0, 1.0).astype(dtype, copy=False)
    return out


__all__ = ["T_OBJ", "time_objective_output"]
