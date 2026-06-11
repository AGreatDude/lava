"""Pooling helpers for CSNN timestamp and feature tensors."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from lava.proc.csnn.latency.utils import INFINITE_TIME


def _as_pair(value: Union[int, Tuple[int, int]], name: str) -> Tuple[int, int]:
    if np.isscalar(value):
        value = int(value)
        return value, value
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or a pair")
    return int(value[0]), int(value[1])


def _validate_spatial_tensor(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(
            "expected a 3D tensor in (width, height, channels) order, "
            f"got shape {arr.shape}")
    return arr


def spike_pool2d(
        spike_times: np.ndarray,
        *,
        kernel_size: Union[int, Tuple[int, int]] = (2, 2),
        stride: Union[int, Tuple[int, int]] = (2, 2),
        padding: Union[int, Tuple[int, int]] = 0,
        dtype=np.float32) -> np.ndarray:
    """Pool CSNN spike-time tensors by keeping the first spike in each window."""

    data = _validate_spatial_tensor(spike_times).astype(dtype, copy=False)
    kernel_w, kernel_h = _as_pair(kernel_size, "kernel_size")
    stride_w, stride_h = _as_pair(stride, "stride")
    pad_w, pad_h = _as_pair(padding, "padding")

    if kernel_w <= 0 or kernel_h <= 0:
        raise ValueError("kernel dimensions must be positive")
    if stride_w <= 0 or stride_h <= 0:
        raise ValueError("stride dimensions must be positive")
    if pad_w < 0 or pad_h < 0:
        raise ValueError("padding must be non-negative")

    padded = np.pad(
        data,
        ((pad_w, pad_w), (pad_h, pad_h), (0, 0)),
        mode="constant",
        constant_values=INFINITE_TIME)
    padded_w, padded_h, channels = padded.shape

    if padded_w < kernel_w or padded_h < kernel_h:
        raise ValueError("kernel is larger than the padded input")

    out_w = (padded_w - kernel_w) // stride_w + 1
    out_h = (padded_h - kernel_h) // stride_h + 1
    out = np.full((out_w, out_h, channels), INFINITE_TIME, dtype=dtype)

    for x in range(out_w):
        x0 = x * stride_w
        x1 = x0 + kernel_w
        for y in range(out_h):
            y0 = y * stride_h
            y1 = y0 + kernel_h
            window = padded[x0:x1, y0:y1, :]
            out[x, y, :] = np.min(window, axis=(0, 1))
    return out


def sum_pool2d(
        features: np.ndarray,
        *,
        target_shape: Tuple[int, int],
        dtype=np.float32) -> np.ndarray:
    """Spatial sum pooling for post-conversion feature tensors."""

    data = _validate_spatial_tensor(features).astype(dtype, copy=False)
    target_w, target_h = int(target_shape[0]), int(target_shape[1])
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target dimensions must be positive")

    width, height, channels = data.shape
    out_w = min(target_w, width)
    out_h = min(target_h, height)
    filter_w = width // out_w
    filter_h = height // out_h
    if filter_w <= 0 or filter_h <= 0:
        raise ValueError("invalid target_shape for input shape")

    out = np.zeros((out_w, out_h, channels), dtype=dtype)
    for x in range(out_w):
        x0 = x * filter_w
        x1 = x0 + filter_w
        for y in range(out_h):
            y0 = y * filter_h
            y1 = y0 + filter_h
            out[x, y, :] = np.sum(data[x0:x1, y0:y1, :], axis=(0, 1))
    return out


def first_spike_pool2x2(spike_times: np.ndarray, *, dtype=np.float32) -> np.ndarray:
    """Convenience wrapper for CSNN ``Pooling(2, 2, 2, 2)``."""

    return spike_pool2d(
        spike_times,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=0,
        dtype=dtype)


__all__ = [
    "_as_pair",
    "first_spike_pool2x2",
    "spike_pool2d",
    "sum_pool2d",
]
