"""Utility functions for fixed-parameter CSNN timestamp convolution."""

from __future__ import annotations

import typing as ty

import numpy as np

from lava.proc.csnn.latency.utils import INFINITE_TIME, finite_spike_events

PairLike = ty.Union[int, ty.Tuple[int, int]]


def _as_pair(value: PairLike | None, name: str) -> ty.Tuple[int, int]:
    """Convert value to a tuple of two integers.

    Parameters
    ----------
    value : PairLike or None
        Scalar integer or a pair of integers.
    name : str
        Name of parameter for error messages.

    Returns
    -------
    tuple of (int, int)
        A pair of integers.
    """
    if value is None:
        raise ValueError(f"{name} must be an int or a pair")

    if np.isscalar(value):
        value = int(value)
        return value, value
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or a pair")
    return int(value[0]), int(value[1])


def compute_output_shape(
    input_shape: ty.Tuple[int, int, int],
    kernel_shape: ty.Tuple[int, int, int, int],
    stride: PairLike = 1,
    padding: PairLike = 0,
) -> ty.Tuple[int, int, int]:
    """Return CSNN convolution output shape in ``(width, height, channels)``."""

    if len(input_shape) != 3:
        raise ValueError("input_shape must be (width, height, channels)")
    if len(kernel_shape) != 4:
        raise ValueError(
            "kernel_shape must be "
            "(filter_width, filter_height, input_depth, output_filters)"
        )

    in_w, in_h, in_depth = (int(v) for v in input_shape)
    filter_w, filter_h, kernel_depth, filters = (int(v) for v in kernel_shape)
    stride_w, stride_h = _as_pair(stride, "stride")
    pad_w, pad_h = _as_pair(padding, "padding")

    if in_w <= 0 or in_h <= 0 or in_depth <= 0:
        raise ValueError("input_shape dimensions must be positive")
    if filter_w <= 0 or filter_h <= 0 or kernel_depth <= 0 or filters <= 0:
        raise ValueError("kernel_shape dimensions must be positive")
    if in_depth != kernel_depth:
        raise ValueError(
            "input channel count must match kernel input depth, "
            f"got {in_depth} and {kernel_depth}"
        )
    if stride_w <= 0 or stride_h <= 0:
        raise ValueError("stride dimensions must be positive")
    if pad_w < 0 or pad_h < 0:
        raise ValueError("padding must be non-negative")
    if in_w + 2 * pad_w < filter_w or in_h + 2 * pad_h < filter_h:
        raise ValueError("kernel is larger than padded input")

    out_w = (in_w + 2 * pad_w - filter_w) // stride_w + 1
    out_h = (in_h + 2 * pad_h - filter_h) // stride_h + 1
    return out_w, out_h, filters


def affected_output_positions(
    x_in: int,
    y_in: int,
    *,
    output_shape: ty.Tuple[int, int, int],
    kernel_size: PairLike,
    stride: PairLike = 1,
    padding: PairLike = 0,
) -> ty.Iterator[ty.Tuple[int, int, int, int]]:
    """Yield output positions affected by one input spike.

    Yields ``(out_x, out_y, weight_x, weight_y)`` using the same coordinate
    mapping as ``Layer3D::forward`` in csnn-simulator.
    """

    out_w, out_h, _ = (int(v) for v in output_shape)
    filter_w, filter_h = _as_pair(kernel_size, "kernel_size")
    stride_w, stride_h = _as_pair(stride, "stride")
    pad_w, pad_h = _as_pair(padding, "padding")

    start_x = (
        (x_in + pad_w - (filter_w - stride_w)) // stride_w
        if x_in + pad_w >= filter_w - stride_w
        else 0
    )
    start_y = (
        (y_in + pad_h - (filter_h - stride_h)) // stride_h
        if y_in + pad_h >= filter_h - stride_h
        else 0
    )
    last_x = (x_in + pad_w) // stride_w
    last_y = (y_in + pad_h) // stride_h

    for out_x in range(start_x, min(last_x + 1, out_w)):
        weight_x = x_in + pad_w - out_x * stride_w
        if weight_x < 0 or weight_x >= filter_w:
            continue
        for out_y in range(start_y, min(last_y + 1, out_h)):
            weight_y = y_in + pad_h - out_y * stride_h
            if weight_y < 0 or weight_y >= filter_h:
                continue
            yield out_x, out_y, weight_x, weight_y


def csnn_convolve(
    spike_times: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
    *,
    stride: PairLike = 1,
    padding: PairLike = 0,
    wta_infer: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """Infer one CSNN convolution layer from fixed weights and thresholds.

    ``spike_times`` is a dense tensor of first-spike timestamps in
    ``(width, height, channels)`` order. Finite entries are processed in sorted
    timestamp order. The output is a dense timestamp tensor with one spike at
    most per output neuron and ``np.inf`` for no-spike entries.
    """

    dtype = np.dtype(dtype)
    input_times = np.asarray(spike_times, dtype=dtype)
    kernel = np.asarray(weights, dtype=np.float32)
    th = np.asarray(thresholds, dtype=np.float32)

    if input_times.ndim != 3:
        raise ValueError(
            "spike_times must have shape (width, height, channels), "
            f"got {input_times.shape}"
        )
    if kernel.ndim != 4:
        raise ValueError(
            "weights must have shape "
            "(filter_width, filter_height, input_depth, output_filters)"
        )

    out_filters = kernel.shape[3]
    if th.ndim == 0:
        th = np.full((out_filters,), float(th), dtype=np.float32)
    elif th.shape != (out_filters,):
        raise ValueError(
            "thresholds must be scalar or have shape (output_filters,), "
            f"got {th.shape}"
        )

    stride = _as_pair(stride, "stride")
    padding = _as_pair(padding, "padding")
    output_shape = compute_output_shape(
        input_times.shape, kernel.shape, stride=stride, padding=padding
    )
    out_w, out_h, _ = output_shape
    filter_w, filter_h = kernel.shape[:2]

    activation = np.zeros(output_shape, dtype=np.float32)
    has_spiked = np.zeros(output_shape, dtype=bool)
    wta_used = np.zeros((out_w, out_h), dtype=bool)
    output = np.full(output_shape, INFINITE_TIME, dtype=dtype)

    for event in finite_spike_events(input_times):
        if len(event.index) != 3:
            raise ValueError("spike event indices must be 3D")
        x_in, y_in, z_in = event.index
        for out_x, out_y, weight_x, weight_y in affected_output_positions(
            x_in,
            y_in,
            output_shape=output_shape,
            kernel_size=(filter_w, filter_h),
            stride=stride,
            padding=padding,
        ):
            if wta_infer and wta_used[out_x, out_y]:
                continue
            for out_filter in range(out_filters):
                if has_spiked[out_x, out_y, out_filter]:
                    continue
                activation[out_x, out_y, out_filter] += kernel[
                    weight_x, weight_y, z_in, out_filter
                ]
                if activation[out_x, out_y, out_filter] >= th[out_filter]:
                    output[out_x, out_y, out_filter] = event.time
                    has_spiked[out_x, out_y, out_filter] = True
                    if wta_infer:
                        wta_used[out_x, out_y] = True
                        break
    return output


__all__ = [
    "PairLike",
    "affected_output_positions",
    "compute_output_shape",
    "csnn_convolve",
]
