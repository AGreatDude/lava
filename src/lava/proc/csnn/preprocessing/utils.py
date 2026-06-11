"""Preprocessing utilities for CSNN MNIST experiments."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def ensure_spatial_tensor(sample: np.ndarray, *, dtype=np.float32) -> np.ndarray:
    """Return ``sample`` as ``(width, height, channels)`` float tensor."""

    dtype = np.dtype(dtype)
    arr = np.asarray(sample, dtype=dtype)
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    if arr.ndim == 3:
        return arr
    raise ValueError(
        "expected a 2D image or 3D spatial tensor, "
        f"got shape {arr.shape}")


def create_dog_filter(
        filter_size: int = 7,
        center_sigma: float = 1.0,
        surround_sigma: float = 4.0,
        *,
        dtype=np.float32) -> np.ndarray:
    """Create the normalized Difference-of-Gaussians filter used by CSNN."""

    dtype = np.dtype(dtype)
    scalar_type = dtype.type
    if filter_size <= 0:
        raise ValueError("filter_size must be positive")
    if center_sigma <= 0 or surround_sigma <= 0:
        raise ValueError("Gaussian sigmas must be positive")
    if filter_size % 2 == 0:
        filter_size += 1

    half = filter_size // 2
    axis = np.arange(-half, half + 1, dtype=dtype)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    distance_sq = xx * xx + yy * yy

    center = np.exp(-distance_sq / scalar_type(2.0 * center_sigma**2))
    surround = np.exp(-distance_sq / scalar_type(2.0 * surround_sigma**2))
    center = center / np.sum(center)
    surround = surround / np.sum(surround)
    return (center - surround).astype(dtype)


def default_on_off_filter(
        sample: np.ndarray,
        *,
        filter_size: int = 7,
        center_sigma: float = 1.0,
        surround_sigma: float = 4.0,
        dtype=np.float32) -> np.ndarray:
    """Apply CSNN default On/Off DoG preprocessing."""

    dtype = np.dtype(dtype)
    scalar_type = dtype.type
    img = ensure_spatial_tensor(sample, dtype=dtype)
    width, height, channels = img.shape
    kernel = create_dog_filter(
        filter_size, center_sigma, surround_sigma, dtype=dtype)
    k_width, k_height = kernel.shape
    half_w = k_width // 2
    half_h = k_height // 2

    out = np.zeros((width, height, channels * 2), dtype=dtype)
    for x in range(width):
        for y in range(height):
            for channel in range(channels):
                value = scalar_type(0.0)
                for fx in range(k_width):
                    x_in = min(max(x + fx - half_w, 0), width - 1)
                    for fy in range(k_height):
                        y_in = min(max(y + fy - half_h, 0), height - 1)
                        value += img[x_in, y_in, channel] * kernel[fx, fy]
                out[x, y, 2 * channel] = max(scalar_type(0.0), value)
                out[x, y, 2 * channel + 1] = max(scalar_type(0.0), -value)
    return out


class FeatureScaler:
    """Per-element train-set min/max scaler matching CSNN FeatureScaling."""

    def __init__(
            self,
            min_: Optional[np.ndarray] = None,
            max_: Optional[np.ndarray] = None,
            dtype: np.dtype = np.float32) -> None:
        self.min_ = min_
        self.max_ = max_
        self.dtype = np.dtype(dtype)

    def fit(self, samples: np.ndarray) -> "FeatureScaler":
        """Compute per-element min/max over a batch of training samples."""

        data = np.asarray(samples, dtype=self.dtype)
        if data.ndim < 2:
            raise ValueError("samples must include a batch dimension")
        self.min_ = np.min(data, axis=0).astype(self.dtype)
        self.max_ = np.max(data, axis=0).astype(self.dtype)
        return self

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Scale samples using fitted train-set statistics."""

        if self.min_ is None or self.max_ is None:
            raise RuntimeError("FeatureScaler must be fitted before transform")

        return feature_scale(samples, self.min_, self.max_, dtype=self.dtype)

    def fit_transform(self, samples: np.ndarray) -> np.ndarray:
        """Fit on training samples and return their scaled values."""

        return self.fit(samples).transform(samples)


def feature_scale(
        values: np.ndarray,
        min_values: np.ndarray,
        max_values: np.ndarray,
        *,
        dtype=np.float32) -> np.ndarray:
    """Scale values with per-element min/max arrays."""

    dtype = np.dtype(dtype)
    data = np.asarray(values, dtype=dtype)
    min_values = np.asarray(min_values, dtype=dtype)
    max_values = np.asarray(max_values, dtype=dtype)
    denom = max_values - min_values
    scaled = np.zeros_like(data, dtype=dtype)
    np.divide(data - min_values, denom, out=scaled, where=denom != 0)
    return scaled.astype(dtype, copy=False)


def feature_scale_train_test(
        train: np.ndarray,
        test: Optional[np.ndarray] = None,
        *,
        dtype=np.float32) -> Tuple[np.ndarray, Optional[np.ndarray], FeatureScaler]:
    """Fit train-set scaling and apply it to train and optional test data."""

    scaler = FeatureScaler(dtype=dtype)
    train_scaled = scaler.fit_transform(train)
    test_scaled = None if test is None else scaler.transform(test)
    return train_scaled, test_scaled, scaler


__all__ = [
    "FeatureScaler",
    "create_dog_filter",
    "default_on_off_filter",
    "ensure_spatial_tensor",
    "feature_scale",
    "feature_scale_train_test",
]
