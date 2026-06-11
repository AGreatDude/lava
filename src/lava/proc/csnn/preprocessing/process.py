"""Lava process interfaces for CSNN preprocessing."""

from __future__ import annotations

import numpy as np

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class CSNNDefaultOnOffFilter(AbstractProcess):
    """Lava process for simulator-compatible DefaultOnOffFilter."""

    def __init__(
            self,
            *,
            input_shape: tuple[int, int, int] = (28, 28, 1),
            filter_size: int = 7,
            center_sigma: float = 1.0,
            surround_sigma: float = 4.0,
            name: str | None = None,
            log_config: LogConfig | None = None) -> None:
        input_shape = tuple(int(v) for v in input_shape)
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (width, height, channels)")
        output_shape = (input_shape[0], input_shape[1], input_shape[2] * 2)
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            filter_size=int(filter_size),
            center_sigma=float(center_sigma),
            surround_sigma=float(surround_sigma),
            name=name,
            log_config=log_config)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.s_in = InPort(shape=input_shape)
        self.s_out = OutPort(shape=output_shape)


class CSNNFeatureScaling(AbstractProcess):
    """Lava process for simulator-compatible per-element FeatureScaling."""

    def __init__(
            self,
            *,
            shape: tuple[int, ...],
            min_values: np.ndarray,
            max_values: np.ndarray,
            name: str | None = None,
            log_config: LogConfig | None = None) -> None:
        shape = tuple(int(v) for v in shape)
        min_values = np.asarray(min_values, dtype=np.float32).reshape(shape)
        max_values = np.asarray(max_values, dtype=np.float32).reshape(shape)
        super().__init__(shape=shape, name=name, log_config=log_config)
        self.shape = shape
        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.min_values = Var(shape=shape, init=min_values)
        self.max_values = Var(shape=shape, init=max_values)


__all__ = ["CSNNDefaultOnOffFilter", "CSNNFeatureScaling"]
