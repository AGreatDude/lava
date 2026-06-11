"""CSNN event-driven convolution process with fixed parameters."""

from __future__ import annotations

import typing as ty

import numpy as np

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.proc.csnn.conv.utils import PairLike, _as_pair, compute_output_shape


class CSNNConvolution(AbstractProcess):
    """Fixed-parameter CSNN convolution process.

    This process performs inference. Input and output are full float spike-time
    tensors, so one Lava runtime step corresponds to one sample.
    """

    def __init__(
            self,
            *,
            weights: np.ndarray,
            thresholds: ty.Union[np.ndarray, float, int],
            input_shape: ty.Tuple[int, int, int],
            stride: ty.Optional[PairLike] = 1,
            padding: ty.Optional[PairLike] = 0,
            wta_infer: ty.Optional[bool] = False,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None) -> None:

        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim != 4:
            raise ValueError(
                "CSNNConvolution weights must have shape "
                "(filter_width, filter_height, input_depth, output_filters)")

        output_filters = weights.shape[3]
        thresholds = np.asarray(thresholds, dtype=np.float32)
        if thresholds.ndim == 0:
            thresholds = np.full(
                (output_filters,), float(thresholds), dtype=np.float32)
        elif thresholds.shape != (output_filters,):
            raise ValueError(
                "thresholds must be scalar or have shape (output_filters,), "
                f"got {thresholds.shape}")

        stride = _as_pair(stride, "stride")
        padding = _as_pair(padding, "padding")
        input_shape = tuple(int(v) for v in input_shape)
        output_shape = compute_output_shape(
            input_shape, weights.shape, stride=stride, padding=padding)
        kernel_size = weights.shape[:2]

        super().__init__(
            weights=weights,
            thresholds=thresholds,
            input_shape=input_shape,
            stride=stride,
            padding=padding,
            wta_infer=wta_infer,
            name=name,
            log_config=log_config)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.s_in = InPort(shape=input_shape)
        self.s_out = OutPort(shape=output_shape)

        self.weights = Var(shape=weights.shape, init=weights)
        self.thresholds = Var(shape=thresholds.shape, init=thresholds)
        self.kernel_size = Var(shape=(2,), init=kernel_size)
        self.stride = Var(shape=(2,), init=stride)
        self.padding = Var(shape=(2,), init=padding)
        self.wta_infer = Var(shape=(1,), init=int(bool(wta_infer)))


CSNNConv = CSNNConvolution

__all__ = ["CSNNConvolution", "CSNNConv"]
