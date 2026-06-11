"""Lava process interfaces for CSNN pooling."""

from __future__ import annotations

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.proc.csnn.pooling.utils import _as_pair


class CSNNSpikePooling(AbstractProcess):
    """Lava process for first-spike spatial pooling.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input as `(width, height, channels)`.
    kernel_size : int or tuple of int, optional
        Size of the pooling window. Default is (2, 2).
    stride : int or tuple of int, optional
        Stride of the pooling window. Default is (2, 2).
    padding : int or tuple of int, optional
        Zero-padding added to both sides of the input. Default is 0.
    name : str, optional
        Name of the process.
    log_config : LogConfig, optional
        Logging configuration.
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        kernel_size: int | tuple[int, int] = (2, 2),
        stride: int | tuple[int, int] = (2, 2),
        padding: int | tuple[int, int] = 0,
        name: str | None = None,
        log_config: LogConfig | None = None,
    ) -> None:
        """Initialize the CSNNSpikePooling process."""
        input_shape = tuple(int(v) for v in input_shape)
        kernel_size = _as_pair(kernel_size, "kernel_size")
        stride = _as_pair(stride, "stride")
        padding = _as_pair(padding, "padding")
        padded_w = input_shape[0] + 2 * padding[0]
        padded_h = input_shape[1] + 2 * padding[1]
        output_shape = (
            (padded_w - kernel_size[0]) // stride[0] + 1,
            (padded_h - kernel_size[1]) // stride[1] + 1,
            input_shape[2],
        )
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            name=name,
            log_config=log_config,
        )
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.s_in = InPort(shape=input_shape)
        self.s_out = OutPort(shape=output_shape)


class CSNNSumPooling(AbstractProcess):
    """Lava process for simulator output SumPooling target size.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input as `(width, height, channels)`.
    target_shape : tuple of int
        Desired target shape as `(width, height)`.
    name : str, optional
        Name of the process.
    log_config : LogConfig, optional
        Logging configuration.
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        target_shape: tuple[int, int],
        name: str | None = None,
        log_config: LogConfig | None = None,
    ) -> None:
        """Initialize the CSNNSumPooling process."""
        input_shape = tuple(int(v) for v in input_shape)
        target_shape = tuple(int(v) for v in target_shape)
        output_shape = (
            min(target_shape[0], input_shape[0]),
            min(target_shape[1], input_shape[1]),
            input_shape[2],
        )
        super().__init__(
            input_shape=input_shape,
            target_shape=target_shape,
            output_shape=output_shape,
            name=name,
            log_config=log_config,
        )
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.s_in = InPort(shape=input_shape)
        self.s_out = OutPort(shape=output_shape)


__all__ = ["CSNNSpikePooling", "CSNNSumPooling"]
