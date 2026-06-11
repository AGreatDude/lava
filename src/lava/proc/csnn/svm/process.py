"""Lava process interface for CSNN SVM classification."""

from __future__ import annotations

from pathlib import Path

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig


class CSNNSVMClassifier(AbstractProcess):
    """Lava process that emits an SVM prediction and decision values.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input port.
    svm_path : str or Path or None
        Path to the serialized SVM model package.
    output_shape : tuple of int, optional
        Shape of the output port. Default is (11,).
    name : str, optional
        Name of the process.
    log_config : LogConfig, optional
        Logging configuration.
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        svm_path: str | Path | None,
        output_shape: tuple[int, ...] = (11,),
        name: str | None = None,
        log_config: LogConfig | None = None,
    ) -> None:
        """Initialize the CSNNSVMClassifier process."""
        input_shape = tuple(int(v) for v in input_shape)
        output_shape = tuple(int(v) for v in output_shape)
        super().__init__(
            input_shape=input_shape,
            svm_path=None if svm_path is None else str(svm_path),
            output_shape=output_shape,
            name=name,
            log_config=log_config,
        )
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.s_in = InPort(shape=input_shape)
        self.pred_out = OutPort(shape=output_shape)


__all__ = ["CSNNSVMClassifier"]
