"""Lava process interfaces for CSNN analysis statistics."""

from __future__ import annotations

import numpy as np

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class CSNNActivity(AbstractProcess):
    """Lava process that emits Activity stats for one feature tensor.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input port.
    name : str, optional
        Name of the process.
    log_config : LogConfig, optional
        Logging configuration.
    """

    output_shape = (3,)

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        name: str | None = None,
        log_config: LogConfig | None = None,
    ) -> None:
        """Initialize the CSNNActivity process."""
        input_shape = tuple(int(v) for v in input_shape)
        super().__init__(
            input_shape=input_shape, name=name, log_config=log_config
        )
        self.input_shape = input_shape
        self.s_in = InPort(shape=input_shape)
        self.stats_out = OutPort(shape=self.output_shape)


class CSNNCoherence(AbstractProcess):
    """Lava process that emits Coherence stats for a layer's weights.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input port.
    weights : np.ndarray
        Layer weights to calculate coherence stats from.
    name : str, optional
        Name of the process.
    log_config : LogConfig, optional
        Logging configuration.
    """

    output_shape = (7,)

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        weights: np.ndarray,
        name: str | None = None,
        log_config: LogConfig | None = None,
    ) -> None:
        """Initialize the CSNNCoherence process."""
        input_shape = tuple(int(v) for v in input_shape)
        weights = np.asarray(weights, dtype=np.float32)
        super().__init__(
            input_shape=input_shape,
            weights_shape=weights.shape,
            name=name,
            log_config=log_config,
        )
        self.input_shape = input_shape
        self.s_in = InPort(shape=input_shape)
        self.stats_out = OutPort(shape=self.output_shape)
        self.weights = Var(shape=weights.shape, init=weights)


class CSNNAnalysisMediator(AbstractProcess):
    """Lava process that combines frontend analysis vectors for one layer.

    Parameters
    ----------
    has_coherence : bool
        Whether coherence inputs are present.
    has_svm : bool, optional
        Whether SVM inputs are present. Default is True.
    svm_shape : tuple of int, optional
        Shape of the SVM input port. Default is (11,).
    name : str, optional
        Name of the process.
    log_config : LogConfig, optional
        Logging configuration.
    """

    def __init__(
        self,
        *,
        has_coherence: bool,
        has_svm: bool = True,
        svm_shape: tuple[int, ...] = (11,),
        name: str | None = None,
        log_config: LogConfig | None = None,
    ) -> None:
        """Initialize the CSNNAnalysisMediator process."""
        svm_shape = tuple(int(v) for v in svm_shape)
        output_size = 3
        if has_coherence:
            output_size += 7
        if has_svm:
            output_size += int(np.prod(svm_shape))
        super().__init__(
            has_coherence=bool(has_coherence),
            has_svm=bool(has_svm),
            svm_shape=svm_shape,
            output_shape=(output_size,),
            name=name,
            log_config=log_config,
        )
        self.has_coherence = bool(has_coherence)
        self.has_svm = bool(has_svm)
        self.output_shape = (output_size,)
        self.activity_in = InPort(shape=(3,))
        self.coherence_in = InPort(shape=(7,))
        self.svm_in = InPort(shape=svm_shape)
        self.analysis_out = OutPort(shape=self.output_shape)


__all__ = ["CSNNActivity", "CSNNAnalysisMediator", "CSNNCoherence"]
