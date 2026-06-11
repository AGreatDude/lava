"""Lava process interface for CSNN objective output conversion."""

from __future__ import annotations

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.proc.csnn.objectives.utils import T_OBJ


class CSNNTimeObjectiveOutput(AbstractProcess):
    """Lava process for simulator TimeObjectiveOutput.

    Parameters
    ----------
    shape : tuple of int
        Shape of the input and output ports.
    t_obj : float, optional
        Time objective target for the conversion. Default is T_OBJ.
    name : str, optional
        Name of the process.
    log_config : LogConfig, optional
        Logging configuration.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        t_obj: float = T_OBJ,
        name: str | None = None,
        log_config: LogConfig | None = None,
    ) -> None:
        """Initialize the CSNNTimeObjectiveOutput process."""
        shape = tuple(int(v) for v in shape)
        super().__init__(
            shape=shape, t_obj=float(t_obj), name=name, log_config=log_config
        )
        self.shape = shape
        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)


__all__ = ["CSNNTimeObjectiveOutput"]
