"""Lava process interface for CSNN objective output conversion."""

from __future__ import annotations

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.proc.csnn.objectives.utils import T_OBJ


class CSNNTimeObjectiveOutput(AbstractProcess):
    """Lava process for simulator TimeObjectiveOutput."""

    def __init__(
            self,
            *,
            shape: tuple[int, ...],
            t_obj: float = T_OBJ,
            name: str | None = None,
            log_config: LogConfig | None = None) -> None:
        shape = tuple(int(v) for v in shape)
        super().__init__(shape=shape, t_obj=float(t_obj), name=name, log_config=log_config)
        self.shape = shape
        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)


__all__ = ["CSNNTimeObjectiveOutput"]
