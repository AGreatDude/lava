"""Lava process interface for CSNN latency coding."""

from __future__ import annotations

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig


class CSNNLatencyCoding(AbstractProcess):
    """Lava process for CSNN latency coding."""

    def __init__(
            self,
            *,
            shape: tuple[int, ...],
            max_timestamp: float | None = None,
            name: str | None = None,
            log_config: LogConfig | None = None) -> None:
        shape = tuple(int(v) for v in shape)
        super().__init__(
            shape=shape,
            max_timestamp=max_timestamp,
            name=name,
            log_config=log_config)
        self.shape = shape
        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)


__all__ = ["CSNNLatencyCoding"]
