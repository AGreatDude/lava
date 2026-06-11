"""ProcessModels for CSNN latency coding."""

from __future__ import annotations

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.csnn.latency.process import CSNNLatencyCoding
from lava.proc.csnn.latency.utils import latency_code


@implements(proc=CSNNLatencyCoding, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNLatencyCodingModel(PyLoihiProcessModel):
    """CPU ProcessModel for CSNN latency coding."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.max_timestamp = proc_params.get("max_timestamp")

    def run_spk(self) -> None:
        self.s_out.send(latency_code(self.s_in.recv(), max_timestamp=self.max_timestamp))


__all__ = ["PyCSNNLatencyCodingModel"]
