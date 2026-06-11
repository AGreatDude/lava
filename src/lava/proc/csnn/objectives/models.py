"""ProcessModels for CSNN objective output conversion."""

from __future__ import annotations

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.csnn.objectives.process import CSNNTimeObjectiveOutput
from lava.proc.csnn.objectives.utils import time_objective_output


@implements(proc=CSNNTimeObjectiveOutput, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNTimeObjectiveOutputModel(PyLoihiProcessModel):
    """CPU ProcessModel for CSNN objective output conversion."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        """Initialize the objective output model."""
        super().__init__(proc_params)
        self.t_obj = float(proc_params["t_obj"])

    def run_spk(self) -> None:
        """Run the objective output model step."""
        self.s_out.send(
            time_objective_output(self.s_in.recv(), t_obj=self.t_obj)
        )


__all__ = ["PyCSNNTimeObjectiveOutputModel"]
