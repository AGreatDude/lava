"""ProcessModels for CSNN pooling."""

from __future__ import annotations

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.csnn.pooling.process import CSNNSpikePooling, CSNNSumPooling
from lava.proc.csnn.pooling.utils import spike_pool2d, sum_pool2d


@implements(proc=CSNNSpikePooling, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNSpikePoolingModel(PyLoihiProcessModel):
    """CPU ProcessModel for first-spike spatial pooling."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        """Initialize the spike pooling model."""
        super().__init__(proc_params)
        self.kernel_size = tuple(int(v) for v in proc_params["kernel_size"])
        self.stride = tuple(int(v) for v in proc_params["stride"])
        self.padding = tuple(int(v) for v in proc_params["padding"])

    def run_spk(self) -> None:
        """Run the spike pooling simulation step."""
        self.s_out.send(
            spike_pool2d(
                self.s_in.recv(),
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        )


@implements(proc=CSNNSumPooling, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNSumPoolingModel(PyLoihiProcessModel):
    """CPU ProcessModel for CSNN feature sum pooling."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        """Initialize the sum pooling model."""
        super().__init__(proc_params)
        self.target_shape = tuple(int(v) for v in proc_params["target_shape"])

    def run_spk(self) -> None:
        """Run the sum pooling simulation step."""
        self.s_out.send(
            sum_pool2d(self.s_in.recv(), target_shape=self.target_shape)
        )


__all__ = ["PyCSNNSpikePoolingModel", "PyCSNNSumPoolingModel"]
