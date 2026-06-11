"""ProcessModels for CSNN convolution."""

from __future__ import annotations

import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.csnn.conv.process import CSNNConvolution
from lava.proc.csnn.conv.utils import csnn_convolve


@implements(proc=CSNNConvolution, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNConvolutionModel(PyLoihiProcessModel):
    """CPU model for fixed-parameter CSNN timestamp convolution."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    thresholds: np.ndarray = LavaPyType(np.ndarray, float)
    kernel_size: np.ndarray = LavaPyType(np.ndarray, np.int32)
    stride: np.ndarray = LavaPyType(np.ndarray, np.int32)
    padding: np.ndarray = LavaPyType(np.ndarray, np.int32)
    wta_infer: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def run_spk(self) -> None:
        """Run the CSNN convolution step by processing incoming spike times."""
        input_times = self.s_in.recv()
        output_times = csnn_convolve(
            input_times,
            self.weights,
            self.thresholds,
            stride=tuple(int(v) for v in np.asarray(self.stride).flat),
            padding=tuple(int(v) for v in np.asarray(self.padding).flat),
            wta_infer=bool(np.asarray(self.wta_infer).item()),
            dtype=np.float32,
        )
        self.s_out.send(output_times)


__all__ = ["PyCSNNConvolutionModel"]
