"""ProcessModels for CSNN analysis statistics."""

from __future__ import annotations

import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.csnn.stats.process import (
    CSNNActivity,
    CSNNAnalysisMediator,
    CSNNCoherence,
)
from lava.proc.csnn.stats.utils import (
    activity_stats,
    activity_to_vector,
    coherence_stats,
    coherence_to_vector,
)


@implements(proc=CSNNActivity, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNActivityModel(PyLoihiProcessModel):
    """CPU ProcessModel for CSNN Activity stats."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    stats_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        self.stats_out.send(activity_to_vector(activity_stats(self.s_in.recv())))


@implements(proc=CSNNCoherence, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNCoherenceModel(PyLoihiProcessModel):
    """CPU ProcessModel for CSNN Coherence stats."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    stats_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self) -> None:
        _ = self.s_in.recv()
        self.stats_out.send(coherence_to_vector(coherence_stats(self.weights)))


@implements(proc=CSNNAnalysisMediator, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNAnalysisMediatorModel(PyLoihiProcessModel):
    """CPU ProcessModel for combining CSNN analysis vectors."""

    activity_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    coherence_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    svm_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    analysis_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.has_coherence = bool(proc_params["has_coherence"])
        self.has_svm = bool(proc_params["has_svm"])

    def run_spk(self) -> None:
        parts = [np.asarray(self.activity_in.recv(), dtype=np.float32).reshape(-1)]
        if self.has_coherence:
            parts.append(np.asarray(self.coherence_in.recv(), dtype=np.float32).reshape(-1))
        if self.has_svm:
            parts.append(np.asarray(self.svm_in.recv(), dtype=np.float32).reshape(-1))
        self.analysis_out.send(np.concatenate(parts).astype(np.float32, copy=False))


__all__ = [
    "PyCSNNActivityModel",
    "PyCSNNAnalysisMediatorModel",
    "PyCSNNCoherenceModel",
]
