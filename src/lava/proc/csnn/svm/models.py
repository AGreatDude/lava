"""ProcessModels for CSNN SVM classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.csnn.svm.process import CSNNSVMClassifier
from lava.proc.csnn.svm.utils import (
    empty_svm_output,
    load_svm_bundle,
    predict_svm_bundle,
)


@implements(proc=CSNNSVMClassifier, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNSVMClassifierModel(PyLoihiProcessModel):
    """CPU ProcessModel for SVM classification."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    pred_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.output_shape = tuple(int(v) for v in proc_params["output_shape"])
        self.output_size = int(np.prod(self.output_shape))
        self.bundle = None
        svm_path = proc_params.get("svm_path")
        if svm_path:
            path = Path(svm_path)
            if path.exists():
                self.bundle = load_svm_bundle(path)

    def _empty_output(self) -> np.ndarray:
        return empty_svm_output(self.output_shape)

    def run_spk(self) -> None:
        features = np.asarray(self.s_in.recv(), dtype=np.float32).reshape(-1)
        if self.bundle is None:
            self.pred_out.send(self._empty_output())
            return
        svm = self.bundle["svm"]
        expected = getattr(svm, "n_features_in_", None)
        if expected is not None and int(expected) != int(features.size):
            self.pred_out.send(self._empty_output())
            return
        prediction, decision = predict_svm_bundle(self.bundle, features)
        out = self._empty_output().reshape(-1)
        out[0] = float(prediction)
        if decision is not None:
            flat_decision = decision.reshape(-1)
            n = min(self.output_size - 1, flat_decision.size)
            out[1:1 + n] = flat_decision[:n]
        self.pred_out.send(out.reshape(self.output_shape))


__all__ = ["PyCSNNSVMClassifierModel"]
