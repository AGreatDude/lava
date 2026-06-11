"""ProcessModels for CSNN preprocessing."""

from __future__ import annotations

import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.csnn.preprocessing.process import (
    CSNNDefaultOnOffFilter,
    CSNNFeatureScaling,
)
from lava.proc.csnn.preprocessing.utils import (
    default_on_off_filter,
    feature_scale,
)


@implements(proc=CSNNDefaultOnOffFilter, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNDefaultOnOffFilterModel(PyLoihiProcessModel):
    """CPU ProcessModel for CSNN default on/off filtering."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        """Initialize the default on/off filter model."""
        super().__init__(proc_params)
        self.filter_size = int(proc_params["filter_size"])
        self.center_sigma = float(proc_params["center_sigma"])
        self.surround_sigma = float(proc_params["surround_sigma"])

    def run_spk(self) -> None:
        """Run the default on/off filter simulation step."""
        self.s_out.send(
            default_on_off_filter(
                self.s_in.recv(),
                filter_size=self.filter_size,
                center_sigma=self.center_sigma,
                surround_sigma=self.surround_sigma,
            )
        )


@implements(proc=CSNNFeatureScaling, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyCSNNFeatureScalingModel(PyLoihiProcessModel):
    """CPU ProcessModel for CSNN feature scaling."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    min_values: np.ndarray = LavaPyType(np.ndarray, float)
    max_values: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self) -> None:
        """Run the feature scaling simulation step."""
        self.s_out.send(
            feature_scale(
                self.s_in.recv(),
                self.min_values,
                self.max_values,
                dtype=np.float32,
            )
        )


__all__ = ["PyCSNNDefaultOnOffFilterModel", "PyCSNNFeatureScalingModel"]
