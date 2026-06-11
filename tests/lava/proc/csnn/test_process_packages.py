from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.csnn.latency.process import CSNNLatencyCoding
from lava.proc.csnn.objectives.process import CSNNTimeObjectiveOutput
from lava.proc.csnn.pooling.process import CSNNSpikePooling, CSNNSumPooling
from lava.proc.csnn.preprocessing.process import (
    CSNNDefaultOnOffFilter,
    CSNNFeatureScaling,
)
from lava.proc.csnn.stats.process import (
    CSNNActivity,
    CSNNAnalysisMediator,
    CSNNCoherence,
)
from lava.proc.csnn.svm.process import CSNNSVMClassifier
from lava.proc.io.sink import RingBuffer as Sink
from lava.proc.io.source import RingBuffer as Source

import lava.proc.csnn.latency.models  # noqa: F401
import lava.proc.csnn.objectives.models  # noqa: F401


ROOT = Path(__file__).resolve().parents[4]
CSNN_SRC = ROOT / "src" / "lava" / "proc" / "csnn"
DELETED_MODULE_NAMES = {
    "analysis",
    "l3_inference",
    "l3_lava_network",
    "mnist_data",
    "model_io",
    "network_models",
    "network_processes",
    "svm_cache",
    "tools",
    "tools.train_mnist_l3_svm",
}
DELETED_MODULES = {f"lava.proc.csnn.{name}" for name in DELETED_MODULE_NAMES}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_package_specific_process_imports_exist() -> None:
    assert CSNNDefaultOnOffFilter
    assert CSNNFeatureScaling
    assert CSNNLatencyCoding
    assert CSNNSpikePooling
    assert CSNNSumPooling
    assert CSNNTimeObjectiveOutput
    assert CSNNActivity
    assert CSNNAnalysisMediator
    assert CSNNCoherence
    assert CSNNSVMClassifier


def test_process_model_modules_do_not_import_high_level_helpers() -> None:
    model_paths = [
        CSNN_SRC / "conv" / "models.py",
        CSNN_SRC / "latency" / "models.py",
        CSNN_SRC / "objectives" / "models.py",
        CSNN_SRC / "pooling" / "models.py",
        CSNN_SRC / "preprocessing" / "models.py",
        CSNN_SRC / "stats" / "models.py",
        CSNN_SRC / "svm" / "models.py",
    ]
    violations = [
        f"{path.relative_to(ROOT)} imports {module}"
        for path in model_paths
        for module in _imports(path)
        if module in DELETED_MODULES
    ]
    assert violations == []


def test_deleted_modules_are_not_present() -> None:
    module_paths = [
        CSNN_SRC / "analysis.py",
        CSNN_SRC / "l3_inference.py",
        CSNN_SRC / "l3_lava_network.py",
        CSNN_SRC / "mnist_data.py",
        CSNN_SRC / "model_io.py",
        CSNN_SRC / "network_models.py",
        CSNN_SRC / "network_processes.py",
        CSNN_SRC / "svm_cache.py",
        CSNN_SRC / "tools",
    ]
    assert [path for path in module_paths if path.exists()] == []


def test_small_csnn_process_graph_runs() -> None:
    source = Source(data=np.array([[0.25], [0.0]], dtype=np.float32))
    latency = CSNNLatencyCoding(shape=(2,), name="test_latency")
    objective = CSNNTimeObjectiveOutput(shape=(2,), name="test_objective")
    sink = Sink(shape=(2,), buffer=1)

    source.s_out.connect(latency.s_in)
    latency.s_out.connect(objective.s_in)
    objective.s_out.connect(sink.a_in)

    try:
        sink.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg(select_tag="floating_pt"))
        np.testing.assert_allclose(
            sink.data.get()[..., -1],
            np.array([1.0, 0.0], dtype=np.float32))
    finally:
        sink.stop()
