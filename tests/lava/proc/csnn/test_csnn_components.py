from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.csnn.conv.process import CSNNConvolution
from lava.proc.csnn.conv.utils import (
    _as_pair as conv_as_pair,
    affected_output_positions,
    compute_output_shape,
    csnn_convolve,
)
from lava.proc.csnn.latency.utils import (
    SpikeEvent,
    finite_spike_events,
    iter_finite_spike_events,
    latency_code,
    raster_to_first_spike_times,
    spike_times_to_raster,
)
from lava.proc.csnn.objectives.utils import time_objective_output
from lava.proc.csnn.pooling.process import CSNNSpikePooling
from lava.proc.csnn.pooling.utils import (
    _as_pair as pooling_as_pair,
    spike_pool2d,
    sum_pool2d,
)
from lava.proc.csnn.preprocessing.process import (
    CSNNDefaultOnOffFilter,
    CSNNFeatureScaling,
)
from lava.proc.csnn.preprocessing.utils import (
    FeatureScaler,
    default_on_off_filter,
    feature_scale,
)
from lava.proc.csnn.stats.process import (
    CSNNActivity,
    CSNNAnalysisMediator,
    CSNNCoherence,
)
from lava.proc.csnn.stats.utils import (
    ActivityStats,
    CoherenceStats,
    activity_from_vector,
    activity_stats,
    activity_to_vector,
    coherence_from_vector,
    coherence_stats,
    coherence_to_vector,
    finite_spike_count,
)
from lava.proc.csnn.svm.process import CSNNSVMClassifier
from lava.proc.csnn.svm.utils import (
    default_cache_dir,
    default_svm_path,
    empty_svm_output,
    load_svm_bundle,
    predict_svm_bundle,
    save_svm_bundle,
)
from lava.proc.io.sink import RingBuffer as Sink
from lava.proc.io.source import RingBuffer as Source

# Import models to register them with the compiler
import lava.proc.csnn.conv.models  # noqa: F401
import lava.proc.csnn.latency.models  # noqa: F401
import lava.proc.csnn.objectives.models  # noqa: F401
import lava.proc.csnn.pooling.models  # noqa: F401
import lava.proc.csnn.preprocessing.models  # noqa: F401
import lava.proc.csnn.stats.models  # noqa: F401
import lava.proc.csnn.svm.models  # noqa: F401


# Dummy estimator to mock sklearn SVM for testing
class DummySVM(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.n_features_in_ = 3

    def predict(self, X):
        return np.array([1])

    def decision_function(self, X):
        return np.array([[0.5, 0.2]])


def test_latency_helpers() -> None:
    # test latency_code
    vals = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    res = latency_code(vals)
    assert np.allclose(res, [np.inf, 0.5, 0.0])

    res_max = latency_code(vals, max_timestamp=0.8)
    assert np.allclose(res_max, [np.inf, 0.5, 0.0])

    # test SpikeEvent
    se1 = SpikeEvent(0.5, (1, 2))
    se2 = SpikeEvent(0.5, (1, 3))
    se3 = SpikeEvent(0.6, (1, 2))
    assert se1 < se2
    assert se2 < se3
    assert se1 == SpikeEvent(0.5, (1, 2))
    assert se1 != "not_a_spike_event"
    assert repr(se1) == "SpikeEvent(time=0.5, index=(1, 2))"

    # test finite_spike_events
    times = np.array([[0.5, np.inf], [0.2, np.inf]])
    events = finite_spike_events(times)
    assert len(events) == 2
    assert events[0].time == 0.2
    assert events[1].time == 0.5

    # test iter_finite_spike_events
    iter_events = list(iter_finite_spike_events(times))
    assert len(iter_events) == 2

    # test spike_times_to_raster & raster_to_first_spike_times
    times = np.array([0.0, 0.5, 1.0, np.inf])
    raster = spike_times_to_raster(times, time_steps=3)
    # 3 steps: step 0 (time 0), step 1 (time 0.5), step 2 (time 1)
    assert np.array_equal(
        raster,
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ],
    )

    with pytest.raises(ValueError):
        spike_times_to_raster(times, time_steps=0)

    # raster back to times
    recon = raster_to_first_spike_times(raster)
    assert np.allclose(recon, [0.0, 0.5, 1.0, np.inf])

    with pytest.raises(ValueError):
        raster_to_first_spike_times(np.array(True))  # scalar (0D)


def test_objectives_helpers() -> None:
    times = np.array([0.2, np.inf])
    res = time_objective_output(times, t_obj=0.5)
    assert np.allclose(res, [1.0, 0.0])


def test_conv_helpers() -> None:
    # test _as_pair
    assert conv_as_pair(3, "test") == (3, 3)
    assert conv_as_pair((1, 2), "test") == (1, 2)
    with pytest.raises(ValueError):
        conv_as_pair(None, "test")
    with pytest.raises(ValueError):
        conv_as_pair((1,), "test")

    # test compute_output_shape
    in_shape = (10, 10, 3)
    kernel_shape = (3, 3, 3, 5)
    out_shape = compute_output_shape(
        in_shape, kernel_shape, stride=1, padding=0
    )
    assert out_shape == (8, 8, 5)

    with pytest.raises(ValueError):
        compute_output_shape((10, 10), kernel_shape)
    with pytest.raises(ValueError):
        compute_output_shape(in_shape, (3, 3, 3))
    with pytest.raises(ValueError):
        compute_output_shape((-10, 10, 3), kernel_shape)
    with pytest.raises(ValueError):
        compute_output_shape(in_shape, kernel_shape, stride=-1)
    with pytest.raises(ValueError):
        compute_output_shape(in_shape, kernel_shape, padding=-1)
    with pytest.raises(ValueError):
        compute_output_shape(
            in_shape, (15, 15, 3, 5)
        )  # kernel larger than input
    with pytest.raises(ValueError):
        compute_output_shape(in_shape, (3, 3, 4, 5))  # mismatch depth

    # test affected_output_positions
    affected = list(
        affected_output_positions(
            0,
            0,
            output_shape=(8, 8, 5),
            kernel_size=(3, 3),
            stride=1,
            padding=0,
        )
    )
    # position (0, 0) affects (0, 0) with weight (0, 0)
    assert len(affected) == 1
    assert affected[0] == (0, 0, 0, 0)

    # test csnn_convolve
    spikes = np.full((5, 5, 2), np.inf)
    spikes[1, 1, 0] = 0.2
    weights = np.ones((3, 3, 2, 2))
    thresholds = 0.5
    out = csnn_convolve(spikes, weights, thresholds, stride=1, padding=1)
    # input spike at (1,1,0) should affect spatial positions
    # (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
    assert np.allclose(out[1, 1, 0], 0.2)

    with pytest.raises(ValueError):
        csnn_convolve(np.ones((5, 5)), weights, thresholds)
    with pytest.raises(ValueError):
        csnn_convolve(spikes, np.ones((3, 3, 2)), thresholds)
    with pytest.raises(ValueError):
        csnn_convolve(spikes, weights, np.array([0.5, 0.5, 0.5]))


def test_pooling_helpers() -> None:
    # test _as_pair
    assert pooling_as_pair(3, "test") == (3, 3)

    # test spike_pool2d
    spikes = np.full((4, 4, 1), np.inf)
    spikes[0, 0, 0] = 0.5
    spikes[0, 1, 0] = 0.2
    out = spike_pool2d(spikes, kernel_size=(2, 2), stride=(2, 2))
    assert out.shape == (2, 2, 1)
    assert np.allclose(out[0, 0, 0], 0.2)

    with pytest.raises(ValueError):
        spike_pool2d(spikes, kernel_size=-1)
    with pytest.raises(ValueError):
        spike_pool2d(spikes, stride=-1)
    with pytest.raises(ValueError):
        spike_pool2d(spikes, padding=-1)

    # test sum_pool2d
    feats = np.ones((4, 4, 2))
    out_sum = sum_pool2d(feats, target_shape=(2, 2))
    assert out_sum.shape == (2, 2, 2)
    assert np.allclose(out_sum[0, 0, 0], 4.0)


def test_preprocessing_helpers() -> None:
    # test default_on_off_filter
    img = np.ones((10, 10, 1))
    out = default_on_off_filter(img, filter_size=3)
    assert out.shape == (10, 10, 2)

    with pytest.raises(ValueError):
        default_on_off_filter(img, filter_size=0)
    with pytest.raises(ValueError):
        default_on_off_filter(img, center_sigma=0)

    # test feature_scale and FeatureScaler
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = FeatureScaler()
    scaler.fit(samples)
    assert np.allclose(scaler.min_, [1.0, 2.0])
    assert np.allclose(scaler.max_, [3.0, 4.0])

    with pytest.raises(ValueError):
        scaler.fit(np.array([1.0, 2.0]))  # no batch dim

    scaled = scaler.transform(np.array([[2.0, 3.0]]))
    assert np.allclose(scaled, [[0.5, 0.5]])

    unfitted = FeatureScaler()
    with pytest.raises(RuntimeError):
        unfitted.transform(samples)

    # test direct feature_scale func
    res = feature_scale(
        np.array([2.0, 3.0]), np.array([1.0, 2.0]), np.array([3.0, 4.0])
    )
    assert np.allclose(res, [0.5, 0.5])


def test_stats_helpers() -> None:
    # test activity_stats
    sample = np.array([0.0, 1.0, 2.0])
    stats = activity_stats(sample)
    assert isinstance(stats, ActivityStats)
    assert stats.quiet is False

    with pytest.raises(ValueError):
        activity_stats(np.array([]))

    # test activity vector conversions
    vec = activity_to_vector(stats)
    recon = activity_from_vector(vec)
    assert np.allclose(recon.sparsity, stats.sparsity)

    # test coherence_stats
    weights = np.ones((3, 3, 2, 4))
    stats_coh = coherence_stats(weights)
    assert isinstance(stats_coh, CoherenceStats)

    with pytest.raises(ValueError):
        coherence_stats(np.ones((3, 3, 2)))

    # test coherence vector conversions
    vec_coh = coherence_to_vector(stats_coh)
    recon_coh = coherence_from_vector(vec_coh)
    assert np.allclose(recon_coh.mean_weights, stats_coh.mean_weights)
    assert coherence_from_vector(np.array([0.0])) is None

    # test finite_spike_count
    assert finite_spike_count(np.array([1.0, np.inf])) == 1


def test_svm_helpers() -> None:
    # test default paths
    assert default_cache_dir("root") == Path("root/lava_mnist_l3_svm")
    assert default_svm_path("root") == Path(
        "root/lava_mnist_l3_svm/mnist_l3_svm.pkl"
    )

    # test empty svm output
    out = empty_svm_output((3,))
    assert out.shape == (3,)
    assert out[0] == -1.0
    assert np.isnan(out[1])

    # test save/load and predict SVM bundle
    with tempfile.TemporaryDirectory() as tmpdir:
        svm = DummySVM()
        bundle = {"svm": svm}
        path = Path(tmpdir) / "svm.pkl"
        save_svm_bundle(path, bundle)
        assert path.exists()

        loaded = load_svm_bundle(path)
        assert "svm" in loaded

        pred, dec = predict_svm_bundle(loaded, np.array([0.1, 0.2, 0.3]))
        assert pred == 1
        assert dec is not None

        # load non-bundle
        bad_path = Path(tmpdir) / "bad.pkl"
        with bad_path.open("wb") as f:
            import pickle

            pickle.dump({"not_svm": 1}, f)

        with pytest.raises(ValueError):
            load_svm_bundle(bad_path)


def test_processes_and_models() -> None:
    # Run test for CSNNConvolution
    source = Source(
        data=np.array(
            [[[[1.0]], [[0.5]]], [[[0.0]], [[0.8]]]], dtype=np.float32
        )
    )
    conv = CSNNConvolution(
        weights=np.ones((2, 2, 1, 1)),
        thresholds=0.5,
        input_shape=(2, 2, 1),
        stride=1,
        padding=0,
    )
    sink = Sink(shape=(1, 1, 1), buffer=1)
    source.s_out.connect(conv.s_in)
    conv.s_out.connect(sink.a_in)

    try:
        sink.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg(select_tag="floating_pt"),
        )
        data = sink.data.get()
        assert data.shape == (1, 1, 1, 1)
    finally:
        sink.stop()

    # Run test for CSNNSpikePooling & CSNNSumPooling
    source = Source(
        data=np.array(
            [[[[0.2]], [[0.5]]], [[[np.inf]], [[0.8]]]], dtype=np.float32
        )
    )
    pooling = CSNNSpikePooling(
        input_shape=(2, 2, 1), kernel_size=(2, 2), stride=(2, 2), padding=0
    )
    sink = Sink(shape=(1, 1, 1), buffer=1)
    source.s_out.connect(pooling.s_in)
    pooling.s_out.connect(sink.a_in)

    try:
        sink.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg(select_tag="floating_pt"),
        )
        assert sink.data.get().shape == (1, 1, 1, 1)
    finally:
        sink.stop()

    # Test CSNNDefaultOnOffFilter
    source = Source(data=np.ones((10, 10, 1, 1), dtype=np.float32))
    onoff = CSNNDefaultOnOffFilter(input_shape=(10, 10, 1), filter_size=3)
    sink = Sink(shape=(10, 10, 2), buffer=1)
    source.s_out.connect(onoff.s_in)
    onoff.s_out.connect(sink.a_in)
    try:
        sink.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg(select_tag="floating_pt"),
        )
        assert sink.data.get().shape == (10, 10, 2, 1)
    finally:
        sink.stop()

    # Test CSNNFeatureScaling
    source = Source(data=np.array([[[2.0], [3.0]]], dtype=np.float32))
    scaling = CSNNFeatureScaling(
        shape=(1, 2),
        min_values=np.array([[1.0, 2.0]]),
        max_values=np.array([[3.0, 4.0]]),
    )
    sink = Sink(shape=(1, 2), buffer=1)
    source.s_out.connect(scaling.s_in)
    scaling.s_out.connect(sink.a_in)
    try:
        sink.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg(select_tag="floating_pt"),
        )
        assert np.allclose(sink.data.get()[..., -1], [[0.5, 0.5]])
    finally:
        sink.stop()

    # Test CSNNActivity, CSNNCoherence, CSNNAnalysisMediator, CSNNSVMClassifier
    source_act = Source(data=np.array([[1.0], [0.0], [0.5]], dtype=np.float32))
    act = CSNNActivity(input_shape=(3,))
    source_act.s_out.connect(act.s_in)

    source_coh = Source(data=np.array([[0.5], [0.5], [0.5]], dtype=np.float32))
    coh = CSNNCoherence(input_shape=(3,), weights=np.ones((3, 3, 1, 2)))
    source_coh.s_out.connect(coh.s_in)

    # Test classifier with dummy SVM path
    with tempfile.TemporaryDirectory() as tmpdir:
        svm = DummySVM()
        bundle = {"svm": svm}
        svm_path = Path(tmpdir) / "svm.pkl"
        save_svm_bundle(svm_path, bundle)

        source_features = Source(
            data=np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        )
        classifier = CSNNSVMClassifier(
            input_shape=(3,), svm_path=svm_path, output_shape=(3,)
        )
        source_features.s_out.connect(classifier.s_in)

        mediator = CSNNAnalysisMediator(
            has_coherence=True, has_svm=True, svm_shape=(3,)
        )
        act.stats_out.connect(mediator.activity_in)
        coh.stats_out.connect(mediator.coherence_in)
        classifier.pred_out.connect(mediator.svm_in)

        sink_med = Sink(shape=(13,), buffer=1)
        mediator.analysis_out.connect(sink_med.a_in)

        try:
            sink_med.run(
                condition=RunSteps(num_steps=1),
                run_cfg=Loihi1SimCfg(select_tag="floating_pt"),
            )
            res = sink_med.data.get()[..., -1]
            assert res.shape == (13,)
        finally:
            sink_med.stop()


def test_models_directly() -> None:
    from unittest.mock import MagicMock
    from lava.proc.csnn.svm.models import PyCSNNSVMClassifierModel
    from lava.proc.csnn.pooling.models import (
        PyCSNNSpikePoolingModel,
        PyCSNNSumPoolingModel,
    )
    from lava.proc.csnn.preprocessing.models import (
        PyCSNNDefaultOnOffFilterModel,
        PyCSNNFeatureScalingModel,
    )
    from lava.proc.csnn.stats.models import (
        PyCSNNActivityModel,
        PyCSNNCoherenceModel,
        PyCSNNAnalysisMediatorModel,
    )
    from lava.proc.csnn.conv.models import PyCSNNConvolutionModel

    # 1. PyCSNNSVMClassifierModel
    model_svm = PyCSNNSVMClassifierModel(
        {"output_shape": (3,), "svm_path": None}
    )
    model_svm.s_in = MagicMock()
    model_svm.s_in.recv.return_value = np.array([1.0, 2.0, 3.0])
    model_svm.pred_out = MagicMock()
    model_svm.run_spk()
    model_svm.pred_out.send.assert_called_once()

    # 2. PyCSNNSpikePoolingModel
    model_spike = PyCSNNSpikePoolingModel(
        {"kernel_size": (2, 2), "stride": (2, 2), "padding": (0, 0)}
    )
    model_spike.s_in = MagicMock()
    model_spike.s_in.recv.return_value = np.ones((4, 4, 1))
    model_spike.s_out = MagicMock()
    model_spike.run_spk()
    model_spike.s_out.send.assert_called_once()

    # 3. PyCSNNSumPoolingModel
    model_sum = PyCSNNSumPoolingModel({"target_shape": (2, 2)})
    model_sum.s_in = MagicMock()
    model_sum.s_in.recv.return_value = np.ones((4, 4, 1))
    model_sum.s_out = MagicMock()
    model_sum.run_spk()
    model_sum.s_out.send.assert_called_once()

    # 4. PyCSNNDefaultOnOffFilterModel
    model_onoff = PyCSNNDefaultOnOffFilterModel(
        {"filter_size": 3, "center_sigma": 1.0, "surround_sigma": 4.0}
    )
    model_onoff.s_in = MagicMock()
    model_onoff.s_in.recv.return_value = np.ones((10, 10, 1))
    model_onoff.s_out = MagicMock()
    model_onoff.run_spk()
    model_onoff.s_out.send.assert_called_once()

    # 5. PyCSNNFeatureScalingModel
    model_scaling = PyCSNNFeatureScalingModel({})
    model_scaling.s_in = MagicMock()
    model_scaling.s_in.recv.return_value = np.array([2.0, 3.0])
    model_scaling.min_values = np.array([1.0, 2.0])
    model_scaling.max_values = np.array([3.0, 4.0])
    model_scaling.s_out = MagicMock()
    model_scaling.run_spk()
    model_scaling.s_out.send.assert_called_once()

    # 6. PyCSNNActivityModel
    model_act = PyCSNNActivityModel({})
    model_act.s_in = MagicMock()
    model_act.s_in.recv.return_value = np.array([1.0, 0.0, 0.5])
    model_act.stats_out = MagicMock()
    model_act.run_spk()
    model_act.stats_out.send.assert_called_once()

    # 7. PyCSNNCoherenceModel
    model_coh = PyCSNNCoherenceModel({})
    model_coh.s_in = MagicMock()
    model_coh.s_in.recv.return_value = np.array([0.5, 0.5, 0.5])
    model_coh.weights = np.ones((3, 3, 1, 2))
    model_coh.stats_out = MagicMock()
    model_coh.run_spk()
    model_coh.stats_out.send.assert_called_once()

    # 8. PyCSNNAnalysisMediatorModel
    model_med = PyCSNNAnalysisMediatorModel(
        {"has_coherence": True, "has_svm": True}
    )
    model_med.activity_in = MagicMock()
    model_med.activity_in.recv.return_value = np.array([1.0, 2.0, 3.0])
    model_med.coherence_in = MagicMock()
    model_med.coherence_in.recv.return_value = np.array([4.0, 5.0, 6.0])
    model_med.svm_in = MagicMock()
    model_med.svm_in.recv.return_value = np.array([7.0, 8.0, 9.0])
    model_med.analysis_out = MagicMock()
    model_med.run_spk()
    model_med.analysis_out.send.assert_called_once()

    # 9. PyCSNNConvolutionModel
    model_conv = PyCSNNConvolutionModel({})
    model_conv.s_in = MagicMock()
    model_conv.s_in.recv.return_value = np.ones((5, 5, 1))
    model_conv.weights = np.ones((3, 3, 1, 1))
    model_conv.thresholds = np.array([0.5])
    model_conv.stride = np.array([1, 1])
    model_conv.padding = np.array([0, 0])
    model_conv.wta_infer = np.array([0])
    model_conv.s_out = MagicMock()
    model_conv.run_spk()
    model_conv.s_out.send.assert_called_once()
