"""Cache helpers for MNIST L3 SVM classification."""

from __future__ import annotations

from pathlib import Path
import pickle
import typing as ty

import numpy as np


DEFAULT_CACHE_DIRNAME = "lava_mnist_l3_svm"
DEFAULT_SVM_FILENAME = "mnist_l3_svm.pkl"


def default_cache_dir(model_root: str | Path) -> Path:
    return Path(model_root) / DEFAULT_CACHE_DIRNAME


def default_svm_path(model_root: str | Path) -> Path:
    return default_cache_dir(model_root) / DEFAULT_SVM_FILENAME


def save_svm_bundle(path: str | Path, bundle: dict[str, ty.Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_svm_bundle(path: str | Path) -> dict[str, ty.Any]:
    path = Path(path)
    with path.open("rb") as file:
        bundle = pickle.load(file)
    if "svm" not in bundle:
        raise ValueError(f"{path} is not an MNIST L3 SVM bundle")
    return bundle


def predict_svm_bundle(
        bundle: dict[str, ty.Any],
        features: np.ndarray) -> tuple[int, np.ndarray | None]:
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    svm = bundle["svm"]
    prediction = int(svm.predict(x)[0])
    decision = None
    if hasattr(svm, "decision_function"):
        decision = np.asarray(svm.decision_function(x), dtype=np.float32)
    return prediction, decision


def empty_svm_output(output_shape: tuple[int, ...]) -> np.ndarray:
    output_size = int(np.prod(output_shape))
    out = np.full((output_size,), np.nan, dtype=np.float32)
    out[0] = -1.0
    return out.reshape(output_shape)


__all__ = [
    "DEFAULT_CACHE_DIRNAME",
    "DEFAULT_SVM_FILENAME",
    "default_cache_dir",
    "default_svm_path",
    "empty_svm_output",
    "load_svm_bundle",
    "predict_svm_bundle",
    "save_svm_bundle",
]
