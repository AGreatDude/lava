"""Cache helpers for MNIST L3 SVM classification."""

from __future__ import annotations

from pathlib import Path
import pickle
import typing as ty

import numpy as np

DEFAULT_CACHE_DIRNAME = "lava_mnist_l3_svm"
DEFAULT_SVM_FILENAME = "mnist_l3_svm.pkl"


def default_cache_dir(model_root: str | Path) -> Path:
    """Return the default cache directory path under model_root.

    Parameters
    ----------
    model_root : str or Path
        Root directory of the model.

    Returns
    -------
    Path
        Default cache directory path.
    """
    return Path(model_root) / DEFAULT_CACHE_DIRNAME


def default_svm_path(model_root: str | Path) -> Path:
    """Return the default SVM model file path under model_root.

    Parameters
    ----------
    model_root : str or Path
        Root directory of the model.

    Returns
    -------
    Path
        Default SVM model file path.
    """
    return default_cache_dir(model_root) / DEFAULT_SVM_FILENAME


def save_svm_bundle(path: str | Path, bundle: dict[str, ty.Any]) -> None:
    """Save an SVM bundle dictionary to a file using pickle.

    Parameters
    ----------
    path : str or Path
        Target file path.
    bundle : dict
        SVM bundle dictionary containing key 'svm'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_svm_bundle(path: str | Path) -> dict[str, ty.Any]:
    """Load an SVM bundle dictionary from a file.

    Parameters
    ----------
    path : str or Path
        Source file path.

    Returns
    -------
    dict
        SVM bundle dictionary.
    """
    path = Path(path)
    with path.open("rb") as file:
        bundle = pickle.load(file)
    if "svm" not in bundle:
        raise ValueError(f"{path} is not an MNIST L3 SVM bundle")
    return bundle


def predict_svm_bundle(
    bundle: dict[str, ty.Any], features: np.ndarray
) -> tuple[int, np.ndarray | None]:
    """Predict label and decision function value using an SVM bundle.

    Parameters
    ----------
    bundle : dict
        Fitted SVM bundle dictionary.
    features : np.ndarray
        Input features array.

    Returns
    -------
    int
        Predicted class label.
    np.ndarray or None
        Decision function outputs if available, else None.
    """
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    svm = bundle["svm"]
    prediction = int(svm.predict(x)[0])
    decision = None
    if hasattr(svm, "decision_function"):
        decision = np.asarray(svm.decision_function(x), dtype=np.float32)
    return prediction, decision


def empty_svm_output(output_shape: tuple[int, ...]) -> np.ndarray:
    """Create an empty SVM output array initialized with NaNs and a -1 prediction.

    Parameters
    ----------
    output_shape : tuple of int
        Desired shape of the output array.

    Returns
    -------
    np.ndarray
        Empty SVM output array.
    """
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
