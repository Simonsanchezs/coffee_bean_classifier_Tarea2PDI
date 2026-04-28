# models/svm_classifier.py
"""SVM con kernel RBF. Pipeline: StandardScaler → SVC."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config.settings import MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)


def _path(feature_set: str) -> Path:
    return Path(MODELS_DIR) / f"svm_{feature_set}.pkl"


def train_svm(X_train: np.ndarray, y_train: np.ndarray, feature_set: str) -> Pipeline:
    """Entrena SVM y guarda en disco."""
    logger.info("Entrenando SVM [%s] — %d muestras", feature_set.upper(), len(X_train))
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale",
                    probability=True, class_weight="balanced",
                    random_state=RANDOM_STATE)),
    ]).fit(X_train, y_train)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    with open(_path(feature_set), "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("SVM [%s] guardado.", feature_set.upper())
    return pipeline


def predict_svm(model: Pipeline, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (clases, probabilidades)."""
    return model.predict(X), model.predict_proba(X)


def load_svm(feature_set: str) -> Pipeline:
    """Carga SVM desde disco."""
    p = _path(feature_set)
    if not p.exists():
        raise FileNotFoundError(
            f"Modelo SVM [{feature_set}] no encontrado en '{p}'.\n"
            "Ejecuta: python main.py"
        )
    with open(p, "rb") as f:
        return pickle.load(f)
