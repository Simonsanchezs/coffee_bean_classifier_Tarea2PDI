# models/nn_classifier.py
"""Red Neuronal Densa (MLP) con TensorFlow/Keras."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from config.settings import (
    MODELS_DIR, NN_BATCH_SIZE, NN_DROPOUT_RATE,
    NN_EPOCHS, NN_HIDDEN_UNITS, NN_LEARNING_RATE, RANDOM_STATE,
)

logger = logging.getLogger(__name__)
tf.random.set_seed(RANDOM_STATE)


def _path(feature_set: str) -> Path:
    return Path(MODELS_DIR) / f"nn_{feature_set}.keras"


def _build(input_dim: int, n_classes: int) -> keras.Model:
    model = keras.Sequential(name="CoffeeBeanMLP")
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    for units in NN_HIDDEN_UNITS:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(NN_DROPOUT_RATE))
    model.add(layers.Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_nn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    n_classes: int, feature_set: str,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Entrena la red neuronal y guarda el mejor checkpoint."""
    logger.info("Entrenando NN [%s] — %d muestras", feature_set.upper(), len(X_train))
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    model = _build(X_train.shape[1], n_classes)
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(str(_path(feature_set)), monitor="val_accuracy",
                                  save_best_only=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE,
        callbacks=cbs, verbose=1,
    )
    logger.info("NN [%s] guardada.", feature_set.upper())
    return model, history


def predict_nn(model: keras.Model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (clases, probabilidades)."""
    probs = model.predict(X, verbose=0)
    return np.argmax(probs, axis=1), probs


def load_nn(feature_set: str) -> keras.Model:
    """Carga la red neuronal desde disco."""
    p = _path(feature_set)
    if not p.exists():
        raise FileNotFoundError(
            f"Modelo NN [{feature_set}] no encontrado en '{p}'.\n"
            "Ejecuta: python main.py"
        )
    return keras.models.load_model(str(p))
