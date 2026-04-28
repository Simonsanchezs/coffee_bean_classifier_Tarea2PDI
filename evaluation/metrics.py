# evaluation/metrics.py
"""Métricas: Accuracy, reporte por clase, matriz de confusión, curvas NN."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score,
    classification_report, confusion_matrix,
)

from config.settings import RESULTS_DIR

logger = logging.getLogger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    acc = accuracy_score(y_true, y_pred)
    logger.info("Accuracy: %.4f (%.2f%%)", acc, acc * 100)
    return acc


def _save_confusion_matrix(y_true, y_pred, class_names, name, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=class_names
    ).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusión — {name}", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cm_{name.replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)


def _save_history(history, name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (tr, val), title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Pérdida"],
    ):
        ax.plot(history.history[tr],  label="Train")
        ax.plot(history.history[val], label="Validación")
        ax.set_title(f"{title} — {name}")
        ax.set_xlabel("Época"); ax.set_ylabel(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"hist_{name.replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)


def evaluate_all(
    results: Dict[str, dict],
    class_names: Optional[List[str]] = None,
    output_dir: str = RESULTS_DIR,
) -> None:
    """
    Imprime reportes y guarda gráficas para todos los modelos.

    results = {
        "SVM-HOG":  {"y_true": ..., "y_pred": ...},
        "NN-HOG":   {"y_true": ..., "y_pred": ..., "history": <Keras History>},
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    summary: List[dict] = []

    for name, data in results.items():
        y_true, y_pred = data["y_true"], data["y_pred"]
        acc = compute_accuracy(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(f"\n{'═'*52}\n{name}\n{'═'*52}\n{report}")
        _save_confusion_matrix(y_true, y_pred, class_names, name, output_dir)
        if "history" in data:
            _save_history(data["history"], name, output_dir)
        summary.append({"model": name, "accuracy": acc})

    print(f"\n{'═'*52}\nACCURACY RESUMEN\n{'═'*52}")
    for item in sorted(summary, key=lambda x: x["accuracy"], reverse=True):
        bar = "█" * int(item["accuracy"] * 30)
        print(f"  {item['model']:<18} {item['accuracy']*100:6.2f}%  {bar}")
