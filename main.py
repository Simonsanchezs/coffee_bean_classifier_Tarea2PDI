#!/usr/bin/env python3
# main.py
"""Coffee Bean Classifier — Pipeline completo."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from config.settings import DATA_DIR, MODELS_DIR, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from data.loader import download_dataset, load_images
from evaluation.metrics import evaluate_all
from features.hog_extractor import extract_hog_batch
from features.sift_extractor import build_vocabulary, extract_sift_batch
from models.nn_classifier import predict_nn, train_nn
from models.svm_classifier import predict_svm, train_svm
from preprocessing.gaussian import apply_gaussian_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_SLUG = "coffee-bean-dataset-resized-224-x-224"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coffee Bean Classifier")
    p.add_argument("--skip-download", action="store_true",
                   help="Usa el dataset ya descargado.")
    p.add_argument("--skip-train", action="store_true",
                   help="Omite entrenamiento (modelos ya guardados).")
    p.add_argument("--gui", action="store_true",
                   help="Lanza la interfaz gráfica al terminar.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    for d in (MODELS_DIR, RESULTS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.skip_train:
        logger.info("Entrenamiento omitido — cargando modelos existentes.")
        if args.gui:
            _launch_gui()
        return

    # 1. Descarga
    if args.skip_download:
        dataset_root = os.path.join(DATA_DIR, DATASET_SLUG)
        logger.info("Dataset local: %s", dataset_root)
    else:
        dataset_root = download_dataset(DATA_DIR)

    # 2. Carga de imágenes
    X, y, class_names = load_images(dataset_root)
    n_classes = len(class_names)
    logger.info("Clases: %s", class_names)

    # 3. Preprocesamiento — Filtro Gaussiano
    logger.info("Aplicando filtro Gaussiano …")
    X_proc = apply_gaussian_batch(X)

    # 4. Split train / test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_proc, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d  |  Test: %d", len(X_tr), len(X_te))

    # 5. Características
    logger.info("─── HOG ──────────────────────────────────────────")
    X_hog_tr = extract_hog_batch(X_tr)
    X_hog_te = extract_hog_batch(X_te)

    logger.info("─── SIFT-BoW ─────────────────────────────────────")
    kmeans   = build_vocabulary(X_tr)
    X_sft_tr = extract_sift_batch(X_tr, kmeans)
    X_sft_te = extract_sift_batch(X_te, kmeans)

    # 6. SVM
    logger.info("─── SVM + HOG ────────────────────────────────────")
    svm_hog           = train_svm(X_hog_tr, y_tr, "hog")
    y_pred_svm_hog, _ = predict_svm(svm_hog, X_hog_te)

    logger.info("─── SVM + SIFT ───────────────────────────────────")
    svm_sft           = train_svm(X_sft_tr, y_tr, "sift")
    y_pred_svm_sft, _ = predict_svm(svm_sft, X_sft_te)

    # 7. Red Neuronal — split adicional train/val (15 %)
    X_hog_t, X_hog_v, y_t, y_v = train_test_split(
        X_hog_tr, y_tr, test_size=0.15, random_state=RANDOM_STATE, stratify=y_tr
    )
    X_sft_t, X_sft_v, _, _ = train_test_split(
        X_sft_tr, y_tr, test_size=0.15, random_state=RANDOM_STATE, stratify=y_tr
    )

    logger.info("─── Red Neuronal + HOG ───────────────────────────")
    nn_hog, hist_hog     = train_nn(X_hog_t, y_t, X_hog_v, y_v, n_classes, "hog")
    y_pred_nn_hog, _     = predict_nn(nn_hog, X_hog_te)

    logger.info("─── Red Neuronal + SIFT ──────────────────────────")
    nn_sft, hist_sft     = train_nn(X_sft_t, y_t, X_sft_v, y_v, n_classes, "sift")
    y_pred_nn_sft, _     = predict_nn(nn_sft, X_sft_te)

    # 8. Evaluación
    evaluate_all(
        results={
            "SVM-HOG":  {"y_true": y_te, "y_pred": y_pred_svm_hog},
            "SVM-SIFT": {"y_true": y_te, "y_pred": y_pred_svm_sft},
            "NN-HOG":   {"y_true": y_te, "y_pred": y_pred_nn_hog,  "history": hist_hog},
            "NN-SIFT":  {"y_true": y_te, "y_pred": y_pred_nn_sft,  "history": hist_sft},
        },
        class_names=class_names,
        output_dir=RESULTS_DIR,
    )
    logger.info("Resultados en '%s'", RESULTS_DIR)

    if args.gui:
        _launch_gui()


def _launch_gui() -> None:
    logger.info("Lanzando interfaz gráfica …")
    from gui.app import run_app
    run_app()


if __name__ == "__main__":
    main()
