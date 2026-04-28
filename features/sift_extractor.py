# features/sift_extractor.py
"""
Extracción de características SIFT con codificación Bag-of-Words (K-Means).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from config.settings import MODELS_DIR, SIFT_CLUSTERS, SIFT_N_KEYPOINTS

logger = logging.getLogger(__name__)
_VOCAB_PATH = Path(MODELS_DIR) / "sift_vocabulary.pkl"


def _raw_descriptors(img: np.ndarray) -> Optional[np.ndarray]:
    """Retorna descriptores SIFT (K, 128) o None si no hay keypoints."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, desc = cv2.SIFT_create(nfeatures=SIFT_N_KEYPOINTS).detectAndCompute(gray, None)
    return desc


def build_vocabulary(images: np.ndarray, n_clusters: int = SIFT_CLUSTERS) -> MiniBatchKMeans:
    """
    Construye vocabulario visual con K-Means sobre el set de entrenamiento.

    Args:
        images:    (N, H, W, 3) uint8
        n_clusters: tamaño del vocabulario

    Returns:
        MiniBatchKMeans ajustado y guardado en disco
    """
    logger.info("Construyendo vocabulario SIFT …")
    all_desc = [d for img in tqdm(images, desc="SIFT descriptors") if (d := _raw_descriptors(img)) is not None]

    if not all_desc:
        raise RuntimeError("No se obtuvieron descriptores SIFT.")

    stacked = np.vstack(all_desc).astype(np.float32)
    logger.info("%d descriptores — ajustando K-Means …", len(stacked))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=3)
    kmeans.fit(stacked)

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    with open(_VOCAB_PATH, "wb") as f:
        pickle.dump(kmeans, f)
    logger.info("Vocabulario guardado en '%s'", _VOCAB_PATH)
    return kmeans


def load_vocabulary() -> MiniBatchKMeans:
    """Carga el vocabulario SIFT desde disco."""
    if not _VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Vocabulario SIFT no encontrado en '{_VOCAB_PATH}'.\n"
            "Ejecuta: python main.py"
        )
    with open(_VOCAB_PATH, "rb") as f:
        return pickle.load(f)


def encode_bow_single(img: np.ndarray, kmeans: MiniBatchKMeans) -> np.ndarray:
    """
    Codifica una imagen como histograma BoW normalizado.

    Args:
        img:    (H, W, 3) uint8
        kmeans: vocabulario ajustado

    Returns:
        (n_clusters,) float32, suma = 1
    """
    n = kmeans.n_clusters
    desc = _raw_descriptors(img)
    if desc is None:
        return np.zeros(n, dtype=np.float32)
    hist, _ = np.histogram(
        kmeans.predict(desc.astype(np.float32)), bins=np.arange(n + 1)
    )
    hist = hist.astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else hist


def extract_sift_batch(images: np.ndarray, kmeans: Optional[MiniBatchKMeans] = None) -> np.ndarray:
    """
    Codifica un lote con SIFT-BoW.

    Args:
        images: (N, H, W, 3) uint8
        kmeans: vocabulario (None = cargar desde disco)

    Returns:
        (N, n_clusters) float32
    """
    if kmeans is None:
        kmeans = load_vocabulary()
    logger.info("Codificando %d imágenes SIFT-BoW …", len(images))
    feats = [encode_bow_single(img, kmeans) for img in tqdm(images, desc="SIFT-BoW", unit="img")]
    X = np.array(feats, dtype=np.float32)
    logger.info("SIFT-BoW shape: %s", X.shape)
    return X
