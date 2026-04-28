# features/hog_extractor.py
"""
Extracción de características HOG (Histogram of Oriented Gradients).
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

from config.settings import HOG_CELLS_PER_BLOCK, HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL

logger = logging.getLogger(__name__)


def extract_hog_single(
    img: np.ndarray,
    visualize: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extrae descriptor HOG de una imagen RGB.

    Args:
        img:       (H, W, 3) uint8
        visualize: si True devuelve también la imagen HOG visual

    Returns:
        Vector 1-D float32, o (vector, imagen_hog) si visualize=True
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=visualize,
        feature_vector=True,
    )


def extract_hog_batch(images: np.ndarray) -> np.ndarray:
    """
    Extrae HOG de un lote de imágenes.

    Args:
        images: (N, H, W, 3) uint8

    Returns:
        (N, D) float32
    """
    logger.info("Extrayendo HOG de %d imágenes …", len(images))
    feats = [extract_hog_single(img) for img in tqdm(images, desc="HOG", unit="img")]
    X = np.array(feats, dtype=np.float32)
    logger.info("HOG shape: %s", X.shape)
    return X
