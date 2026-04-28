# preprocessing/gaussian.py
"""
Filtro Gaussiano para preprocesamiento de imágenes.
Reduce ruido antes de la extracción de características HOG y SIFT.
"""

from __future__ import annotations

import cv2
import numpy as np

from config.settings import GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA


def apply_gaussian(
    img: np.ndarray,
    kernel_size: int = GAUSSIAN_KERNEL_SIZE,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """
    Aplica filtro Gaussiano a una imagen RGB.

    Args:
        img:         (H, W, 3) uint8
        kernel_size: tamaño del kernel, debe ser impar
        sigma:       desviación estándar (0 = automático)

    Returns:
        Imagen suavizada (H, W, 3) uint8
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size debe ser impar, recibido: {kernel_size}")
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def apply_gaussian_batch(images: np.ndarray) -> np.ndarray:
    """
    Aplica filtro Gaussiano a un lote de imágenes.

    Args:
        images: (N, H, W, 3) uint8

    Returns:
        (N, H, W, 3) uint8
    """
    return np.array([apply_gaussian(img) for img in images], dtype=np.uint8)
