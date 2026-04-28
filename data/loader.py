# data/loader.py
"""
Descarga y carga del dataset Coffee Bean desde Kaggle.

Autenticación: lee KAGGLE_API_TOKEN del entorno y escribe
~/.kaggle/kaggle.json automáticamente. Compatible con kaggle >= 2.0.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config.settings import (
    CLASSES,
    DATA_DIR,
    IMAGE_SIZE,
    KAGGLE_DATASET,
    MAX_SAMPLES_PER_CLASS,
)

logger = logging.getLogger(__name__)


def _write_credentials() -> None:
    """Escribe kaggle.json desde la variable de entorno KAGGLE_API_TOKEN."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        return
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    creds_path = kaggle_dir / "kaggle.json"
    creds_path.write_text(json.dumps({"username": "user", "key": token}))
    try:
        creds_path.chmod(0o600)
    except Exception:
        pass
    logger.info("Credenciales escritas en '%s'", creds_path)


def download_dataset(download_dir: str = DATA_DIR) -> str:
    """
    Descarga el dataset con la CLI de kaggle. Compatible con kaggle 1.x y 2.x.

    Args:
        download_dir: Carpeta destino.

    Returns:
        Ruta raíz del dataset extraído.
    """
    _write_credentials()
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # En kaggle>=2 no siempre existe `python -m kaggle`; preferimos el ejecutable.
    kaggle_cmd = shutil.which("kaggle")
    if not kaggle_cmd:
        scripts_dir = Path(sys.executable).resolve().parent
        candidates = [scripts_dir / "kaggle"]
        if os.name == "nt":
            candidates.insert(0, scripts_dir / "kaggle.exe")
        for candidate in candidates:
            if candidate.exists():
                kaggle_cmd = str(candidate)
                break

    if not kaggle_cmd:
        raise RuntimeError(
            "No se encontró la CLI de Kaggle en el entorno actual.\n"
            "Instala/verifica el paquete con: pip install kaggle"
        )

    logger.info("Descargando '%s' …", KAGGLE_DATASET)
    result = subprocess.run(
        [
            kaggle_cmd,
            "datasets", "download",
            "--dataset", KAGGLE_DATASET,
            "--path", download_dir,
            "--unzip",
        ],
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Error al descargar el dataset.\n"
            "Verifica que KAGGLE_API_TOKEN sea válido."
        )

    slug = KAGGLE_DATASET.split("/")[-1]
    candidate = os.path.join(download_dir, slug)
    root = candidate if os.path.isdir(candidate) else download_dir
    logger.info("Dataset en: %s", root)
    return root


def load_images(
    dataset_root: str,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    max_per_class: int | None = MAX_SAMPLES_PER_CLASS,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Lee las imágenes del disco.

    Returns:
        images  : (N, H, W, 3) uint8
        labels  : (N,) int32
        classes : lista de nombres de clase
    """
    images: List[np.ndarray] = []
    labels: List[int]        = []
    classes_found: List[str] = []

    for class_idx, class_name in enumerate(CLASSES):
        dirs = glob.glob(os.path.join(dataset_root, "**", class_name), recursive=True)
        if not dirs:
            logger.warning("Clase '%s' no encontrada.", class_name)
            continue

        paths: List[str] = []
        for d in dirs:
            paths += glob.glob(os.path.join(d, "*.jpg"))
            paths += glob.glob(os.path.join(d, "*.png"))

        if max_per_class is not None:
            paths = paths[:max_per_class]

        logger.info("'%s': %d imágenes", class_name, len(paths))
        for p in tqdm(paths, desc=class_name, leave=False):
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(class_idx)

        classes_found.append(class_name)

    if not images:
        raise FileNotFoundError(
            f"Sin imágenes en '{dataset_root}'. Verifica la descarga."
        )

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=np.int32)
    logger.info("Total: %d imágenes, %d clases", len(X), len(classes_found))
    return X, y, classes_found
