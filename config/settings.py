# config/settings.py
"""Parámetros globales del proyecto. Ajusta aquí sin tocar el código."""

# ── Dataset ───────────────────────────────────────────────────────────────────
KAGGLE_DATASET        = "gpiosenka/coffee-bean-dataset-resized-224-x-224"
DATA_DIR              = "data/coffee_beans"
IMAGE_SIZE            = (224, 224)
CLASSES               = ["Dark", "Green", "Light", "Medium"]
MAX_SAMPLES_PER_CLASS = 200   # None = todo el dataset

# ── Preprocesamiento — Filtro Gaussiano ───────────────────────────────────────
GAUSSIAN_KERNEL_SIZE  = 5     # debe ser impar
GAUSSIAN_SIGMA        = 0.0   # 0 = OpenCV lo calcula automáticamente

# ── HOG ──────────────────────────────────────────────────────────────────────
HOG_ORIENTATIONS      = 9
HOG_PIXELS_PER_CELL   = (8, 8)
HOG_CELLS_PER_BLOCK   = (2, 2)

# ── SIFT + Bag-of-Words ───────────────────────────────────────────────────────
SIFT_N_KEYPOINTS      = 128
SIFT_CLUSTERS         = 100

# ── Entrenamiento ─────────────────────────────────────────────────────────────
TEST_SIZE             = 0.20
RANDOM_STATE          = 42

# ── Red Neuronal ──────────────────────────────────────────────────────────────
NN_EPOCHS             = 30
NN_BATCH_SIZE         = 32
NN_LEARNING_RATE      = 1e-3
NN_DROPOUT_RATE       = 0.4
NN_HIDDEN_UNITS       = (512, 256, 128)

# ── Rutas de salida ───────────────────────────────────────────────────────────
MODELS_DIR            = "saved_models"
RESULTS_DIR           = "results"
