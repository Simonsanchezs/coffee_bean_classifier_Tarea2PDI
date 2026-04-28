# Clasificador de Granos de Café (Coffee Bean Classifier)

Este proyecto implementa un pipeline completo de Procesamiento Digital de Imágenes y Machine Learning para clasificar el nivel de tueste de granos de café en cuatro categorías: **Dark, Green, Light y Medium**.

**Integrantes:**
* Andrea Correa Arango
* Julián Isaza Marin
* Simón Sánchez Sepúlveda

## Características Principales

El flujo de trabajo automatizado incluye:
* **Preprocesamiento:** Reducción de ruido mediante la aplicación de un Filtro Gaussiano.
* **Extracción de Características (Descriptores):**
  * **HOG** (Histogram of Oriented Gradients) para el análisis de formas y contornos.
  * **SIFT** (Scale-Invariant Feature Transform) usando codificación Bag-of-Words (Vocabulario Visual mediante K-Means).
* **Modelos de Clasificación:**
  * Máquinas de Vectores de Soporte (**SVM**) con kernel RBF.
  * Red Neuronal Multicapa (**MLP**) construida con TensorFlow/Keras.
* **Interfaz Gráfica (GUI):** Aplicación de escritorio interactiva construida con Tkinter para probar predicciones sobre nuevas imágenes de forma intuitiva.

---

## Instalación y Configuración del Entorno

Sigue estos pasos para clonar el proyecto, preparar el entorno virtual y descargar las dependencias.

**1. Crear y activar el entorno virtual:**
Abre tu terminal en la raíz del proyecto y ejecuta:

    python -m venv venv

Para activar el entorno virtual (en Windows):


    venv\Scripts\activate

(Si usas Linux o macOS, el comando de activación es: source venv/bin/activate)

**2. Instalar dependencias:**
Con el entorno virtual activado, instala los paquetes requeridos:

    pip install -r requirements.txt

## Configuración del Dataset (Kaggle)
El dataset se descarga automáticamente en tiempo de ejecución utilizando la API de Kaggle. Por seguridad, necesitas configurar tu propio token de acceso.

1. Obtener tu Token de Kaggle

2. Configurar la variable de entorno:
En la misma terminal donde ejecutarás el proyecto, configura tu token:

* En Windows (PowerShell):

        $env:KAGGLE_API_TOKEN="TU_TOKEN_AQUI"
        
        
* En Windows (CMD):

        set KAGGLE_API_TOKEN=TU_TOKEN_AQUI
        
* En Linux / macOS:

        export KAGGLE_API_TOKEN="TU_TOKEN_AQUI"

## Ejecución del Proyecto
Para ejecutar todo el pipeline y finalmente abrir la interfaz gráfica, utiliza el siguiente comando:

        python main.py --gui


## Estructura del Directorio

* config/ : Archivo settings.py con parámetros globales del proyecto (Rutas, Hiperparámetros, tamaño de imágenes).

* data/ : Lógica para la autenticación en Kaggle, descarga y lectura de las imágenes.

* evaluation/ : Funciones para calcular Accuracy, reportes de clasificación y generar matrices de confusión en gráficos.

* features/ : Scripts para la extracción matemática de descriptores HOG y SIFT (BoW).

* gui/ : Código fuente de la interfaz gráfica desarrollada en Tkinter.

* models/ : Arquitectura y lógicas de entrenamiento para la Red Neuronal y la máquina SVM.

* preprocessing/ : Aplicación del filtro gaussiano para preparación de datos.