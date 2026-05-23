# Clasificación de contaminaciones en una línea de producción simulada

Este repositorio contiene el código y los datos para entrenar y evaluar modelos de clasificación binaria que detectan contaminaciones en imágenes de una línea de producción simulada.

## Estructura principal
- `preprocessing.py`: preprocessing de imágenes, generación de imágenes binarizadas y creación de archivos CSV.
- `model_training_all_datasets.py`: entrenamiento y evaluación de modelos sobre los archivos `dataset*.csv`, comparación de tres configuraciones y exportación del modelo SVM afinado.
- `images/positive` y `images/negative`: carpetas de datos originales de entrada.
- `preprocessed_images/`: imágenes procesadas generadas por `preprocessing.py`.
- `dataset.csv`: conjuntos de datos CSV usados en el entrenamiento.

## Requisitos
Usar Python 3.12 o una versión compatible.

Para instalar dependencias en el entorno virtual:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generar el dataset
El script `preprocessing.py` crea el CSV principal a partir de las imágenes en `images/positive` y `images/negative`.

```bash
python preprocessing.py
```

Esto genera:
- `preprocessed_images/`: imágenes binarizadas de 128x128 píxeles
- `dataset.csv`: matriz de características con la etiqueta en la última columna

## Entrenar los modelos
Ejecutar el script de entrenamiento y evaluación:

```bash
python model_training_all_datasets.py
```

El script realiza:
- evaluación de cuatro modelos: Decision Tree, Naive Bayes, KNN y SVM
- comparación entre baseline, normalización por dataset y PCA
- afinación de hiperparámetros para SVM en datos normalizados
- exportación del modelo optimizado a `svm_dataset_norm.joblib`

## Inferencia
Para usar el modelo afinado en un nuevo ejemplo:

```python
from joblib import load
from preprocessing import process_image_to_matrix
import numpy as np

model = load('svm_dataset_norm.joblib')
matrix, binary = process_image_to_matrix('ruta/de/imagen.jpg')
X_new = matrix.flatten().reshape(1, -1)
pred = model.predict(X_new)
print('Predicción:', pred[0])
```

## Archivos generados
- `confusion_matrices_baseline.png`
- `model_comparison_baseline.png`
- `confusion_matrices_dataset_level_normalization.png`
- `model_comparison_dataset_level_normalization.png`
- `svm_dataset_norm.joblib`

