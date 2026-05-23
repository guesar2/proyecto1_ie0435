# Model Card

## Model name + version
- Nombre: SVM normalizado por dataset
- Versión: 1.0

## Intended use / out-of-scope
- Uso previsto: detección binaria de contaminaciones en imágenes de una línea de producción simulada.
- Fuera de alcance: no está diseñado para clasificación de otro tipo de defectos, detección de objetos genéricos, reconocimiento de piezas o análisis en entornos con cámaras muy diferentes a las usadas en el entrenamiento.

## Data summary
- El conjunto de datos se recolectó a partir de imágenes organizadas en carpetas `positive` y `negative`.
- Se aplicó un pipeline de preprocesamiento para generar imágenes binarizadas de `128x128` y vectores de características.
- Las variaciones cubren diferencias de iluminación y de cámaras, pero no abarcan todas las condiciones de producción reales.

## Labeling process
- El etiquetado se realizó manualmente, incluyendo en la carpeta `positivas` las imágenes con contaminaciones y en la carpeta `negativas` las imágenes sin contaminaciones.
- No se utilizó una herramienta de anotación especializada; la calidad depende de la consistencia de las carpetas.
- No hay un proceso de validación de etiquetas con múltiples anotadores documentado.

## Metrics
- Métricas reportadas: accuracy, precision, recall y F1.
- Evaluación realizada con validación cruzada estratificada de 5 pliegues (`StratifiedKFold`).
- La métrica principal usada para comparar modelos fue F1 ponderado.

## Ethical/safety notes
- El modelo puede ser sesgado por condiciones de iluminación, tipos de cámaras y fondos de las muestras de entrenamiento.
- Podría fallar cuando se aplique a imágenes muy distintas de las usadas en el entrenamiento.
- No debe usarse como único criterio de calidad en un entorno de producción real sin pruebas adicionales.

## Limitations
- Rendimiento limitado con objetos pequeños o poco contrastados.
- Sensible a oclusiones, sombras fuertes y desenfoque.
- Las imágenes con fondos complejos o texturas inusuales pueden degradar la predicción.
- El modelo se entrenó con un conjunto de datos reducido; no garantiza generalización amplia.

## Reproducibility
Comandos exactos:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python preprocessing.py
python model_training_all_datasets.py
```
Hardware aproximado:
- CPU Intel® Core™ i5-9600K.
- Sistema operativo Linux.
- Python 3.12 en entorno virtual `.venv`.
