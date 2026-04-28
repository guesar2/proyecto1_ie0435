# Clasificación de “contaminaciones” en una líneade producción simulada
En este repositorio se encuentra el código usado para generar el conjunto de datos para entrenar modelos de IA que permitan resolver el problema de clasificación de contaminaciones.

El conjunto de datos completo se encuentra en el archivo archivo `datatset.csv`.  Este archivo se puede generar a partir del directorio con imágenes `imagenes` con el script de Python `preprocessing.py`. Dicho directorio debe contener dos subdirectorios `negativas` y `positivas`, que corresponden a las dos etiquetas del problema de clasificación.

Para ejecutar `preprocessing.py` es necesario instalar OpenCV-Python. La forma más sencilla es por medio de `pip` con el comando:
```bash
pip install opencv-python
```
Una vez instalada dicha biblioteca, se puede ejecutar `preprocessing.py` con el comando:
```bash
python preprocessing.py
```
Además de producir el archivo CSV se generará el directorio `preprocessed_images` con las imágenes en blanco y negro recortadas.