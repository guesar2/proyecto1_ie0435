# Dataset

## Origen
El dataset principal se genera a partir de las carpetas `images/positive` y `images/negative` mediante el script `preprocessing.py`.

- `positive`: imágenes con contaminaciones visibles.
- `negative`: imágenes sin contaminaciones.

## Proceso de recolección
Las imágenes fueron recolectadas tomando fotografías que simulan la línea de producción. Posteriormente se aplicó un preprocesamiento que:
- convierte la imagen a escala de grises,
- aplica umbral adaptativo y operaciones morfológicas,
- redimensiona a `128x128` píxeles,
- binariza el resultado final.

La función `gen_csv` transforma cada imagen binarizada en un vector de 16.384 características y anexa la etiqueta (`1` para positiva, `0` para negativa) en la última columna.

## Archivos incluidos
- `dataset.csv`: archivo principal generado por `preprocessing.py`.

## Variaciones de iluminación y cámara
El dataset contiene variaciones en las condiciones de captura, pero no todas las combinaciones posibles están cubiertas. Las diferencias principales son:
- cambios leves de iluminación ambiental,
- distintas posiciones de cámara en algunos conjuntos,
- variabilidad en el fondo y la textura del plano de la imagen.

## Limitaciones
- El dataset aún es pequeño y puede no representar todas las condiciones reales de producción.
- Hay sesgos hacia los tipos de luz y cámaras utilizadas en las capturas existentes.
- Las piezas muy pequeñas pueden perderse tras la binarización y el redimensionado.
- La presencia de objetos oclusos, sombras intensas o desenfoque puede reducir la calidad del etiquetado.
- El etiquetado se deriva de la estructura de carpetas, por lo que la consistencia depende de la calidad de la organización original.
