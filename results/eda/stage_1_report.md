# ETAPA 1 - Analisis automatico y EDA de RAF-DB

## Objetivo de la etapa

Analizar automaticamente el archivo local `Archive(2).zip`, detectar su estructura interna, extraer el dataset sin descargar datos externos y producir un analisis exploratorio reproducible para el reconocimiento de emociones faciales.

## Explicacion teorica

El analisis exploratorio de datos es una fase critica antes del entrenamiento de modelos CNN. En reconocimiento de emociones faciales permite verificar:

- Si las clases estan correctamente etiquetadas.
- Si existe desbalance entre emociones.
- Si hay imagenes corruptas o duplicadas.
- Si las resoluciones y formatos son consistentes.
- Si la division train/test ya viene definida por el dataset.

RAF-DB utiliza siete emociones basicas:

| Etiqueta | Emocion |
|---:|---|
| 1 | surprise |
| 2 | fear |
| 3 | disgust |
| 4 | happiness |
| 5 | sadness |
| 6 | anger |
| 7 | neutral |

## Estructura detectada

El ZIP contiene:

- Carpeta principal: `DATASET/`
- Split de entrenamiento: `DATASET/train/`
- Split de prueba: `DATASET/test/`
- Carpetas por clase: `1`, `2`, `3`, `4`, `5`, `6`, `7`
- Archivos de etiquetas: `train_labels.csv`, `test_labels.csv`
- Imagenes: archivos `.jpg`

## Resultados principales

| Metrica | Valor |
|---|---:|
| Total de imagenes | 15339 |
| Imagenes de entrenamiento | 12271 |
| Imagenes de prueba | 3068 |
| Numero de clases | 7 |
| Imagenes danadas | 0 |
| Etiquetas nulas | 0 |
| Imagenes duplicadas | 6 |
| Discrepancias carpeta/CSV | 0 |
| Resolucion promedio | 100 x 100 |
| Resolucion minima | 100 x 100 |
| Resolucion maxima | 100 x 100 |
| Formato | `.jpg` |
| Ratio de desbalance max/min | 16.7803 |

## Distribucion por emocion

| Etiqueta | Emocion | Total | Porcentaje |
|---:|---|---:|---:|
| 1 | surprise | 1619 | 10.55 |
| 2 | fear | 355 | 2.31 |
| 3 | disgust | 877 | 5.72 |
| 4 | happiness | 5957 | 38.84 |
| 5 | sadness | 2460 | 16.04 |
| 6 | anger | 867 | 5.65 |
| 7 | neutral | 3204 | 20.89 |

## Conclusion sobre desbalance

El dataset esta fuertemente desbalanceado. La clase mayoritaria es `happiness` con 5957 imagenes, mientras que la clase minoritaria es `fear` con 355 imagenes. El ratio max/min es 16.7803, por encima del umbral practico de 2.0. En etapas posteriores se recomienda usar augmentation, class weights, metricas macro-promediadas y analisis por clase.

## Codigo implementado

### `main.py`

- Define el punto de entrada del proyecto.
- Ejecuta `python main.py --stage eda`.
- Crea carpetas necesarias.
- Busca automaticamente el ZIP en la raiz o en `dataset/`.
- Invoca `RAFDBInspector`.
- Guarda `detected_structure.json`.
- Invoca `EDAReport`.
- Imprime un resumen final en consola.

### `src/config/settings.py`

- Centraliza rutas del proyecto.
- Define extensiones de imagen validas.
- Define nombres candidatos del ZIP.
- Define el mapa canonico de etiquetas RAF-DB.
- Define la semilla global.

### `src/utils/files.py`

- `ensure_directories`: crea carpetas si no existen.
- `find_dataset_zip`: localiza el ZIP sin asumir mayusculas o ubicacion unica.

### `src/utils/reproducibility.py`

- `set_global_seed`: fija semillas para Python, NumPy y opcionalmente TensorFlow.

### `src/preprocessing/dataset_inspector.py`

- Lee el indice interno del ZIP.
- Extrae el dataset si aun no fue extraido.
- Detecta la raiz real del dataset.
- Detecta splits como `train` y `test`.
- Lee archivos CSV de etiquetas.
- Recorre imagenes.
- Calcula resolucion, canales, formato, brillo promedio, hash SHA1, duplicados, imagenes danadas y discrepancias entre carpeta y CSV.

### `src/evaluation/eda.py`

- Genera tablas CSV.
- Genera resumen JSON.
- Exporta listados de problemas: duplicados, danadas, etiquetas nulas y discrepancias.
- Genera graficas: barras, pie chart, histogramas y ejemplos por clase.

## Artefactos generados

Tablas y resumen:

- `results/eda/eda_summary.json`
- `results/eda/detected_structure.json`
- `results/eda/image_metadata.csv`
- `results/eda/class_distribution.csv`
- `results/eda/split_distribution.csv`
- `results/eda/zip_index.csv`
- `results/eda/labels_detected.csv`
- `results/eda/duplicate_images.csv`
- `results/eda/damaged_images.csv`
- `results/eda/null_label_images.csv`
- `results/eda/label_mismatches.csv`

Graficas:

- `figures/eda/class_distribution_bar.png`
- `figures/eda/class_distribution_pie.png`
- `figures/eda/split_class_distribution_bar.png`
- `figures/eda/resolution_and_size_histograms.png`
- `figures/eda/examples_per_class.png`

## Como ejecutar

```bash
python main.py --stage eda
```

## Posibles errores y soluciones

| Error | Causa probable | Solucion |
|---|---|---|
| No se encontro el ZIP | El archivo no esta en raiz ni en `dataset/` | Colocar `Archive(2).zip` en una de esas ubicaciones |
| `ModuleNotFoundError: cv2` | Falta OpenCV | Ejecutar `pip install opencv-python` |
| Error al extraer | Archivos abiertos o permisos | Cerrar visores/procesos y volver a ejecutar |
| Graficas no se generan | Falta Matplotlib/Seaborn | Ejecutar `pip install matplotlib seaborn` |

## Buenas practicas aplicadas

- No se descarga el dataset.
- No se asume una estructura fija antes de inspeccionar el ZIP.
- Se guardan artefactos reproducibles.
- Se separa configuracion, inspeccion, EDA y utilidades.
- Se calcula hash para detectar duplicados reales.
- Se documenta el desbalance para decisiones metodologicas posteriores.
