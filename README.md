# Comparacion de arquitecturas CNN para reconocimiento de emociones faciales con RAF-DB

Proyecto cientifico reproducible para comparar arquitecturas CNN en RAF-DB.

##  Guía de Inicio Rápido (Paso a Paso)

Sigue estos pasos para ejecutar el proyecto desde cero y generar tus propios modelos:

### 1. Clonar el repositorio e instalar dependencias
Abre tu terminal y ejecuta los siguientes comandos:
```bash
# Clonar el proyecto
git clone https://github.com/MijaGod-Creator/ia-con-sin-transfer-learning-fine-tuning.git
cd ia-con-sin-transfer-learning-fine-tuning

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # En Windows

# Instalar PyTorch con soporte CUDA (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar dependencias restantes
pip install -r requirements.txt
```

### 2. Configurar el Dataset
El dataset original `archive (2).zip` ya está incluido en el repositorio para facilitar la ejecución. Solo debes correr el script de preparación para extraerlo e iniciar el análisis exploratorio:
```bash
python main.py --stage setup
```

### 3. Crear los Splits de Datos (Preprocesamiento)
Genera las particiones de entrenamiento, validación y prueba:
```bash
python main.py --stage preprocess --validation-size 0.15
```

### 4. Entrenar tus Modelos
Puedes entrenar un modelo específico para generar sus pesos. Por ejemplo, entrena una CNN personalizada desde cero (`scratch`):
```bash
python main.py --stage train --architecture custom_cnn --scenario scratch --epochs 10
```
*(Puedes sustituir `custom_cnn` por otras arquitecturas como `resnet50`, `vgg16`, `mobilenetv2`, etc., y cambiar `--scenario` a `transfer` o `fine_tuning`).*

Para ejecutar la búsqueda completa de todos los modelos (esto puede tardar varias horas):
```bash
python main.py --stage train-grid --architecture all --scenario all --epochs 30
```

### 5. Evaluar los Modelos y Comparar
Una vez que entrenes uno o más modelos, evalúa sus métricas:
```bash
python main.py --stage evaluate --architecture custom_cnn --scenario scratch
```
Luego genera la tabla comparativa con:
```bash
python main.py --stage compare
```

### 6. Levantar la Aplicación Web
Para probar los modelos con imágenes en tiempo real, inicia el servidor local Flask:
```bash
python app.py
```
Abre en tu navegador `http://127.0.0.1:5000/`. El servidor cargará automáticamente el modelo entrenado con el mejor F1-Score que esté guardado en tu carpeta `saved_models/`.

## Estructura

```text
dataset/
  Archive(2).zip
  raw/
  processed/
src/
  config/
  preprocessing/
  augmentation/
  models/
  training/
  evaluation/
  utils/
results/
  eda/
figures/
  eda/
logs/
saved_models/
notebooks/
main.py
requirements.txt
```

## Instalacion

Esta version del proyecto usa PyTorch y esta pensada para Windows con GPU nativa.
Usa Python 3.10 o 3.11 y instala el build CUDA de PyTorch que coincida con tu driver/NVIDIA Toolkit.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
# Instala primero PyTorch con soporte CUDA desde el sitio oficial de PyTorch.
# Ejemplo:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Si ya tenias creado el entorno con TensorFlow, crea uno nuevo para evitar conflictos de dependencias.

## Ejecutar ETAPA 1

```bash
python main.py --stage eda
```

El script:

- Detecta automaticamente `Archive(2).zip` en la raiz o en `dataset/`.
- Extrae el dataset en `dataset/raw/rafdb/`.
- Identifica splits, carpetas de clases, CSV de etiquetas e imagenes.
- Calcula conteos por clase, resoluciones, formatos, duplicados, etiquetas nulas, imagenes danadas y desbalance.
- Guarda tablas en `results/eda/`.
- Guarda graficas en `figures/eda/`.

## Salidas principales

- `results/eda/eda_summary.json`
- `results/eda/image_metadata.csv`
- `results/eda/class_distribution.csv`
- `results/eda/split_distribution.csv`
- `results/eda/zip_index.csv`
- `figures/eda/class_distribution_bar.png`
- `figures/eda/class_distribution_pie.png`
- `figures/eda/split_class_distribution_bar.png`
- `figures/eda/resolution_and_size_histograms.png`
- `figures/eda/examples_per_class.png`

## Preparar proyecto completo

Ejecuta EDA, crea splits train/validation/test, guarda reporte de entorno y genera el borrador de articulo:

```bash
python main.py --stage setup
```

## ETAPA 2 - Preprocesamiento

```bash
python main.py --stage preprocess --validation-size 0.15
```

Genera:

- `dataset/processed/train.csv`
- `dataset/processed/validation.csv`
- `dataset/processed/test.csv`

El pipeline de PyTorch implementa lectura, resize, normalizacion, `DataLoader`, `pin_memory` y uso de GPU cuando esta disponible.

## ETAPA 3 - Data Augmentation

Activar augmentation durante entrenamiento:

```bash
python main.py --stage train --architecture custom_cnn --scenario scratch --augmentation
```

Activar MixUp o CutMix:

```bash
python main.py --stage train --architecture custom_cnn --scenario scratch --augmentation --mixup
python main.py --stage train --architecture custom_cnn --scenario scratch --augmentation --cutmix
```

## ETAPAS 4 y 5 - Arquitecturas y escenarios

Arquitecturas disponibles:

- `custom_cnn`
- `vgg16`
- `resnet50`
- `mobilenetv2`
- `efficientnetb0`
- `densenet121`

Escenarios:

- `scratch`
- `transfer`
- `fine_tuning`

Ejemplos:

```bash
python main.py --stage train --architecture vgg16 --scenario scratch --epochs 30
python main.py --stage train --architecture resnet50 --scenario transfer --epochs 30
python main.py --stage train --architecture efficientnetb0 --scenario fine_tuning --epochs 20 --learning-rate 0.0001
```

Grid completo, con y sin augmentation:

```bash
python main.py --stage train-grid --architecture all --scenario all --epochs 30
```

## ETAPA 6 - Optimizacion de hiperparametros

```bash
python main.py --stage tune --architecture mobilenetv2 --scenario transfer --max-trials 15
```

Optimiza learning rate, batch size, optimizer, dropout, dense units y weight decay mediante una busqueda simple en PyTorch.

## ETAPA 7 - Callbacks

El entrenamiento usa:

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- TensorBoard
- CSVLogger
- LearningRateScheduler

## ETAPA 8 - Evaluacion

```bash
python main.py --stage evaluate --architecture mobilenetv2 --scenario transfer
```

Calcula accuracy, precision, recall, F1, specificity, top-2, top-3, ROC, AUC, confusion matrix, classification report, MCC, balanced accuracy y Cohen Kappa.

## ETAPAS 9 y 10 - Visualizaciones y comparacion

Luego de evaluar varios modelos:

```bash
python main.py --stage compare
```

Genera `results/model_comparison.csv`, `results/model_comparison.xlsx` y graficas comparativas.

## ETAPA 11 - Articulo cientifico IEEE

```bash
python main.py --stage article
```

Genera:

- `results/article/rafdb_cnn_ieee_article.md`

## Posibles errores

- **No se encontro el ZIP**: coloque `Archive(2).zip` en la raiz del proyecto o en `dataset/`.
- **Falta OpenCV**: ejecute `pip install opencv-python`.
- **Permisos al extraer**: cierre visores de imagenes o procesos que esten usando `dataset/raw/rafdb/`.
- **No detecta GPU en Windows**: revise que el entorno use el build CUDA de PyTorch y que `torch.cuda.is_available()` devuelva `True`.
- **PyTorch en CPU**: si instalaste la rueda CPU por error, reinstala con el indice CUDA oficial de PyTorch.

## Buenas practicas aplicadas

- Semillas aleatorias configuradas.
- Codigo modular por responsabilidad.
- No se descarga RAF-DB; solo se usa el ZIP local.
- El EDA se guarda como artefactos reproducibles en CSV, JSON y PNG.
- Entrenamientos, logs, modelos y metricas se guardan automaticamente.
- El grid completo queda parametrizado para ejecucion reproducible en GPU.
