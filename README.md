# 🧠 Reconocimiento de Emociones Faciales con Redes Neuronales Profundas

> **Evaluación comparativa de modelos CNN, Híbridos y Transformers en RAF-DB bajo escenarios de Fine-Tuning, Transfer Learning y Scratch**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)


---

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de **Reconocimiento de Emociones Faciales (FER)** que entrena y evalúa **9 familias de modelos** de aprendizaje profundo sobre el dataset RAF-DB (Real-world Affective Faces Database) con **7 emociones básicas**: Sorpresa, Miedo, Disgusto, Felicidad, Tristeza, Ira y Neutral.

### 🏆 Resultados Principales

| Modelo | Escenario | Aug | Accuracy | F1 Macro | Latencia | Params |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **POSTER++ (Poster V2)** 🥇 | Scratch | ✅ | **85.63%** | **0.7917** | 4.09ms | 28.9M |
| **QCS (Cross-Similarity)** 🥈 | Scratch | ✅ | **84.91%** | **0.7770** | 4.27ms | 41.2M |
| VGG16 | Fine-Tuning | ❌ | 77.71% | 0.6920 | 8.32ms | 21.1M |
| DenseNet121 | Fine-Tuning | ✅ | 75.78% | 0.6848 | 6.19ms | 7.2M |
| MobileNetV2 | Scratch | ❌ | 72.13% | 0.6310 | 3.56ms | 2.5M |
| EfficientNetB0 | Fine-Tuning | ❌ | 70.01% | 0.6183 | 3.67ms | 4.3M |
| SwinFace (Swin ViT) | Scratch | ✅ | 69.39% | 0.6040 | 8.69ms | 38.8M |
| ResNet50 | Fine-Tuning | ✅ | 67.93% | 0.6068 | 5.30ms | 24.0M |
| CustomCNN (baseline) | Scratch | ❌ | 59.13% | 0.4891 | 6.09ms | 1.3M |

---

## 🏗️ Arquitectura del Proyecto

```
📂 Proyecto FER/
├── 📂 src/                          # Código fuente principal
│   ├── 📂 augmentation/             # MixUp, CutMix, Albumentations
│   │   ├── augmentations.py         # MixUp + CutMix batch-level
│   │   └── albumentations_ops.py    # Pipeline de augmentation avanzado
│   ├── 📂 config/                   # Configuración global
│   │   ├── settings.py              # Constantes, rutas, label maps
│   │   └── experiment.py            # ExperimentConfig dataclass
│   ├── 📂 evaluation/               # Evaluación y métricas
│   │   ├── metrics.py               # ModelEvaluator (Accuracy, F1, ROC, MCC)
│   │   ├── plots.py                 # Gráficas de entrenamiento
│   │   ├── comparison.py            # Comparativa entre modelos
│   │   ├── explainability.py        # Grad-CAM y visualización
│   │   └── error_analysis.py        # Análisis de errores
│   ├── 📂 models/                   # Arquitecturas de modelos
│   │   ├── model_factory.py         # Factory pattern + congelamiento de capas
│   │   ├── applications.py          # CNNs estándar (VGG16, ResNet50, etc.)
│   │   ├── custom_cnn.py            # CNN personalizada (baseline)
│   │   ├── qcs.py                   # ⭐ QCS - Quadruplet Cross-Similarity
│   │   ├── poster_v2.py             # ⭐ POSTER++ con landmark queries
│   │   ├── swin_face.py             # ⭐ SwinFace - Swin ViT + CBAM
│   │   └── deit.py                  # ⭐ DeiT - Data-efficient ViT
│   ├── 📂 preprocessing/            # Carga y split de datos
│   │   ├── data_split.py            # Split estratificado train/val/test
│   │   ├── dataset_inspector.py     # EDA y análisis del dataset
│   │   └── tfdata.py                # DataLoaders y transformaciones
│   ├── 📂 training/                 # Bucle de entrenamiento
│   │   ├── trainer.py               # ExperimentTrainer (SCN, Mixed Precision)
│   │   ├── tuner.py                 # Hyperparameter tuning
│   │   └── callbacks.py             # Callbacks de entrenamiento
│   └── 📂 utils/                    # Utilidades generales
├── 📂 saved_models/                 # Pesos entrenados (Git LFS)
│   ├── poster_v2_scratch_aug/       # 🥇 Campeón - 85.63%
│   ├── qcs_scratch_aug/             # 🥈 Subcampeón - 84.91%
│   ├── swin_face_scratch_aug/       # Swin Transformer
│   └── deit_scratch_aug/            # DeiT Transformer
├── 📂 results/                      # Resultados experimentales
│   ├── 📂 training/                 # Logs de entrenamiento (CSV)
│   ├── 📂 evaluation/               # Métricas, confusion matrices, ROC
│   └── model_comparison.csv         # Tabla comparativa global
├── 📂 figures/                      # Gráficas generadas
│   ├── 📂 report/                   # Figuras del artículo
│   └── 📂 evaluation/               # Comparativas de evaluación
├── 📂 DOCUMENTACION/                # Documentación científica (.docx)
├── 📂 PARA_ANDROID_STUDIO/          # Modelos exportados para Android
├── main.py                          # Punto de entrada CLI
├── requirements.txt                 # Dependencias
└── README.md                        # Este archivo
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos
- **Python** 3.10+
- **CUDA** 11.8+ (GPU NVIDIA recomendada, funciona con CPU)
- **RAM** 8 GB mínimo
- **GPU VRAM** 4 GB mínimo (optimizado para RTX 3050)

### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/MijaGod-Creator/Entrenamiento-de-Redes-Neuronales-en-escenarios-FineTuning-TransferLearning-Scratch-.git
cd Entrenamiento-de-Redes-Neuronales-en-escenarios-FineTuning-TransferLearning-Scratch-
```

### Paso 2: Instalar dependencias (Elige una opción)

**Opción A: Usando Conda / Anaconda (Recomendado - Evita errores en Windows)**
Si utilizas Anaconda o Miniconda, ejecuta el siguiente comando en tu terminal (Anaconda Prompt) para crear un entorno virtual e instalar todas las dependencias pre-compiladas (incluyendo controladores CUDA de GPU):
```bash
conda env create -f environment.yml
conda activate fer_env
```

**Opción B: Usando pip estándar**
Si prefieres usar `pip` de Python directo:
```bash
pip install -r requirements.txt
```
> 💡 **Solución de problemas (Windows):** Si al instalar por `pip` obtienes un error relacionado con la compilación de `stringzilla` (como `Microsoft Visual C++ 14.0 or greater is required`), puedes solucionarlo ejecutando:
> ```bash
> pip install "stringzilla<3.10"
> ```


### Paso 3: Descomprimir el dataset RAF-DB (Ya incluido)
El dataset ya viene incluido en el repositorio en `dataset/Archive(2).zip` (~39.5 MB). Solo debes descomprimirlo dentro de la carpeta `dataset/raw/`.

**En Windows (PowerShell):**
```powershell
Expand-Archive -Path dataset/Archive(2).zip -DestinationPath dataset/raw
```

**En Linux / macOS:**
```bash
unzip dataset/Archive(2).zip -d dataset/raw
```

La estructura final de la carpeta debe quedar de la siguiente forma:
```
dataset/raw/
├── DATASET/
│   ├── train/
│   │   ├── 1/   (surprise)
│   │   ├── 2/   (fear)
│   │   ├── ...
│   │   └── 7/   (neutral)
│   └── test/
│       ├── 1/
│       └── ...
```

### Paso 4: Ejecutar el análisis exploratorio y preprocesamiento
```bash
python main.py --stage setup
python main.py --stage eda
python main.py --stage preprocess
```

---

## 🎯 Uso: Entrenar Modelos

### Entrenar un modelo específico
```bash
# POSTER++ (Campeón) - Scratch con augmentation
python main.py --stage train --architecture poster_v2 --scenario scratch --epochs 100 --learning-rate 0.000035 --augmentation --mixup --cutmix --patience 30 --self-cure

# QCS (Subcampeón) - Scratch con augmentation
python main.py --stage train --architecture qcs --scenario scratch --epochs 100 --learning-rate 0.000035 --augmentation --mixup --cutmix --patience 30 --self-cure

# SwinFace - Scratch con augmentation
python main.py --stage train --architecture swin_face --scenario scratch --epochs 100 --learning-rate 0.000035 --augmentation --mixup --cutmix --patience 30 --self-cure

# DeiT - Scratch con augmentation
python main.py --stage train --architecture deit --scenario scratch --epochs 100 --learning-rate 0.000035 --augmentation --mixup --cutmix --patience 30 --self-cure

# VGG16 - Fine-Tuning estándar
python main.py --stage train --architecture vgg16 --scenario fine_tuning --epochs 30 --augmentation
```

### Entrenar todos los modelos (grid completo)
```bash
python main.py --stage train-grid
```

### Reanudar un entrenamiento interrumpido
```bash
python main.py --stage train --architecture poster_v2 --scenario scratch --epochs 100 --resume
```

---

## 📊 Evaluar Modelos

```bash
# Evaluar un modelo específico
python main.py --stage evaluate --architecture poster_v2 --scenario scratch --augmentation

# Comparar todos los modelos entrenados
python main.py --stage compare
```

---

## 🔧 Técnicas Implementadas

### Preprocesamiento
- **Alineación Facial Geométrica**: MTCNN detecta 5 landmarks → cálculo de ángulo interocular → rotación afín → crop 224×224
- **Normalización ImageNet**: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

### Data Augmentation
- **Nivel de imagen**: RandomHorizontalFlip, RandomRotation(15°), RandomAffine, ColorJitter, RandomResizedCrop, RandomErasing
- **Nivel de batch**: MixUp (α=0.2, distribución Beta) y CutMix (α=0.35)
- **Albumentations**: GaussNoise, Blur, CoarseDropout

### Función de Pérdida
- **Soft-Target Cross-Entropy**: compatible con las etiquetas suaves de MixUp/CutMix
- **Class Weights Balanceados**: sklearn `compute_class_weight('balanced')` para mitigar el desbalance
- **Self-Cure Network (SCN)**: pesos dinámicos que suprimen etiquetas ruidosas (umbral 0.85/0.15)

### Optimización
- **Optimizador**: AdamW (weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (épocas ≥ 45) o ReduceLROnPlateau
- **Mixed Precision**: torch.amp.GradScaler para reducir VRAM
- **Early Stopping**: paciencia de 30 épocas

### Escenarios de Entrenamiento
| Escenario | Descripción |
|:---|:---|
| **Scratch** | Todas las capas entrenables. Pesos iniciales de VGGFace2 (modelos faciales) o ImageNet (CNNs) |
| **Transfer Learning** | Backbone completamente congelado. Solo se entrena la cabeza clasificadora |
| **Fine-Tuning** | Backbone congelado excepto el último bloque (block8/layers[-1]/blocks[-1]) |

---

## 📱 Despliegue en Android

Los modelos exportados para Android se encuentran en `PARA_ANDROID_STUDIO/`:
- Archivos `.ptl` (PyTorch Lite) para PyTorch Mobile
- Archivos `.onnx` para ONNX Runtime

```kotlin
// Ejemplo de uso en Android (Kotlin)
val module = LiteModuleLoader.load("poster_v2.ptl")
val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
    bitmap, floatArrayOf(0.485f, 0.456f, 0.406f),
    floatArrayOf(0.229f, 0.224f, 0.225f)
)
val output = module.forward(IValue.from(inputTensor)).toTensor()
```

---

## 🔬 Modelos SOTA Implementados

### POSTER++ (Poster V2) - 🥇 Campeón (85.63%)
Red híbrida que usa **68 landmark queries aprendibles** (sin detector explícito) y **atención bidireccional** (Image→Landmarks→Image) sobre un backbone InceptionResnetV1 pre-entrenado en VGGFace2.

### QCS (Quadruplet Cross-Similarity) - 🥈 (84.91%)
Usa **Cross-Similarity Attention** que durante el entrenamiento cruza features de diferentes muestras del batch mediante `torch.roll()`, forzando representaciones discriminativas. En inferencia usa self-attention.

### SwinFace (Swin Transformer + CBAM) - (69.39%)
Backbone **Swin Transformer** con ventanas desplazadas (W-MSA/SW-MSA), extracción multi-escala (local 1344ch + global 768ch = 2112ch), **CBAM** (Channel + Spatial attention) y TaskSpecificSubnet.

### DeiT (Data-efficient Image Transformer)
**Vision Transformer** con destilación via `timm`. Usa `deit_tiny_patch16_224` (5.7M params) optimizado para datasets pequeños.

---

## ⚡ Optimización de Hardware

Este proyecto fue diseñado para funcionar en **GPUs de gama portátil** (RTX 3050 con 4GB VRAM):

| Técnica | Ahorro |
|:---|:---|
| Congelamiento selectivo de capas | VRAM: 3.2GB → 1.2GB (62%) |
| Mixed Precision (FP16) | Velocidad: 3x más rápido |
| Batch size optimizado (32) | Estabilidad de gradientes |
| DataLoader sin workers (Windows) | Compatibilidad sin deadlocks |

---

## 📄 Documentación

El directorio `DOCUMENTACION/` contiene:
- **Resultados del Proyecto de Reconocimiento de Emociones (1).docx**: Registro completo de avance con 12 figuras, 2 tablas, análisis descriptivos y 12 anexos de código fuente

---

## 📦 Dependencias Principales

```
torch >= 2.0
torchvision >= 0.15
timm >= 0.9
facenet-pytorch >= 2.5
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
Pillow >= 10.0
matplotlib >= 3.7
tqdm >= 4.65
python-docx >= 0.8
```

---

## 👤 Autor

**Mijamin Taipe** - Proyecto de Inteligencia Artificial  
Universidad - Reconocimiento de Emociones Faciales


---

## 🙏 Agradecimientos

- Dataset [RAF-DB](http://www.whdeng.cn/raf/model1.html) por S. Li, W. Deng, y J. Du
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) por Tim Esler
- [timm](https://github.com/huggingface/pytorch-image-models) por Ross Wightman
- Inspirado en los papers de POSTER++, QCS y SwinFace
