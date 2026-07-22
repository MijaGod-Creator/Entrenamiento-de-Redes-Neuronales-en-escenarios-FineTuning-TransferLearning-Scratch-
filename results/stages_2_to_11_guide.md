# Guia de ETAPAS 2 a 11

## ETAPA 2 - Preprocesamiento

### Explicacion teorica

El preprocesamiento homogeniza las entradas antes del entrenamiento. En este proyecto se aplica resize a 224 x 224, normalizacion a `[0, 1]`, codificacion one-hot y division estratificada de train en train/validation, manteniendo el test original de RAF-DB.

### Codigo

- `src/preprocessing/data_split.py`
- `src/preprocessing/tfdata.py`

### Ejecutar

```bash
python main.py --stage preprocess
```

### Posibles errores

- Si falta `results/eda/image_metadata.csv`, ejecutar primero `python main.py --stage eda`.

### Buenas practicas

- Mantener test separado hasta la evaluacion final.
- Usar validation estratificado por clase.
- Usar `tf.data.AUTOTUNE` para rendimiento.

## ETAPA 3 - Data Augmentation

### Explicacion teorica

La augmentation reduce sobreajuste y ayuda ante desbalance. Se implementan transformaciones geometricas, fotometricas y tecnicas de regularizacion como Cutout, MixUp y CutMix.

### Codigo

- `src/augmentation/augmentations.py`
- `src/augmentation/albumentations_ops.py`

### Ejecutar

```bash
python main.py --stage train --architecture custom_cnn --scenario scratch --augmentation
python main.py --stage train --architecture custom_cnn --scenario scratch --augmentation --mixup
python main.py --stage train --architecture custom_cnn --scenario scratch --augmentation --cutmix
```

### Posibles errores

- MixUp y CutMix deben usarse despues de batching; el modulo ya lo gestiona.

### Buenas practicas

- Comparar siempre con y sin augmentation.
- Evitar transformaciones extremas que destruyan expresiones faciales.

## ETAPAS 4 y 5 - Modelos y escenarios

### Explicacion teorica

Se comparan modelos con distinta capacidad y costo computacional. Cada arquitectura se evalua en scratch, transfer learning y fine tuning.

### Codigo

- `src/models/custom_cnn.py`
- `src/models/applications.py`
- `src/models/model_factory.py`
- `src/training/trainer.py`

### Ejecutar

```bash
python main.py --stage train --architecture vgg16 --scenario transfer --epochs 30
python main.py --stage train --architecture resnet50 --scenario fine_tuning --epochs 20 --learning-rate 0.0001
python main.py --stage train-grid --architecture all --scenario all --epochs 30
```

### Posibles errores

- Si no hay GPU, reducir `--batch-size`.
- Si hay falta de memoria, usar `mobilenetv2` o batch size 16.

### Buenas practicas

- Usar learning rate menor en fine tuning.
- Guardar el mejor checkpoint por `val_loss`.

## ETAPA 6 - Optimizacion de hiperparametros

### Explicacion teorica

La busqueda bayesiana explora configuraciones prometedoras sin probar exhaustivamente todas las combinaciones.

### Codigo

- `src/training/tuner.py`

### Ejecutar

```bash
python main.py --stage tune --architecture mobilenetv2 --scenario transfer --max-trials 15
```

### Posibles errores

- El tuning puede tardar bastante; iniciar con 5 trials si no hay GPU.

### Buenas practicas

- Tunar primero modelos eficientes.
- Reentrenar con los mejores hiperparametros antes de evaluar en test.

## ETAPA 7 - Callbacks

### Codigo

- `src/training/callbacks.py`

### Incluye

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- TensorBoard
- CSVLogger
- LearningRateScheduler

### Ejecutar TensorBoard

```bash
tensorboard --logdir logs
```

## ETAPA 8 - Evaluacion

### Explicacion teorica

En datasets desbalanceados, accuracy no basta. Por eso se reportan metricas macro, weighted, specificity, top-k, AUC, MCC, balanced accuracy y Cohen Kappa.

### Codigo

- `src/evaluation/metrics.py`

### Ejecutar

```bash
python main.py --stage evaluate --architecture mobilenetv2 --scenario transfer
```

## ETAPA 9 - Visualizaciones

### Codigo

- `src/evaluation/plots.py`
- `src/evaluation/error_analysis.py`
- `src/evaluation/explainability.py`

### Genera

- Curvas accuracy/loss.
- ROC.
- Matriz de confusion.
- Comparaciones de arquitecturas.
- Correctos/incorrectos.
- Grad-CAM.
- Feature maps.
- Saliency maps.

## ETAPA 10 - Comparacion

### Codigo

- `src/evaluation/comparison.py`

### Ejecutar

```bash
python main.py --stage compare
```

### Salidas

- `results/model_comparison.csv`
- `results/model_comparison.xlsx`
- Figuras en `figures/evaluation/`

## ETAPA 11 - Articulo cientifico

### Codigo

- `src/utils/reporting.py`

### Ejecutar

```bash
python main.py --stage article
```

### Salida

- `results/article/rafdb_cnn_ieee_article.md`

## Recomendacion de ejecucion cientifica

1. `python main.py --stage setup`
2. Entrenar primero `mobilenetv2 transfer` como baseline rapido.
3. Evaluar baseline.
4. Ejecutar tuning.
5. Reentrenar mejores configuraciones.
6. Ejecutar grid completo en GPU.
7. Evaluar todos los modelos.
8. Ejecutar comparacion.
9. Regenerar articulo.
