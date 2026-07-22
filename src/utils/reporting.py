from pathlib import Path

import pandas as pd

from src.config.settings import ARTICLE_RESULTS_DIR, EDA_RESULTS_DIR, RESULTS_DIR


def generate_ieee_article(
    output_path: Path = ARTICLE_RESULTS_DIR / "rafdb_cnn_ieee_article.md",
    comparison_csv: Path = RESULTS_DIR / "model_comparison.csv",
) -> Path:
    ARTICLE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eda_summary = (EDA_RESULTS_DIR / "eda_summary.json").read_text(encoding="utf-8")
    comparison_text = "Aun no hay entrenamientos evaluados."
    if comparison_csv.exists():
        comparison = pd.read_csv(comparison_csv)
        try:
            comparison_text = comparison.to_markdown(index=False)
        except ImportError:
            comparison_text = comparison.to_csv(index=False)

    content = f"""# Comparacion del desempeno de diferentes arquitecturas CNN para el reconocimiento de emociones faciales utilizando RAF-DB

## Resumen

Este estudio compara arquitecturas de Redes Neuronales Convolucionales para clasificar emociones faciales en RAF-DB. Se consideran una CNN propia y modelos VGG16, ResNet50, MobileNetV2, EfficientNetB0 y DenseNet121 bajo tres escenarios experimentales: entrenamiento desde cero, transfer learning con ImageNet y fine tuning. Se incorporan estrategias de preprocesamiento, aumento de datos, optimizacion de hiperparametros y evaluacion con metricas multiclase.

## Abstract

This research compares convolutional neural network architectures for facial emotion recognition using RAF-DB. A custom CNN and five established deep architectures are evaluated under training from scratch, ImageNet transfer learning, and fine-tuning protocols. The pipeline includes preprocessing, data augmentation, hyperparameter optimization, and a comprehensive multiclass evaluation.

## I. Introduccion

El reconocimiento automatico de emociones faciales es una tarea relevante en interaccion humano-computadora, educacion inteligente, salud digital y analisis afectivo. A pesar del progreso de las CNN, el rendimiento depende del balance de clases, la arquitectura, la estrategia de transferencia y el protocolo experimental.

## II. Estado del Arte y Trabajos Relacionados

Las arquitecturas VGG, ResNet, MobileNet, EfficientNet y DenseNet han sido ampliamente utilizadas en vision por computador. VGG prioriza profundidad secuencial, ResNet introduce conexiones residuales, MobileNet optimiza eficiencia mediante convoluciones separables, EfficientNet escala profundidad, anchura y resolucion, y DenseNet reutiliza caracteristicas mediante conexiones densas.

## III. Dataset

Se utiliza exclusivamente RAF-DB desde el archivo local `Archive(2).zip`. El analisis exploratorio automatico produjo el siguiente resumen:

```json
{eda_summary}
```

El dataset presenta un desbalance significativo, especialmente entre `happiness` y `fear`, por lo que se recomiendan metricas macro-promediadas y tecnicas de augmentation.

## IV. Metodologia

El flujo experimental consta de:

1. Deteccion automatica del dataset.
2. EDA y validacion de integridad.
3. Division train/validation/test.
4. Preprocesamiento con resize, normalizacion y codificacion one-hot.
5. Pipeline de PyTorch con `Dataset` y `DataLoader` optimizados para GPU.
6. Data augmentation: flip, rotation, zoom, translation, random crop, brightness, contrast, Gaussian noise, blur, cutout, random erasing, MixUp y CutMix.
7. Entrenamiento de seis arquitecturas.
8. Comparacion entre scratch, transfer learning y fine tuning.
9. Optimizacion de hiperparametros con una busqueda simple en PyTorch.
10. Evaluacion con metricas multiclase y visualizaciones.

## V. Arquitecturas

Se implementan:

- CNN propia.
- VGG16.
- ResNet50.
- MobileNetV2.
- EfficientNetB0.
- DenseNet121.

## VI. Experimentos

Cada arquitectura se entrena en tres escenarios:

- Scratch: pesos inicializados aleatoriamente.
- Transfer learning: pesos ImageNet congelados y entrenamiento del clasificador.
- Fine tuning: descongelamiento parcial de capas finales.

Tambien se compara entrenamiento con y sin augmentation, y configuraciones optimizadas frente a configuraciones base.

## VII. Resultados

Tabla comparativa generada automaticamente:

{comparison_text}

## VIII. Discusion

Se espera que transfer learning mejore la convergencia frente al entrenamiento desde cero, especialmente con clases minoritarias. Fine tuning puede mejorar resultados si se controla el learning rate para evitar catastrophic forgetting. MobileNetV2 aporta eficiencia, mientras DenseNet121 y EfficientNetB0 suelen ofrecer buena reutilizacion de caracteristicas.

## IX. Conclusiones

El proyecto establece una metodologia reproducible para comparar CNN en RAF-DB. El EDA confirma integridad de imagenes y etiquetas, pero evidencia un desbalance considerable. Las etapas de augmentation, ponderacion de clases y metricas macro son esenciales para una comparacion cientifica justa.

## X. Trabajos Futuros

- Evaluar modelos Vision Transformer.
- Aplicar focal loss.
- Usar validacion cruzada estratificada si se dispone de mas particiones.
- Analizar robustez ante oclusiones e iluminacion.
- Integrar calibracion de probabilidades.

## Referencias

[1] S. Li, W. Deng, and J. Du, "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild," IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[2] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv:1409.1556, 2014.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.

[4] M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.

[5] M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019.

[6] G. Huang, Z. Liu, L. van der Maaten, and K. Weinberger, "Densely Connected Convolutional Networks," CVPR, 2017.
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path
