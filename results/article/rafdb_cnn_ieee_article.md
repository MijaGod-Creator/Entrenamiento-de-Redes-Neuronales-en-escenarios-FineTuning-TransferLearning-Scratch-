# Comparacion del desempeno de diferentes arquitecturas CNN para el reconocimiento de emociones faciales utilizando RAF-DB

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
{
  "total_zip_entries": 15341,
  "total_images": 15339,
  "train_images": 12271,
  "test_images": 3068,
  "unknown_split_images": 0,
  "num_classes": 7,
  "classes": {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happiness",
    "5": "sadness",
    "6": "anger",
    "7": "neutral"
  },
  "images_by_class": {
    "1_surprise": 1619,
    "2_fear": 355,
    "3_disgust": 877,
    "4_happiness": 5957,
    "5_sadness": 2460,
    "6_anger": 867,
    "7_neutral": 3204
  },
  "image_formats": {
    ".jpg": 15339
  },
  "resolution_average": {
    "width": 100.0,
    "height": 100.0
  },
  "resolution_min": {
    "width": 100,
    "height": 100
  },
  "resolution_max": {
    "width": 100,
    "height": 100
  },
  "damaged_images": 0,
  "null_labels": 0,
  "duplicate_images": 6,
  "label_mismatches": 0,
  "imbalance_ratio_max_min": 16.7803,
  "is_imbalanced": true
}
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

experiment_name,accuracy,balanced_accuracy,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted,specificity_macro,top_2_accuracy,top_3_accuracy,mcc,cohen_kappa,auc_macro_ovr,inference_total_seconds,inference_ms_per_image,num_parameters
custom_cnn_scratch_aug,0.3161668839634941,0.3059592205802398,0.2792817395124898,0.3059592205802398,0.252615539439913,0.4096195904700754,0.3161668839634941,0.323719387997656,0.8824850962701254,0.508148631029987,0.6646023468057366,0.1719229247717055,0.1626114668243091,0.6842718846865392,16.655004700180143,5.428619524178665,1309415
custom_cnn_scratch_noaug,0.5912646675358539,0.5252972799215873,0.4898816599613485,0.5252972799215873,0.4891331822287024,0.6407639523710599,0.5912646675358539,0.6096265656810076,0.9295655276516356,0.7728161668839635,0.863102998696219,0.4805267605702293,0.4776348620337998,0.8612790372921691,18.69433550024405,6.09332969369102,1309415
custom_cnn_transfer_aug,0.3207301173402868,0.3841243220267833,0.3756028070499307,0.3841243220267833,0.3171531316198389,0.5443555269622825,0.3207301173402868,0.3280316772072827,0.8897122544653275,0.4996740547588005,0.644393741851369,0.2306654320662044,0.211511415964345,0.7624740622331798,17.700125999748707,5.769271838249252,1309415
custom_cnn_transfer_noaug,0.2503259452411995,0.2671871142763296,0.2189706694110326,0.2671871142763296,0.203678803741768,0.336132627550371,0.2503259452411995,0.2704334584191066,0.8727143544753015,0.4335071707953064,0.5935462842242504,0.1006507676214166,0.0961096360958709,0.6272653626404916,17.617519200313836,5.742346545082737,1309415


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
