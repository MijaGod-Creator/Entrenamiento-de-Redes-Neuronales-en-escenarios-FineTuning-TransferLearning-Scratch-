# 📱 Modelos Exportados para Android Studio

Esta carpeta contiene los modelos de Inteligencia Artificial de tu tesis exportados a formatos de producción optimizados para dispositivos móviles (**Android**).

---

## 📂 Archivos en esta Carpeta

Para cada modelo se han generado dos versiones:
1. **`.ptl` (TorchScript Lite Interpreter):** Formato oficial de **PyTorch Mobile**. Está altamente optimizado y reducido para cargarse directamente con la biblioteca nativa de PyTorch en Android (Java o Kotlin).
2. **`.onnx` (Open Neural Network Exchange):** Formato universal, ideal para cargarse usando **ONNX Runtime Mobile SDK**, que es excelente para aceleración por hardware en procesadores móviles.

### 📌 Modelos Disponibles:
* **Poster V2 (Campeón - 85.63%):**
  * `poster_v2_scratch_aug_mobile.ptl`
  * `poster_v2_scratch_aug.onnx`
* **QCS (84.91%):**
  * `qcs_scratch_aug_mobile.ptl`
  * `qcs_scratch_aug.onnx`
* **SwinFace (Swin Transformer - 69.39%):**
  * `swin_face_scratch_aug_mobile.ptl` *(Nota: Debido a operaciones de atención dinámica, solo el formato .ptl está disponible).*
* **MobileNetV2 (Ligero para celulares):**
  * `mobilenetv2_scratch_noaug_mobile.ptl`
  * `mobilenetv2_scratch_noaug.onnx`

---

## 🛠️ Cómo integrarlos en tu App de Android Studio

### Paso 1: Agregar las dependencias en `build.gradle` (Module:app)

Para usar la versión oficial de **PyTorch Mobile** (archivos `.ptl`), añade esto a tus dependencias:

```groovy
dependencies {
    // Dependencias básicas de PyTorch Mobile
    implementation 'org.pytorch:pytorch_android_lite:2.1.0'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:2.1.0'
}
```

O si prefieres usar **ONNX Runtime** (archivos `.onnx`):

```groovy
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
}
```

### Paso 2: Colocar los modelos en tu App
Copia el archivo del modelo que elijas (ej. `poster_v2_scratch_aug_mobile.ptl`) dentro de la carpeta de recursos de Android Studio:
📁 `app/src/main/assets/`

---

### Paso 3: Código de Carga y Predicción en Java/Kotlin (Ejemplo PyTorch Mobile)

Aquí tienes una plantilla para cargar el modelo y hacer la predicción sobre el rostro recortado por la cámara:

#### ☕ Kotlin Code Example:
```kotlin
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import android.graphics.Bitmap

// 1. Cargar el modelo desde la carpeta assets
val module = LiteModuleLoader.loadModuleFromAsset(assets, "poster_v2_scratch_aug_mobile.ptl")

// 2. Preprocesamiento de la imagen (Rostro de 224x224 píxeles)
// Redimensionar el Bitmap recortado por la cámara a 224x224
val bitmapResized = Bitmap.createScaledBitmap(faceBitmap, 224, 224, true)

// Normalizar la imagen utilizando la media y desviación estándar de ImageNet
val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
    bitmapResized,
    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
    TensorImageUtils.TORCHVISION_NORM_STD_RGB
)

// 3. Ejecutar inferencia (Inferencia rápida)
val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
val scores = outputTensor.dataAsFloatArray

// 4. Mapear las 7 emociones de salida (Mismo orden que RAF-DB)
val emociones = arrayOf("Sorpresa", "Miedo", "Disgusto", "Felicidad", "Tristeza", "Ira", "Neutral")

// Obtener el índice con el valor máximo de confianza (Softmax)
var maxIndex = 0
var maxScore = -Float.MAX_VALUE
for (i in scores.indices) {
    if (scores[i] > maxScore) {
        maxScore = scores[i]
        maxIndex = i
    }
}

val emocionDetectada = emociones[maxIndex]
println("La emoción detectada es: $emocionDetectada")
```

---

## 💡 Recomendaciones para la App Android:
1. **Detección de Rostros:** No le pases la foto completa de la cámara al modelo. Usa el SDK gratuito de **Google ML Kit Face Detection** para recortar el rostro primero en tiempo real, y pásale únicamente ese recorte de la cara a tu modelo.
2. **Modelo Recomendado:** 
   * **Para rendimiento y precisión:** Usa `poster_v2_scratch_aug_mobile.ptl`. Es súper rápido e increíblemente exacto.
   * **Para celulares de gama baja:** Usa `mobilenetv2_scratch_noaug.onnx` o `mobilenetv2_scratch_noaug_mobile.ptl` para menor consumo de batería y CPU.
