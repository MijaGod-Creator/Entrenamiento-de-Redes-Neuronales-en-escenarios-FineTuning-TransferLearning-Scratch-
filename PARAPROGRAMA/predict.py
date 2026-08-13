import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

# Agregar directorio actual al PATH para poder importar la carpeta src
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from src.config.experiment import ExperimentConfig
from src.models.model_factory import build_model
from src.preprocessing.tfdata import build_transform

# Configurar el dispositivo de computo (GPU CUDA si esta disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cargar_modelo(path_pesos):
    """
    Carga la arquitectura del modelo y sus pesos entrenados desde un archivo .pt
    """
    if not os.path.exists(path_pesos):
        raise FileNotFoundError(f"No se encontro el archivo de pesos en: {path_pesos}")
        
    print(f"Cargando modelo en {device}...")
    checkpoint = torch.load(path_pesos, map_location=device)
    config_data = checkpoint["config"]
    
    # Reconstruir la configuracion guardada en el entrenamiento
    allowed_keys = set(ExperimentConfig.__dataclass_fields__.keys())
    config = ExperimentConfig(**{k: v for k, v in config_data.items() if k in allowed_keys})
    
    # Construir la arquitectura del modelo y cargar los pesos de entrenamiento
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print("¡Modelo cargado exitosamente!")
    return model

def predecir_emocion(model, path_imagen):
    """
    Recibe la imagen de un rostro recortado, realiza la inferencia y retorna
    la emocion predicha con sus probabilidades.
    """
    # Cargar y preprocesar la imagen
    img = Image.open(path_imagen).convert("RGB")
    
    # build_transform redimensiona la imagen a 224x224, la normaliza para PyTorch y la convierte a tensor
    transform = build_transform(image_size=(224, 224), train=False, use_augmentation=False)
    tensor_img = transform(img).unsqueeze(0).to(device)
    
    # Inferencia sin calcular gradientes
    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
    EMOTION_CLASSES = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
    EMOTION_TRANSLATION = {
        "surprise": "Sorpresa",
        "fear": "Miedo",
        "disgust": "Disgusto",
        "happiness": "Felicidad",
        "sadness": "Tristeza",
        "anger": "Ira",
        "neutral": "Neutral"
    }
    
    # Obtener el resultado
    pred_idx = probs.argmax()
    emocion_ingles = EMOTION_CLASSES[pred_idx]
    emocion_espanol = EMOTION_TRANSLATION.get(emocion_ingles, emocion_ingles)
    confianza = probs[pred_idx]
    
    print(f"\n--- RESULTADO DE LA PREDICCION ---")
    print(f"Emocion Detectada: {emocion_espanol} ({confianza*100:.2f}%)")
    print("----------------------------------")
    for idx, e in enumerate(EMOTION_CLASSES):
        print(f"{EMOTION_TRANSLATION[e]}: {probs[idx]*100:.2f}%")
        
    return emocion_espanol, confianza, probs

if __name__ == "__main__":
    # Ruta al archivo de pesos
    ruta_pesos = os.path.join("saved_models", "poster_v2_scratch_aug", "best_model.pt")
    
    # Ruta de prueba (Reemplaza con tu imagen de rostro)
    ruta_imagen_prueba = "prueba_rostro.jpg"
    
    # Crear una imagen negra de prueba por si no existe una foto real
    if not os.path.exists(ruta_imagen_prueba):
        from PIL import ImageDraw
        img_temp = Image.new("RGB", (224, 224), color=(128, 128, 128))
        d = ImageDraw.Draw(img_temp)
        d.text((50, 100), "Rostro de Prueba", fill=(255, 255, 255))
        img_temp.save(ruta_imagen_prueba)
        print(f"Creado archivo temporal de prueba en: {ruta_imagen_prueba}")
        
    # Cargar y predecir
    try:
        modelo = cargar_modelo(ruta_pesos)
        predecir_emocion(modelo, ruta_imagen_prueba)
    except Exception as e:
        print(f"Error durante la ejecucion: {e}")
