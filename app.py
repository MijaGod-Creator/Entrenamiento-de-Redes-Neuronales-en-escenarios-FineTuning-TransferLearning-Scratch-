import os
import sys
import json
import base64
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, request, render_template

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.config.experiment import ExperimentConfig
from src.models.model_factory import build_model
from src.preprocessing.tfdata import build_transform

app = Flask(__name__)

# Constants
EMOTION_CLASSES = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
EMOTION_COLOR_MAP = {
    "surprise": "#C77DFF",   # Violet/Purple
    "fear": "#3A86C8",       # Deep Blue
    "disgust": "#38B000",    # Vibrant Green
    "happiness": "#FFD166",  # Bright Yellow/Gold
    "sadness": "#00B4D8",    # Cyan/Teal
    "anger": "#EF476F",      # Vibrant Red
    "neutral": "#6C757D"     # Cool Gray
}
EMOTION_TRANSLATION = {
    "surprise": "sorpresa",
    "fear": "miedo",
    "disgust": "disgusto",
    "happiness": "felicidad",
    "sadness": "tristeza",
    "anger": "ira",
    "neutral": "neutral"
}

# Global variables for model state
model_lock = threading.Lock()
current_model = None
current_model_name = None
current_model_config = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Face Detector
try:
    if hasattr(cv2, 'data') and hasattr(cv2, 'CascadeClassifier'):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        face_cascade = None
except Exception as e:
    print(f"Aviso detector facial Haar: {e}")
    face_cascade = None

def get_best_model_name():
    """Reads model_comparison.csv to find the model with the highest F1-Score."""
    comparison_csv = PROJECT_ROOT / "results" / "model_comparison.csv"
    if comparison_csv.exists():
        try:
            df = pd.read_csv(comparison_csv)
            if not df.empty and "f1_macro" in df.columns:
                best_row = df.sort_values(by="f1_macro", ascending=False).iloc[0]
                return best_row["experiment_name"]
        except Exception as e:
            print(f"Error reading comparison CSV: {e}")
    return "vgg16_fine_tuning_noaug" # Default fallback

models_cache = {}

def load_pytorch_model(model_name):
    """Loads a PyTorch model dynamically into memory with caching."""
    global current_model, current_model_name, current_model_config
    
    if model_name in models_cache:
        current_model, current_model_config = models_cache[model_name]
        current_model_name = model_name
        print(f"Model {model_name} loaded from memory cache.")
        return
        
    model_path = PROJECT_ROOT / "saved_models" / model_name / "best_model.pt"
    # Detect if file is missing or is a Git LFS text pointer (< 10 KB)
    if not model_path.exists() or model_path.stat().st_size < 10000:
        print(f"Descargando archivo binario real de {model_name} desde Hugging Face...")
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            hf_repo = "MijaKun/Reconocimiento-de-Emociones-Faciales-RAFDB"
            rel_path = f"saved_models/{model_name}/best_model.pt"
            downloaded = hf_hub_download(repo_id=hf_repo, filename=rel_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloaded, model_path)
        except Exception as e:
            print(f"Error descargando modelo desde Hugging Face: {e}")
            raise FileNotFoundError(f"Model file not found at {model_path}: {e}")
        
    print(f"Loading model: {model_name} on {device}...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    config_data = checkpoint["config"]
    
    # Reconstruct config
    allowed_keys = set(ExperimentConfig.__dataclass_fields__.keys())
    config = ExperimentConfig(**{k: v for k, v in config_data.items() if k in allowed_keys})
    
    # Build architecture and load state dict
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    models_cache[model_name] = (model, config)
    current_model = model
    current_model_name = model_name
    current_model_config = config
    print(f"Model {model_name} loaded successfully.")

# Load the best model on startup
try:
    best_model = get_best_model_name()
    load_pytorch_model(best_model)
except Exception as e:
    print(f"Warning: Could not load initial model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Returns a list of all available models, their metrics, and which one is recommended."""
    models_dir = PROJECT_ROOT / "saved_models"
    available_models = []
    
    # Read metrics comparison
    comparison_csv = PROJECT_ROOT / "results" / "model_comparison.csv"
    metrics_map = {}
    best_model_name = get_best_model_name()
    
    if comparison_csv.exists():
        try:
            df = pd.read_csv(comparison_csv)
            for _, row in df.iterrows():
                metrics_map[row["experiment_name"]] = {
                    "accuracy": float(row["accuracy"]),
                    "f1_macro": float(row["f1_macro"]),
                    "inference_ms": float(row["inference_ms_per_image"]),
                    "num_parameters": int(row["num_parameters"])
                }
        except Exception as e:
            print(f"Error parsing model comparison metrics: {e}")

    model_names_to_check = set()
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir() and "smoke_test" not in d.name:
                model_names_to_check.add(d.name)
    if metrics_map:
        for k in metrics_map.keys():
            if "smoke_test" not in k:
                model_names_to_check.add(k)
                
    for name in model_names_to_check:
        metrics = metrics_map.get(name, {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "inference_ms": 0.0,
            "num_parameters": 0
        })
        
        available_models.append({
            "name": name,
            "is_current": name == current_model_name,
            "is_recommended": name == best_model_name,
            "metrics": metrics
        })
            
    # Sort available models: recommended first, then by F1 score descending
    available_models.sort(key=lambda x: (x["is_recommended"], x["metrics"]["f1_macro"]), reverse=True)
    return jsonify(available_models)

@app.route('/api/select_model', methods=['POST'])
def select_model():
    """Changes the active model."""
    data = request.get_json() or {}
    model_name = data.get("model_name")
    
    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
        
    try:
        with model_lock:
            load_pytorch_model(model_name)
        return jsonify({
            "status": "success",
            "current_model": current_model_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Processes an image, detects the face, and runs emotion prediction."""
    global current_model, current_model_name
    if current_model is None:
        target = current_model_name or get_best_model_name() or "poster_v2_scratch_aug"
        try:
            with model_lock:
                load_pytorch_model(target)
        except Exception as e:
            print(f"Auto-load model error: {e}")
            return jsonify({"error": f"No model loaded and failed to auto-load: {str(e)}"}), 500
        
    # Get image from request
    file = request.files.get("image")
    if file:
        # Standard file upload
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        # Check base64 JSON upload (common for webcam frames)
        data = request.get_json() or {}
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "No image data received"}), 400
            
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
            
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400
        
    height, width = img.shape[:2]
    
    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    faces_results = []
    
    if len(faces) > 0:
        for idx, (x, y, w, h) in enumerate(faces):
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.05)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(width, x + w + margin_x)
            y2 = min(height, y + h + margin_y)
            
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
                
            try:
                # Preprocess cropped face
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                
                # Build standard transform
                transform = build_transform(image_size=(224, 224), train=False, use_augmentation=False)
                face_tensor = transform(face_pil).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    with model_lock:
                        logits = current_model(face_tensor)
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Format results
                predictions = []
                for e_idx, name in enumerate(EMOTION_CLASSES):
                    spanish_name = EMOTION_TRANSLATION.get(name, name)
                    predictions.append({
                        "emotion": spanish_name,
                        "probability": float(probs[e_idx]),
                        "color": EMOTION_COLOR_MAP.get(name, "#6C757D")
                    })
                predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
                primary = predictions[0]
                
                faces_results.append({
                    "id": idx + 1,
                    "face_box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "primary_emotion": primary["emotion"],
                    "probability": primary["probability"],
                    "color": primary["color"],
                    "predictions": predictions
                })
            except Exception as e:
                print(f"Error processing face {idx}: {e}")
    else:
        # Fallback to the entire image if no face is detected
        try:
            face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            transform = build_transform(image_size=(224, 224), train=False, use_augmentation=False)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                with model_lock:
                    logits = current_model(face_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            predictions = []
            for e_idx, name in enumerate(EMOTION_CLASSES):
                spanish_name = EMOTION_TRANSLATION.get(name, name)
                predictions.append({
                    "emotion": spanish_name,
                    "probability": float(probs[e_idx]),
                    "color": EMOTION_COLOR_MAP.get(name, "#6C757D")
                })
            predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
            primary = predictions[0]
            
            faces_results.append({
                "id": 1,
                "face_box": {
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height
                },
                "primary_emotion": primary["emotion"],
                "probability": primary["probability"],
                "color": primary["color"],
                "predictions": predictions,
                "is_fallback": True
            })
        except Exception as e:
            return jsonify({"error": f"Error running fallback inference: {str(e)}"}), 500
            
    return jsonify({
        "faces": faces_results,
        "face_count": len(faces) if len(faces) > 0 else 0,
        "model_used": current_model_name
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
