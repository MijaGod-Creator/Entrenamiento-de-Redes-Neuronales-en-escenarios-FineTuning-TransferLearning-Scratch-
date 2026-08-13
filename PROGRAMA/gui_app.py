import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import threading
import time

# Agregar directorio raíz del proyecto al PATH para importar componentes
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config.experiment import ExperimentConfig
from src.models.model_factory import build_model
from src.preprocessing.tfdata import build_transform

# Constantes del proyecto
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
EMOTION_COLOR_MAP = {
    "surprise": (255, 125, 199),   # Morado/Rosa
    "fear": (200, 134, 58),       # Azul Profundo (OpenCV usa BGR)
    "disgust": (0, 176, 56),       # Verde
    "happiness": (102, 209, 255),  # Amarillo/Oro
    "sadness": (216, 180, 0),      # Cian/Celeste
    "anger": (111, 71, 239),       # Rojo/Rosa
    "neutral": (120, 117, 108)     # Gris
}

class DesktopFERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Emociones Faciales - Aplicación de Escritorio")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1e1e24")
        
        # Estado de la aplicación
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.model_name = None
        self.cap = None
        self.running_camera = False
        self.static_image = None
        self.lock = threading.Lock()
        
        # Estilos visuales
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#1e1e24", foreground="#ffffff")
        self.style.configure("TLabel", background="#1e1e24", foreground="#ffffff", font=("Helvetica", 10))
        self.style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), foreground="#ffd166")
        self.style.configure("TCombobox", fieldbackground="#2b2d42", background="#2b2d42", foreground="#ffffff")
        self.style.configure("TButton", background="#e9c46a", foreground="#1e1e24", font=("Helvetica", 10, "bold"))
        self.style.map("TButton", background=[("active", "#f4a261")])
        
        # Crear estructura de paneles
        self.create_widgets()
        
        # Cargar lista de modelos disponibles
        self.scan_available_models()
        
        # Iniciar con el mejor modelo
        self.auto_load_best_model()

    def create_widgets(self):
        # 1. Panel Superior (Header y Controles del Modelo)
        top_frame = tk.Frame(self.root, bg="#2b2d42", height=80)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        title_label = ttk.Label(top_frame, text="SISTEMA DE RECONOCIMIENTO DE EMOCIONES FACIALES (FER)", style="Header.TLabel")
        title_label.pack(side=tk.LEFT, padx=15, pady=15)
        
        # Selector de modelos
        model_frame = tk.Frame(top_frame, bg="#2b2d42")
        model_frame.pack(side=tk.RIGHT, padx=15, pady=15)
        
        model_label = ttk.Label(model_frame, text="Modelo Activo:", font=("Helvetica", 10, "bold"))
        model_label.pack(side=tk.LEFT, padx=5)
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=35, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_changed)
        
        self.status_label = ttk.Label(top_frame, text="Cargando modelo...", font=("Helvetica", 9, "italic"), foreground="#00b4d8")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # 2. Panel Izquierdo (Visor de Video / Imagen)
        self.left_frame = tk.Frame(self.root, bg="#1e1e24")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.video_canvas = tk.Canvas(self.left_frame, bg="#101014", highlightthickness=1, highlightbackground="#e9c46a")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Botones de control del visor
        control_frame = tk.Frame(self.left_frame, bg="#1e1e24")
        control_frame.pack(fill=tk.X, pady=5)
        
        self.btn_webcam = ttk.Button(control_frame, text="Iniciar Cámara", command=self.toggle_webcam)
        self.btn_webcam.pack(side=tk.LEFT, padx=10, pady=5, expand=True, fill=tk.X)
        
        self.btn_upload = ttk.Button(control_frame, text="Subir Foto", command=self.upload_image)
        self.btn_upload.pack(side=tk.LEFT, padx=10, pady=5, expand=True, fill=tk.X)

        # 3. Panel Derecho (Métricas y Gráfico de Barras de Emociones)
        right_frame = tk.Frame(self.root, bg="#2b2d42", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=5)
        
        panel_title = ttk.Label(right_frame, text="Resultados de Detección", font=("Helvetica", 12, "bold"), foreground="#ffd166")
        panel_title.pack(anchor="w", padx=15, pady=15)
        
        # Etiquetas de estado
        self.emotion_label = ttk.Label(right_frame, text="Emoción: Esperando...", font=("Helvetica", 14, "bold"))
        self.emotion_label.pack(anchor="w", padx=15, pady=5)
        
        self.confidence_label = ttk.Label(right_frame, text="Confianza: --", font=("Helvetica", 11))
        self.confidence_label.pack(anchor="w", padx=15, pady=5)
        
        self.latency_label = ttk.Label(right_frame, text="Inferencia: -- ms", font=("Helvetica", 10, "italic"), foreground="#a8dadc")
        self.latency_label.pack(anchor="w", padx=15, pady=5)
        
        # Contenedor para barras de probabilidad
        bars_title = ttk.Label(right_frame, text="Probabilidades por emoción:", font=("Helvetica", 10, "bold"))
        bars_title.pack(anchor="w", padx=15, pady=15)
        
        self.bar_frames = {}
        self.bar_widgets = {}
        
        for emotion in EMOTION_CLASSES:
            esp_name = EMOTION_TRANSLATION.get(emotion, emotion)
            
            frame = tk.Frame(right_frame, bg="#2b2d42")
            frame.pack(fill=tk.X, padx=15, pady=4)
            
            lbl = ttk.Label(frame, text=f"{esp_name}:", width=12, anchor="w", font=("Helvetica", 9))
            lbl.pack(side=tk.LEFT)
            
            canvas = tk.Canvas(frame, height=14, bg="#101014", highlightthickness=0)
            canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            val_lbl = ttk.Label(frame, text="0.0%", width=6, anchor="e", font=("Helvetica", 9))
            val_lbl.pack(side=tk.RIGHT)
            
            self.bar_frames[emotion] = canvas
            self.bar_widgets[emotion] = val_lbl

    def scan_available_models(self):
        """Busca modelos guardados en la carpeta saved_models."""
        models_dir = PROJECT_ROOT / "saved_models"
        available = []
        if models_dir.exists():
            for p in models_dir.glob("*/best_model.pt"):
                if "smoke_test" not in p.parent.name:
                    available.append(p.parent.name)
        
        # Ordenar alfabéticamente
        available.sort()
        self.model_combo['values'] = available
        return available

    def auto_load_best_model(self):
        """Busca y carga automáticamente el mejor modelo basado en la comparativa."""
        models = self.model_combo['values']
        if not models:
            self.status_label.config(text="⚠️ No se encontraron modelos en saved_models/", foreground="#e63946")
            return
            
        best_model = "poster_v2_scratch_aug"  # Fallback recomendado
        comparison_csv = PROJECT_ROOT / "results" / "model_comparison.csv"
        
        if comparison_csv.exists():
            try:
                df = pd.read_csv(comparison_csv)
                if not df.empty and "f1_macro" in df.columns:
                    best_row = df.sort_values(by="f1_macro", ascending=False).iloc[0]
                    best_model = best_row["experiment_name"]
            except Exception:
                pass
                
        if best_model in models:
            self.model_var.set(best_model)
        else:
            self.model_var.set(models[0])
            
        self.on_model_changed()

    def load_model_thread(self, name):
        """Carga el modelo en segundo plano para no congelar la interfaz."""
        model_path = PROJECT_ROOT / "saved_models" / name / "best_model.pt"
        try:
            self.status_label.config(text="Cargando modelo...", foreground="#00b4d8")
            checkpoint = torch.load(model_path, map_location=self.device)
            config_data = checkpoint["config"]
            
            allowed_keys = set(ExperimentConfig.__dataclass_fields__.keys())
            config = ExperimentConfig(**{k: v for k, v in config_data.items() if k in allowed_keys})
            
            # Construir modelo y cargar pesos
            new_model = build_model(config)
            new_model.load_state_dict(checkpoint["model_state_dict"])
            new_model.to(self.device)
            new_model.eval()
            
            with self.lock:
                self.model = new_model
                self.model_name = name
                
            self.status_label.config(text="⚡ Modelo Cargado y Activo", foreground="#ffd166")
            # Forzar re-evaluación si hay una foto estática cargada
            if not self.running_camera and self.static_image is not None:
                self.process_static_image()
                
        except Exception as e:
            self.status_label.config(text="❌ Error al cargar modelo", foreground="#e63946")
            messagebox.showerror("Error de carga", f"No se pudo cargar el modelo:\n{str(e)}")

    def on_model_changed(self, event=None):
        name = self.model_var.get()
        if name:
            threading.Thread(target=self.load_model_thread, args=(name,), daemon=True).start()

    def toggle_webcam(self):
        if self.running_camera:
            self.running_camera = False
            self.btn_webcam.config(text="Iniciar Cámara")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.video_canvas.delete("all")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error de Cámara", "No se pudo abrir la cámara web.")
                return
            self.running_camera = True
            self.btn_webcam.config(text="Detener Cámara")
            self.btn_upload.config(state="disabled")
            self.static_image = None
            self.show_frame()

    def show_frame(self):
        if not self.running_camera:
            self.btn_upload.config(state="normal")
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Procesar detección de emociones en la cámara
            processed_frame = self.process_frame(frame)
            
            # Convertir imagen para tkinter
            cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            
            # Redimensionar para ajustar al canvas
            canvas_w = self.video_canvas.winfo_width()
            canvas_h = self.video_canvas.winfo_height()
            
            if canvas_w > 10 and canvas_h > 10:
                img = img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
                
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.imgtk = imgtk
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            
        self.root.after(15, self.show_frame)

    def process_frame(self, frame):
        if self.model is None:
            return frame
            
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        for (x, y, fw, fh) in faces:
            # Cropear la cara
            margin_x = int(fw * 0.05)
            margin_y = int(fh * 0.05)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + fw + margin_x)
            y2 = min(h, y + fh + margin_y)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
                
            # Preprocesar
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            transform = build_transform(image_size=(224, 224), train=False, use_augmentation=False)
            face_tensor = transform(face_pil).unsqueeze(0).to(self.device)
            
            start_t = time.perf_counter()
            with torch.no_grad():
                with self.lock:
                    logits = self.model(face_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            latency_ms = (time.perf_counter() - start_t) * 1000
            
            best_idx = np.argmax(probs)
            emotion = EMOTION_CLASSES[best_idx]
            prob = probs[best_idx]
            
            # Dibujar rectangulos y texto en OpenCV
            color = EMOTION_COLOR_MAP.get(emotion, (128, 128, 128))
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 3)
            
            esp_label = f"{EMOTION_TRANSLATION[emotion]} ({prob*100:.1f}%)"
            cv2.putText(frame, esp_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # Actualizar panel lateral en la UI principal
            self.update_gui_metrics(emotion, prob, probs, latency_ms)
            
        return frame

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not file_path:
            return
            
        self.static_image = cv2.imread(file_path)
        if self.static_image is None:
            messagebox.showerror("Error de Imagen", "No se pudo cargar el archivo seleccionado.")
            return
            
        self.process_static_image()

    def process_static_image(self):
        if self.static_image is None:
            return
            
        frame = self.static_image.copy()
        processed_frame = self.process_frame(frame)
        
        cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        
        # Redimensionar para ajustar al canvas
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        
        # Si el canvas aun no ha sido renderizado del todo, usar dimensiones por defecto
        if canvas_w < 50 or canvas_h < 50:
            canvas_w = 700
            canvas_h = 500
            
        # Calcular relacion de aspecto
        img_w, img_h = img.size
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Centrar la imagen en el canvas
        img_container = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        img_container.paste(img, (offset_x, offset_y))
        
        imgtk = ImageTk.PhotoImage(image=img_container)
        self.video_canvas.imgtk = imgtk
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def update_gui_metrics(self, emotion, prob, all_probs, latency_ms):
        # Actualizar texto principal
        esp_emotion = EMOTION_TRANSLATION.get(emotion, emotion)
        self.emotion_label.config(text=f"Emoción: {esp_emotion}")
        self.confidence_label.config(text=f"Confianza: {prob*100:.1f}%")
        self.latency_label.config(text=f"Inferencia: {latency_ms:.1f} ms")
        
        # Actualizar barras de probabilidad
        for idx, e in enumerate(EMOTION_CLASSES):
            val = all_probs[idx]
            canvas = self.bar_frames[e]
            val_lbl = self.bar_widgets[e]
            
            # Actualizar texto de porcentaje
            val_lbl.config(text=f"{val*100:.1f}%")
            
            # Dibujar barra
            canvas.delete("bar")
            cw = canvas.winfo_width()
            if cw < 10:
                cw = 150 # Fallback si no ha cargado dimensiones de Tkinter
                
            fill_w = int(cw * val)
            
            # Obtener color en formato hex para Tkinter
            bgr = EMOTION_COLOR_MAP.get(e, (128, 128, 128))
            hex_color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}" # Convertir BGR a RGB Hex
            
            canvas.create_rectangle(0, 0, fill_w, 14, fill=hex_color, outline="", tags="bar")

if __name__ == "__main__":
    root = tk.Tk()
    app = DesktopFERApp(root)
    
    # Manejar cierre de la ventana
    def on_closing():
        if app.cap is not None:
            app.cap.release()
        root.destroy()
        sys.exit(0)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
