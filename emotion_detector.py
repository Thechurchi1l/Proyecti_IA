import joblib
import numpy as np
import librosa
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from transformers import AutoFeatureExtractor, HubertModel

# -------------------------------------------------------------
# 1. CLASE EMOTION DETECTOR
# -------------------------------------------------------------

class EmotionDetector:
    def __init__(self, model_dir):
        # model_dir es la ruta al directorio que contiene los archivos h5, pkl, json
        self.model_dir = model_dir
        self.FRECUENCIA_MUESTREO = 16000
        self.MODEL_NAME = "superb/hubert-large-superb-er"
        
        self._load_components()

    def _load_components(self):
        """Carga HuBERT, DNN, Scaler y Mapeo."""
        
        # Construir rutas absolutas dentro del directorio base
        ruta_modelo =  'modelo_dnn_hubert_final.h5'
        ruta_scaler =  'scaler_hubert_final.pkl'
        ruta_mapeo =  'label_to_id_CREM.json'

        # Carga de HuBERT y Modelos (Asumiendo que las librerías están instaladas)
        self.processor = AutoFeatureExtractor.from_pretrained(self.MODEL_NAME)
        self.model_hubert = HubertModel.from_pretrained(self.MODEL_NAME)
        self.modelo_cargado = load_model(ruta_modelo)
        self.scaler_cargado = joblib.load(ruta_scaler)

        # Carga y Mapeo de Etiquetas
        with open(ruta_mapeo, 'r') as f:
            label_to_id = json.load(f)
            self.id_to_label = {int(v): k for k, v in label_to_id.items()}
        
        print("Módulo EmotionDetector listo.")

    def preprocesar_audio(self, audio_path):
        """Implementa el pipeline de 4096 características."""
        # Se omite el try/except externo para simplificar el flujo; se maneja en la GUI.
        
        waveform, sr = librosa.load(audio_path, sr=self.FRECUENCIA_MUESTREO)
        inputs = self.processor(waveform, sampling_rate=self.FRECUENCIA_MUESTREO, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model_hubert(inputs.input_values)
            sequence_embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        # Agregación (4096 features)
        mean_features = np.mean(sequence_embeddings, axis=0)
        std_features = np.std(sequence_embeddings, axis=0)  
        min_features = np.min(sequence_embeddings, axis=0)
        max_features = np.max(sequence_embeddings, axis=0)
        
        return np.concatenate([mean_features, std_features, min_features, max_features]).reshape(1, -1)

    def predecir_emocion(self, audio_path):
        """Método público que ejecuta el pipeline completo."""
        try:
            vector_crudo = self.preprocesar_audio(audio_path)
            
            # Escalado y Predicción
            vector_escalado = self.scaler_cargado.transform(vector_crudo)
            predicciones = self.modelo_cargado.predict(vector_escalado)[0]
            
            # Decodificación
            clase_predicha_id = np.argmax(predicciones)
            emocion_predicha = self.id_to_label[clase_predicha_id]
            confianza = predicciones[clase_predicha_id]

            # Retorna el resultado como tupla para la GUI
            return emocion_predicha, confianza, self.id_to_label, predicciones
        
        except Exception as e:
            # Retorna un tipo de dato diferente para indicar el error
            return f"ERROR durante la predicción: {e}", 0, {}, []
            # --- EN EL ARCHIVO emotion_detector.py ---
import sounddevice as sd
from scipy.io.wavfile import write as write_wav # Para guardar el archivo WAV

# --- NUEVA FUNCIÓN DE GRABACIÓN ---
def grabar_audio_temporal(duracion_segundos=3):
    """
    Graba audio del micrófono y lo guarda en un archivo temporal con la frecuencia
    de muestreo requerida por HuBERT (16 kHz).
    Retorna la ruta del archivo temporal.
    """
    # 1. Parámetros de Grabación
    # HuBERT requiere 16000 Hz, un canal (mono)
    fs = 16000 
    ruta_temporal = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio_input.wav")
    
    print(f"\n📢 GRABANDO... (Duración: {duracion_segundos} segundos)")
    
    # 2. Capturar el audio
    # Grabar en NumPy array. dtype='int16' es un formato común para WAV.
    recording = sd.rec(int(duracion_segundos * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Espera a que termine la grabación
    
    # 3. Guardar el archivo WAV temporal
    write_wav(ruta_temporal, fs, recording)
    
    print(f"✅ Grabación finalizada y guardada en: {ruta_temporal}")
    return ruta_temporal

# Asegúrate de importar 'write' desde 'scipy.io.wavfile' y 'sd' desde 'sounddevice' en la parte superior.

        