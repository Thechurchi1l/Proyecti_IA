import joblib
import numpy as np
import librosa
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from transformers import AutoFeatureExtractor, HubertModel
import sounddevice as sd
from scipy.io.wavfile import write as write_wav

# =============================================================================
# CLASE EMOTIONDETECTOR: MOTOR DE INFERENCIA IA
# =============================================================================
class EmotionDetector:
    """
    Gestiona el pipeline de Inteligencia Artificial:
    1. Carga de modelos (HuBERT, DNN, Scaler).
    2. Extracción de características de audio.
    3. Predicción probabilística de emociones.
    """
    def __init__(self, model_dir):
        """
        Inicializa el motor de IA cargando los archivos binarios necesarios.
        """
        self.model_dir = model_dir
        self.FRECUENCIA_MUESTREO = 16000 # Estándar requerido por HuBERT
        self.MODEL_NAME = "superb/hubert-large-superb-er" # Modelo base de Facebook/Meta
        
        self._load_components()

    def _load_components(self):
        """
        Carga persistente de componentes en memoria (Singleton implícito).
        Carga HuBERT (Transformers), DNN (Keras) y el Escalador (Joblib).
        """
        # Rutas de los activos entrenados
        ruta_modelo = 'modelo_dnn_hubert_final.h5'
        ruta_scaler = 'scaler_hubert_final.pkl'
        ruta_mapeo  = 'label_to_id_CREM.json'

        # --- CARGA DE MODELOS PESADOS ---
        # Procesador y extractor de características HuBERT
        self.processor = AutoFeatureExtractor.from_pretrained(self.MODEL_NAME)
        self.model_hubert = HubertModel.from_pretrained(self.MODEL_NAME)
        
        # Red Neuronal Profunda (DNN) ya entrenada
        self.modelo_cargado = load_model(ruta_modelo)
        
        # Escalador para normalizar los datos antes de la DNN
        self.scaler_cargado = joblib.load(ruta_scaler)

        # --- CARGA DEL DICCIONARIO DE ETIQUETAS ---
        # Traduce los números de salida de la IA (0, 1, 2...) a palabras (Angry, Sad...)
        with open(ruta_mapeo, 'r') as f:
            label_to_id = json.load(f)
            self.id_to_label = {int(v): k for k, v in label_to_id.items()}
        
        print("Módulo EmotionDetector listo y modelos cargados en memoria.")

    def preprocesar_audio(self, audio_path):
        """
        TRANSFORMACIÓN DE AUDIO A VECTOR (PIPELINE 4096):
        1. Carga el audio a 16kHz.
        2. HuBERT genera 'embeddings' (representaciones matemáticas ocultas).
        3. Se calculan estadísticos (Media, Desviación, Mín, Máx) para formar 
           un vector final de 4096 características (1024 * 4).
        """
        # Carga física del archivo WAV
        waveform, sr = librosa.load(audio_path, sr=self.FRECUENCIA_MUESTREO)
        
        # Preparación para HuBERT (PyTorch)
        inputs = self.processor(waveform, sampling_rate=self.FRECUENCIA_MUESTREO, return_tensors="pt")

        with torch.no_grad():
            # Inferencia en el modelo HuBERT para extraer el estado oculto (embeddings)
            outputs = self.model_hubert(inputs.input_values)
            sequence_embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        # --- AGREGACIÓN ESTADÍSTICA (REDUCCIÓN DE DIMENSIONALIDAD) ---
        # Colapsamos la secuencia temporal en estadísticos globales
        mean_features = np.mean(sequence_embeddings, axis=0) # Promedio
        std_features  = np.std(sequence_embeddings, axis=0)  # Dispersión
        min_features  = np.min(sequence_embeddings, axis=0)  # Valores mínimos
        max_features  = np.max(sequence_embeddings, axis=0)  # Valores máximos
        
        # Concatenación final: 1024 + 1024 + 1024 + 1024 = 4096 features
        return np.concatenate([mean_features, std_features, min_features, max_features]).reshape(1, -1)

    def predecir_emocion(self, audio_path):
        """
        FLUJO FINAL DE PREDICCIÓN:
        Toma una ruta de audio y devuelve la emoción más probable y su confianza.
        """
        try:
            # 1. Obtener el vector de 4096 características
            vector_crudo = self.preprocesar_audio(audio_path)
            
            # 2. Normalizar el vector con el StandardScaler (Media 0, Varianza 1)
            vector_escalado = self.scaler_cargado.transform(vector_crudo)
            
            # 3. Clasificación con la Red Neuronal (DNN)
            predicciones = self.modelo_cargado.predict(vector_escalado, verbose=0)[0]
            
            # 4. Decodificación de resultados
            clase_predicha_id = np.argmax(predicciones) # Índice con mayor probabilidad
            emocion_predicha = self.id_to_label[clase_predicha_id]
            confianza = predicciones[clase_predicha_id] # Valor entre 0 y 1

            return emocion_predicha, confianza, self.id_to_label, predicciones
        
        except Exception as e:
            return f"ERROR: {e}", 0, {}, []

# =============================================================================
# FUNCIONES AUXILIARES DE CAPTURA
# =============================================================================
def grabar_audio_temporal(duracion_segundos=3):
    """
    Utilidad para capturar audio directamente desde el laringófono.
    Asegura que el formato coincida con los requisitos de la IA (16kHz, Mono).
    """
    fs = 16000 # Frecuencia de muestreo (Sample Rate)
    # Definimos el nombre del archivo temporal de entrada
    ruta_temporal = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio_input.wav")
    
    # Captura mediante la librería sounddevice
    # Se recomienda forzar device=1 si el laringófono está en ese puerto
    recording = sd.rec(int(duracion_segundos * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait() # Bloquea hasta finalizar la grabación
    
    # Escritura del archivo WAV al disco
    write_wav(ruta_temporal, fs, recording)
    
    return ruta_temporal