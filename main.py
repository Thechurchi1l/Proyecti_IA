import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import numpy as np
import scipy.io.wavfile as wav

# Importamos la clase lógica
try:
    from emotion_detector import EmotionDetector
except ImportError:
    messagebox.showerror("Error de Módulo", "No se encontró 'emotion_detector.py'.")
    sys.exit(1)

MODEL_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 
# --- CONFIGURACIÓN DE SENSIBILIDAD ---
ID_MIC_GARGANTA = 1  
UMBRAL_SILENCIO = 0.03 # <--- Lo subimos un poco para ser más estrictos

class EmotionApp:
    def __init__(self, master):
        self.master = master
        master.title("Detección de Emociones: HuBERT + DNN") 
        master.geometry("680x560") 
        master.resizable(False, False) 
        
        try:
            self.detector = EmotionDetector(MODEL_DIRECTORY)
        except Exception as e:
            messagebox.showerror("Error Crítico", str(e))
            sys.exit(1)

        self.style = ttk.Style()
        self.style.theme_use('clam') 
        COLOR_FONDO = '#f9f9f9'
        
        self.style.configure('TFrame', background=COLOR_FONDO)
        self.style.configure('TLabel', background=COLOR_FONDO, font=('Arial', 11))
        self.style.configure('Header.TLabel', font=('Arial', 18, 'bold'), foreground='#1E88E5') 
        self.style.configure('Accent.TButton', font=('Arial', 13, 'bold'), padding=10, background='#4CAF50', foreground='white')
        self.style.map('Accent.TButton', background=[('active', '#66BB6A')])
        self.style.configure('Secondary.TButton', font=('Arial', 11), padding=8, background='#607D8B', foreground='white')

        main_frame = ttk.Frame(master, padding="25", relief='flat') 
        main_frame.pack(fill='both', expand=True)

        ttk.Label(main_frame, text="Sistema de Detección de Emociones (HuBERT/DNN)", style='Header.TLabel').grid(row=0, column=0, columnspan=3, pady=15, sticky='n')
        ttk.Separator(main_frame, orient='horizontal').grid(row=1, column=0, columnspan=3, sticky='ew', pady=(5, 15))

        ttk.Label(main_frame, text="1. Grabación en Vivo", font=('Arial', 13, 'bold', 'underline')).grid(row=2, column=0, columnspan=3, sticky='w', padx=10, pady=(15, 5))
        
        duration_frame = ttk.Frame(main_frame)
        duration_frame.grid(row=3, column=0, columnspan=3, sticky='w', padx=10)
        ttk.Label(duration_frame, text="Duración (segundos):").pack(side='left', padx=(0, 10))
        self.duration_var = tk.StringVar(value="3")
        self.duration_entry = ttk.Entry(duration_frame, width=5, textvariable=self.duration_var, justify='center', font=('Arial', 11))
        self.duration_entry.pack(side='left')
        
        self.record_button = ttk.Button(main_frame, text="INICIAR GRABACIÓN Y ANÁLISIS", command=self.toggle_recording, style='Accent.TButton', cursor="hand2")
        self.record_button.grid(row=4, column=0, columnspan=3, pady=15, sticky='ew', padx=10)
        
        self.status_label = ttk.Label(main_frame, text="Estado: Listo para grabar", font=('Arial', 11, 'italic'), foreground='blue')
        self.status_label.grid(row=5, column=0, columnspan=3, sticky='w', padx=10, pady=5)

        ttk.Separator(main_frame, orient='horizontal').grid(row=6, column=0, columnspan=3, sticky='ew', pady=(20, 15))
        ttk.Label(main_frame, text="2. Análisis de Archivo Local", font=('Arial', 13, 'bold', 'underline')).grid(row=7, column=0, columnspan=3, sticky='w', padx=10, pady=5)
        
        self.select_button = ttk.Button(main_frame, text="Seleccionar Archivo (.wav/.mp3)", command=self.select_audio, style='Secondary.TButton', cursor="hand2")
        self.select_button.grid(row=8, column=0, sticky='w', padx=10, pady=10)
        self.file_label = ttk.Label(main_frame, text="Archivo: Ninguno seleccionado", foreground='#757575')
        self.file_label.grid(row=8, column=1, columnspan=2, sticky='w', padx=10, pady=10)
        
        ttk.Separator(main_frame, orient='horizontal').grid(row=9, column=0, columnspan=3, sticky='ew', pady=(20, 15))
        ttk.Label(main_frame, text="RESULTADO DE LA PREDICCIÓN", font=('Arial', 14, 'bold', 'underline')).grid(row=10, column=0, columnspan=3, pady=5)
        
        self.result_label = ttk.Label(main_frame, text="Esperando análisis...", font=('Arial', 24, 'bold'), foreground='#FF9800')
        self.result_label.grid(row=11, column=0, columnspan=3, pady=10)
        self.confidence_label = ttk.Label(main_frame, text="Presiona un botón para comenzar", font=('Arial', 12, 'italic'))
        self.confidence_label.grid(row=12, column=0, columnspan=3, pady=5)

    def toggle_recording(self):
        try:
            duration = int(self.duration_entry.get()) 
            self.record_button.config(text="🔴 GRABANDO...", state='disabled')
            self.status_label.config(text=f"Escuchando laringófono...", foreground='red')
            self.master.update()

            import sounddevice as sd
            fs = 16000
            # Grabamos directamente con el ID 1
            grabacion = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=ID_MIC_GARGANTA)
            sd.wait()
            
            temp_file_path = "temp_vivo.wav"
            wav.write(temp_file_path, fs, grabacion)
            
            # --- CÁLCULO DE ENERGÍA ---
            # El valor 'rms' nos dice cuánta potencia hay realmente
            rms = np.sqrt(np.mean(grabacion**2))
            print(f"DEBUG - Energía capturada: {rms:.6f}") 

            if rms < UMBRAL_SILENCIO:
                if os.path.exists(temp_file_path): os.remove(temp_file_path)
                self.result_label.config(text="SILENCIO DETECTADO", foreground='gray')
                self.confidence_label.config(text=f"Voz insuficiente (Potencia: {rms:.4f})")
            else:
                self.analyze_audio_file(temp_file_path, is_temp=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Fallo en la captura: {e}")
        finally:
            self.record_button.config(text="INICIAR GRABACIÓN Y ANÁLISIS", state='normal')
            self.status_label.config(text="Estado: Listo", foreground='blue')

    def select_audio(self):
        file_path = filedialog.askopenfilename(defaultextension=".wav", filetypes=[("Audio", "*.wav *.mp3 *.flac")])
        if file_path:
            self.file_label.config(text=os.path.basename(file_path))
            self.analyze_audio_file(file_path, is_temp=False)

    def analyze_audio_file(self, file_path, is_temp):
        self.result_label.config(text="Analizando...", font=('Arial', 18), foreground='darkorange')
        self.master.update() 
        
        emocion, confianza, id_to_label, predicciones = self.detector.predecir_emocion(file_path)
        if is_temp and os.path.exists(file_path): os.remove(file_path)
        
        color = '#00796B' if confianza >= 0.60 else '#D32F2F' 
        self.result_label.config(text=f"EMOCIÓN: {emocion.upper()}", font=('Arial', 24, 'bold'), foreground=color)
        self.confidence_label.config(text=f"Confianza: {confianza*100:.2f}%", foreground='black')

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()