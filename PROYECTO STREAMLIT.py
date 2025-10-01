import gdown
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import ultralytics
from ultralytics import YOLO
import json
from collections import defaultdict, deque
import zipfile
from io import BytesIO

# ----------------------------
# CONFIGURACIÓN MEJORADA DE DESCARGAS
# ----------------------------
def verificar_archivo_modelo(nombre_archivo):
    """Verificar que el archivo existe y tiene tamaño adecuado"""
    if not os.path.exists(nombre_archivo):
        return False, "No existe"
    
    tamaño = os.path.getsize(nombre_archivo)
    if tamaño < 1000000:  # Menos de 1MB = corrupto
        return False, f"Archivo corrupto ({tamaño} bytes)"
    
    return True, f"OK ({tamaño // 1000000}MB)"

def descargar_modelo_directo(url, output):
    """Descargar modelo con método directo"""
    try:
        # Método 1: gdown directo
        gdown.download(url, output, quiet=False, fuzzy=True)
        return True
    except Exception as e:
        st.warning(f"Intento 1 falló: {e}")
        try:
            # Método 2: gdown con formato alternativo
            file_id = url.split('id=')[1] if 'id=' in url else url
            download_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(download_url, output, quiet=False)
            return True
        except Exception as e2:
            st.error(f"Intento 2 falló: {e2}")
            return False

@st.cache_resource
def inicializar_modelos():
    """Inicializar todos los modelos con verificación robusta"""
    
    modelos_info = {
        "frutas": {
            "path": "w_best.pt", 
            "url": "https://drive.google.com/uc?id=16BNxvPRSwUQEKULlgKhG2jRUyUNnSApu"
        },
        "placas": {
            "path": "W_PLACA.pt", 
            "url": "https://drive.google.com/uc?id=12KSiZvxS262NPQ1s-hdsOxJliHSMS3tS"
        }
    }
    
    modelos_cargados = {}
    
    for nombre, info in modelos_info.items():
        archivo = info["path"]
        url = info["url"]
        
        st.write(f"**Verificando {nombre}...**")
        
        # Verificar si el archivo ya existe y es válido
        existe, mensaje = verificar_archivo_modelo(archivo)
        
        if existe:
            st.success(f"✅ {archivo} - {mensaje}")
            try:
                modelo = YOLO(archivo)
                modelos_cargados[nombre] = modelo
                st.success(f"✅ Modelo {nombre} cargado correctamente")
            except Exception as e:
                st.error(f"❌ Error cargando {archivo}: {e}")
                # Intentar re-descargar
                st.info("🔄 Intentando re-descargar...")
                if descargar_modelo_directo(url, archivo):
                    try:
                        modelo = YOLO(archivo)
                        modelos_cargados[nombre] = modelo
                        st.success(f"✅ Modelo {nombre} cargado después de re-descarga")
                    except Exception as e2:
                        st.error(f"❌ Error persistente con {archivo}: {e2}")
        else:
            st.warning(f"⚠️ {archivo} - {mensaje}")
            st.info("📥 Descargando...")
            
            if descargar_modelo_directo(url, archivo):
                # Verificar descarga
                existe_descarga, mensaje_descarga = verificar_archivo_modelo(archivo)
                if existe_descarga:
                    try:
                        modelo = YOLO(archivo)
                        modelos_cargados[nombre] = modelo
                        st.success(f"✅ Modelo {nombre} descargado y cargado")
                    except Exception as e:
                        st.error(f"❌ Error cargando {archivo} después de descarga: {e}")
                else:
                    st.error(f"❌ Descarga falló: {mensaje_descarga}")
            else:
                st.error(f"❌ No se pudo descargar {archivo}")
    
    return modelos_cargados

# Diccionario para caracteres de placas
ID_TO_CHAR = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z', 36: 'placa'
}

# ----------------------------
# CLASE DE TRACKING
# ----------------------------
class SimpleTracker:
    def __init__(self, max_age=30):
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        
        if not detections:
            self.tracks = [track for track in self.tracks 
                          if self.frame_count - track['last_seen'] < self.max_age]
            return []
        
        tracks_with_ids = []
        
        for det in detections:
            x1, y1, x2, y2, conf = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            best_track = None
            min_distance = float('inf')
            
            for track in self.tracks:
                if self.frame_count - track['last_seen'] < self.max_age:
                    track_center_x = (track['x1'] + track['x2']) / 2
                    track_center_y = (track['y1'] + track['y2']) / 2
                    
                    distance = np.sqrt((center_x - track_center_x)**2 + 
                                     (center_y - track_center_y)**2)
                    
                    if distance < 100 and distance < min_distance:
                        min_distance = distance
                        best_track = track
            
            if best_track:
                best_track.update({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf,
                    'last_seen': self.frame_count
                })
                tracks_with_ids.append((x1, y1, x2, y2, conf, best_track['id']))
            else:
                new_track = {
                    'id': self.next_id,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf,
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count
                }
                self.tracks.append(new_track)
                tracks_with_ids.append((x1, y1, x2, y2, conf, self.next_id))
                self.next_id += 1
        
        self.tracks = [track for track in self.tracks 
                      if self.frame_count - track['last_seen'] < self.max_age]
        
        return tracks_with_ids

# ----------------------------
# FUNCIONES DE PROCESAMIENTO
# ----------------------------
def procesar_imagen_frutas(modelo, imagen, confianza_min=0.5):
    """Procesar imagen para detectar frutas"""
    try:
        resultados = modelo.predict(
            source=imagen,
            conf=confianza_min,
            imgsz=640,
            verbose=False
        )
        
        detecciones = []
        img_resultado = None
        
        for r in resultados:
            img_resultado = r.plot()
            for box in r.boxes:
                clase_id = int(box.cls[0].item())
                clase = modelo.names[clase_id]
                confianza = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detecciones.append({
                    "clase": clase,
                    "confianza": round(confianza, 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tipo": "fruta"
                })
        
        return img_resultado, sorted(detecciones, key=lambda x: x['confianza'], reverse=True)
    
    except Exception as e:
        st.error(f"Error procesando frutas: {str(e)}")
        return None, []

def procesar_frame_con_tracking(modelo, frame, tracker, confianza_min=0.5):
    """Procesar frame con tracking para evitar duplicados"""
    try:
        resultados = modelo.predict(
            source=frame,
            conf=confianza_min,
            imgsz=640,
            verbose=False
        )
        
        detecciones_raw = []
        
        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confianza = float(box.conf[0].item())
                detecciones_raw.append((x1, y1, x2, y2, confianza))
        
        # Aplicar tracking
        tracks = tracker.update(detecciones_raw)
        
        return tracks
    
    except Exception as e:
        st.error(f"Error en tracking: {str(e)}")
        return []

def extraer_recorte_placa(frame, bbox, padding=15):
    """Extraer recorte de placa con padding"""
    x1, y1, x2, y2 = bbox
    
    # Agregar padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    
    recorte = frame[y1:y2, x1:x2]
    return recorte

# ----------------------------
# CONFIGURACIÓN DE LA APLICACIÓN
# ----------------------------
def init_session_state():
    """Inicializar variables de sesión"""
    if "imagen_actual" not in st.session_state:
        st.session_state.imagen_actual = None
    if "detecciones_historial" not in st.session_state:
        st.session_state.detecciones_historial = []
    if "resultado_actual" not in st.session_state:
        st.session_state.resultado_actual = None
    if "texto_placa_actual" not in st.session_state:
        st.session_state.texto_placa_actual = ""
    if "tracker" not in st.session_state:
        st.session_state.tracker = SimpleTracker(max_age=30)
    if "recortes_placas" not in st.session_state:
        st.session_state.recortes_placas = []
    if "modelos_inicializados" not in st.session_state:
        st.session_state.modelos_inicializados = False

def configurar_pagina():
    """Configurar página de Streamlit"""
    st.set_page_config(
        page_title="🌴🚗 Sistema Dual CNN",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ----------------------------
# INTERFAZ PRINCIPAL
# ----------------------------
def main():
    configurar_pagina()
    init_session_state()
    
    st.title("🔬 Sistema Dual CNN - Detección Inteligente")
    st.markdown("Sistema de detección con dos redes neuronales especializadas")
    
    # PANEL DE INICIALIZACIÓN
    st.sidebar.header("🚀 Inicialización del Sistema")
    
    if not st.session_state.modelos_inicializados:
        st.info("🔄 **Inicializando modelos...**")
        
        with st.spinner("Cargando modelos YOLO..."):
            modelos = inicializar_modelos()
            
            if "frutas" in modelos and "placas" in modelos:
                st.session_state.modelo_frutas = modelos["frutas"]
                st.session_state.modelo_placas = modelos["placas"]
                st.session_state.modelos_inicializados = True
                st.success("🎉 ¡Sistema inicializado correctamente!")
                st.rerun()
            else:
                st.error("❌ No se pudieron cargar todos los modelos")
                
                # Mostrar solución paso a paso
                st.markdown("""
                ### 🔧 Solución Manual:
                
                1. **Abre una terminal en la carpeta de tu proyecto**
                2. **Ejecuta estos comandos:**
                ```bash
                # Navega a tu carpeta del proyecto
                cd /ruta/a/tu/proyecto
                
                # Elimina archivos problemáticos
                rm -f w_best.pt W_PLACA.pt
                
                # Descarga manualmente
                gdown "https://drive.google.com/uc?id=16BNxvPRSwUQEKULlgKhG2jRUyUNnSApu" -O w_best.pt
                gdown "https://drive.google.com/uc?id=12KSiZvxS262NPQ1s-hdsOxJliHSMS3tS" -O W_PLACA.pt
                
                # Verifica que se descargaron
                ls -la *.pt
                ```
                3. **Recarga esta página**
                """)
                
                if st.button("🔄 Reintentar Inicialización"):
                    st.rerun()
                
                return
    
    # SIDEBAR CON CONFIGURACIÓN
    st.sidebar.header("⚙️ Configuración")
    confianza = st.sidebar.slider("🎚️ Confianza mínima", 0.0, 1.0, 0.5, 0.01)
    
    # Estado del sistema
    st.sidebar.subheader("📊 Estado del Sistema")
    st.sidebar.success("✅ Modelo frutas cargado")
    st.sidebar.success("✅ Modelo placas cargado")
    
    total_detecciones = len(st.session_state.detecciones_historial)
    st.sidebar.metric("Detecciones totales", total_detecciones)
    
    # Botón de limpieza
    if st.sidebar.button("🗑️ Limpiar Historial", use_container_width=True):
        st.session_state.detecciones_historial = []
        st.session_state.resultado_actual = None
        st.session_state.texto_placa_actual = ""
        st.session_state.recortes_placas = []
        st.session_state.tracker = SimpleTracker(max_age=30)
        st.success("Historial limpiado")
        time.sleep(1)
        st.rerun()
    
    # INTERFAZ PRINCIPAL CON TABS
    tab1, tab2, tab3 = st.tabs(["📸 Cargar Imagen", "🌴 Detectar Frutas", "🚗 Detectar Placas"])
    
    with tab1:
        st.header("📸 Cargar imagen")
        
        metodo = st.radio(
            "Método de entrada:",
            ["📁 Subir archivo", "📷 Cámara web"],
            key="metodo_carga"
        )
        
        if metodo == "📁 Subir archivo":
            archivo = st.file_uploader(
                "Selecciona una imagen",
                type=['jpg', 'jpeg', 'png'],
                key="uploader_imagen"
            )
            if archivo is not None:
                try:
                    imagen_pil = Image.open(archivo)
                    imagen = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
                    st.session_state.imagen_actual = imagen
                    st.success("✅ Imagen cargada correctamente")
                except Exception as e:
                    st.error(f"Error cargando imagen: {str(e)}")
        
        elif metodo == "📷 Cámara web":
            st.info("📱 Este método funciona perfectamente en dispositivos móviles")
            foto = st.camera_input("Toma una foto", key="camera_input")
            
            if foto is not None:
                try:
                    imagen_pil = Image.open(foto)
                    imagen = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
                    st.session_state.imagen_actual = imagen
                    st.success("✅ Imagen capturada correctamente")
                except Exception as e:
                    st.error(f"❌ Error procesando imagen: {str(e)}")
        
        # Mostrar imagen actual
        if st.session_state.imagen_actual is not None:
            st.subheader("🖼️ Imagen cargada")
            st.image(st.session_state.imagen_actual, channels="BGR", use_column_width=True)
    
    with tab2:
        st.header("🌴 Detección de Frutas")
        
        if st.session_state.imagen_actual is None:
            st.warning("⚠️ Primero carga una imagen en la pestaña 'Cargar Imagen'")
        else:
            if st.button("🔍 Detectar Frutas", type="primary", key="btn_frutas", use_container_width=True):
                with st.spinner("🧠 Analizando frutas..."):
                    img_resultado, detecciones = procesar_imagen_frutas(
                        st.session_state.modelo_frutas, 
                        st.session_state.imagen_actual, 
                        confianza
                    )
                    
                    if detecciones and img_resultado is not None:
                        st.session_state.resultado_actual = img_resultado
                        st.session_state.detecciones_historial.extend(detecciones)
                        st.success(f"✅ {len(detecciones)} frutas detectadas")
                    else:
                        st.warning("🔍 No se detectaron frutas con la confianza especificada")
            
            if st.session_state.resultado_actual is not None:
                st.subheader("🎯 Resultado de la detección")
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)
    
    with tab3:
        st.header("🚗 Detección de Placas")
        
        if st.session_state.imagen_actual is None:
            st.warning("⚠️ Primero carga una imagen en la pestaña 'Cargar Imagen'")
        else:
            if st.button("🔍 Detectar Placas", type="primary", key="btn_placas", use_container_width=True):
                with st.spinner("🧠 Analizando placas con tracking..."):
                    tracks = procesar_frame_con_tracking(
                        st.session_state.modelo_placas,
                        st.session_state.imagen_actual,
                        st.session_state.tracker,
                        confianza
                    )
                    
                    if tracks:
                        img_resultado = st.session_state.imagen_actual.copy()
                        
                        for track in tracks:
                            x1, y1, x2, y2, conf, track_id = track
                            
                            cv2.rectangle(img_resultado, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(img_resultado, f"ID:{track_id} {conf:.2f}", 
                                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            recorte = extraer_recorte_placa(st.session_state.imagen_actual, (x1, y1, x2, y2))
                            
                            recorte_data = {
                                'id': track_id,
                                'imagen': recorte,
                                'confianza': conf,
                                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                            }
                            
                            st.session_state.recortes_placas.append(recorte_data)
                        
                        st.session_state.resultado_actual = img_resultado
                        st.success(f"✅ {len(tracks)} placa(s) detectada(s)")
                    else:
                        st.warning("🔍 No se detectaron placas")
            
            if st.session_state.resultado_actual is not None:
                st.subheader("🎯 Resultado de la detección")
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
