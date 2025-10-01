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
import requests

# ----------------------------
# SISTEMA ROBUSTO DE DESCARGA DE MODELOS
# ----------------------------
def descargar_modelo_seguro(url, nombre_archivo):
    """Descargar modelo con múltiples métodos de respaldo"""
    st.info(f"📥 Intentando descargar {nombre_archivo}...")
    
    # Método 1: gdown (principal)
    try:
        st.write("🔄 Usando gdown...")
        gdown.download(url, nombre_archivo, quiet=False)
        if os.path.exists(nombre_archivo) and os.path.getsize(nombre_archivo) > 1000000:  # >1MB
            st.success(f"✅ {nombre_archivo} descargado con gdown")
            return True
    except Exception as e:
        st.warning(f"⚠️ gdown falló: {e}")
    
    # Método 2: requests (respaldo)
    try:
        st.write("🔄 Usando requests...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(nombre_archivo, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            if os.path.exists(nombre_archivo):
                st.success(f"✅ {nombre_archivo} descargado con requests")
                return True
    except Exception as e:
        st.warning(f"⚠️ requests falló: {e}")
    
    # Método 3: URL alternativa
    try:
        st.write("🔄 Probando URL alternativa...")
        # Convertir URL de Google Drive a formato directo
        file_id = url.split('id=')[1] if 'id=' in url else url.split('/')[-1]
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        response = session.get(direct_url, stream=True)
        
        # Manejar confirmación de archivos grandes
        for key, value in response.cookies.items():
            if 'download_warning' in key:
                direct_url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                response = session.get(direct_url, stream=True)
                break
        
        with open(nombre_archivo, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.exists(nombre_archivo) and os.path.getsize(nombre_archivo) > 1000000:
            st.success(f"✅ {nombre_archivo} descargado con URL alternativa")
            return True
    except Exception as e:
        st.warning(f"⚠️ URL alternativa falló: {e}")
    
    return False

@st.cache_resource
def cargar_modelo_frutas():
    """Cargar modelo de frutas con sistema robusto"""
    modelo_path = "w_best.pt"
    url = "https://drive.google.com/uc?id=16BNxvPRSwUQEKULlgKhG2jRUyUNnSApu"
    
    # Verificar si el modelo existe y es válido
    if not os.path.exists(modelo_path) or os.path.getsize(modelo_path) < 1000000:
        st.warning("🔄 Modelo de frutas no encontrado o corrupto, descargando...")
        if not descargar_modelo_seguro(url, modelo_path):
            st.error("❌ No se pudo descargar el modelo de frutas")
            return None
    
    try:
        modelo = YOLO(modelo_path)
        st.success("✅ Modelo de frutas cargado correctamente")
        return modelo
    except Exception as e:
        st.error(f"❌ Error cargando modelo de frutas: {e}")
        return None

@st.cache_resource
def cargar_modelo_placas():
    """Cargar modelo de placas con sistema robusto"""
    modelo_path = "W_PLACA.pt"
    url = "https://drive.google.com/uc?id=12KSiZvxS262NPQ1s-hdsOxJliHSMS3tS"
    
    # Verificar si el modelo existe y es válido
    if not os.path.exists(modelo_path) or os.path.getsize(modelo_path) < 1000000:
        st.warning("🔄 Modelo de placas no encontrado o corrupto, descargando...")
        if not descargar_modelo_seguro(url, modelo_path):
            st.error("❌ No se pudo descargar el modelo de placas")
            return None
    
    try:
        modelo = YOLO(modelo_path)
        st.success("✅ Modelo de placas cargado correctamente")
        return modelo
    except Exception as e:
        st.error(f"❌ Error cargando modelo de placas: {e}")
        return None

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
# CONFIGURACIÓN DE LA APLICACIÓN
# ----------------------------
def init_session_state():
    """Inicializar variables de sesión"""
    defaults = {
        "imagen_actual": None,
        "detecciones_historial": [],
        "resultado_actual": None,
        "texto_placa_actual": "",
        "recortes_placas": [],
        "rtsp_url": "",
        "roi_coords": None,
        "modelos_cargados": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def configurar_pagina():
    """Configurar página de Streamlit"""
    st.set_page_config(
        page_title="🌴🚗 Sistema Dual CNN",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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
# INTERFAZ PRINCIPAL
# ----------------------------
def main():
    configurar_pagina()
    init_session_state()
    
    st.title("🔬 Sistema Dual CNN - Detección Inteligente")
    st.markdown("Sistema de detección con dos redes neuronales especializadas")
    
    # PANEL DE CONTROL DE MODELOS
    st.sidebar.header("🔧 Panel de Control de Modelos")
    
    # Verificar estado de modelos
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Verificar Modelos", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Limpiar Cache", use_container_width=True):
            try:
                # Limpiar cache de modelos
                cargar_modelo_frutas.clear()
                cargar_modelo_placas.clear()
                st.success("Cache limpiado")
                st.rerun()
            except:
                st.rerun()
    
    # Mostrar estado de modelos
    st.sidebar.subheader("📊 Estado de Modelos")
    
    modelo_frutas_ok = os.path.exists("w_best.pt") and os.path.getsize("w_best.pt") > 1000000
    modelo_placas_ok = os.path.exists("W_PLACA.pt") and os.path.getsize("W_PLACA.pt") > 1000000
    
    if modelo_frutas_ok:
        st.sidebar.success("✅ Frutas: w_best.pt")
    else:
        st.sidebar.error("❌ Frutas: Faltante")
    
    if modelo_placas_ok:
        st.sidebar.success("✅ Placas: W_PLACA.pt")
    else:
        st.sidebar.error("❌ Placas: Faltante")
    
    # CARGAR MODELOS
    if not modelo_frutas_ok or not modelo_placas_ok:
        st.warning("⚠️ Algunos modelos no están disponibles")
        
        if st.button("🚀 Descargar Modelos Automáticamente"):
            with st.spinner("Descargando modelos..."):
                # Descargar ambos modelos
                modelo_frutas = cargar_modelo_frutas()
                modelo_placas = cargar_modelo_placas()
                
                if modelo_frutas and modelo_placas:
                    st.success("🎉 Todos los modelos cargados correctamente!")
                    st.session_state.modelos_cargados = True
                    st.rerun()
                else:
                    st.error("❌ No se pudieron cargar todos los modelos")
    
    else:
        # Cargar modelos si existen
        with st.spinner("🔄 Cargando modelos YOLO..."):
            modelo_frutas = cargar_modelo_frutas()
            modelo_placas = cargar_modelo_placas()
            
            if modelo_frutas and modelo_placas:
                st.session_state.modelos_cargados = True
            else:
                st.error("❌ Error crítico cargando modelos")
                st.stop()
    
    # Si los modelos no están cargados, mostrar opciones de descarga
    if not st.session_state.modelos_cargados:
        st.error("""
        ❌ **No se pudieron cargar los modelos necesarios**
        
        **Opciones de solución:**
        
        1. **Haz clic en 'Descargar Modelos Automáticamente'** arriba
        2. **Descarga manual desde tu terminal:**
        ```bash
        # Instalar gdown si no lo tienes
        pip install gdown requests
        
        # Descargar modelos
        gdown "https://drive.google.com/uc?id=16BNxvPRSwUQEKULlgKhG2jRUyUNnSApu" -O w_best.pt
        gdown "https://drive.google.com/uc?id=12KSiZvxS262NPQ1s-hdsOxJliHSMS3tS" -O W_PLACA.pt
        ```
        3. **Verifica tu conexión a internet**
        4. **Reinicia la aplicación**
        """)
        
        # Opción para forzar recarga
        if st.button("🔄 Reintentar Carga"):
            st.rerun()
        
        return
    
    # CONTINUAR CON LA APLICACIÓN SI LOS MODELOS ESTÁN CARGADOS
    st.success("🎉 ¡Sistema listo! Todos los modelos cargados correctamente")
    
    # Configuración principal
    st.sidebar.header("⚙️ Configuración")
    confianza = st.sidebar.slider("🎚️ Confianza mínima", 0.0, 1.0, 0.5, 0.01)
    
    # Inicializar tracker
    if "tracker" not in st.session_state:
        st.session_state.tracker = SimpleTracker(max_age=30)
    
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
                        modelo_frutas, 
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
                        modelo_placas,
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


