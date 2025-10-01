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
# Configuración y descarga de modelos
# ----------------------------
@st.cache_resource
def descargar_y_cargar_modelo_frutas():
    """Descargar y cargar modelo de frutas"""
    ruta_modelo = "CNN_FRUTA.pt" 
    if not os.path.exists(ruta_modelo):
        st.info("📥 Descargando modelo de frutas...") 
        url = "https://drive.google.com/uc?id=13f5QbkwbR-SpQHqkb2tsEcondQMYKNnx" 
        gdown.download(url, ruta_modelo, quiet=False)
    
    return YOLO(ruta_modelo)

@st.cache_resource
def descargar_y_cargar_modelo_placas():
    """Descargar y cargar modelo de placas"""
    ruta_modelo = "W_PLACA.pt" 
    if not os.path.exists(ruta_modelo):
        st.info("📥 Descargando modelo de placas...")
        url = "https://drive.google.com/uc?id=12KSiZvxS262NPQ1s-hdsOxJliHSMS3tS"
        gdown.download(url, ruta_modelo, quiet=False)
    
    return YOLO(ruta_modelo)

# Para RTSP usaremos el mismo modelo de placas
@st.cache_resource 
def descargar_y_cargar_modelo_rtsp():
    ruta_modelo = "det_placa.pt" 
    if not os.path.exists(ruta_modelo):
        st.info("📥 Descargando modelo de placas...")
        url = "https://drive.google.com/uc?id=1MlKp1RDi90XFpz1L3W73kqqhoRnMEdO4"
        gdown.download(url, ruta_modelo, quiet=False)
    
    return YOLO(ruta_modelo)
       
# ----------------------------
# CLASE DE TRACKING PARA RTSP
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
                    
                    distance = np.sqrt((center_x - track_center_x)**2 + (center_y - track_center_y)**2)
                    
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
                    'last_seen': self.frame_count,
                    'save_count': 0,
                    'last_save_frame': 0
                }
                self.tracks.append(new_track)
                tracks_with_ids.append((x1, y1, x2, y2, conf, self.next_id))
                self.next_id += 1
        
        self.tracks = [track for track in self.tracks 
                      if self.frame_count - track['last_seen'] < self.max_age]
        
        return tracks_with_ids

class RTSPCaptureSystem:
    def __init__(self, modelo, rtsp_url, roi=None):
        self.modelo = modelo
        self.rtsp_url = rtsp_url
        self.roi = roi
        
        self.tracker = SimpleTracker(max_age=30)
        self.OUTPUT_DIR = "placas_capturadas"
        
        # Crear directorio si no existe
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
        
        # Inicializar captura de video
        self.cap = cv2.VideoCapture(rtsp_url)
        self.is_running = False
        self.current_frame = None
        self.frame_count = 0
        self.save_count = 0
        
        if not self.cap.isOpened():
            raise ConnectionError(f"No se pudo conectar a la cámara RTSP: {rtsp_url}")

    def process_frame(self):
        """Procesar un frame y detectar placas"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        
        # Aplicar ROI si está configurado
        if self.roi:
            x1, y1, x2, y2 = self.roi
            frame_roi = frame[y1:y2, x1:x2]
        else:
            frame_roi = frame
        
        # USAR EL MODELO RTSP PARA DETECCIÓN
        resultados = self.modelo.predict(
            source=frame_roi,
            conf=0.5,
            imgsz=640,
            verbose=False
        )
        
        # Procesar detecciones
        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                
                # Ajustar coordenadas si hay ROI
                if self.roi:
                    x1 += self.roi[0]
                    y1 += self.roi[1]
                    x2 += self.roi[0]
                    y2 += self.roi[1]
                
                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Placa {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Guardar placa automáticamente cada 30 frames
                if self.frame_count % 30 == 0 and conf > 0.7:
                    self._guardar_placa(frame, (x1, y1, x2, y2))
        
        # Dibujar ROI si está configurado
        if self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "ROI", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Información en pantalla
        cv2.putText(frame, f"Frames: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Placas guardadas: {self.save_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        self.current_frame = frame
        return frame

    def _guardar_placa(self, frame, bbox):
        """Guardar imagen de placa detectada"""
        x1, y1, x2, y2 = bbox
        
        # Agregar padding
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        placa_recorte = frame[y1:y2, x1:x2]
        
        if placa_recorte.size > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"placa_{timestamp}.jpg"
            filepath = os.path.join(self.OUTPUT_DIR, filename)
            
            if cv2.imwrite(filepath, placa_recorte):
                self.save_count += 1
                return True
        
        return False

    def start_capture(self):
        self.is_running = True

    def stop_capture(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()

    def get_stats(self):
        return {
            'frames_procesados': self.frame_count,
            'placas_guardadas': self.save_count
        }

    def get_saved_plates_count(self):
        if os.path.exists(self.OUTPUT_DIR):
            return len([f for f in os.listdir(self.OUTPUT_DIR) if f.endswith('.jpg')])
        return 0

# ----------------------------
# DICCIONARIO PARA CARACTERES DE PLACAS
# ----------------------------
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
# Funciones de procesamiento
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

def procesar_imagen_placas(modelo, imagen, confianza_min=0.5):
    """Procesar imagen para detectar placas"""
    try:
        resultados = modelo.predict(
            source=imagen,
            conf=confianza_min,
            imgsz=640,
            verbose=False
        )
        
        detecciones = []
        img_resultado = None
        texto_placa = ""
        
        for r in resultados:
            img_resultado = r.plot()
            caracteres_detectados = []
            
            for box in r.boxes:
                clase_id = int(box.cls[0].item())
                char = ID_TO_CHAR.get(clase_id, '')
                confianza = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detecciones.append({
                    "clase": char,
                    "confianza": round(confianza, 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tipo": "placa"
                })
                
                if char != 'placa' and char:
                    caracteres_detectados.append({
                        'caracter': char,
                        'x': x1,
                        'confianza': confianza
                    })
            
            # Ordenar caracteres por posición X y formar texto
            if caracteres_detectados:
                caracteres_ordenados = sorted(caracteres_detectados, key=lambda x: x['x'])
                texto_placa = ''.join([c['caracter'] for c in caracteres_ordenados])
        
        return img_resultado, detecciones, texto_placa
    
    except Exception as e:
        st.error(f"Error procesando placas: {str(e)}")
        return None, [], ""

# ----------------------------
# Funciones de exportación JSON
# ----------------------------
def generar_datos_json():
    """Generar datos para exportar en formato JSON"""
    # Obtener placa actual
    placa = st.session_state.texto_placa_actual if st.session_state.texto_placa_actual else "No detectada"
    
    # Obtener frutas del historial
    frutas_detectadas = [d for d in st.session_state.detecciones_historial if d.get('tipo') == 'fruta']
    
    # Contar cantidad total de frutas
    cantidad_fruta = len(frutas_detectadas)
    
    # Clasificar frutas por estado usando las clases YOLO reales
    clasificacion_por_estado = {}
    
    for fruta in frutas_detectadas:
        estado_fruta = fruta['clase']
        confianza = fruta['confianza']
        
        if estado_fruta not in clasificacion_por_estado:
            clasificacion_por_estado[estado_fruta] = {
                "cantidad": 0,
                "confianza_promedio": 0,
                "confianzas": []
            }
        
        clasificacion_por_estado[estado_fruta]["cantidad"] += 1
        clasificacion_por_estado[estado_fruta]["confianzas"].append(confianza)
    
    # Calcular confianza promedio para cada estado
    for estado in clasificacion_por_estado:
        confianzas = clasificacion_por_estado[estado]["confianzas"]
        clasificacion_por_estado[estado]["confianza_promedio"] = round(np.mean(confianzas), 3)
        del clasificacion_por_estado[estado]["confianzas"]
    
    # Estructura JSON final
    datos_json = {
        "placa": placa,
        "cantidad_fruta": cantidad_fruta,
        "clasificacion_fruta_por_estado": clasificacion_por_estado,
        "timestamp": datetime.now().isoformat(),
        "resumen": {
            "total_detecciones": len(st.session_state.detecciones_historial),
            "estados_fruta_detectados": len(clasificacion_por_estado),
            "confianza_promedio_general": round(
                np.mean([d['confianza'] for d in frutas_detectadas]) if frutas_detectadas else 0, 3
            )
        }
    }
    
    return datos_json

# ----------------------------
# Funciones de visualización
# ----------------------------
def crear_grafico_frutas(detecciones):
    """Crear gráfico de barras para frutas"""
    frutas = [d for d in detecciones if d.get('tipo') == 'fruta']
    if not frutas:
        return None
    
    # Contar frutas por clase
    clases = [d['clase'] for d in frutas]
    conteo = pd.Series(clases).value_counts().reset_index()
    conteo.columns = ['clase', 'cantidad']
    
    fig = px.bar(
        conteo,
        x='clase',
        y='cantidad',
        title="🌴 Frutas detectadas por clase",
        text='cantidad',
        color='clase',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Tipo de fruta",
        yaxis_title="Cantidad",
        showlegend=False,
        height=400
    )
    
    return fig

def crear_grafico_confianza(detecciones, tipo_filtro=None):
    """Crear gráfico de confianza"""
    if tipo_filtro:
        detecciones_filtradas = [d for d in detecciones if d.get('tipo') == tipo_filtro]
    else:
        detecciones_filtradas = detecciones
    
    if not detecciones_filtradas:
        return None
    
    df = pd.DataFrame(detecciones_filtradas)
    
    fig = px.bar(
        df,
        x='clase',
        y='confianza',
        title=f"📊 Confianza de detecciones",
        color='confianza',
        color_continuous_scale='Viridis',
        text='confianza'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Clase detectada",
        yaxis_title="Nivel de confianza",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

# ----------------------------
# Configuración de la aplicación
# ----------------------------
def init_session_state():
    """Inicializar variables de sesión de manera segura"""
    if "imagen_actual" not in st.session_state:
        st.session_state.imagen_actual = None
    if "detecciones_historial" not in st.session_state:
        st.session_state.detecciones_historial = []
    if "resultado_actual" not in st.session_state:
        st.session_state.resultado_actual = None
    if "texto_placa_actual" not in st.session_state:
        st.session_state.texto_placa_actual = ""
    if "rtsp_system" not in st.session_state:
        st.session_state.rtsp_system = None
    if "rtsp_url" not in st.session_state:
        st.session_state.rtsp_url = "rtsp://usuario:contraseña@192.168.1.100:554/stream"

def configurar_pagina():
    """Configurar página de Streamlit"""
    st.set_page_config(
        page_title="🌴🚗 Sistema Dual CNN + RTSP",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ----------------------------
# Interfaz principal
# ----------------------------
def main():
    configurar_pagina()
    init_session_state()
    
    # Título principal
    st.title("🔬 Sistema Dual CNN + Captura RTSP")
    st.markdown("Detección de frutas + placas + Captura continua de placas desde RTSP")
    
    # Cargar modelos de manera segura
    try:
        modelo_frutas = descargar_y_cargar_modelo_frutas()
        modelo_placas = descargar_y_cargar_modelo_placas()
        modelo_rtsp = descargar_y_cargar_modelo_rtsp()
        modelos_ok = True
    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        modelos_ok = False
    
    if not modelos_ok:
        st.stop()
    
    # Sidebar con configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Parámetros
        confianza = st.slider("🎚️ Confianza mínima", 0.0, 1.0, 0.5, 0.01)
        
        # Estado del sistema
        st.subheader("📊 Estado del Sistema")
        st.success("✅ Modelo frutas cargado")
        st.success("✅ Modelo placas cargado")
        st.success("✅ Modelo RTSP cargado")
        
        # Mostrar estadísticas de RTSP si está activo
        if st.session_state.rtsp_system and st.session_state.rtsp_system.is_running:
            stats = st.session_state.rtsp_system.get_stats()
            st.metric("📹 Frames procesados", stats['frames_procesados'])
            st.metric("💾 Placas guardadas", stats['placas_guardadas'])
        
        total_detecciones = len(st.session_state.detecciones_historial)
        st.metric("Detecciones totales", total_detecciones)
        
        # Botón de limpieza
        if st.button("🗑️ Limpiar historial", key="btn_limpiar"):
            st.session_state.detecciones_historial = []
            st.session_state.resultado_actual = None
            st.session_state.texto_placa_actual = ""
            st.success("Historial limpiado")
            time.sleep(1)
            st.rerun()
    
    # Layout principal con tabs
    tab1, tab2, tab3 = st.tabs(["🎥 Cámara RTSP", "🌴 Detectar Frutas", "🚗 Detectar Placas"])
    
    with tab1:
        st.header("🎥 Sistema de Captura RTSP en Tiempo Real")
        st.markdown("**Captura automática de placas desde cámara en tiempo real**")
        
        # Configuración RTSP
        col1, col2 = st.columns([3, 1])
        with col1:
            rtsp_url = st.text_input(
                "URL RTSP de la cámara:",
                value=st.session_state.rtsp_url,
                placeholder="rtsp://usuario:contraseña@ip:puerto/ruta",
                key="input_rtsp_url"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("💾 Guardar URL", key="btn_save_rtsp"):
                st.session_state.rtsp_url = rtsp_url
                st.success("✅ URL guardada")
        
        # Configuración ROI
        with st.expander("⚙️ Configurar ROI (Región de Interés)"):
            st.info("Define el área específica donde buscar placas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x_start = st.number_input("X inicio", 0, 2000, 275, key="roi_x_start")
            with col2:
                y_start = st.number_input("Y inicio", 0, 2000, 200, key="roi_y_start")
            with col3:
                x_end = st.number_input("X fin", 0, 2000, 1550, key="roi_x_end")
            with col4:
                y_end = st.number_input("Y fin", 0, 2000, 600, key="roi_y_end")
            
            roi = [x_start, y_start, x_end, y_end]
            st.info(f"ROI: ({x_start}, {y_start}) a ({x_end}, {y_end})")
        
        # Controles RTSP
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🎬 Iniciar Captura", type="primary", key="btn_start_rtsp"):
                if not rtsp_url:
                    st.error("❌ Ingresa una URL RTSP válida")
                else:
                    try:
                        # Inicializar sistema RTSP
                        st.session_state.rtsp_system = RTSPCaptureSystem(
                            modelo=modelo_rtsp,
                            rtsp_url=rtsp_url,
                            roi=roi
                        )
                        st.session_state.rtsp_system.start_capture()
                        st.success("✅ Sistema RTSP iniciado")
                    except Exception as e:
                        st.error(f"❌ Error iniciando RTSP: {e}")
        
        with col2:
            if st.button("⏹️ Detener Captura", key="btn_stop_rtsp"):
                if st.session_state.rtsp_system:
                    st.session_state.rtsp_system.stop_capture()
                    st.info("🛑 Captura detenida")
                else:
                    st.warning("⚠️ No hay sistema RTSP activo")
        
        with col3:
            if st.button("🔄 Actualizar Vista", key="btn_refresh_rtsp"):
                st.rerun()
        
        # Mostrar vista en tiempo real
        if st.session_state.rtsp_system and st.session_state.rtsp_system.is_running:
            st.subheader("📹 Vista en Tiempo Real")
            
            # Procesar y mostrar frame actual
            frame = st.session_state.rtsp_system.process_frame()
            if frame is not None:
                # Convertir BGR a RGB para Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, use_column_width=True, caption="Vista en tiempo real con detección de placas")
            
            # Estadísticas en tiempo real
            stats = st.session_state.rtsp_system.get_stats()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frames", stats['frames_procesados'])
            with col2:
                st.metric("Placas Guardadas", stats['placas_guardadas'])
            with col3:
                total_guardadas = st.session_state.rtsp_system.get_saved_plates_count()
                st.metric("Total en Memoria", total_guardadas)
            
            # Botón para ver placas capturadas
            if total_guardadas > 0:
                if st.button("📁 Ver Placas Capturadas", key="btn_view_captured"):
                    # Mostrar miniaturas de las placas capturadas
                    st.subheader("🖼️ Placas Capturadas")
                    output_dir = "placas_capturadas"
                    if os.path.exists(output_dir):
                        plate_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
                        if plate_files:
                            # Mostrar las últimas 6 placas
                            recent_plates = plate_files[-6:]
                            cols = st.columns(3)
                            for idx, plate_file in enumerate(recent_plates):
                                with cols[idx % 3]:
                                    plate_path = os.path.join(output_dir, plate_file)
                                    plate_img = cv2.imread(plate_path)
                                    if plate_img is not None:
                                        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                                        st.image(plate_rgb, caption=plate_file, width=200)
                            
                            # Botón para descargar todas
                            if st.button("📥 Descargar Todas las Placas"):
                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for plate_file in plate_files:
                                        plate_path = os.path.join(output_dir, plate_file)
                                        with open(plate_path, 'rb') as f:
                                            zip_file.writestr(plate_file, f.read())
                                
                                zip_buffer.seek(0)
                                st.download_button(
                                    label="📦 Descargar ZIP",
                                    data=zip_buffer,
                                    file_name=f"placas_capturadas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip"
                                )
        
        else:
            st.info("🎥 Configure la URL RTSP y haga clic en 'Iniciar Captura' para comenzar")
    
    with tab2:
        st.header("🌴 Detección de Frutas")
        
        if st.session_state.imagen_actual is None:
            st.warning("⚠️ Primero carga una imagen en la pestaña 'Cargar Imagen'")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
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
            
            # Mostrar resultados de frutas
            if st.session_state.resultado_actual is not None:
                st.subheader("🎯 Resultado de la detección")
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)
                
                # Obtener solo las detecciones de frutas más recientes
                frutas_detectadas = [d for d in st.session_state.detecciones_historial if d.get('tipo') == 'fruta']
                if frutas_detectadas:
                    # Mostrar gráfico
                    fig = crear_grafico_frutas(frutas_detectadas)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="grafico_frutas")
    
    with tab3:
        st.header("🚗 Detección de Placas")
        
        if st.session_state.imagen_actual is None:
            st.warning("⚠️ Primero carga una imagen en la pestaña 'Cargar Imagen'")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔍 Detectar Placas", type="primary", key="btn_placas", use_container_width=True):
                    with st.spinner("🧠 Analizando placas..."):
                        img_resultado, detecciones, texto_placa = procesar_imagen_placas(
                            modelo_placas,
                            st.session_state.imagen_actual,
                            confianza
                        )
                        
                        if detecciones and img_resultado is not None:
                            st.session_state.resultado_actual = img_resultado
                            st.session_state.texto_placa_actual = texto_placa
                            st.session_state.detecciones_historial.extend(detecciones)
                            
                            if texto_placa:
                                st.success(f"✅ Placa detectada: **{texto_placa}**")
                                
                                # Guardar resultado en JSON
                                resultado_json = {
                                    "imagen": "imagen_procesada",
                                    "placa": texto_placa,
                                    "timestamp": datetime.now().isoformat(),
                                    "confianza_promedio": np.mean([d['confianza'] for d in detecciones])
                                }
                                
                                try:
                                    with open("resultado_placa.json", "w", encoding="utf-8") as f:
                                        json.dump(resultado_json, f, indent=4, ensure_ascii=False)
                                    st.info("💾 Resultado guardado en resultado_placa.json")
                                except Exception as e:
                                    st.warning(f"No se pudo guardar el archivo: {str(e)}")
                            else:
                                st.success("✅ Elementos de placa detectados pero no se pudo formar texto completo")
                        else:
                            st.warning("🔍 No se detectaron placas con la confianza especificada")
            
            # Mostrar resultados de placas
            if st.session_state.resultado_actual is not None and st.session_state.texto_placa_actual:
                st.subheader("🎯 Resultado de la detección")
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)
                
                # Mostrar texto de placa en formato destacado
                st.subheader("🚗 Placa detectada")
                st.code(st.session_state.texto_placa_actual, language="text")
    
    # Sección de carga de imagen (común para todos los tabs)
    st.markdown("---")
    st.header("📸 Cargar Imagen")
    
    # Métodos de carga
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

if __name__ == "__main__":
    main()




