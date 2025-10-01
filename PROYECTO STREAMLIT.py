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
    ruta_modelo = "w_best.pt" 
    if not os.path.exists(ruta_modelo):
        st.info("📥 Descargando modelo de frutas...") 
        url = "https://drive.google.com/uc?id=16BNxvPRSwUQEKULlgKhG2jRUyUNnSApu"
        try:
            gdown.download(url, ruta_modelo, quiet=False)
            st.success("✅ Modelo de frutas descargado correctamente")
        except Exception as e:
            st.error(f"❌ Error descargando modelo de frutas: {str(e)}")
            return None
    
    try:
        modelo = YOLO(ruta_modelo)
        st.success("✅ Modelo de frutas cargado correctamente")
        return modelo
    except Exception as e:
        st.error(f"❌ Error cargando modelo de frutas: {str(e)}")
        return None

@st.cache_resource
def descargar_y_cargar_modelo_placas():
    """Descargar y cargar modelo de placas"""
    ruta_modelo = "W_PLACA.pt" 
    if not os.path.exists(ruta_modelo):
        st.info("📥 Descargando modelo de placas...")
        url = "https://drive.google.com/uc?id=12KSiZvxS262NPQ1s-hdsOxJliHSMS3tS"
        try:
            gdown.download(url, ruta_modelo, quiet=False)
            st.success("✅ Modelo de placas descargado correctamente")
        except Exception as e:
            st.error(f"❌ Error descargando modelo de placas: {str(e)}")
            return None
    
    try:
        modelo = YOLO(ruta_modelo)
        st.success("✅ Modelo de placas cargado correctamente")
        return modelo
    except Exception as e:
        st.error(f"❌ Error cargando modelo de placas: {str(e)}")
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
# Clase de Tracking
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
# Funciones RTSP
# ----------------------------
def capturar_frame_rtsp(rtsp_url, roi=None):
    """Capturar un frame desde cámara RTSP"""
    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return None, "No se pudo conectar a la cámara RTSP"
        
        # Esperar un momento para que la cámara se inicialice
        time.sleep(0.1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, "No se pudo leer el frame"
        
        # Aplicar ROI si existe
        if roi:
            x_start, y_start, x_end, y_end = roi
            frame = frame[y_start:y_end, x_start:x_end]
        
        return frame, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"

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

def crear_grafico_temporal(detecciones):
    """Crear gráfico temporal de detecciones"""
    if not detecciones:
        return None
    
    df = pd.DataFrame(detecciones)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_por_hora = df.groupby([df['timestamp'].dt.hour, 'clase']).size().reset_index(name='cantidad')
    
    fig = px.line(
        df_por_hora,
        x='timestamp',
        y='cantidad',
        color='clase',
        title="📈 Evolución temporal de detecciones",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Hora del día",
        yaxis_title="Cantidad de detecciones",
        height=400
    )
    
    return fig

# ----------------------------
# Funciones de utilidad
# ----------------------------
def convertir_imagen_a_base64(imagen):
    """Convertir imagen OpenCV a base64 para descarga"""
    _, buffer = cv2.imencode('.jpg', imagen)
    imagen_bytes = buffer.tobytes()
    return base64.b64encode(imagen_bytes).decode()

def crear_resumen_ejecucion():
    """Crear resumen de la ejecución actual"""
    total_frutas = len([d for d in st.session_state.detecciones_historial if d['tipo'] == 'fruta'])
    total_placas = len([d for d in st.session_state.detecciones_historial if d['tipo'] == 'placa'])
    
    resumen = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_detecciones": len(st.session_state.detecciones_historial),
        "frutas_detectadas": total_frutas,
        "placas_detectadas": total_placas,
        "recortes_guardados": len(st.session_state.recortes_placas),
        "confianza_promedio": round(np.mean([d['confianza'] for d in st.session_state.detecciones_historial]), 3) if st.session_state.detecciones_historial else 0
    }
    
    return resumen

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
    
    # Para RTSP y recortes
    if "tracker" not in st.session_state:
        st.session_state.tracker = SimpleTracker(max_age=30)
    if "recortes_placas" not in st.session_state:
        st.session_state.recortes_placas = []
    if "rtsp_conectado" not in st.session_state:
        st.session_state.rtsp_conectado = False
    if "rtsp_url" not in st.session_state:
        st.session_state.rtsp_url = ""
    if "roi_coords" not in st.session_state:
        st.session_state.roi_coords = None
    
    # Nuevas variables para mejor control
    if "procesamiento_automatico" not in st.session_state:
        st.session_state.procesamiento_automatico = False
    if "ultima_placa_detectada" not in st.session_state:
        st.session_state.ultima_placa_detectada = ""
    if "estadisticas" not in st.session_state:
        st.session_state.estadisticas = {
            "inicio_sesion": datetime.now(),
            "total_procesamientos": 0,
            "errores": 0
        }

def configurar_pagina():
    """Configurar página de Streamlit"""
    st.set_page_config(
        page_title="🌴🚗 Sistema Dual CNN",
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
    st.title("🔬 Sistema Dual CNN - Detección Inteligente + RTSP")
    st.markdown("Sistema de detección con dos redes neuronales especializadas + Captura RTSP")
    
    # Cargar modelos de manera segura
    with st.spinner("🔄 Cargando modelos YOLO..."):
        try:
            modelo_frutas = descargar_y_cargar_modelo_frutas()
            modelo_placas = descargar_y_cargar_modelo_placas()
            
            if modelo_frutas is None or modelo_placas is None:
                st.error("❌ No se pudieron cargar uno o ambos modelos. Verifica la conexión a internet.")
                st.stop()
                
            modelos_ok = True
            st.session_state.estadisticas["total_procesamientos"] += 1
            
        except Exception as e:
            st.error(f"❌ Error crítico cargando modelos: {str(e)}")
            st.session_state.estadisticas["errores"] += 1
            modelos_ok = False
    
    if not modelos_ok:
        st.stop()
    
    # Sidebar con configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Parámetros
        confianza = st.slider("🎚️ Confianza mínima", 0.0, 1.0, 0.5, 0.01)
        
        # Opciones avanzadas
        with st.expander("🔧 Opciones avanzadas"):
            padding_placas = st.slider("📏 Padding recortes placas", 5, 30, 15)
            max_age_tracking = st.slider("🕒 Máxima edad tracking", 10, 60, 30)
            st.session_state.tracker.max_age = max_age_tracking
            
            # Procesamiento automático
            st.session_state.procesamiento_automatico = st.checkbox(
                "🔄 Procesamiento automático RTSP", 
                value=st.session_state.procesamiento_automatico
            )
        
        # Estado del sistema
        st.subheader("📊 Estado del Sistema")
        st.success("✅ Modelo frutas cargado")
        st.success("✅ Modelo placas cargado")
        
        total_detecciones = len(st.session_state.detecciones_historial)
        st.metric("Detecciones totales", total_detecciones)
        st.metric("Recortes en memoria", len(st.session_state.recortes_placas))
        
        # Resumen de estadísticas
        with st.expander("📈 Estadísticas de sesión"):
            st.write(f"**Inicio de sesión:** {st.session_state.estadisticas['inicio_sesion'].strftime('%H:%M:%S')}")
            st.write(f"**Procesamientos:** {st.session_state.estadisticas['total_procesamientos']}")
            st.write(f"**Errores:** {st.session_state.estadisticas['errores']}")
            st.write(f"**Tiempo activo:** {(datetime.now() - st.session_state.estadisticas['inicio_sesion']).seconds // 60} min")
        
        # Botones de gestión
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Limpiar historial", key="btn_limpiar", use_container_width=True):
                st.session_state.detecciones_historial = []
                st.session_state.resultado_actual = None
                st.session_state.texto_placa_actual = ""
                st.session_state.recortes_placas = []
                st.session_state.tracker = SimpleTracker(max_age=30)
                st.success("Historial limpiado")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("💾 Exportar JSON", key="btn_exportar", use_container_width=True):
                datos_json = generar_datos_json()
                json_str = json.dumps(datos_json, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 Descargar JSON",
                    data=json_str,
                    file_name=f"resumen_detecciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
    
    # Layout principal con tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Cargar Imagen", "🌴 Detectar Frutas", "🚗 Detectar Placas", "📊 Dashboard"])
    
    with tab1:
        st.header("📸 Cargar imagen")
        
        # Métodos de carga
        metodo = st.radio(
            "Método de entrada:",
            ["📁 Subir archivo", "📷 Cámara web", "🎥 Cámara RTSP (Báscula)"],
            key="metodo_carga"
        )
        
        imagen_cargada = False
        
        if metodo == "📁 Subir archivo":
            archivo = st.file_uploader(
                "Selecciona una imagen",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key="uploader_imagen"
            )
            if archivo is not None:
                try:
                    imagen_pil = Image.open(archivo)
                    imagen = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
                    st.session_state.imagen_actual = imagen
                    imagen_cargada = True
                    st.success(f"✅ Imagen cargada: {archivo.name}")
                except Exception as e:
                    st.error(f"❌ Error cargando imagen: {str(e)}")
                    st.session_state.estadisticas["errores"] += 1
        
        elif metodo == "📷 Cámara web":
            st.info("📱 Este método funciona perfectamente en dispositivos móviles")
            foto = st.camera_input("Toma una foto", key="camera_input")
            
            if foto is not None:
                try:
                    imagen_pil = Image.open(foto)
                    imagen = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
                    st.session_state.imagen_actual = imagen
                    imagen_cargada = True
                    st.success("✅ Imagen capturada correctamente")
                    st.session_state.estadisticas["total_procesamientos"] += 1
                except Exception as e:
                    st.error(f"❌ Error procesando imagen: {str(e)}")
                    st.session_state.estadisticas["errores"] += 1
        
        elif metodo == "🎥 Cámara RTSP (Báscula)":
            st.subheader("🎥 Configuración de Cámara RTSP")
            
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
                if st.button("💾 Guardar URL", key="btn_save_rtsp", use_container_width=True):
                    st.session_state.rtsp_url = rtsp_url
                    st.success("✅ URL guardada")
            
            # Configuración ROI (opcional)
            with st.expander("⚙️ Configurar ROI (Región de Interés) - Opcional"):
                st.info("Define una región específica de la imagen para analizar")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x_start = st.number_input("X inicio", 0, 2000, 275, key="roi_x_start")
                with col2:
                    y_start = st.number_input("Y inicio", 0, 2000, 200, key="roi_y_start")
                with col3:
                    x_end = st.number_input("X fin", 0, 2000, 1550, key="roi_x_end")
                with col4:
                    y_end = st.number_input("Y fin", 0, 2000, 600, key="roi_y_end")
                
                usar_roi = st.checkbox("✅ Usar ROI", key="usar_roi")
                
                if usar_roi:
                    st.session_state.roi_coords = [x_start, y_start, x_end, y_end]
                    st.info(f"ROI configurado: ({x_start}, {y_start}) a ({x_end}, {y_end})")
                else:
                    st.session_state.roi_coords = None
            
            # Botones de captura
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("📸 Capturar Frame", type="primary", key="btn_capturar_rtsp", use_container_width=True):
                    if not st.session_state.rtsp_url:
                        st.error("❌ Por favor ingresa una URL RTSP")
                    else:
                        with st.spinner("📡 Conectando a cámara..."):
                            frame, error = capturar_frame_rtsp(
                                st.session_state.rtsp_url,
                                st.session_state.roi_coords
                            )
                            
                            if error:
                                st.error(f"❌ {error}")
                                st.session_state.estadisticas["errores"] += 1
                            else:
                                st.session_state.imagen_actual = frame
                                imagen_cargada = True
                                st.success("✅ Frame capturado exitosamente")
                                st.session_state.estadisticas["total_procesamientos"] += 1
                
                # Procesamiento automático
                if st.session_state.procesamiento_automatico and st.session_state.rtsp_url:
                    if st.button("🔄 Procesamiento Continuo", key="btn_continuo", use_container_width=True):
                        placeholder = st.empty()
                        for i in range(5):  # Procesar 5 frames
                            with placeholder.container():
                                st.info(f"Procesando frame {i+1}/5...")
                                frame, error = capturar_frame_rtsp(
                                    st.session_state.rtsp_url,
                                    st.session_state.roi_coords
                                )
                                if frame is not None:
                                    st.session_state.imagen_actual = frame
                                    # Procesar automáticamente
                                    tracks = procesar_frame_con_tracking(
                                        modelo_placas,
                                        frame,
                                        st.session_state.tracker,
                                        confianza
                                    )
                                    if tracks:
                                        st.success(f"Frame {i+1}: {len(tracks)} placas detectadas")
                                time.sleep(2)
        
        # Mostrar imagen actual
        if st.session_state.imagen_actual is not None:
            st.subheader("🖼️ Imagen cargada")
            
            # Mostrar información de la imagen
            height, width, channels = st.session_state.imagen_actual.shape
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ancho", f"{width} px")
            with col2:
                st.metric("Alto", f"{height} px")
            with col3:
                st.metric("Canales", channels)
            
            st.image(st.session_state.imagen_actual, channels="BGR", use_column_width=True)
            
            # Botón para descargar imagen original
            if st.button("📥 Descargar imagen original", key="btn_descargar_original"):
                imagen_base64 = convertir_imagen_a_base64(st.session_state.imagen_actual)
                href = f'<a href="data:image/jpeg;base64,{imagen_base64}" download="imagen_original_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg">Descargar imagen</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    with tab2:
        st.header("🌴 Detección de Frutas")
        
        if st.session_state.imagen_actual is None:
            st.warning("⚠️ Primero carga una imagen en la pestaña 'Cargar Imagen'")
        else:
            col1, col2, col3 = st.columns([1, 2, 1])
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
                            st.session_state.estadisticas["total_procesamientos"] += 1
                            
                            st.success(f"✅ {len(detecciones)} frutas detectadas")
                            
                            # Mostrar resumen rápido
                            clases_detectadas = set([d['clase'] for d in detecciones])
                            st.info(f"📋 Clases detectadas: {', '.join(clases_detectadas)}")
                        else:
                            st.warning("🔍 No se detectaron frutas con la confianza especificada")
            
            # Mostrar resultados de frutas
            if st.session_state.resultado_actual is not None:
                st.subheader("🎯 Resultado de la detección")
                
                # Mostrar imagen con detecciones
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)
                
                # Obtener solo las detecciones de frutas más recientes
                frutas_detectadas = [d for d in st.session_state.detecciones_historial if d.get('tipo') == 'fruta']
                if frutas_detectadas:
                    # Mostrar tabla de detecciones
                    with st.expander("📋 Detalles de detecciones"):
                        df_frutas = pd.DataFrame(frutas_detectadas)
                        st.dataframe(df_frutas[['clase', 'confianza', 'timestamp']], use_container_width=True)
                    
                    # Mostrar gráfico
                    fig = crear_grafico_frutas(frutas_detectadas)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="grafico_frutas")
                    
                    # Gráfico de confianza
                    fig_conf = crear_grafico_confianza(frutas_detectadas, 'fruta')
                    if fig_conf:
                        st.plotly_chart(fig_conf, use_container_width=True, key="grafico_confianza_frutas")
    
    with tab3:
        st.header("🚗 Detección de Placas")
        
        if st.session_state.imagen_actual is None:
            st.warning("⚠️ Primero carga una imagen en la pestaña 'Cargar Imagen'")
        else:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Detectar Placas", type="primary", key="btn_placas", use_container_width=True):
                    with st.spinner("🧠 Analizando placas con tracking..."):
                        # Usar tracking para evitar duplicados
                        tracks = procesar_frame_con_tracking(
                            modelo_placas,
                            st.session_state.imagen_actual,
                            st.session_state.tracker,
                            confianza
                        )
                        
                        if tracks:
                            # Dibujar detecciones
                            img_resultado = st.session_state.imagen_actual.copy()
                            
                            for track in tracks:
                                x1, y1, x2, y2, conf, track_id = track
                                
                                # Dibujar bbox
                                cv2.rectangle(img_resultado, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(img_resultado, f"ID:{track_id} {conf:.2f}", 
                                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                                # Extraer y guardar recorte en memoria
                                recorte = extraer_recorte_placa(st.session_state.imagen_actual, (x1, y1, x2, y2), padding_placas)
                                
                                recorte_data = {
                                    'id': track_id,
                                    'imagen': recorte,
                                    'confianza': conf,
                                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                                    'bbox': (x1, y1, x2, y2)
                                }
                                
                                # Evitar duplicados
                                if not any(r['id'] == track_id for r in st.session_state.recortes_placas):
                                    st.session_state.recortes_placas.append(recorte_data)
                            
                            st.session_state.resultado_actual = img_resultado
                            st.session_state.estadisticas["total_procesamientos"] += 1
                            st.success(f"✅ {len(tracks)} placa(s) detectada(s) y guardada(s) en memoria")
                        else:
                            st.warning("🔍 No se detectaron placas con la confianza especificada")
            
            # Mostrar resultados de placas
            if st.session_state.resultado_actual is not None:
                st.subheader("🎯 Resultado de la detección")
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)
    
    with tab4:
        st.header("📊 Dashboard de Análisis")
        
        if not st.session_state.detecciones_historial:
            st.warning("⚠️ No hay datos de detecciones para mostrar. Realiza algunas detecciones primero.")
        else:
            # Resumen ejecutivo
            st.subheader("📈 Resumen Ejecutivo")
            resumen = crear_resumen_ejecucion()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Detecciones", resumen["total_detecciones"])
            with col2:
                st.metric("Frutas Detectadas", resumen["frutas_detectadas"])
            with col3:
                st.metric("Placas Detectadas", resumen["placas_detectadas"])
            with col4:
                st.metric("Confianza Promedio", resumen["confianza_promedio"])
            
            # Gráficos principales
            col1, col2 = st.columns(2)
            
            with col1:
                fig_frutas = crear_grafico_frutas(st.session_state.detecciones_historial)
                if fig_frutas:
                    st.plotly_chart(fig_frutas, use_container_width=True)
                else:
                    st.info("No hay datos de frutas para graficar")
            
            with col2:
                fig_temporal = crear_grafico_temporal(st.session_state.detecciones_historial)
                if fig_temporal:
                    st.plotly_chart(fig_temporal, use_container_width=True)
                else:
                    st.info("No hay suficientes datos para gráfico temporal")
            
            # Análisis detallado
            st.subheader("🔍 Análisis Detallado")
            
            with st.expander("📋 Historial Completo de Detecciones"):
                df_completo = pd.DataFrame(st.session_state.detecciones_historial)
                st.dataframe(df_completo, use_container_width=True)
                
                # Opciones de filtrado
                col1, col2 = st.columns(2)
                with col1:
                    filtro_tipo = st.selectbox("Filtrar por tipo:", ["Todos", "fruta", "placa"])
                with col2:
                    filtro_confianza = st.slider("Confianza mínima:", 0.0, 1.0, 0.0, 0.1)
                
                # Aplicar filtros
                df_filtrado = df_completo.copy()
                if filtro_tipo != "Todos":
                    df_filtrado = df_filtrado[df_filtrado['tipo'] == filtro_tipo]
                df_filtrado = df_filtrado[df_filtrado['confianza'] >= filtro_confianza]
                
                st.metric("Detecciones filtradas", len(df_filtrado))
            
            # Exportación de datos
            st.subheader("💾 Exportación de Datos")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Exportar CSV", key="btn_export_csv"):
                    df_completo = pd.DataFrame(st.session_state.detecciones_historial)
                    csv = df_completo.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar CSV",
                        data=csv,
                        file_name=f"detecciones_completas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
            
            with col2:
                if st.button("📷 Exportar Imágenes", key="btn_export_imgs"):
                    if st.session_state.resultado_actual is not None:
                        imagen_base64 = convertir_imagen_a_base64(st.session_state.resultado_actual)
                        href = f'<a href="data:image/jpeg;base64,{imagen_base64}" download="resultado_deteccion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg">Descargar imagen resultado</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.warning("No hay imagen de resultado para exportar")
            
            with col3:
                if st.button("📋 Reporte Ejecutivo", key="btn_reporte"):
                    datos_json = generar_datos_json()
                    json_str = json.dumps(datos_json, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📥 Descargar Reporte JSON",
                        data=json_str,
                        file_name=f"reporte_ejecutivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_reporte"
                    )
    
    # Sección de recortes guardados en memoria
    if st.session_state.recortes_placas:
        st.markdown("---")
        st.header("🗂️ Recortes de Placas en Memoria")
        
        st.info(f"📦 Tienes {len(st.session_state.recortes_placas)} recorte(s) guardado(s) en esta sesión")
        
        # Mostrar recortes en grid
        cols = st.columns(4)
        for idx, recorte_data in enumerate(st.session_state.recortes_placas):
            with cols[idx % 4]:
                st.image(recorte_data['imagen'], channels="BGR", 
                        caption=f"ID: {recorte_data['id']} - Conf: {recorte_data['confianza']:.2f}")
                
                # Botón de descarga individual
                _, buffer = cv2.imencode('.jpg', recorte_data['imagen'])
                st.download_button(
                    label="⬇️ Descargar",
                    data=buffer.tobytes(),
                    file_name=f"placa_ID{recorte_data['id']}_{recorte_data['timestamp']}.jpg",
                    mime="image/jpeg",
                    key=f"download_recorte_{idx}"
                )
        
        # Botón para descargar todos
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📥 Descargar Todos los Recortes", key="btn_download_all_crops"):
                # Crear ZIP en memoria
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for recorte_data in st.session_state.recortes_placas:
                        _, img_buffer = cv2.imencode('.jpg', recorte_data['imagen'])
                        filename = f"placa_ID{recorte_data['id']}_{recorte_data['timestamp']}.jpg"
                        zip_file.writestr(filename, img_buffer.tobytes())
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📦 Descargar ZIP",
                    data=zip_buffer,
                    file_name=f"recortes_placas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    key="download_zip_crops"
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            🚀 Sistema Dual CNN - Desarrollado con Streamlit, YOLO y OpenCV
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
