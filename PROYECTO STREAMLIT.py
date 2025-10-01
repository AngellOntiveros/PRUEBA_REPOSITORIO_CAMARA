import gdown
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import plotly.express as px
import pandas as pd
from ultralytics import YOLO
import json
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

@st.cache_resource 
def descargar_y_cargar_modelo_rtsp():
    """Modelo específico para detección de placas completas en RTSP"""
    ruta_modelo = "det_placa.pt" 
    if not os.path.exists(ruta_modelo):
        st.info("📥 Descargando modelo RTSP...")
        url = "https://drive.google.com/uc?id=1MlKp1RDi90XFpz1L3W73kqqhoRnMEdO4"
        gdown.download(url, ruta_modelo, quiet=False)
    return YOLO(ruta_modelo)

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
# Funciones RTSP MEJORADAS
# ----------------------------
def capturar_frame_rtsp(rtsp_url, roi=None, timeout=10):
    """
    Captura UN frame desde RTSP con timeout y manejo de errores
    """
    cap = None
    try:
        # Intentar diferentes backends
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Configuración optimizada para RTSP
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_TIMEOUT, timeout * 1000)  # timeout en ms
        
        if not cap.isOpened():
            return None, "No se pudo conectar a la cámara RTSP. Verifica la URL y credenciales."
        
        # Intentar leer frame con múltiples intentos
        max_attempts = 3
        for attempt in range(max_attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                # Aplicar ROI si existe
                if roi and len(roi) == 4:
                    x_start, y_start, x_end, y_end = roi
                    if 0 <= x_start < x_end and 0 <= y_start < y_end:
                        frame = frame[y_start:y_end, x_start:x_end]
                
                return frame, None
            time.sleep(0.5)
        
        return None, "No se pudo leer el frame después de varios intentos"
    
    except Exception as e:
        return None, f"Error de conexión: {str(e)}"
    
    finally:
        if cap is not None:
            cap.release()

def detectar_placas_en_frame(modelo, frame, confianza_min=0.4, min_area=1000):
    """
    Detecta placas en un frame y retorna las detecciones con recortes
    """
    try:
        resultados = modelo.predict(
            source=frame,
            conf=confianza_min,
            imgsz=640,
            verbose=False
        )
        
        detecciones = []
        frame_anotado = frame.copy()
        
        for r in resultados:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            for idx, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confianza = float(box.conf[0].cpu().numpy())
                
                # Filtrar por área mínima
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    continue
                
                # Extraer recorte con padding
                padding = 15
                x1_crop = max(0, x1 - padding)
                y1_crop = max(0, y1 - padding)
                x2_crop = min(frame.shape[1], x2 + padding)
                y2_crop = min(frame.shape[0], y2 + padding)
                
                recorte = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                # Dibujar bbox en frame anotado
                color = (0, 255, 0)
                cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame_anotado, f"{confianza:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                detecciones.append({
                    'bbox': (x1, y1, x2, y2),
                    'confianza': confianza,
                    'recorte': recorte,
                    'area': area,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                })
        
        return frame_anotado, detecciones
    
    except Exception as e:
        st.error(f"Error en detección: {str(e)}")
        return frame, []

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
    """Procesar imagen para detectar caracteres de placas"""
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
            
            if caracteres_detectados:
                caracteres_ordenados = sorted(caracteres_detectados, key=lambda x: x['x'])
                texto_placa = ''.join([c['caracter'] for c in caracteres_ordenados])
        
        return img_resultado, detecciones, texto_placa
    
    except Exception as e:
        st.error(f"Error procesando placas: {str(e)}")
        return None, [], ""

# ----------------------------
# Funciones de visualización
# ----------------------------
def crear_grafico_frutas(detecciones):
    """Crear gráfico de barras para frutas"""
    frutas = [d for d in detecciones if d.get('tipo') == 'fruta']
    if not frutas:
        return None
    
    clases = [d['clase'] for d in frutas]
    conteo = pd.Series(clases).value_counts().reset_index()
    conteo.columns = ['clase', 'cantidad']
    
    fig = px.bar(
        conteo,
        x='clase',
        y='cantidad',
        title="Frutas detectadas por clase",
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

def generar_datos_json():
    """Generar datos JSON para exportar"""
    placa = st.session_state.texto_placa_actual if st.session_state.texto_placa_actual else "No detectada"
    frutas_detectadas = [d for d in st.session_state.detecciones_historial if d.get('tipo') == 'fruta']
    
    clasificacion_por_estado = {}
    for fruta in frutas_detectadas:
        estado = fruta['clase']
        if estado not in clasificacion_por_estado:
            clasificacion_por_estado[estado] = {"cantidad": 0, "confianzas": []}
        clasificacion_por_estado[estado]["cantidad"] += 1
        clasificacion_por_estado[estado]["confianzas"].append(fruta['confianza'])
    
    for estado in clasificacion_por_estado:
        confianzas = clasificacion_por_estado[estado]["confianzas"]
        clasificacion_por_estado[estado]["confianza_promedio"] = round(np.mean(confianzas), 3)
        del clasificacion_por_estado[estado]["confianzas"]
    
    return {
        "placa": placa,
        "cantidad_fruta": len(frutas_detectadas),
        "clasificacion_fruta_por_estado": clasificacion_por_estado,
        "timestamp": datetime.now().isoformat()
    }

# ----------------------------
# Configuración
# ----------------------------
def init_session_state():
    """Inicializar estado de sesión"""
    defaults = {
        "imagen_actual": None,
        "detecciones_historial": [],
        "resultado_actual": None,
        "texto_placa_actual": "",
        "recortes_placas_rtsp": [],  # Recortes de RTSP en memoria
        "rtsp_url": "",
        "roi_coords": None,
        "ultimo_frame_rtsp": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def configurar_pagina():
    """Configurar página"""
    st.set_page_config(
        page_title="Sistema Dual CNN + RTSP",
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
    
    st.title("Sistema Dual CNN + Captura RTSP")
    st.markdown("Detección de frutas y placas con captura desde cámara RTSP")
    
    # Cargar modelos
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
    
    # Sidebar
    with st.sidebar:
        st.header("Configuración")
        confianza = st.slider("Confianza mínima", 0.0, 1.0, 0.5, 0.01)
        
        st.subheader("Estado del Sistema")
        st.success("Modelo frutas cargado")
        st.success("Modelo placas cargado")
        st.success("Modelo RTSP cargado")
        
        st.metric("Detecciones totales", len(st.session_state.detecciones_historial))
        st.metric("Recortes RTSP", len(st.session_state.recortes_placas_rtsp))
        
        if st.button("Limpiar historial"):
            st.session_state.detecciones_historial = []
            st.session_state.resultado_actual = None
            st.session_state.texto_placa_actual = ""
            st.session_state.recortes_placas_rtsp = []
            st.success("Historial limpiado")
            st.rerun()
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "Cámara RTSP", 
        "Cargar Imagen",
        "Detectar Frutas", 
        "Detectar Placas"
    ])
    
    with tab1:
        st.header("Captura desde Cámara RTSP")
        st.info("Captura frames individuales desde tu cámara de báscula")
        
        # Configuración RTSP
        col1, col2 = st.columns([3, 1])
        with col1:
            rtsp_url = st.text_input(
                "URL RTSP:",
                value=st.session_state.rtsp_url,
                placeholder="rtsp://usuario:contraseña@192.168.1.100:554/stream",
                help="Ejemplo: rtsp://admin:password@192.168.11.147:554/cam/realmonitor?channel=1&subtype=0"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("Guardar URL"):
                st.session_state.rtsp_url = rtsp_url
                st.success("URL guardada")
        
        # Configuración ROI
        with st.expander("Configurar ROI (Región de Interés)"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x_start = st.number_input("X inicio", 0, 2000, 275)
            with col2:
                y_start = st.number_input("Y inicio", 0, 2000, 200)
            with col3:
                x_end = st.number_input("X fin", 0, 2000, 1550)
            with col4:
                y_end = st.number_input("Y fin", 0, 2000, 600)
            
            usar_roi = st.checkbox("Usar ROI")
            if usar_roi:
                st.session_state.roi_coords = [x_start, y_start, x_end, y_end]
                st.info(f"ROI activo: ({x_start}, {y_start}) → ({x_end}, {y_end})")
            else:
                st.session_state.roi_coords = None
        
        # Botón de captura
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Capturar Frame", type="primary", use_container_width=True):
                if not st.session_state.rtsp_url:
                    st.error("Ingresa una URL RTSP válida")
                else:
                    with st.spinner("Conectando a cámara..."):
                        frame, error = capturar_frame_rtsp(
                            st.session_state.rtsp_url,
                            st.session_state.roi_coords,
                            timeout=15
                        )
                        
                        if error:
                            st.error(f"Error: {error}")
                            st.info("Verifica: URL correcta, credenciales, conexión de red, puerto abierto")
                        else:
                            st.session_state.ultimo_frame_rtsp = frame
                            
                            # Detectar placas automáticamente
                            frame_anotado, detecciones = detectar_placas_en_frame(
                                modelo_rtsp,
                                frame,
                                confianza_min=0.4
                            )
                            
                            if detecciones:
                                st.success(f"{len(detecciones)} placa(s) detectada(s)")
                                
                                # Guardar recortes en memoria
                                for det in detecciones:
                                    recorte_data = {
                                        'id': len(st.session_state.recortes_placas_rtsp) + 1,
                                        'imagen': det['recorte'],
                                        'confianza': det['confianza'],
                                        'timestamp': det['timestamp']
                                    }
                                    st.session_state.recortes_placas_rtsp.append(recorte_data)
                                
                                st.session_state.ultimo_frame_rtsp = frame_anotado
                            else:
                                st.warning("No se detectaron placas en el frame")
        
        # Mostrar último frame capturado
        if st.session_state.ultimo_frame_rtsp is not None:
            st.subheader("Último Frame Capturado")
            frame_rgb = cv2.cvtColor(st.session_state.ultimo_frame_rtsp, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_column_width=True)
        
        # Mostrar recortes guardados
        if st.session_state.recortes_placas_rtsp:
            st.markdown("---")
            st.subheader(f"Recortes de Placas ({len(st.session_state.recortes_placas_rtsp)})")
            
            cols = st.columns(4)
            for idx, recorte in enumerate(st.session_state.recortes_placas_rtsp):
                with cols[idx % 4]:
                    recorte_rgb = cv2.cvtColor(recorte['imagen'], cv2.COLOR_BGR2RGB)
                    st.image(recorte_rgb, caption=f"ID: {recorte['id']} ({recorte['confianza']:.2f})")
                    
                    # Botón descarga individual
                    _, buffer = cv2.imencode('.jpg', recorte['imagen'])
                    st.download_button(
                        label="Descargar",
                        data=buffer.tobytes(),
                        file_name=f"placa_ID{recorte['id']}_{recorte['timestamp']}.jpg",
                        mime="image/jpeg",
                        key=f"download_rtsp_{idx}"
                    )
            
            # Descargar todas en ZIP
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for recorte in st.session_state.recortes_placas_rtsp:
                        _, img_buffer = cv2.imencode('.jpg', recorte['imagen'])
                        filename = f"placa_ID{recorte['id']}_{recorte['timestamp']}.jpg"
                        zip_file.writestr(filename, img_buffer.tobytes())
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="Descargar Todas (ZIP)",
                    data=zip_buffer,
                    file_name=f"placas_rtsp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
    
    with tab2:
        st.header("Cargar Imagen")
        
        metodo = st.radio(
            "Método de entrada:",
            ["Subir archivo", "Cámara web"]
        )
        
        if metodo == "Subir archivo":
            archivo = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png'])
            if archivo:
                try:
                    imagen_pil = Image.open(archivo)
                    imagen = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
                    st.session_state.imagen_actual = imagen
                    st.success("Imagen cargada correctamente")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif metodo == "Cámara web":
            foto = st.camera_input("Toma una foto")
            if foto:
                try:
                    imagen_pil = Image.open(foto)
                    imagen = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
                    st.session_state.imagen_actual = imagen
                    st.success("Imagen capturada")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.session_state.imagen_actual is not None:
            st.subheader("Imagen cargada")
            st.image(st.session_state.imagen_actual, channels="BGR", use_column_width=True)
    
    with tab3:
        st.header("Detección de Frutas")
        
        if st.session_state.imagen_actual is None:
            st.warning("Primero carga una imagen en 'Cargar Imagen'")
        else:
            if st.button("Detectar Frutas", type="primary"):
                with st.spinner("Analizando frutas..."):
                    img_resultado, detecciones = procesar_imagen_frutas(
                        modelo_frutas,
                        st.session_state.imagen_actual,
                        confianza
                    )
                    
                    if detecciones and img_resultado is not None:
                        st.session_state.resultado_actual = img_resultado
                        st.session_state.detecciones_historial.extend(detecciones)
                        st.success(f"{len(detecciones)} frutas detectadas")
                    else:
                        st.warning("No se detectaron frutas")
            
            if st.session_state.resultado_actual is not None:
                st.subheader("Resultado")
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)
                
                frutas_detectadas = [d for d in st.session_state.detecciones_historial if d.get('tipo') == 'fruta']
                if frutas_detectadas:
                    fig = crear_grafico_frutas(frutas_detectadas)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Detección de Placas")
        
        if st.session_state.imagen_actual is None:
            st.warning("Primero carga una imagen en 'Cargar Imagen'")
        else:
            if st.button("Detectar Placas", type="primary"):
                with st.spinner("Analizando placas..."):
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
                            st.success(f"Placa detectada: **{texto_placa}**")
                        else:
                            st.success("Elementos de placa detectados")
                    else:
                        st.warning("No se detectaron placas")
            
            if st.session_state.resultado_actual is not None and st.session_state.texto_placa_actual:
                st.subheader("Resultado")
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)
                st.code(st.session_state.texto_placa_actual)
    
    # Historial
    if st.session_state.detecciones_historial:
        st.markdown("---")
        st.header("Historial y Estadísticas")
        
        frutas_total = len([d for d in st.session_state.detecciones_historial if d.get('tipo') == 'fruta'])
        placas_total = len([d for d in st.session_state.detecciones_historial if d.get('tipo') == 'placa'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Frutas detectadas", frutas_total)
        col2.metric("Placas detectadas", placas_total)
        col3.metric("Total detecciones", len(st.session_state.detecciones_historial))
        
        ultimas = st.session_state.detecciones_historial[-10:]
        if ultimas:
            df = pd.DataFrame(ultimas)[['tipo', 'clase', 'confianza', 'timestamp']]
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Botón JSON
        if st.button("Descargar Reporte JSON"):
            datos_json = generar_datos_json()
            json_str = json.dumps(datos_json, indent=4, ensure_ascii=False)
            
            st.download_button(
                label="Descargar JSON",
                data=json_str,
                file_name=f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
