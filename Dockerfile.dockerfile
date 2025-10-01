FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema primero
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "PROYECTO STREAMLIT.py", "--server.port=8501", "--server.address=0.0.0.0"]
