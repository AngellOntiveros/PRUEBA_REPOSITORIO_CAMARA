#!/bin/bash

# Construir la imagen
docker build -t streamlit-dual-cnn .

# Ejecutar el contenedor
docker run -d \
  -p 8501:8501 \
  --name dual-cnn-app \
  streamlit-dual-cnn

echo "Aplicación ejecutándose en http://localhost:8501"