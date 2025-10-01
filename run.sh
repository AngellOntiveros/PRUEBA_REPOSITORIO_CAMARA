#!/bin/bash

# Ejecutar con docker-compose
docker-compose up -d

echo "Aplicación ejecutándose en http://localhost:8501"
echo "Para ver los logs: docker-compose logs -f"