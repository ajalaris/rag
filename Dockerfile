FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev cmake  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt y la aplicación
COPY requirements.txt .
COPY app.py .

# Crear carpetas necesarias
RUN mkdir -p documentos faiss_index

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de Streamlit
EXPOSE 8501

# Ejecutar la aplicación
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
