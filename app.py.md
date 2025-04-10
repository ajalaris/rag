# Consulta de Documentos con Mistral

Esta es una aplicación basada en **Streamlit** que permite cargar, gestionar y consultar documentos (PDF y TXT) utilizando el modelo de lenguaje **Mistral** ejecutado a través de **Ollama**, con soporte para aceleración por GPU. Utiliza **LangChain** para procesar documentos, generar embeddings y realizar búsquedas vectoriales con **FAISS**.

El propósito principal es proporcionar una interfaz sencilla para subir documentos, crear un índice vectorial y realizar consultas basadas en el contenido de esos documentos, con respuestas generadas por Mistral en español.

---

## Características

- **Carga de documentos**: Soporta archivos PDF y TXT.
- **Gestión de documentos**: Subir y eliminar documentos desde la interfaz.
- **Índice vectorial**: Usa FAISS para búsquedas rápidas de similitud.
- **Consultas**: Respuestas generadas por Mistral basadas únicamente en los documentos cargados.
- **Soporte GPU**: Detecta y utiliza CUDA si está disponible.
- **Configuración Docker**: Optimizada para ejecutarse en contenedores.

---

## Requisitos

### Dependencias
- **Python 3.8+**
- Bibliotecas:
  - `streamlit`: Interfaz web interactiva.
  - `langchain_community`: Herramientas para procesamiento de documentos y cadenas de QA.
  - `langchain_huggingface`: Embeddings de Hugging Face.
  - `torch`: Detección y uso de GPU.
  - `requests`: Verificación de conexión con Ollama.
  - `faiss-cpu` o `faiss-gpu`: Almacén vectorial (instalar según hardware).
  - Otras: `os`, `datetime`, `pathlib`, `shutil`.

# Relación entre LangChain, HuggingFace Embeddings, FAISS y Ollama

Cómo funcionan juntos estos componentes y el flujo de datos entre ellos:

## 1. El papel de cada componente

- **LangChain**: Es un framework que facilita la creación de aplicaciones basadas en LLMs (Large Language Models). Proporciona abstracciones para conectar diferentes componentes como cargadores de documentos, modelos, bases de datos vectoriales, etc.
- **HuggingFace Embeddings**: Convierte texto en vectores numéricos (embeddings) que capturan el significado semántico del texto. En tu aplicación usas específicamente `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`, un modelo diseñado para generar representaciones vectoriales de textos en múltiples idiomas.
- **FAISS (Facebook AI Similarity Search)**: Es una biblioteca de búsqueda de similitud de vectores optimizada para encontrar rápidamente vectores similares en grandes conjuntos de datos. Permite indexar y buscar embeddings de manera eficiente.
- **Ollama**: Plataforma que ejecuta modelos de lenguaje localmente. En tu caso, proporciona el modelo `Mistral:7b` para generar respuestas a las consultas.

## 2. Flujo de datos entre los componentes

### Fase de indexación:

1. **Carga de documentos**:
   - LangChain usa cargadores como `PyPDFLoader` y `TextLoader` para extraer texto de archivos PDF y TXT.

2. **Procesamiento de documentos**:
   - El texto se divide en fragmentos más pequeños (chunks) usando `RecursiveCharacterTextSplitter`.

3. **Generación de embeddings**:
   - HuggingFace Embeddings convierte cada fragmento de texto en un vector numérico (típicamente de 384 dimensiones con el modelo usado).

4. **Indexación con FAISS**:
   - Los vectores se almacenan en un índice FAISS junto con el texto original.
   - FAISS crea estructuras de datos optimizadas para búsquedas fluidas por similitud.
   - El índice se serializa y guarda en disco usando `pickle` para uso futuro.

### Fase de consulta:

1. **Procesamiento de la pregunta**:
   - La pregunta del usuario se convierte en un vector usando el mismo modelo de HuggingFace.

2. **Búsqueda de similitud**:
   - FAISS busca en su índice los vectores más similares al vector de la pregunta.
   - Recupera los `k` fragmentos de texto más relevantes (en este caso, `k=4`).

3. **Generación de respuesta**:
   - Los fragmentos recuperados se envían a Ollama (`Mistral:7b`) junto con la pregunta.
   - LangChain formatea esto con un patrón de `"stuff"` que coloca todos los fragmentos en un solo prompt.
   - Ollama genera una respuesta basada en el contexto proporcionado y la pregunta.

4. **Presentación**:
   - La respuesta y las fuentes de información se muestran al usuario.

## 3. Representación visual del flujo

```
Documentos → LangChain (carga) → Fragmentación → HuggingFace (embeddings) → FAISS (indexación) → Almacenamiento
Pregunta → HuggingFace (embedding) → Búsqueda de similitud en FAISS → Fragmentos relevantes → Ollama → Respuesta
```

En la aplicación, **LangChain** actúa como coordinador, orquestando el flujo entre los demás componentes y proporcionando una interfaz unificada para trabajar con todos ellos.

---

## Estructura del Código

### Imports
```python
import os  # Manejo de sistema de archivos
import streamlit as st  # Interfaz web
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # Carga de documentos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # División de texto
from langchain_huggingface import HuggingFaceEmbeddings  # Embeddings
from langchain_community.vectorstores import FAISS  # Almacén vectorial
from langchain_community.llms import Ollama  # Modelo de lenguaje
from langchain.chains import RetrievalQA  # Cadena de preguntas y respuestas
import torch  # Soporte GPU
import requests  # Solicitudes HTTP
from datetime import datetime  # Marcas temporales
from pathlib import Path  # Rutas (no usado)
import shutil  # Operaciones de archivos (no usado)
```

- **Alternativas**:
  - `PyPDFLoader`: `PDFMinerLoader` (más preciso para PDFs complejos).
  - `HuggingFaceEmbeddings`: `OpenAIEmbeddings` (requiere API), `FastEmbeddings` (más ligero).
  - `FAISS`: `Chroma` (persistente), `Annoy` (más ligero).

---

### Configuración Inicial
```python
cuda_available = torch.cuda.is_available()  # Detecta GPU
device = "cuda" if cuda_available else "cpu"  # Selecciona dispositivo
st.set_page_config(page_title="Consulta de documentos con Mistral", layout="wide")  # Configuración de Streamlit
st.title("Consulta de documentos con Mistral:7b")  # Título
```

- **`layout`**: `"wide"` usa el ancho completo. Alternativa: `"centered"`.

---

### Barra Lateral (Configuración)
```python
with st.sidebar:
    docs_folder = st.text_input("Carpeta de documentos", "documentos")  # Carpeta de documentos
    default_ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")  # URL por defecto
    ollama_base_url = st.text_input("URL de Ollama", default_ollama_url)  # URL configurable
    model_name = st.text_input("Nombre del modelo", "mistral:7b-instruct-q2_K")  # Modelo
    index_path = st.text_input("Ruta para guardar el índice", "faiss_index")  # Índice FAISS
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.1)  # Creatividad
    max_tokens = st.slider("Tokens máximos", 100, 2000, 1000)  # Longitud de respuesta
```

- **`model_name`**: Opciones: `"llama2"`, `"mixtral:8x7b"`.
- **`temperature`**: 0.0 (determinista) a 1.0 (creativo).
- **`max_tokens`**: Ajusta según detalle deseado.

---

## Funciones Principales

### `upload_documents(folder_path)`
Sube documentos PDF/TXT a la carpeta especificada.
- **Parámetros**:
  - `folder_path`: Ruta donde se guardan los documentos.
- **Detalles**:
  - Usa `st.file_uploader` con soporte para `"pdf"` y `"txt"`.
  - Genera nombres únicos con timestamps.
- **Alternativas**:
  - Agregar soporte para `"docx"`, `"csv"`.
  - Usar UUID en lugar de timestamps.

### `delete_documents(folder_path)`
Elimina documentos del repositorio.
- **Opciones**:
  - Eliminar individuales (con `st.multiselect`).
  - Eliminar todos (con confirmación `"CONFIRMAR"`).
- **Ventajas**: Interfaz segura.
- **Desventajas**: Sin opción de deshacer.

### `manage_documents_section(docs_folder)`
Gestiona documentos en la barra lateral con pestañas "Subir" y "Eliminar".
- **Detalles**:
  - Usa `st.tabs` para organización.
- **Alternativas**: Usar `st.expander` para colapsar secciones.

### `load_documents(folder_path)`
Carga documentos PDF/TXT en memoria.
- **Cargadores**:
  - `PyPDFLoader`: Para PDFs.
  - `TextLoader`: Para TXT.
- **Alternativas**: `UnstructuredFileLoader` para formatos mixtos.

### `get_embeddings()`
Genera embeddings para los documentos.
- **Modelo**: `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`.
  - **Ventajas**: Multilingüe, ligero.
  - **Alternativas**:
    - `"all-MiniLM-L6-v2"`: Más rápido.
    - `"all-mpnet-base-v2"`: Más preciso.
- **Respaldo**: `FakeEmbeddings` para pruebas.

### `get_vector_index(docs_folder, index_path, force_rebuild=False)`
Crea o carga el índice FAISS.
- **Parámetros**:
  - `chunk_size=1000`, `chunk_overlap=100` en `RecursiveCharacterTextSplitter`.
- **Alternativas**:
  - `CharacterTextSplitter`: Más simple.
  - `TokenTextSplitter`: Divide por tokens.
- **Ventajas**: Rápido con FAISS.
- **Desventajas**: No persistente sin guardado.

### `setup_ollama_model(base_url, model_name, temperature, max_tokens)`
Configura el modelo Ollama.
- **Parámetros**:
  - `system`: Respuestas siempre en español.
- **Alternativas**: Usar prompts específicos por consulta.

### `check_ollama_connection(base_url, model_name)`
Verifica la conexión con Ollama.
- **Detalles**: Usa `requests.get` para listar modelos.
- **Alternativas**: Agregar timeouts con `requests.Session`.

---

## Configuración Avanzada

### Embeddings
- **Opciones**: 
  - `"all-MiniLM-L6-v2"`: Rápido, menos preciso.
  - `"all-mpnet-base-v2"`: Preciso, más pesado.
- **Consideraciones**: Ajusta según idioma y recursos.

### Text Splitter
- **Parámetros**:
  - `chunk_size`: Tamaño del fragmento.
  - `chunk_overlap`: Superposición para contexto.
- **Alternativas**: `TokenTextSplitter` para modelos tokenizados.

### RetrievalQA
- **chain_type**:
  - `"stuff"`: Combina todos los documentos (rápido).
  - `"map_reduce"`: Procesa por separado (escalable).
  - `"refine"`: Refina iterativamente (preciso).
- **search_kwargs**: `k=4` (número de documentos recuperados).

---

## Notas sobre Docker

- **Volúmenes**:
  - `/documentos`: Almacena los archivos subidos.
  - `/faiss_index`: Guarda el índice vectorial.
- **GPU**: Detecta y usa CUDA automáticamente si está disponible.

---
