# Sistema RAG con Mistral
## Aplicación Streamlit y API FastAPI

---

## ¿Qué es un sistema RAG?

- **R**etrieval **A**ugmented **G**eneration
- Combina recuperación de información con generación de texto
- Permite consultas en lenguaje natural sobre documentos específicos
- Respuestas basadas en evidencia documental

---

## Componentes del Sistema

1. **Aplicación web con Streamlit** (`app.py`)
   - Interfaz gráfica de usuario
   - Carga y gestión de documentos
   - Consultas interactivas

2. **API REST con FastAPI** (`rag_api.py`) 
   - Integración programática
   - Endpoints para consultas
   - Mismo motor RAG en segundo plano

---

## Arquitectura del Sistema

![esquema](https://github.com/ajalaris/rag/blob/master/compose-viz.VizFormats.svg)

1. **Carga de documentos** (PDFs/TXTs)
2. **Procesamiento de texto**
3. **Indexación vectorial** (FAISS)
4. **Consultas** con modelo LLM (Mistral)
5. **Respuestas** con citas de fuentes

---

## Flujo de Trabajo: App Streamlit

1. **Configuración** de parámetros
2. **Carga** de documentos (PDF/TXT)
3. **Creación** del índice vectorial
4. **Consulta** en lenguaje natural
5. **Visualización** de respuestas y fuentes

---

## Funcionalidades de `app.py`

- **Interfaz intuitiva** con Streamlit
- **Configuración flexible** mediante sidebar
  - Carpeta de documentos
  - URL de Ollama
  - Modelo a utilizar
  - Parámetros de generación
- **Visualización** de respuestas y fuentes
- **Instrucciones** integradas para usuarios
![rag-app](https://github.com/ajalaris/rag/blob/master/rag-app.png)

---

## Demostración: App Streamlit


- **Carga** de documentos
- **Indexación** vectorial
- **Consulta** al sistema
- **Visualización** de resultados con fuentes

---

## Funcionalidades de `rag_api.py`

- **API REST** con FastAPI
- **Endpoint** principal: `/query_documents`
- **Parámetros configurables**:
  - Query (consulta)
  - Temperature (creatividad)
  - Max tokens (longitud respuesta)
- **Respuesta estructurada** con fuentes
- **Manejo de errores** integrado

---

## Ejemplo de Request a la API

```json
POST /query_documents
{
  "query": "¿Cuáles son los beneficios de la energía solar?",
  "temperature": 0.1,
  "max_tokens": 500
}
```
Ejemplo de Response de la API
```json
{
  "answer": "Los beneficios de la energía solar incluyen...",
  "sources": [
    {
      "source": "energia_renovable.pdf",
      "page": 12,
      "content": "La energía solar ofrece múltiples ventajas..."
    }
  ]
}
```

## Tecnologías Implementadas

- **Streamlit**: Interfaz web interactiva
- **FastAPI**: API RESTful de alto rendimiento
- **LangChain**: Framework para aplicaciones LLM
- **Ollama**: Ejecución local de modelos LLM
- **FAISS**: Índice vectorial eficiente
- **HuggingFace Embeddings**: Vectorización de texto
- **Docker**: Contenedorización y despliegue
- **Ollama WEBUI**: Interfaz web para Ollama local 
 ![Ollama](https://github.com/ajalaris/rag/blob/master/ollama-webui.png)
- **N8N**: Ejecución local de Workflown varios
![N8N_Workflow](https://github.com/ajalaris/rag/blob/master/n8n_workflow.png)
---

## Componentes Técnicos Destacados

1. **Modelo de embeddings multilingüe**
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

2. **Índice vectorial FAISS**
   - Búsqueda eficiente por similitud semántica

3. **Chunking de documentos**
   - División en fragmentos de 1000 caracteres
   - Overlap de 100 caracteres

4. **Modelo LLM: Mistral 7B**
   - Versión optimizada para inferencia rápida

---

## Ventajas del Sistema

- **Respuestas basadas en documentos propios**
- **Privacidad de datos** (procesamiento local)
- **Verificabilidad** con citas de fuentes
- **Multilingüe** (embeddings multilingües)
- **Doble interfaz**: Web UI y API
- **Docker ready**: Fácil despliegue

---

## Casos de Uso

- **Asistentes de documentación**
- **Sistemas de soporte interno**
- **Análisis de documentos legales**
- **Investigación académica**
- **Bases de conocimiento corporativas**

---

## Configuración en Docker

- **Contenedor 1**: Aplicación (Streamlit + FastAPI)
- **Contenedor 2**: Ollama con Mistral
- **Volúmenes**:
  - Carpeta `documentos`
  - Índice FAISS
- **Networking**: Comunicación entre contenedores

---

## Configuración Avanzada

- **Variables de entorno personalizables**
- **Parámetros ajustables**:
  - Temperatura (creatividad)
  - Tokens máximos (longitud respuesta)
  - Carpeta de documentos
  - Ruta del índice
- **Soporte para múltiples formatos** (PDF, TXT)

---

## Extensibilidad

- **Soporte para otros modelos** de Ollama
- **Posibilidad de añadir formatos** adicionales
- **Adaptable a otros backends** LLM
- **Personalización de prompts** del sistema

---
# Deploy #
Paso a paso:

1.	**Clonacion del repositorio https://github.com/ajalaris/rag.git**
````
git clone https://github.com/ajalaris/rag.git
````
2.	**cambiar al directorio rag**
````
cd rag/
````
3.	**Modificacion de datos específicos de la implementación:**
   -  a.	En n8n_workflow.json 
````
"fromEmail": "la casilla que recibe las consultas",
````
   -  b. En Docker-compose.yml 
      - i.	Servicio n8n 
         - N8N_BASIC_AUTH_PASSWORD=UnaClaveParaN8N
         - N8N_HOST=<FQDN del dockerhost, sin protocolo> ej: server.miempresa.local
         - N8N_EDITOR_BASE_URL=http://<IP del dockerhost>
         - WEBHOOK_URL=”http://<FQDN del dockerhost>”
         - N8N_ENCRYPTION_KEY=unaClaveDeEncriptaciónMuySegura
      - ii.	En ollama-webui
         - PUBLIC_OLLAMA_API_BASE_URL=http://<FQDN del dockerhost>:11434/api
4.	Crear la imagen y encender Ollama
````
Docker compose up -d n8n postgres ollama ollama-webui 
````
5.	Ejecutar la consola del contenedor de ollama
````
docker compose exec ollama /bin/bash 
````
6.	Cargar el modelo en el contenedor
````
ollama run mistral:7b-instruct-q2_K
````
7.	Copiar los archivos pdf y txt sobre los que se realizarán las consultas en la carpeta ./documentos (si no existe, crearla)
8.	Arrancar los contenedores de api y aplicación
```` Docker compose up -d rag-app rag-api ````
Este ultimo paso va a tomar algún tiempo porque instala una cantidad considerable de paquetes.

## ¡Gracias!
