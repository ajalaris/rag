# app.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Importación corregida para HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
# Intentar usar CUDA si está disponible
import torch
cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
st.sidebar.info(f"Dispositivo de cómputo: {device.upper()}")
if cuda_available:
    st.sidebar.success(f"GPU detectada: {torch.cuda.get_device_name(0)}")
# Configuración de la página
st.set_page_config(page_title="Consulta de documentos con Mistral", layout="wide")
st.title("Consulta de documentos con Mistral:7b")

# Parámetros configurables - con valores predeterminados para Docker
with st.sidebar:
    st.header("Configuración")
    docs_folder = st.text_input("Carpeta de documentos", "documentos")
    # Usar variable de entorno para la URL de Ollama o valor predeterminado
    default_ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
    ollama_base_url = st.text_input("URL de Ollama", default_ollama_url)
    model_name = st.text_input("Nombre del modelo", "mistral:7b-instruct-q2_K")
    index_path = st.text_input("Ruta para guardar el índice", "faiss_index")
    
    st.subheader("Parámetros del modelo")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.1)
    max_tokens = st.slider("Tokens máximos", 100, 2000, 500)
    
    if st.button("Crear/Actualizar Índice"):
        with st.spinner("Procesando documentos..."):
            create_index = True

# Función para cargar documentos
def load_documents(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        st.sidebar.error(f"La carpeta {folder_path} no existe. Creándola...")
        os.makedirs(folder_path, exist_ok=True)
        return documents
        
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                st.sidebar.write(f"✅ Cargado: {file}")
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
                st.sidebar.write(f"✅ Cargado: {file}")
            else:
                st.sidebar.write(f"❌ Formato no soportado: {file}")
        except Exception as e:
            st.sidebar.write(f"❌ Error en {file}: {str(e)}")
    return documents

# Función para crear embeddings compatibles
def get_embeddings():
    try:
        # Detectar si CUDA está disponible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Usar HuggingFaceEmbeddings con el dispositivo detectado
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": device}
        )
    except Exception as e:
        st.error(f"Error al cargar el modelo de embeddings: {str(e)}")
        # Alternativa: usar embeddings más simples si el otro falla
        try:
            from langchain_community.embeddings import FakeEmbeddings
            st.warning("Usando embeddings de respaldo temporales. La calidad puede ser menor.")
            return FakeEmbeddings(size=384)
        except:
            st.error("No se pudo cargar ningún modelo de embeddings.")
            return None
# Función para crear o cargar el índice
@st.cache_resource
def get_vector_index(docs_folder, index_path):
    # Asegurar que la carpeta del índice existe
    os.makedirs(index_path, exist_ok=True)
    
    # Obtener embeddings
    embeddings = get_embeddings()
    if embeddings is None:
        return None
    
    # Intentar cargar el índice existente
    index_files = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_files):
        try:
            vectorstore = FAISS.load_local(index_path, embeddings)
            st.sidebar.success("Índice cargado correctamente")
            return vectorstore
        except Exception as e:
            st.warning(f"No se pudo cargar el índice existente: {str(e)}")
    
    # Crear un nuevo índice
    documents = load_documents(docs_folder)
    if not documents:
        st.warning(f"No se encontraron documentos en {docs_folder}. Por favor, añade algunos archivos PDF o TXT.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    try:
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Guardar el índice para uso futuro
        vectorstore.save_local(index_path)
        st.sidebar.success("Índice creado y guardado correctamente")
        
        return vectorstore
    except Exception as e:
        st.error(f"Error al crear el índice vectorial: {str(e)}")
        return None

# Función para configurar el modelo Ollama
def setup_ollama_model(base_url, model_name, temperature, max_tokens):
    return Ollama(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        num_predict=max_tokens,
        system="Por favor, proporciona siempre tus respuestas en español, independientemente del idioma de la consulta."
    )

# Función para verificar la conexión con Ollama
def check_ollama_connection(base_url, model_name):
    import requests
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            if model_name in model_names or f"{model_name}:latest" in model_names:
                return True, f"Conectado a Ollama. Modelo {model_name} disponible."
            else:
                return False, f"Conectado a Ollama, pero el modelo {model_name} no está disponible. Modelos disponibles: {', '.join(model_names)}"
        else:
            return False, f"Error al conectar con Ollama: {response.status_code}"
    except Exception as e:
        return False, f"No se pudo conectar con Ollama en {base_url}: {str(e)}"

# Comprobar la conexión con Ollama
ollama_status, ollama_message = check_ollama_connection(ollama_base_url, model_name)
if ollama_status:
    st.sidebar.success(ollama_message)
else:
    st.sidebar.error(ollama_message)

# Inicializar o cargar el índice
try:
    with st.spinner("Cargando índice de documentos..."):
        vectorstore = get_vector_index(docs_folder, index_path)
    
    # Configurar la cadena de consulta si el índice está disponible y Ollama está conectado
    if vectorstore is not None and ollama_status:
        # Configurar el modelo Ollama
        ollama_model = setup_ollama_model(ollama_base_url, model_name, temperature, max_tokens)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=ollama_model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            verbose=True
        )
        
        # Interfaz de consulta
        st.subheader("Realiza tu consulta")
        query = st.text_area("Pregunta:", height=100)
        
        if st.button("Consultar"):
            if query:
                with st.spinner("Procesando tu consulta..."):
                    try:
                        result = qa_chain({"query": query})
                        
                        st.subheader("Respuesta:")
                        st.write(result["result"])
                        
                        st.subheader("Fuentes consultadas:")
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Documento {i+1}:**")
                            st.markdown(f"- **Fuente:** {doc.metadata.get('source', 'Desconocido')}")
                            st.markdown(f"- **Página:** {doc.metadata.get('page', 'N/A')}")
                            with st.expander("Ver contenido"):
                                st.markdown(doc.page_content)
                    except Exception as e:
                        st.error(f"Error al procesar la consulta: {str(e)}")
            else:
                st.warning("Por favor, ingresa una pregunta.")
except Exception as e:
    st.error(f"Error general: {str(e)}")

# Instrucciones de uso
with st.expander("Instrucciones de uso"):
    st.markdown("""
    ### Cómo usar esta aplicación
    
    1. **Configuración inicial:**
       - Coloca tus documentos PDF y TXT en la carpeta 'documentos'
       - La aplicación se conecta automáticamente a Ollama en el contenedor 'ollama'
       - Si necesitas cambiar la URL de Ollama, puedes hacerlo en la barra lateral

    2. **Creación del índice:**
       - Haz clic en "Crear/Actualizar Índice" para procesar tus documentos
       - Este proceso extrae el texto, lo divide en fragmentos y crea un índice de búsqueda

    3. **Realización de consultas:**
       - Escribe tu pregunta en el área de texto
       - El sistema buscará información relevante en tus documentos
       - Mistral:7b generará una respuesta basada únicamente en el contenido de tus documentos
       
    4. **Ajustes avanzados:**
       - Temperatura: controla la creatividad del modelo (valores más bajos = respuestas más deterministas)
       - Tokens máximos: limita la longitud de las respuestas
    """)

# Información sobre Docker
with st.expander("Información sobre Docker"):
    st.markdown("""
    ### Configuración Docker
    
    Esta aplicación está ejecutándose en un entorno Docker con:
    
    - **Contenedor de la aplicación:** Ejecutando Streamlit
    - **Contenedor de Ollama:** Ejecutando el modelo Mistral:7b
    
    Los documentos se guardan en la carpeta 'documentos' que está montada como volumen.
    El índice FAISS se guarda en la carpeta 'faiss_index' que también está montada como volumen.
    
    Para añadir nuevos documentos, colócalos en la carpeta 'documentos' y haz clic en "Crear/Actualizar Índice".
    """)
