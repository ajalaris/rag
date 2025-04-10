# app.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Importación corregida para HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI  # Añadido para OpenAI
from langchain.chains import RetrievalQA
import torch
import requests
from datetime import datetime
from pathlib import Path
import shutil

# Intentar usar CUDA si está disponible
cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"

# Configuración de la página
st.set_page_config(page_title="Consulta de documentos con LLM", layout="wide")
st.title("Consulta de documentos con LLM")

# Parámetros configurables - con valores predeterminados
with st.sidebar:
    st.header("Configuración")
    st.info(f"Dispositivo de cómputo: {device.upper()}")
    if cuda_available:
        st.success(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    
    # Selector de LLM
    llm_provider = st.selectbox(
        "Proveedor de LLM",
        ["Ollama (Mistral)", "OpenAI"],
        index=0
    )
    
    docs_folder = st.text_input("Carpeta de documentos", "documentos")
    index_path = st.text_input("Ruta para guardar el índice", "faiss_index")
    
    # Configuración específica según el proveedor seleccionado
    if llm_provider == "Ollama (Mistral)":
        st.subheader("Configuración de Ollama")
        # Usar variable de entorno para la URL de Ollama o valor predeterminado
        default_ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        ollama_base_url = st.text_input("URL de Ollama", default_ollama_url)
        ollama_model_name = st.text_input("Nombre del modelo", "mistral:7b-instruct-q2_K")
    else:  # OpenAI
        st.subheader("Configuración de OpenAI")
        openai_api_key = st.text_input("OpenAI API Key", os.environ.get("OPENAI_API_KEY", ""), type="password")
        openai_model_name = st.selectbox(
            "Modelo de OpenAI", 
            ["gpt-3.5-turbo", "gpt-4o", "gpt-4", "gpt-4-turbo"],
            index=0
        )
    
    st.subheader("Parámetros del modelo")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.1)
    max_tokens = st.slider("Tokens máximos", 100, 2000, 1000)
    
    if st.button("Crear/Actualizar Índice"):
        with st.spinner("Procesando documentos..."):
            st.session_state['create_index'] = True

# Función para subir documentos
def upload_documents(folder_path):
    """Función para subir documentos a la carpeta del repositorio"""
    st.subheader("Subir nuevos documentos")
    
    # Crear la carpeta si no existe
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
    uploaded_files = st.file_uploader("Selecciona archivos PDF o TXT", 
                                     type=["pdf", "txt"], 
                                     accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Subiendo documentos..."):
            uploaded_count = 0
            for uploaded_file in uploaded_files:
                # Crear un nombre seguro para el archivo
                safe_filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + uploaded_file.name
                file_path = os.path.join(folder_path, safe_filename)
                
                # Guardar el archivo
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    uploaded_count += 1
                    st.success(f"✅ Archivo subido: {safe_filename}")
                except Exception as e:
                    st.error(f"❌ Error al subir {uploaded_file.name}: {str(e)}")
            
            if uploaded_count > 0:
                st.success(f"Se subieron {uploaded_count} documento(s) correctamente")
                # Sugerir recrear el índice
                st.info("Recuerda hacer clic en 'Crear/Actualizar Índice' para procesar los nuevos documentos")
                return True
    return False

# Función para eliminar documentos
def delete_documents(folder_path):
    """Función para eliminar documentos del repositorio"""
    st.subheader("Eliminar documentos")
    
    if not os.path.exists(folder_path):
        st.warning(f"La carpeta {folder_path} no existe")
        return False
    
    # Listar documentos disponibles
    docs = sorted([f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.txt'))])
    
    if not docs:
        st.info("No hay documentos para eliminar")
        return False
    
    # Opciones de eliminación
    delete_option = st.radio(
        "Selecciona una opción:",
        ["Eliminar documentos individuales", "Eliminar todos los documentos"]
    )
    
    if delete_option == "Eliminar documentos individuales":
        selected_docs = st.multiselect("Selecciona los documentos a eliminar:", docs)
        
        if selected_docs and st.button("Eliminar seleccionados", type="primary"):
            with st.spinner("Eliminando documentos..."):
                deleted_count = 0
                for doc in selected_docs:
                    try:
                        os.remove(os.path.join(folder_path, doc))
                        deleted_count += 1
                        st.success(f"✅ Eliminado: {doc}")
                    except Exception as e:
                        st.error(f"❌ Error al eliminar {doc}: {str(e)}")
                
                if deleted_count > 0:
                    st.success(f"Se eliminaron {deleted_count} documento(s) correctamente")
                    st.info("Recuerda hacer clic en 'Crear/Actualizar Índice' para actualizar el índice")
                    return True
    
    elif delete_option == "Eliminar todos los documentos":
        st.warning("⚠️ Esta acción eliminará TODOS los documentos del repositorio")
        
        # Confirmar la eliminación con un campo de texto
        confirmation = st.text_input("Escribe 'CONFIRMAR' para eliminar todos los documentos")
        
        if confirmation == "CONFIRMAR" and st.button("Eliminar todos", type="primary"):
            with st.spinner("Eliminando todos los documentos..."):
                try:
                    # Eliminar cada archivo individualmente
                    deleted_count = 0
                    for doc in docs:
                        os.remove(os.path.join(folder_path, doc))
                        deleted_count += 1
                    
                    st.success(f"Se eliminaron todos los documentos ({deleted_count} archivos)")
                    st.info("Recuerda hacer clic en 'Crear/Actualizar Índice' para actualizar el índice")
                    return True
                except Exception as e:
                    st.error(f"Error al eliminar todos los documentos: {str(e)}")
    
    return False

# Sección para gestionar documentos
def manage_documents_section(docs_folder):
    """Sección para gestionar documentos en la barra lateral"""
    with st.sidebar:
        st.header("Gestión de documentos")
        tab1, tab2 = st.tabs(["Subir", "Eliminar"])
        
        with tab1:
            uploaded = upload_documents(docs_folder)
        
        with tab2:
            deleted = delete_documents(docs_folder)
        
        if uploaded or deleted:
            # Si hubo cambios, sugerir actualizar el índice
            st.sidebar.button("Actualizar índice ahora", on_click=lambda: st.session_state.update({'create_index': True}))

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
def get_vector_index(docs_folder, index_path, force_rebuild=False):
    """
    Crea o carga el índice vectorial.
    
    Args:
        docs_folder: Carpeta donde están los documentos
        index_path: Carpeta donde se guarda el índice
        force_rebuild: Si es True, reconstruye el índice aunque ya exista
    
    Returns:
        FAISS vectorstore o None si hay error
    """
    # Asegurar que la carpeta del índice existe
    os.makedirs(index_path, exist_ok=True)
    
    # Obtener embeddings
    embeddings = get_embeddings()
    if embeddings is None:
        return None
    
    # Intentar cargar el índice existente si no se fuerza la reconstrucción
    index_files = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_files) and not force_rebuild:
        try:
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
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

# Función para configurar el modelo OpenAI
def setup_openai_model(api_key, model_name, temperature, max_tokens):
    if not api_key:
        st.error("Por favor, proporciona una API Key de OpenAI")
        return None
    
    try:
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )
    except Exception as e:
        st.error(f"Error al configurar el modelo de OpenAI: {str(e)}")
        return None

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

# Función para verificar la conexión con OpenAI
def check_openai_connection(api_key):
    if not api_key:
        return False, "No se ha proporcionado una API Key de OpenAI."
    
    try:
        # Crear una instancia temporal para verificar la conexión
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        return True, "Conexión con OpenAI establecida correctamente."
    except Exception as e:
        return False, f"Error al conectar con OpenAI: {str(e)}"

# Añadir la sección de gestión de documentos
manage_documents_section(docs_folder)

# Comprobar la conexión con el proveedor seleccionado
llm_status = False
llm_message = ""

if llm_provider == "Ollama (Mistral)":
    llm_status, llm_message = check_ollama_connection(ollama_base_url, ollama_model_name)
else:  # OpenAI
    if openai_api_key:
        llm_status, llm_message = check_openai_connection(openai_api_key)
    else:
        llm_message = "Por favor, introduce tu API Key de OpenAI en la configuración."

# Mostrar el estado de la conexión
if llm_status:
    st.sidebar.success(llm_message)
else:
    if llm_provider == "OpenAI" and not openai_api_key:
        st.sidebar.warning(llm_message)
    else:
        st.sidebar.error(llm_message)

# Inicializar o cargar el índice
try:
    # Verificar si debemos forzar la reconstrucción del índice
    force_rebuild = st.session_state.get('create_index', False)
    if force_rebuild:
        st.session_state['create_index'] = False  # Resetear el flag
    
    with st.spinner("Cargando índice de documentos..."):
        vectorstore = get_vector_index(docs_folder, index_path, force_rebuild=force_rebuild)
    
    # Configurar la cadena de consulta si el índice está disponible y el proveedor LLM está conectado
    llm_model = None
    
    if vectorstore is not None:
        if llm_provider == "Ollama (Mistral)" and llm_status:
            # Configurar el modelo Ollama
            llm_model = setup_ollama_model(ollama_base_url, ollama_model_name, temperature, max_tokens)
        elif llm_provider == "OpenAI" and openai_api_key:
            # Configurar el modelo OpenAI
            llm_model = setup_openai_model(openai_api_key, openai_model_name, temperature, max_tokens)
        
        if llm_model:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_model,
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
    
    1. **Selección del proveedor LLM:**
       - Puedes elegir entre Ollama (Mistral) y OpenAI
       - Para OpenAI, necesitarás proporcionar una API Key válida
    
    2. **Configuración inicial:**
       - Puedes subir documentos PDF y TXT usando la sección "Gestión de documentos"
       - Para Ollama: configura la URL de conexión si es diferente de la predeterminada
       - Para OpenAI: introduce tu API Key y selecciona el modelo deseado

    3. **Gestión de documentos:**
       - **Subir:** Puedes cargar PDF y TXT directamente desde la interfaz
       - **Eliminar:** Puedes eliminar documentos individualmente o todos a la vez
       - Después de agregar o eliminar documentos, haz clic en "Crear/Actualizar Índice"

    4. **Realización de consultas:**
       - Escribe tu pregunta en el área de texto
       - El sistema buscará información relevante en tus documentos
       - El LLM seleccionado generará una respuesta basada únicamente en el contenido de tus documentos
       
    5. **Ajustes avanzados:**
       - Temperatura: controla la creatividad del modelo (valores más bajos = respuestas más deterministas)
       - Tokens máximos: limita la longitud de las respuestas
    """)

# Información sobre los proveedores
with st.expander("Información sobre los proveedores LLM"):
    st.markdown("""
    ### Ollama (Mistral)
    
    Ollama ejecuta modelos de lenguaje localmente, lo que ofrece:
    
    - **Privacidad:** Los datos nunca salen de tu entorno
    - **Sin costos de API:** Uso ilimitado sin tarifas adicionales
    - **Personalización:** Puedes usar diferentes modelos de Ollama
    
    Esta configuración requiere que Ollama esté instalado y ejecutándose en tu entorno local o en un contenedor Docker accesible.
    
    ### OpenAI
    
    OpenAI ofrece modelos en la nube con:
    
    - **Alto rendimiento:** Modelos de última generación como GPT-4
    - **Escalabilidad:** No requiere recursos locales significativos
    - **Actualizaciones constantes:** Los modelos se actualizan regularmente
    
    Esta configuración requiere una API Key válida de OpenAI y conexión a internet. Se aplican los costos según el uso de la API.
    """)
