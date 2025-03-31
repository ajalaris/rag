import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
sys.path.append('/app')  # Agrega el dir de la api al path de python

# Importa la función ppal desde  app.py
from app import (
    get_vector_index, 
    setup_ollama_model, 
    check_ollama_connection, 
    load_documents
)
from langchain.chains import RetrievalQA

app = FastAPI(title="RAG API for Document Querying")

# Configuración de parameteros (similar a Streamlit app)
DOCS_FOLDER = os.environ.get('DOCS_FOLDER', '/app/documentos')
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')
MODEL_NAME = os.environ.get('MODEL_NAME', 'mistral:7b-instruct-q2_K')
INDEX_PATH = os.environ.get('INDEX_PATH', '/app/faiss_index')
DEFAULT_TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.1))
DEFAULT_MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 500))

class QueryRequest(BaseModel):
    query: str
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.post("/query_documents")
async def query_documents(request: QueryRequest):
    try:
        # testea la conexión con Ollama 
        ollama_status, ollama_message = check_ollama_connection(OLLAMA_BASE_URL, MODEL_NAME)
        if not ollama_status:
            raise HTTPException(status_code=500, detail=f"Ollama connection error: {ollama_message}")

        # carga índice vectorDB 
        vectorstore = get_vector_index(DOCS_FOLDER, INDEX_PATH)
        if vectorstore is None:
            raise HTTPException(status_code=404, detail="No documents found or index creation failed")

        # Configura los parámetros para Ollama
        ollama_model = setup_ollama_model(
            OLLAMA_BASE_URL, 
            MODEL_NAME, 
            request.temperature, 
            request.max_tokens
        )

        # Crea la cadena deconsulta
        qa_chain = RetrievalQA.from_chain_type(
            llm=ollama_model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

        # Procesa los datos de consulta
        result = qa_chain({"query": request.query})

        # Prepara la información de fuentes
        sources = [
            {
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "content": doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content
            } 
            for doc in result['source_documents']
        ]

        return QueryResponse(
            answer=result['result'],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
