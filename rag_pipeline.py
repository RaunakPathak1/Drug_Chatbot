from sentence_transformers import SentenceTransformer
import numpy as np
import os
import glob
import chromadb
from chromadb.config import Settings
from utils import call_llm,call_llm_with_history,safe_eval


MODEL = 'google/gemini-2.5-flash-lite'

MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = r"C:\Users\rauna\projects\llm_engineering\My Projects\Drug Chatbot\ChromaDB\chroma_db"
COLLECTION_NAME = "document_embeddings"

model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

def retrieve_documents(query: str, n_results: int = 5):
    """
    Call your vector store and get top-k relevant chunks.
    """
    # Important: you added embeddings manually, so query with query_embeddings
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

def build_context(results) -> str:
    """
    Convert Chroma results into a single context string.
    """
    docs = results["documents"][0]      # list of texts
    metadatas = results["metadatas"][0]  # list of metadata dicts

    context_blocks = []
    for doc, meta in zip(docs, metadatas):
        filename = meta.get("filename", "unknown_source.txt")
        context_blocks.append(f"Source: {filename}\n{doc}")

    context = "\n\n---\n\n".join(context_blocks)
    return context

def rag_answer(message: str) -> str:
    """
    Full RAG pipeline:
    1) Retrieve from Chroma
    2) Build context
    3) Ask LLM using that context
    """
    results = retrieve_documents(message)
    context = build_context(results)

    system_message = f"""You are a helpful drug information assistant.

Use ONLY the following context to answer the question. 
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {message}

Answer:"""

    answer = call_llm(MODEL, system_message, message)
    return answer

