from sentence_transformers import SentenceTransformer
import numpy as np
import os
import glob
import chromadb
from chromadb.config import Settings
from utils import call_llm,MODEL



MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = r"C:\Users\rauna\projects\llm_engineering\My Projects\Drug Chatbot\ChromaDB\chroma_db"
COLLECTION_NAME = "document_embeddings"

model = SentenceTransformer(MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)



def retrieve_documents_paracetamol(message: str, n_results: int = 5):
    system_message = '''You extract paracetamol related messages from the user message and retrun only 'paracetamol' related questions.
                        Only return the extracted message without any explaniations.
                        Igone any other medicine name that is mentioned.'''
    paracetamol_message = call_llm(MODEL,system_message,message)
    print(paracetamol_message)
    message_embedding = model.encode(paracetamol_message).tolist()

    results = collection.query(
        query_embeddings=[message_embedding],
        n_results=n_results
    )
    return results


def retrieve_documents_insulin(message: str, n_results: int = 10):
    system_message = '''You extract Insulin related messages from the user message and retrun only 'Insulin' related questions.
                        Only return the extracted message without any explaniations.
                        Igone any other medicine name that is mentioned.'''
    insulin_message = call_llm(MODEL,system_message,message)
    print(insulin_message)
    message_embedding = model.encode(insulin_message).tolist()

    results = collection.query(
        query_embeddings=[message_embedding],
        n_results=n_results
    )
    return results



def build_context(documents) -> str:
    docs = documents["documents"][0]      
    metadatas = documents["metadatas"][0]  # list of metadata dicts

    context_blocks = []
    for doc, meta in zip(docs, metadatas):
        filename = meta.get("filename", "unknown_source.txt")
        context_blocks.append(f"Source: {filename}\n{doc}")

    context = "\n\n---\n\n".join(context_blocks)
    return context