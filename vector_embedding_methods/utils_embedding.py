from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import glob
import os

# Universal

DB_PATH = r"C:\Users\rauna\projects\llm_engineering\My Projects\Drug Chatbot\ChromaDB\chroma_db"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Full File Embedding constants

FULL_FOLDER_PATH = r"C:\Users\rauna\projects\llm_engineering\My Projects\Drug Chatbot\archive_synthetic_files"  
full_file_pattern = os.path.join(FULL_FOLDER_PATH, "*.txt")
FULL_FILE_COLLECTION_NAME = 'full_document_embeddings'
file_paths = glob.glob(full_file_pattern) 

full_file_collection = chroma_client.get_or_create_collection(
    name=FULL_FILE_COLLECTION_NAME,
    metadata={"description": "Text file embeddings using all-MiniLM-L6-v2"}
)


# Sentence Embedding constants

SENTENCE_COLLECTION_NAME = 'chunked_by_sentence_collection'
BASE_PATH = Path(r"C:\Users\rauna\projects\llm_engineering\My Projects\Drug Chatbot\Synthetic_files")
subfolders = ["Paracetamol", "Insulin"]

gpt_model = 'gpt-4o-2024-11-20'

sentence_length = 8
overlap = 0.3

sentense_collection = chroma_client.get_or_create_collection(
    name=SENTENCE_COLLECTION_NAME,
    metadata={"description": "Sentence method embeddings using all-MiniLM-L6-v2"}
)

