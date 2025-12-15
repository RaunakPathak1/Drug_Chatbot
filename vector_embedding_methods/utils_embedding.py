#witnessess for embeddings story
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import glob
import os


# Universal constants
DB_PATH = r"C:\Users\rauna\projects\My Projects\Drug Chatbot\ChromaDB"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=DB_PATH)
BASE_PATH = Path(r"C:\Users\rauna\projects\My Projects\Drug Chatbot\Synthetic_files")
subfolders = ["Paracetamol", "Insulin"]


# Sentence Embedding constants
# SENTENCE_COLLECTION_NAME = 'chunked_by_sentence_collection'

NEW_SENTENCE_COLLECTION_NAME = 'new_chunked_by_sentence_collection'
sentence_length = 8
overlap = 0.3

# chroma_client.delete_collection(SENTENCE_COLLECTION_NAME)
sentense_collection = chroma_client.get_or_create_collection(
    name=NEW_SENTENCE_COLLECTION_NAME,
    metadata={"description": "Sentence method embeddings using all-MiniLM-L6-v2 with chunk size of 8 and overlap of 0.3"}
)


# Full File Embedding constants with new data 
NEW_DATA_FF_COLLECTION_NAME = 'new_data_full_document_embeddings'
FULL_FOLDER_PATH = r"C:\Users\rauna\projects\My Projects\Drug Chatbot\Synthetic_files"  
subfolders = ["Paracetamol", "Insulin"]

full_file_pattern = os.path.join(FULL_FOLDER_PATH, "*.txt")
file_paths = glob.glob(full_file_pattern) 

new_data_ff_collection = chroma_client.get_or_create_collection(
    name=NEW_DATA_FF_COLLECTION_NAME,
    metadata={"description": "Text file embeddings using all-MiniLM-L6-v2 with 25 files each of Paracetamol and Insulin"}
)
