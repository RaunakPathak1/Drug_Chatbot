from sentence_transformers import SentenceTransformer
import numpy as np
import os
import glob
import chromadb
from chromadb.config import Settings

model = SentenceTransformer('all-MiniLM-L6-v2')

FOLDER_PATH = r"C:\Users\rauna\projects\llm_engineering\My Projects\Drug Chatbot\Synthetic_files"  # Uncomment and set your path
file_pattern = os.path.join(FOLDER_PATH, "*.txt")
COLLECTION_NAME = "document_embeddings"  # ChromaDB collection name
DB_PATH = r"C:\Users\rauna\projects\llm_engineering\My Projects\Drug Chatbot\ChromaDB\chroma_db"

chroma_client = chromadb.PersistentClient(path=DB_PATH)

file_paths = glob.glob(file_pattern)

texts = []
filenames = []

for file_path in file_paths:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:  # Only add non-empty files
                texts.append(content)
                filenames.append(os.path.basename(file_path))
                print(f"Loaded: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

embeddings = model.encode(texts)

print(f"\n{'='*60}")
print(f"Model: all-MiniLM-L6-v2")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Number of files embedded: {embeddings.shape[0]}")
print(f"{'='*60}\n")


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Text file embeddings using all-MiniLM-L6-v2"}
)

print("Adding embeddings to ChromaDB...")
ids = [f"doc_{i}" for i in range(len(filenames))]
metadatas = [{"filename": fn, "file_path": fp} for fn, fp in zip(filenames, file_paths)]

collection.add(
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas,
    ids=ids
)


print("\n" + "="*60)
print("ChromaDB Collection Info:")
print("="*60)
print(f"Collection name: {COLLECTION_NAME}")
print(f"Total documents: {collection.count()}")
print(f"Database path: {DB_PATH}")
print("\nTo query this collection later, use:")
print(f"  collection = chroma_client.get_collection('{COLLECTION_NAME}')")
print(f"  results = collection.query(query_texts=['your query'], n_results=5)")



