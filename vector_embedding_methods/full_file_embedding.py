##IMPORTS##

import numpy as np
import os
from utils_embedding import full_file_collection,full_file_pattern,DB_PATH
from utils_embedding import FULL_FILE_COLLECTION_NAME,embedding_model,chroma_client,FULL_FOLDER_PATH,file_paths



texts = []
filenames = []

##READING FILES FROM FILE STORAGE PATH
for file_path in file_paths:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:  
                texts.append(content)
                filenames.append(os.path.basename(file_path))
                print(f"Loaded: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


## TURNING TEXT INTO VECTORS
full_file_embeddings = embedding_model.encode(texts)

# Logging vector creation
print(f"\n{'='*60}")
print(f"Model: all-MiniLM-L6-v2")
print(f"Embedding dimension: {full_file_embeddings.shape[1]}")
print(f"Number of files embedded: {full_file_embeddings.shape[0]}")
print(f"{'='*60}\n")


# Fetch existing collection if present, otherwise create a new one.
# Attach some metadata describing what this collection stores.


# Creating uqique id and metadata for chromaDB to be added along with vectors and text

def extract_person(path):
    filename = os.path.basename(path)            # Insulin_file_20.txt
    person = filename.split("_file_")[0]         # "Insulin"
    return {"person": person}

person_dict = [extract_person(p) for p in file_paths]

print("Adding embeddings to ChromaDB...")
ids = [f"doc_{i}" for i in range(len(filenames))]
metadatas = [{"filename": fn, **fp} for fn, fp in zip(filenames, person_dict)]


# Adding vectors, texts, metadata and ids in chromaDB collection
full_file_collection.add(
    embeddings=full_file_embeddings.tolist(),
    documents=texts,
    metadatas=metadatas,
    ids=ids
)


# Logging
print("\n" + "="*60)
print("ChromaDB Collection Info:")
print("="*60)
print(f"Collection name: {FULL_FILE_COLLECTION_NAME}")
print(f"Total documents: {full_file_collection.count()}")
print(f"Database path: {DB_PATH}")
print("\nTo query this collection later, use:")
print(f"  collection = chroma_client.get_collection('{FULL_FILE_COLLECTION_NAME}')")
print(f"  results = collection.query(query_texts=['your query'], n_results=5)")



