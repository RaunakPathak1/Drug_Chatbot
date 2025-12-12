##IMPORTS##

import numpy as np
import os
from utils import embedding_model,chroma_client,file_paths,COLLECTION_NAME,DB_PATH


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
embeddings = embedding_model.encode(texts)

# Logging vector creation
print(f"\n{'='*60}")
print(f"Model: all-MiniLM-L6-v2")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Number of files embedded: {embeddings.shape[0]}")
print(f"{'='*60}\n")


# Fetch existing collection if present, otherwise create a new one.
# Attach some metadata describing what this collection stores.
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Text file embeddings using all-MiniLM-L6-v2"}
)

# Creating uqique id and metadata for chromaDB to be added along with vectors and text
print("Adding embeddings to ChromaDB...")
ids = [f"doc_{i}" for i in range(len(filenames))]
metadatas = [{"filename": fn, "file_path": fp} for fn, fp in zip(filenames, file_paths)]


# Adding vectors, texts, metadata and ids in chromaDB collection
collection.add(
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas,
    ids=ids
)


# Logging
print("\n" + "="*60)
print("ChromaDB Collection Info:")
print("="*60)
print(f"Collection name: {COLLECTION_NAME}")
print(f"Total documents: {collection.count()}")
print(f"Database path: {DB_PATH}")
print("\nTo query this collection later, use:")
print(f"  collection = chroma_client.get_collection('{COLLECTION_NAME}')")
print(f"  results = collection.query(query_texts=['your query'], n_results=5)")



