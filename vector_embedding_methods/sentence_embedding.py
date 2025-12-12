import os
from utils_embedding import BASE_PATH,subfolders,DB_PATH,gpt_model,sentence_length,overlap,chroma_client
from utils_embedding import chroma_client,sentense_collection,embedding_model,SENTENCE_COLLECTION_NAME
import glob
import tiktoken
from pathlib import Path
import chromadb


folder_paths = [ str(BASE_PATH/sf) for sf in subfolders]


file_patterns = []
for folder_path in folder_paths:
    file_patterns.append(os.path.join(folder_path, "*.txt"))


list_file_paths = []
for file_pattern in file_patterns : 
    list_file_paths.append(glob.glob(file_pattern))


file_paths = [item for sublist in list_file_paths for item in sublist]


def chunk_by_lines_with_overlap(file_paths, chunk_size, overlap_ratio=0.3):
    chunks = []
    fn = []
    pn = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        overlap = int(chunk_size * overlap_ratio)
        step = chunk_size - overlap  # how many new lines we move forward each chunk

        for i in range(0, len(lines), step):
            chunk = lines[i : i + chunk_size]
            if chunk:
                chunk_with_name = [f"Filename: {filename}\n"] + chunk
                chunks.append(chunk_with_name)
                fn.append({'file_name' : filename})
                pn.append({'person' : filename.split('_')[0]})
                

            if i + chunk_size >= len(lines):
                break

    metadatas = [{**fil_nm,**per_nm} for fil_nm,per_nm in zip(fn,pn)]

    return chunks,metadatas


chunks = ["".join(chunk) for chunk in chunk_by_lines_with_overlap(file_paths,sentence_length,overlap)[0]]
metadatas = chunk_by_lines_with_overlap(file_paths,sentence_length,overlap)[1]
ids = [f'chunk_{i}' for i in range(len(chunk_by_lines_with_overlap(file_paths,sentence_length,overlap)[0]))] 


sentence_embeddings = embedding_model.encode(chunks)



sentense_collection.add(
    embeddings=sentence_embeddings,
    documents=chunks,
    metadatas=metadatas,
    ids=ids
)


# Logging
print(f"\n{'='*60}")
print("ChromaDB Collection Info:")
print(f"Model: all-MiniLM-L6-v2")
print(f"Collection name: {SENTENCE_COLLECTION_NAME}")
print(f"Embedding dimension: {sentence_embeddings.shape[1]}")
print(f"Number of files embedded: {sentence_embeddings.shape[0]}")
print(f"{'='*60}\n")
