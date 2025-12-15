##IMPORTS##
import os
# from utils_embedding import full_file_collection
from utils_embedding import DB_PATH,BASE_PATH,subfolders,new_data_ff_collection
from utils_embedding import NEW_DATA_FF_COLLECTION_NAME,embedding_model
import glob


texts = []
filenames = []

folder_paths = [ str(BASE_PATH/sf) for sf in subfolders]

file_patterns = []
for folder_path in folder_paths:
    file_patterns.append(os.path.join(folder_path, "*.txt"))

list_file_paths = []
for file_pattern in file_patterns : 
    list_file_paths.append(glob.glob(file_pattern))

file_paths = [item for sublist in list_file_paths for item in sublist]

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



# Creating uqique id and metadata for chromaDB to be added along with vectors and text

def extract_person(path):
    filename = os.path.basename(path)               # Insulin_09.txt or Insulin_09
    name_without_ext = filename.rsplit('.', 1)[0]   # Insulin_09
    person = name_without_ext.split("_")[0]         # "Insulin"
    return {"person": person}

person_dict = [extract_person(p) for p in file_paths]

print("Adding embeddings to ChromaDB...")
ids = [f"doc_{i}" for i in range(len(filenames))]
metadatas = [{"filename": fn, **fp} for fn, fp in zip(filenames, person_dict)]


# Adding vectors, texts, metadata and ids in chromaDB collection
new_data_ff_collection.add(
    embeddings=full_file_embeddings.tolist(),
    documents=texts,
    metadatas=metadatas,
    ids=ids
)


# Logging
print("\n" + "="*60)
print("ChromaDB Collection Info:")
print("="*60)
print(f"Collection name: {NEW_DATA_FF_COLLECTION_NAME}")
print(f"Total documents: {new_data_ff_collection.count()}")
print(f"Database path: {DB_PATH}")
print("\nTo query this collection later, use:")
print(f"  collection = chroma_client.get_collection('{NEW_DATA_FF_COLLECTION_NAME}')")
print(f"  results = collection.query(query_texts=['your query'], n_results=5)")



