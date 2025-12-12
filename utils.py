##IMPORTS

from dotenv import load_dotenv
import os
from openai import OpenAI
import ast
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import glob


##LOADING OPENROUTER
load_dotenv(override=True)
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
openrouter_url = "https://openrouter.ai/api/v1"
openrouter = OpenAI(base_url=openrouter_url, api_key=openrouter_api_key)

##MODEL FOR LLM CALLS
MODEL = 'openai/gpt-4o-2024-11-20'


######CHROMA DB SETUP#######
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

FOLDER_PATH = r"C:\Users\rauna\projects\llm_engineering\My Projects\Drug Chatbot\Synthetic_files"
file_pattern = os.path.join(FOLDER_PATH, "*.txt")
COLLECTION_NAME = "document_embeddings"  
DB_PATH = r"C:\Users\rauna\projects\llm_engineering\My Projects\Drug Chatbot\ChromaDB\chroma_db"

chroma_client = chromadb.PersistentClient(path=DB_PATH)  ##CREATING CHROMA CLIENT ON DISK
file_paths = glob.glob(file_pattern)  ##FILES ALONG WITH THEIR PATHS


##FUNCTION UTILS####
def call_llm(MODEL, system_message, user_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def call_llm_with_history(MODEL, system_message, history, user_message):
    messages = [{"role": "system", "content": system_message}] + history + [
        {"role": "user", "content": user_message}
    ]
    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def safe_eval(text):
    return ast.literal_eval(text)