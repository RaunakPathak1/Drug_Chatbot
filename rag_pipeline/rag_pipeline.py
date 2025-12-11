from sentence_transformers import SentenceTransformer
import numpy as np
import os
import glob
import chromadb
from chromadb.config import Settings
from utils import call_llm,MODEL
from rag_pipeline.context_build import retrieve_documents_paracetamol,retrieve_documents_insulin,build_context




def rag_answer_paracetamol(message):
    system_message = '''You extract paracetamol related messages from the user message and retrun only 'paracetamol' related questions.
                        Only return the extracted message without any explaniations.
                        Igone any other medicine name that is mentioned.'''
    paracetamol_message = call_llm(MODEL,system_message,message)    
 
    results = retrieve_documents_paracetamol(paracetamol_message)
    context = build_context(results)

    system_message = f"""You are a assistant that answers about the person 'paracetamol'.
                        Answer only based on the context that is provided to you.
                        Use ONLY the following context to answer the question. 
                        If the answer is not in the context, say you don't know.

                        Context:
                        {context}

                        Question: {paracetamol_message}

                        Answer:"""

    answer = call_llm(MODEL, system_message, paracetamol_message)
    return answer


def rag_answer_insulin(message):
    system_message = '''You extract insulin related messages from the user message and retrun only 'insulin' related questions.
                        Only return the extracted message without any explaniations.
                        Igone any other medicine name that is mentioned.'''
    insulin_message = call_llm(MODEL,system_message,message)    
 
    results = retrieve_documents_insulin(insulin_message)
    context = build_context(results)

    system_message = f"""You are a helpful assistant that answers about the person 'insulin'.
                        Answer only based on the context that is provided to you.
                        Use ONLY the following context to answer the question. 
                        If the answer is not in the context, say you don't know.

                        Context:
                        {context}

                        Question: {insulin_message}

                        Answer:"""
    answer = call_llm(MODEL, system_message, insulin_message)
    return answer



