# List of witnesses (imports)
from utils import call_llm,MODEL
from vector_embedding_methods.utils_embedding import embedding_model,chroma_client,SENTENCE_COLLECTION_NAME


sentense_doc_collection = chroma_client.get_collection(SENTENCE_COLLECTION_NAME)



def sen_retrieve_documents_paracetamol(message: str, n_results: int = 5):
    print('reteriving sentence documents for paracetamol...')
    system_message = '''You extract paracetamol related messages from the user message and retrun only 'paracetamol' related questions.
                        Only return the extracted message without any explaniations.
                        Igone any other medicine name that is mentioned.'''
    paracetamol_message = call_llm(MODEL,system_message,message)
    print(paracetamol_message)
    message_embedding = embedding_model.encode(paracetamol_message).tolist()

    results = sentense_doc_collection.query(
        query_embeddings=[message_embedding],
        n_results=n_results
    )
    return results


def sen_retrieve_documents_insulin(message: str, n_results: int = 5):
    print('reteriving sentence documents for insulin...')
    system_message = '''You extract Insulin related messages from the user message and retrun only 'Insulin' related questions.
                        Only return the extracted message without any explaniations.
                        Igone any other medicine name that is mentioned.'''
    insulin_message = call_llm(MODEL,system_message,message)
    print(insulin_message)
    message_embedding = embedding_model.encode(insulin_message).tolist()

    results = sentense_doc_collection.query(
        query_embeddings=[message_embedding],
        n_results=n_results
    )
    return results



def sen_build_context(documents) -> str:
    print('building sentence context...')
    docs = documents["documents"][0]      
    metadatas = documents["metadatas"][0]  # list of metadata dicts

    context_blocks = []
    for doc, meta in zip(docs, metadatas):
        filename = meta.get("filename", "unknown_source.txt")
        person_name = meta.get("person", "Unknown Person")
        context_blocks.append(f"Source: {person_name}\n{filename}\n{doc}")

    context = "\n\n---\n\n".join(context_blocks)
    return context