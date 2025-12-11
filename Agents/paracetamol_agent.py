from utils import call_llm,call_llm_with_history,safe_eval,MODEL
from rag_pipeline.rag_pipeline import rag_answer_paracetamol



def paracetamol_agent(message):
    print('paracetamol_agent is running')
    raw_paractamol_output = rag_answer_paracetamol(message)
    return raw_paractamol_output