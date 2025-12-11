from utils import call_llm,call_llm_with_history,safe_eval
from rag_pipeline import rag_answer_paracetamol

MODEL = 'google/gemini-2.5-flash-lite'


def paracetamol_agent(message):
    print('paracetamol_agent is running')
    raw_paractamol_output = rag_answer_paracetamol(message)
    return raw_paractamol_output