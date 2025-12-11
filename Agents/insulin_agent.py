from utils import call_llm,call_llm_with_history,safe_eval
from rag_pipeline import rag_answer_insulin

MODEL = 'google/gemini-2.5-flash-lite'


def insulin_agent(message):
    print('insulin_agent is running')
    raw_insulin_output = rag_answer_insulin(message)
    return raw_insulin_output