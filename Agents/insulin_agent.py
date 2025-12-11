from utils import call_llm,call_llm_with_history,safe_eval,MODEL
from rag_pipeline.rag_pipeline import rag_answer_insulin


def insulin_agent(message):
    print('insulin_agent is running')
    raw_insulin_output = rag_answer_insulin(message)
    return raw_insulin_output