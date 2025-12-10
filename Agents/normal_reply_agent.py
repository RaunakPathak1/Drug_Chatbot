from utils import call_llm,call_llm_with_history,safe_eval

MODEL = 'google/gemini-2.5-flash-lite'


def reply_normal_agent(message,history):
    system_message = 'You are a helpful assistant that can answer questions and help with tasks.'
    response = call_llm_with_history(MODEL, system_message,history, message)
    return response