from utils import call_llm,call_llm_with_history,safe_eval
from rag_pipeline import rag_answer

MODEL = 'google/gemini-2.5-flash-lite'


def paracetamol_agent(message):
    print('paracetamol_agent is running')
    # system_message = '''You are a helpful assistant that answers questions only related to paracetamol and nothing else.
    #                     If any other drug is mentioned then completly ignore its mention and do not talk about it.
    #                     Return results in Dict format where the keys are different medicine names
    #                     Return ONLY a Python dict.
    #                     Do NOT wrap the response in backticks.
    #                     Do NOT use JSON.
    #                     Do NOT add explanations.
    #                     Example : user message : what is paracetamol. what is insulin
    #                               assistant message : {'paracetamol' : 'details about paracetamol',
    #                                                     'insulin' : 'details about insulin'}'''
    raw_paractamol_output = rag_answer(message)
    # paracetamol = safe_eval(raw_paractamol_output)
    # return paracetamol['paracetamol']
    return raw_paractamol_output