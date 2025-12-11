from utils import call_llm,call_llm_with_history,safe_eval,MODEL
from Agents.paracetamol_agent import paracetamol_agent
from Agents.insulin_agent import insulin_agent



def drug_reply(message):
    system_message = '''You are a helpful assistant that picks out all the drug names in a query and gives a list of the drug names present in the query.
                        Return only python list
                        Fix the typos as well if you find any'''
    drugs_raw = call_llm(MODEL, system_message, message)
    drugs = safe_eval(drugs_raw)
    print(drugs)

    results = []

    for drug in drugs:
        if drug.lower() == 'paracetamol':
            results.append(paracetamol_agent(message))
        elif drug.lower() == 'insulin' :
            results.append(insulin_agent(message)) 
        else :
            results.append(f"Invalid drug {drug}. I only answer questions related to paracetamol and insulin")

    return results