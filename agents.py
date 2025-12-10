

def cat_agent(message):
    system_message = '''You are a helpful assistant that catergorizes the given message into two catergories.
                        Category 1 : General Query
                        Category 2 : Drug Information
                        If the message contains any drug or medicine name then put it into Category 2 : Drug Information
                        Return only the category decision of the whole message.
    '''
    category = call_llm(MODEL, system_message, message)
    print(category)
    return category

def reply_normal_agent(message,history):
    system_message = 'You are a helpful assistant that can answer questions and help with tasks.'
    response = call_llm_with_history(MODEL, system_message,history, message)
    return response


def paracetamol_agent(message):
    print('paracetamol_agent is running')
    system_message = '''You are a helpful assistant that answers questions only related to paracetamol and nothing else.
                        If any other drug is mentioned then completly ignore its mention and do not talk about it.
                        Return results in Dict format where the keys are different medicine names
                        Return ONLY a Python dict.
                        Do NOT wrap the response in backticks.
                        Do NOT use JSON.
                        Do NOT add explanations.
                        Example : user message : what is paracetamol. what is insulin
                                  assistant message : {'paracetamol' : 'details about paracetamol',
                                                        'insulin' : 'details about insulin'}'''
    raw_paractamol_output = call_llm(MODEL, system_message, message)
    paracetamol = safe_eval(raw_paractamol_output)
    return paracetamol['paracetamol']


def insulin_agent(message):
    print('insulin_agent is running')
    system_message = '''You are a helpful assistant that answers questions only related to insulin and nothing else.
                        If any other drug is mentioned then completly ignore its mention and do not talk about it.
                        Return results in Dict format where the keys are different medicine names
                        Return ONLY a Python dict.
                        Do NOT wrap the response in backticks.
                        Do NOT use JSON.
                        Do NOT add explanations.
                        Example : user message : what is paracetamol. what is insulin
                                  assistant message : {'paracetamol' : 'details about paracetamol',
                                                        'insulin' : 'details about insulin'}'''
    raw_insulin_output = call_llm(MODEL, system_message, message)
    insulin = safe_eval(raw_insulin_output)
    return insulin['insulin']

def drug_id(message):
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




