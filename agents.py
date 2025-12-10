from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(override=True)

openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
openrouter_url = "https://openrouter.ai/api/v1"
openrouter = OpenAI(base_url=openrouter_url, api_key=openrouter_api_key)

MODEL = 'google/gemini-2.5-flash-lite'

def cat_agent(message):
    system_message = '''You are a helpful assistant that catergorizes the given message into two catergories.
                        Category 1 : General Query
                        Category 2 : Drug Information
                        If the message contains any drug or medicine name then put it into Category 2 : Drug Information
    '''
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": message}]
    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    category = response.choices[0].message.content
    print(category)
    return category

def reply_normal_agent(message,history):
    system_message = 'You are a helpful assistant that can answer questions and help with tasks.'
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

def paracetamol_agent(message):
    system_message = 'You are a helpful assistant that answers questions only related to paracetamol and nothing else.'
    messages = [{"role": "system", "content": system_message},{"role": "user", "content": message}]
    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

def insulin_agent(message):
    system_message = 'You are a helpful assistant that answers questions only related to insulin and nothing else.'
    messages = [{"role": "system", "content": system_message},{"role": "user", "content": message}]
    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

def drug_id(message):
    system_message = '''You are a helpful assistant that picks out all the drug names in a query and gives a list of the drug names present in the query.
                        Return only python list
                        Fix the typos as well if you find any'''
    messages = [{"role": "system", "content": system_message},{"role": "user", "content": message}]
    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    drugs = response.choices[0].message.content
    print(drugs.lower())
    if 'paracetamol' in drugs.lower() :
        return paracetamol_agent(message)
    if 'insulin' in drugs.lower() :
        return insulin_agent(message)    
    else :
        return 'invalid drug. I only answer questions related to paracetamol and insulin'


