##IMPORTS

from dotenv import load_dotenv
import os
from openai import OpenAI
import ast


##LOADING OPENROUTER
load_dotenv(override=True)
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
openrouter_url = "https://openrouter.ai/api/v1"
openrouter = OpenAI(base_url=openrouter_url, api_key=openrouter_api_key)

##MODEL FOR LLM CALLS
MODEL = 'openai/gpt-4o-2024-11-20'
gpt_model = 'gpt-4o-2024-11-20'


##FUNCTION UTILS####
def call_llm(MODEL, system_message, user_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def call_llm_with_history(MODEL, system_message, history, user_message):
    messages = [{"role": "system", "content": system_message}] + history + [
        {"role": "user", "content": user_message}
    ]
    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def safe_eval(text):
    return ast.literal_eval(text)