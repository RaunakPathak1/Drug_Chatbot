from utils import call_llm,call_llm_with_history,safe_eval

MODEL = 'google/gemini-2.5-flash-lite'

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