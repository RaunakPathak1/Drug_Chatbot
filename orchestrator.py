from agents import cat_agent, reply_normal_agent, drug_id

def orch(message,history) :
    catergory = cat_agent(message)
    if 'General Query' in catergory:
        return reply_normal_agent(message,history)
    elif 'Drug Information' in catergory:
        return drug_id(message)
    else :
        return "Sorry, I could not categorize your question. Please try rephrasing it."