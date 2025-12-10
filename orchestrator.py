from agents import cat_agent, reply_normal_agent, drug_id

def orch(message,history) :
    catergory = cat_agent(message)
    if 'Category 1 : General Query' in catergory:
        return reply_normal_agent(message,history)
    elif 'Category 2 : Drug Information' in catergory:
        return drug_id(message,history)
    else :
        return "Sorry, I could not categorize your question. Please try rephrasing it."