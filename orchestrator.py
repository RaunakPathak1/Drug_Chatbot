from Agents.categorization_agent import cat_agent
from Agents.normal_reply_agent import reply_normal_agent
from Agents.drug_reply_agent import drug_reply

def orch(message,history) :
    catergory = cat_agent(message)
    if 'General Query' in catergory:
        return reply_normal_agent(message,history)
    elif 'Drug Information' in catergory:
        return drug_reply(message)
    else :
        return "Sorry, I could not categorize your question. Please try rephrasing it."