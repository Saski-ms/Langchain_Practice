from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model=ChatOpenAI()


chat_history=[
    SystemMessage(content="You are a helpful assistant"),
]

while True:
    user=input('User:')
    chat_history.append(HumanMessage(content=user))
    if user.lower() in ["exit","quit"]:
        print("Exiting the chatbot. Goodbye!")
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)

print("Chat history:",chat_history)