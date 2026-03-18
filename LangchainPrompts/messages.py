from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
 

load_dotenv()
model=ChatOpenAI()
messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is the capital of Maharashtra?")
]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))


print(messages)