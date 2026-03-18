from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_template=ChatPromptTemplate([
   ('system','You are a helpful {domain} expert.'),
   ('human','Explain in simple terms, what is {topic}')]
)

prompt=chat_template.invoke({'domain':'Barbie','topic':'Dollhouse'})

llm=ChatOpenAI()
response=llm.invoke(prompt)

print(response.content)