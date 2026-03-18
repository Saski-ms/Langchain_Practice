from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

prompt=PromptTemplate(template='Write an essay for 5 year old about {Season}',input_variables=['Season'])

model=ChatOpenAI()

parser=StrOutputParser()

chain= prompt | model | parser

result=chain.invoke({'Season':'Rainy'})

print(result)



