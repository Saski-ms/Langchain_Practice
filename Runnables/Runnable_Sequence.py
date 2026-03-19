from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

from dotenv import load_dotenv

load_dotenv()


prompt=PromptTemplate(
    template='What is {topic}?',
    input_variables='topic'
)
print(prompt)

parser=StrOutputParser()

model=ChatOpenAI()


chain=RunnableSequence(prompt,model,parser)
print(chain.invoke({'topic':'AI'}))
