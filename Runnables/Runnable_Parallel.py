from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence

from dotenv import load_dotenv

load_dotenv()


prompt1=PromptTemplate(
    template='What is content for reddit :{topic}?',
    input_variables='topic'
)
prompt2=PromptTemplate(
    template='What is content for linkedin :{topic}?',
    input_variables='topic'
)

parser=StrOutputParser()

model=ChatOpenAI()


chain=RunnableParallel(
    {
        'reddit':RunnableSequence(prompt1,model,parser),
        'linkedin':RunnableSequence(prompt2,model,parser)
    }
)
print(chain.invoke({'topic':'AI'}))
