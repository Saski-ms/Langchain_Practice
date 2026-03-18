from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel
from typing import Literal


from dotenv import load_dotenv

load_dotenv()


model1=ChatOpenAI()
parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['Positive','Negative']

parser2=PydanticOutputParser(pydantic_object=Feedback)


prompt1=PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative{feedback} \n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)
classifier_chain=prompt1 |model1|parser2

prompt2=PromptTemplate(
    template='Write an appropriate reposnse to this positive feedback {feedback}',
    input_variables=['feedback'],
)
prompt3=PromptTemplate(
    template='Write an appropriate reposnse to this negative feedback {feedback}',
    input_variables=['feedback'],
)



# result=classifier_chain.invoke({'feedback':'You are the best person'})


#branch if else
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | model1 | parser),
    (lambda x: x.sentiment == "Negative", prompt3 | model1 | parser),
    RunnableLambda(lambda x: "Could not find sentiment"),
)



chain=classifier_chain| branch_chain
print(chain.invoke({'feedback':'This is a very wrong behaviour'}))

chain.get_graph().print_ascii()


