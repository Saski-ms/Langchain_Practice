from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv

load_dotenv()


model1=ChatOpenAI()

model2=ChatAnthropic()

promp1=PromptTemplate(
    template='Generate a short and simple notes from text{text}', input_variable=['text'])

promp2=PromptTemplate(
    template='Generate 5 short question answer from text:{text}',
    input_variables=['text']
)

promp3=PromptTemplate(
    template='Merge the notes and quiz into single document {notes} and {text}',
    input_variable=['notes','quiz']
)
parser=StrOutputParser()



parallel_chain=RunnableParallel({
    'notes':promp1 | model1 |parser,
    'text':promp2 | model2 |parser
})

merge_chain= promp3 |model1 | parser

chain=parallel_chain | merge_chain

text="""
This playlist is absolute gold! 🙌 Easily one of the best resources for learning LangChain.

Really hoping you cover AI agents / multi-agent systems next — especially from a startup-building perspective. Would love a step-by-step breakdown on how to build real agentic AI apps or even SaaS tools.

Stuff like choosing between AutoGen, CrewAI, ReAct, designing agent workflows, connecting tools, and making it production-ready. This next wave of AI agents feels like a game-changer — and your teaching style makes it all feel possible.

Can’t wait for that part!

"""

result=chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii
