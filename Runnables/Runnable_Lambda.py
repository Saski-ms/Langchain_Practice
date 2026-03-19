from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
# from langchain_core.runnables.passthrough import RunnablePassthrough

# from dotenv import load_dotenv

# load_dotenv()


def word_counter(text):
    return len(text.split())

runnable_word_counter=RunnableLambda(word_counter)
print(runnable_word_counter.invoke('Heyy'))
