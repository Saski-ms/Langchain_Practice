from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)


docs=["What is my name?",
      "What is your name?",
      "What is the capital of Maharashtra?"]
result=embeddings.embed_documents(docs)
print(result)