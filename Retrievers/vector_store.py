from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


documents=[
    Document(page_content="My name is Ram"),
    Document(page_content="Langchain is more powerful for building AI Agents"),
    Document(page_content="Langsmith is tool used for observations")

]
embeddings = OpenAIEmbeddings()

vector_Store=Chroma.from_documents(documents=documents,
                                   embedding=embeddings,
                                   collection_name="my_collection")

retriver=vector_Store.as_retriever(search_kwargs={"k": 2})
vectorstore = Chroma("langchain_store", embeddings)

query="What is langchain?"
result=retriver.invoke(query)

print(result)