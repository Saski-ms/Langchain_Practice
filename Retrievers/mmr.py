from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


documents=[
    Document(page_content="An AI agent is an autonomous entity that observes its environment through sensors, makes decisions based on goals, and takes actions to achieve specific outcomes."),
    Document(page_content="Unlike simple automation, AI agents use large language models (LLMs) to reason, plan, and break down complex tasks into smaller steps."),
    Document(page_content="AI agents can actively use external tools—such as web browsers, APIs, and calendars—to interact with the world and complete tasks on behalf of users."),
    Document(page_content="Multi-agent systems allow specialized AI agents to collaborate with one another, coordinating efforts to solve complex problems faster than a single agent.")
]
embeddings = OpenAIEmbeddings()

vector_Store=FAISS.from_documents(documents=documents,
                                   embedding=embeddings
                                   )

retriver=vector_Store.as_retriever(search_type="mmr",search_kwargs={"k": 2})
# vectorstore = FAISS("langchain_store", embeddings)

query="What is the use-case of AI-Agent"
result=retriver.invoke(query)

print(result)