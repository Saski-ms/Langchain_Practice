
from langchain_community.document_loaders import TextLoader


loader = TextLoader("desc.txt", encoding="utf-8")


documents = loader.load()

print(documents[0])