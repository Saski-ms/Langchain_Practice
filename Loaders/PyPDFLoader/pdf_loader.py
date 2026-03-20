
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("IS Officer Syllabus.pdf")


documents = loader.load()

print(documents)