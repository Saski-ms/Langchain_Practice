from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

pdf_path = (
    Path(__file__).resolve().parent.parent
    / "Loaders"
    / "PyPDFLoader"
    / "ISOfficerSyllabus.pdf"
)
loader = PyPDFLoader(str(pdf_path))
docs=loader.load()

# text="""
# pypdf is a free and open-source pure-python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files. It can also add custom data, viewing options, and passwords to PDF files. pypdf can retrieve text and metadata from PDFs as well.

# See pdfly for a CLI application that uses pypdf to interact with PDFs.
# """

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result=splitter.split_documents(docs)

print(result[0])

