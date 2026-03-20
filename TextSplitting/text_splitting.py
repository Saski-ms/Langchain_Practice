from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)


text="""
Across Wealth Management, Goldman Sachs helps empower clients and customers around the world to reach their financial goals. Our advisor-led wealth management businesses provide financial planning, investment management, banking and comprehensive advice to a wide range of clients, including ultra-high net worth and high net worth individuals, as well as family offices, foundations and endowments, and corporations and their employees. Across Wealth Management, our growth is driven by a relentless focus on our people, our clients and customers, and leading-edge technology, data and design.
"""

result=splitter.split_text(text)
print(result)
