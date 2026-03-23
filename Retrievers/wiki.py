from langchain_community.retrievers.wikipedia import WikipediaRetriever


retriver=WikipediaRetriever(top_k_results=3,lang="en")

query="World war IRAN and Dubai"

docs=retriver.invoke(query)

for i,doc in enumerate(docs):
    print(f"-----Result{i+1}-----")
    print(f"Content is:{doc.page_content}.....")

print(docs)