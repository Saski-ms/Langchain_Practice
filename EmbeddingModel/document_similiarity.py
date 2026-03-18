from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=100)

documents=["Mango is a fruit","I like to eat mangoes","The sky is blue","The sun is bright"]

doc_embeddings=embedding.embed_documents(documents)


query="What do I like to eat ?"
query_embedding=embedding.embed_query(query)


similiarity_score=cosine_similarity([query_embedding],doc_embeddings)

index,score=sorted(list(enumerate(similiarity_score[0])),key=lambda x:x[1])[-1]
print(f"Most similiar document: {documents[index]} with score {score}")