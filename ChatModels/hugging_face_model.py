from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="Olmo-3-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="HUGGINGFACEHUB_API_TOKEN",
)

print(llm.invoke("What is AI?"))
