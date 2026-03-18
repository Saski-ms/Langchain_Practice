# from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline


# llm=HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
#     task="text-generation",
# )


# pipeline=ChatHuggingFace(llm=llm,template="{input}")
# result=pipeline.invoke("What is AI?")
# print(result)

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=100
)

llm = HuggingFacePipeline(pipeline=pipe)

response = llm.invoke("Where is India located?")

print(response)