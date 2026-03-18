from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-pro-latest")
result=model.invoke("Where is India located?")
print(result.text)