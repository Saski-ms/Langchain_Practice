from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(model="claude-opus-4-6",temperature=1,max_tokens=10)

result=model.invoke("Where is India located?")
print(result.content)