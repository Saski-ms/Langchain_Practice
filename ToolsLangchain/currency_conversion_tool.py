from langchain.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

# 🔹 Tool 1: Currency Converter
@tool
def currency_converter(base_curr: str, target_curr: str) -> dict:
    """Function for conversion factor calculation"""
    url = f"https://v6.exchangerate-api.com/v6/4f939714a5b073cce0624c76/pair/{base_curr}/{target_curr}"
    response = requests.get(url)
    return response.json()


# 🔹 Tool 2: Convert amount
@tool
def convert_base_curr(base_curr_val: int, conversion_rate: float) -> float:
    """convert base to converted"""
    return base_curr_val * conversion_rate


# 🔹 LLM
llm = ChatOpenAI(model="gpt-4o-mini")

llm_with_tools = llm.bind_tools([currency_converter, convert_base_curr])

# 🔹 User Query
messages = [
    HumanMessage(
        content="What is the conversion factor between USD and INR and also convert 10 USD to INR"
    )
]

# 1️⃣ First LLM call
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

# 2️⃣ Execute tools manually
conversion_rate = None  # store value

for tool_call in ai_message.tool_calls:

    # 🔹 Tool 1 execution
    if tool_call["name"] == "currency_converter":
        tool_result = currency_converter.invoke(tool_call["args"])

        # extract conversion rate
        conversion_rate = tool_result["conversion_rate"]

        # append tool result
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )
        )

    # 🔹 Tool 2 execution
    if tool_call["name"] == "convert_base_curr":
        # inject conversion rate manually
        tool_call["args"]["conversion_rate"] = conversion_rate

        tool_result2 = convert_base_curr.invoke(tool_call["args"])

        messages.append(
            ToolMessage(
                content=str(tool_result2),
                tool_call_id=tool_call["id"],
            )
        )

# 3️⃣ Final LLM response
final_response = llm_with_tools.invoke(messages)

print("\n✅ Final Answer:\n")
print(final_response.content)