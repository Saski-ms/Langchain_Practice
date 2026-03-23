from langchain_core.tools.structured import StructuredTool

from pydantic import BaseModel,Field

class add_num(BaseModel):
    a:int=Field(required=True,description="first number")
    b:int=Field(required=True,description="Second number")

def add(a:int,b:int)->int:
    return a+b

add_tool=StructuredTool.from_function(
    func=add,
    name="add",
    description="Addition",
    args_schema=add_num
)

res=add_tool.invoke({"a":2,"b":9})
print(res)