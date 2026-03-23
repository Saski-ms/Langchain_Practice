from langchain_core.tools import tool

def add(a,b):
    """Add 2 numbers"""
    return a+b

def add(a:int,b:int)->int:
    """ Add numbers"""
    return a+b

@tool
def add(a:int,b:int)->int:
    """Add numbers"""
    return a+b


result=add.invoke({"a":3,"b":3})
print(result)


print(add.name)
print(add.description)
print(add.args)