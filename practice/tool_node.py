from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from settings import MODEL_API_KEY

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=MODEL_API_KEY,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    print(f"Multiplying {a} and {b}")
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    print(f"Dividing {a} by {b}")
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
# print(tools_by_name)
model_with_tools = model.bind_tools(tools)

# tools_by_name dictionary will look like this:
"""
{'add': StructuredTool(name='add', description='Adds `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int', args_schema=<class 'langchain_core.utils.pydantic.add'>, func=<function add at 0x71e3028da340>), 'multiply': StructuredTool(name='multiply', description='Multiply `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int', args_schema=<class 'langchain_core.utils.pydantic.multiply'>, func=<function multiply at 0x71e302d58040>), 'divide': StructuredTool(name='divide', description='Divide `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int', args_schema=
"""
""" 
tool_call:dictionary  brlow
--- Calling Tool Node ---
Tool call: {'name': 'add', 'args': {'a': 10, 'b': 3}, 'id': '63c2cded-f99b-443d-ab80-1e30827ed301', 'type': 'tool_call'}
AI: The result of adding 10 and 3 is 13. 
"""
def tool_node(state: dict):
    """Tool node that takes in a state with messages and calls the model with tools."""
    print("--- Calling Tool Node ---")
    result = []
    # print(f"Tool Calls: {state['messages']}")
    for tool_call in state["messages"][-1].tool_calls:
        
        tool = tools_by_name[tool_call["name"]]  
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

