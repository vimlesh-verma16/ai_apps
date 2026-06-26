from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from settings import MODEL_API_KEY
from mcp_client import initialize_client, list_tools,invoke_tool
import json

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=MODEL_API_KEY,
)


_, session_id = initialize_client()
print(f"Initialized. Session ID: {session_id}")

# 2. List Tools
tools_response = list_tools(session_id)
print("Available tools:", json.dumps(tools_response, indent=2))

mcp_tools = tools_response["result"]["tools"]

# Augment the LLM with tools
tools_by_name = [
    {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["inputSchema"],
    }
    for tool in mcp_tools
]
model_with_tools = model.bind_tools(tools_by_name)

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
        observation = invoke_tool(session_id, tool_call["name"], tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

