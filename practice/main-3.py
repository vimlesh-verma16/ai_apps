from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from settings import MODEL_API_KEY
from tool_node import model_with_tools, tool_node
import asyncio


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1, google_api_key=MODEL_API_KEY)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if getattr(last_message, "tool_calls", None):
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def llm_call(state: AgentState):
    """LLM decides whether to call a tool or not"""
    print("--- Calling LLM Node ---")

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Build workflow
agent_builder = StateGraph(AgentState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()


# import os

# # Capture the raw PNG data
# png_data = agent.get_graph(xray=True).draw_mermaid_png()

# # Define the filename and path where you want to save it
# output_filename = "graph_diagram.png" # Saves in the current working directory

# Open the file in binary write mode ('wb') and write the data
# try:
#     with open(output_filename, 'wb') as f:
#         f.write(png_data)
#     print(f"Graph diagram saved successfully as '{output_filename}'")
# except Exception as e:
#     print(f"Error saving file: {e}")


async def main():
    inputs: AgentState = {"messages": [HumanMessage(content= "add 20 and 30 ")]}
    final_state = agent.invoke(inputs)
    print(f"AI: {final_state['messages'][-1].content}")

    
asyncio.run(main())
