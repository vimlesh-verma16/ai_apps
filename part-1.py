
import os
from pathlib import Path
from typing import TypedDict, Annotated, List, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from settings import MODEL_API_KEY



llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1, google_api_key=MODEL_API_KEY)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def call_llm(state: AgentState):
    print("--- Calling LLM Node ---")
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("llm", call_llm)
workflow.set_entry_point("llm")
workflow.set_finish_point("llm")
app = workflow.compile()


# --- 5. Run the Graph ---
inputs: AgentState = {"messages": [HumanMessage(content="Hello! How are you today?")]}
final_state = app.invoke(inputs)
print(f"AI: {final_state['messages'][-1].content}")