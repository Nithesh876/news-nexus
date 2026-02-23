import operator
from typing import Annotated, List, TypedDict

from langgraph.graph import StateGraph, END

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)

from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    researcher_data: List[str]
    chart_data: List[dict]

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

def researcher_node(state: AgentState):
    """Research information from user query"""

    messages = state["messages"]

    system_prompt = SystemMessage(
        content="You are a research assistant. Extract useful information and insights."
    )

    response = llm.invoke([system_prompt] + messages)

    return {
        "messages": [response],
        "researcher_data": [response.content]
    }

def chart_node(state: AgentState):
    """Convert research data into structured chart data"""

    research_text = state["researcher_data"][-1]

    prompt = f"""
    Convert the following research into chart-friendly JSON data.

    Return only JSON.

    Research:
    {research_text}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    chart_data = {
        "raw": response.content
    }

    return {
        "messages": [response],
        "chart_data": [chart_data]
    }

def answer_node(state: AgentState):
    """Generate final response using research + chart"""

    research = state["researcher_data"][-1]
    chart = state["chart_data"][-1]

    prompt = f"""
    Provide a final answer using the research and chart data.

    Research:
    {research}

    Chart Data:
    {chart}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [response]
    }

def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_node("final_answer", answer_node)

    workflow.set_entry_point("researcher")

    workflow.add_edge("researcher", "chart_generator")
    workflow.add_edge("chart_generator", "final_answer")
    workflow.add_edge("final_answer", END)

    return workflow.compile()

app = build_graph()

def run_agent(user_input: str):
    """Main function to run agent"""

    result = app.invoke({
        "messages": [HumanMessage(content=user_input)],
        "researcher_data": [],
        "chart_data": []
    })

    return result["messages"][-1].content

if __name__ == "__main__":
    output = run_agent("Give analysis of AI growth in last 5 years")
    print(output)