import sys
from typing import Literal

# LangGraph Imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

# Import our Memory Manager
from memory_store import MemoryStore

# Import existing logic (Reusing your work!)
from agents import (
    AgentState, 
    analyst_node, 
    writer_node, 
    llm_with_tools, 
    lookup_policy_docs, 
    web_search_stub,
    rss_feed_search
)

# Initialize Memory
memory_store = MemoryStore()

# --- 1. Define the NEW Memory-Aware Researcher ---
# We are replacing the old researcher_node with this smarter one.

def researcher_with_memory_node(state: AgentState):
    print("\n--- [Agent: Researcher + Memory] is starting ---")
    
    # 1. Check Long-Term Memory FIRST (Step 5.2)
    last_message = state["messages"][0]
    user_topic = last_message.content
    
    print(f"   > Checking archive for '{user_topic}'...")
    memory_context = memory_store.check_memory(user_topic)
    print(f"   > Memory Report: {memory_context}")
    
    # 2. Update System Prompt with Memory Context (Synced with agents.py)
    system_prompt = f"""You are a data gatherer. 
    The current date is February 22, 2026.
    Use tools to find facts about the user's topic.
    Do not analyze, just report facts.
    ALWAYS use 'lookup_policy_docs', 'web_search_stub', and 'rss_feed_search' to gather a mix of PDF, Web, and Industry news.
    
    CRITICAL MEMORY CONTEXT:
    {memory_context}
    
    If the memory says we already covered this recently, mention it in your findings and prioritize finding NEW information."""
    
    # 3. Standard Agent Execution (Same as before)
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt), last_message])
    
    research_findings = []
    
    # Tool Execution Loop
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"   > Executing Tool: {tool_name}")
            
            q = tool_args.get('query')
            if isinstance(q, dict): q = q.get('value', str(q))
            
            # Fallback for Llama 3.2 schema confusion
            if not q or q == "{'type': 'string'}":
                q = tool_args.get('__arg1', tool_args.get('input', user_topic))
                
            q = str(q)

            if tool_name == "lookup_policy_docs":
                res = lookup_policy_docs.invoke(q)
            elif tool_name == "web_search_stub":
                res = web_search_stub.invoke(q)
            elif tool_name == "rss_feed_search":
                res = rss_feed_search.invoke(q)
            
            research_findings.append(f"Source: {tool_name}\nData: {res}")
    else:
        # If the agent decided NOT to call tools, we report that.
        research_findings.append(f"Agent decided not to research. Reason: {response.content}")

    return {"messages": [response], "research_data": research_findings}


# --- 2. The Approval Nodes (Same as Phase 4) ---

def human_approval_node(state: AgentState):
    return state

def route_after_human(state: AgentState) -> Literal["Writer", "__end__"]:
    last_msg = state["messages"][-1].content.lower()
    if "approve" in last_msg:
        return "__end__"
    else:
        return "Writer"

# --- 3. Build the Final Graph ---

workflow = StateGraph(AgentState)

# Use the NEW Researcher, but Keep Old Analyst/Writer
workflow.add_node("Researcher", researcher_with_memory_node) 
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("human_approval", human_approval_node)

workflow.set_entry_point("Researcher")
workflow.add_edge("Researcher", "Analyst")
workflow.add_edge("Analyst", "Writer")
workflow.add_edge("Writer", "human_approval")
workflow.add_conditional_edges("human_approval", route_after_human)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])

# --- 4. Main Execution Loop ---

if __name__ == "__main__":
    print("===============================================")
    print("   NEWS NEXUS FINAL: MEMORY & PERSISTENCE")
    print("===============================================")
    
    # Ask for topic
    user_topic = input("Enter topic: ")
    
    config = {"configurable": {"thread_id": "final_session"}}
    inputs = {"messages": [HumanMessage(content=user_topic)], "research_data": []}
    
    # Run to Gate
    for output in app.stream(inputs, config):
        pass
    
    while True:
        state = app.get_state(config)
        draft = state.values['messages'][-1].content
        
        print("\n=== DRAFT ===\n" + draft + "\n=============\n")
        
        feedback = input(">> 'Approve' to Publish & Save, or give feedback: ")
        
        app.update_state(config, {"messages": [HumanMessage(content=feedback)]})
        
        if "approve" in feedback.lower():
            print("\n[System] Publishing...")
            
            # --- STEP 5.1: SAVE TO LONG TERM MEMORY ---
            memory_store.save_memory(user_topic, draft)
            print("[System] This newsletter has been archived to Long-Term Memory.")
            break
        else:
            print("[System] Revising...")
            for output in app.stream(None, config):
                pass