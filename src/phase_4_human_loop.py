import sys
import operator
from typing import Literal

# LangGraph Imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# --- IMPORT FROM PHASE 3 ---
# We reuse the logic you already built!
from agents import (
    AgentState, 
    researcher_node, 
    analyst_node, 
    writer_node, 
    llm_with_tools # We need the LLM for the routing decision
)

# --- 1. Define New Phase 4 Logic (The Human Layer) ---

def human_approval_node(state: AgentState):
    """
    This node acts as a pause point (a toll booth).
    The graph stops here, allowing the human to inspect the state.
    """
    # We don't modify the state here, just pass it through.
    return state

def route_after_human(state: AgentState) -> Literal["Writer", "__end__"]:
    """
    Decides where to go based on what the human typed during the pause.
    """
    # Get the last message (which will be the human's input)
    last_msg = state["messages"][-1].content.lower()
    
    if "approve" in last_msg:
        print("\n--- [System] Content Approved. Publishing... ---")
        return "__end__"
    else:
        print("\n--- [System] Feedback received. Routing back to Writer... ---")
        return "Writer"

# --- 2. Build the Advanced Graph ---
# We are reconstructing the graph but adding the new "Human" node.

workflow = StateGraph(AgentState)

# Add the Phase 3 Nodes (Reused)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)

# Add the Phase 4 Node (New)
workflow.add_node("human_approval", human_approval_node)

# --- 3. Define the Edges (The New Flow) ---
workflow.set_entry_point("Researcher")
workflow.add_edge("Researcher", "Analyst")
workflow.add_edge("Analyst", "Writer")

# NEW: Instead of ending, the Writer goes to Approval
workflow.add_edge("Writer", "human_approval")

# NEW: Conditional Logic
workflow.add_conditional_edges(
    "human_approval",
    route_after_human
)

# Compile with Memory (Required for pausing)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])

# --- 4. The Interactive Execution Loop ---

if __name__ == "__main__":
    print("===============================================")
    print("   NEWS NEXUS: HUMAN-IN-THE-LOOP MODE")
    print("===============================================")
    
    user_topic = input("Enter a topic (e.g., 'AI trends in 2024'): ")
    
    # Config keeps the session memory alive
    config = {"configurable": {"thread_id": "session_phase4"}}
    inputs = {"messages": [HumanMessage(content=user_topic)], "research_data": []}
    
    print(f"\n[System] Starting Agents...")
    
    # 1. Run until the "interrupt" (Human Approval Node)
    # The graph will execute Researcher -> Analyst -> Writer -> PAUSE
    for output in app.stream(inputs, config):
        pass 
    
    # 2. The Loop: Review -> Feedback -> Refine
    while True:
        # Fetch the current state (The Draft created by the Writer)
        state = app.get_state(config)
        
        # Safety check in case something failed
        if not state.values:
            print("Error: No state found.")
            break
            
        current_draft = state.values['messages'][-1].content
        
        print("\n" + "="*40)
        print("        CURRENT DRAFT FOR REVIEW")
        print("="*40)
        print(current_draft)
        print("="*40 + "\n")
        
        # 3. Get Human Input
        user_feedback = input(">> Type 'Approve' to finish, or give feedback (e.g., 'Too long'): ")
        
        # 4. Update the State
        # We pretend the user just sent this message to the bot
        app.update_state(config, {"messages": [HumanMessage(content=user_feedback)]})
        
        # 5. Decide: Exit or Resume?
        if "approve" in user_feedback.lower():
            # If approved, we break the loop.
            # In a real app, you might trigger a 'Publish' function here.
            print("\n[System] Newsletter Finalized and Published!")
            break
        else:
            # If feedback, we resume the graph. 
            # The 'route_after_human' function will see the feedback and send it to 'Writer'
            print(f"\n[System] Agents are revising based on feedback...")
            for output in app.stream(None, config):
                pass