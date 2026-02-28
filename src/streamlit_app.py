import streamlit as st
import os
import time
from datetime import datetime

# --- Import our Backend Logic ---
# We assume these files exist from previous steps
from ingestion import ingest_documents
from tools import get_llm_with_tools, lookup_policy_docs, web_search_stub
from agents import app as agent_app  # Import the graph we built
from memory_store import MemoryStore
from langchain_core.messages import HumanMessage

# --- Paths Configuration ---
PROJECT_ROOT = r"D:\skill-dp\rag-agentic-ai\news-nexus"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# --- Page Config ---
st.set_page_config(page_title="NewsNexus AI", page_icon="📰", layout="wide")

# --- Custom CSS for "Good HTML" Preview ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #4B5563;}
    .agent-box {border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 10px; background-color: #f9f9f9;}
    .success-box {background-color: #D1FAE5; padding: 15px; border-radius: 10px; border: 1px solid #10B981;}
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_data" not in st.session_state:
    st.session_state.research_data = []
if "chart_data" not in st.session_state:
    st.session_state.chart_data = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{int(time.time())}"
if "current_step" not in st.session_state:
    st.session_state.current_step = "idle" # idle, researching, reviewing, finished
if "draft_content" not in st.session_state:
    st.session_state.draft_content = ""

# --- PDF Export Utility ---
def export_as_pdf(html_content):
    from io import BytesIO
    from xhtml2pdf import pisa
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(html_content, dest=pdf_buffer)
    if pisa_status.err:
        return None
    return pdf_buffer.getvalue()

# --- Sidebar: Data Management ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2593/2593240.png", width=80)
    st.title("NewsNexus Control")
    st.divider()
    
    st.subheader("📂 Knowledge Base")
    
    # 1. Show existing files
    existing_pdfs = []
    if os.path.exists(DATA_PATH):
        existing_pdfs = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    
    if existing_pdfs:
        with st.expander(f"Documents in Library ({len(existing_pdfs)})", expanded=False):
            for pdf in existing_pdfs:
                st.write(f"📄 {pdf}")
    else:
        st.caption("No documents in library yet.")

    st.divider()
    
    # 2. Upload new files
    uploaded_files = st.file_uploader("Add New Reports (PDF)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(DATA_PATH, uploaded_file.name)
            os.makedirs(DATA_PATH, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} files!")
        st.rerun() # Refresh to show in list
        
    if st.button("🧠 Build/Update Vector Index"):
        if not existing_pdfs:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Processing Library..."):
                try:
                    pages, chunks = ingest_documents() 
                    st.success(f"Success! Processed {pages} pages into {chunks} chunks.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    # Check system status
    db_ready = os.path.exists(DB_PATH) and os.listdir(DB_PATH)
    status_msg = "✅ Database: ACTIVE" if db_ready else "⚠️ Database: MISSING"
    st.info(f"System Status: {status_msg}")
    st.caption(f"Mode: {'Hybrid (PDF+Web)' if existing_pdfs else 'Web Search Only'}")
    st.caption("LLM: Llama 3.2 (768-dim Ollama)")

# --- Main Interface ---

st.markdown('<div class="main-header">📰 NewsNexus: Corporate Intelligence Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous Multi-Agent System with Human-in-the-Loop</div>', unsafe_allow_html=True)
st.divider()

# Input Area
topic = st.text_input("Enter Research Topic:", placeholder="e.g., 'Impact of Generative AI on Banking sector 2024'")

if st.button("🚀 Start Agents", disabled=st.session_state.current_step != "idle") and topic:
    # --- SMART INITIALIZATION LOGIC ---
    db_exists = os.path.exists(DB_PATH) and os.listdir(DB_PATH)
    raw_pdfs_exist = os.path.exists(DATA_PATH) and any(f.endswith(".pdf") for f in os.listdir(DATA_PATH))
    
    if db_exists:
        st.success("📂 Using existing Knowledge Base...")
    elif raw_pdfs_exist:
        with st.info("🔍 Indexing your library for the first time..."):
            try:
                pages, chunks = ingest_documents()
                st.success(f"Library Indexed! ({pages} pages, {chunks} chunks)")
            except Exception as e:
                st.error(f"Auto-index failed: {e}")
                st.stop()
    else:
        st.warning("🌐 No PDFs found. Proceeding with Web Search only.")
    # -------------------------

    st.session_state.current_step = "researching"
    st.session_state.messages = [HumanMessage(content=topic)]
    st.session_state.research_data = []
    
    # Initialize Memory Store
    try:
        mem_store = MemoryStore()
        with st.spinner("Checking historical archives..."):
            past_memory = mem_store.check_memory(topic)
    except Exception as e:
        st.error(f"Memory Store failed: {e}")
        st.stop()
    
    if "WARNING" in past_memory:
        st.warning(past_memory)

# --- Visualization Logic ---

if st.session_state.current_step == "researching":
    
    # Create containers for real-time updates
    col1, col2, col3 = st.columns(3)
    with col1:
        research_status = st.status("🕵️ Researcher Agent", expanded=True)
    with col2:
        analyst_status = st.status("🧠 Analyst Agent", state="running", expanded=False)
    with col3:
        writer_status = st.status("✍️ Writer Agent", state="running", expanded=False)

    # Run the Graph Stream
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs = {"messages": st.session_state.messages, "research_data": [], "chart_data": []}
    
    try:
        print(f"\n[Streamlit] Starting graph for topic: '{topic}'")
        for event in agent_app.stream(inputs, config):
            if "Researcher" in event:
                research_output = event["Researcher"]
                st.session_state.research_data = research_output.get("research_data", [])
                with research_status:
                    for item in st.session_state.research_data:
                        st.markdown(f"--- \n{item}")
                research_status.update(label=f"Researcher: Found {len(st.session_state.research_data)} items", state="complete", expanded=False)
                analyst_status.update(expanded=True)

            if "Analyst" in event:
                analyst_output = event["Analyst"]
                st.session_state.chart_data = analyst_output.get("chart_data", [])
                with analyst_status:
                    st.write("Identified Trends & Extracted Data:")
                    if st.session_state.chart_data:
                        st.json(st.session_state.chart_data)
                    else:
                        st.write("No numeric trends found.")
                analyst_status.update(label="Analyst: Complete", state="complete", expanded=False)
                writer_status.update(expanded=True)

            if "Writer" in event:
                writer_output = event["Writer"]
                st.session_state.draft_content = writer_output["messages"][-1].content
                with writer_status:
                    st.success("Draft Generated!")
                    st.code(st.session_state.draft_content[:200] + "...", language="html")
                writer_status.update(label="Writer: Complete", state="complete")
        
        st.session_state.current_step = "reviewing"
        st.rerun()
    except Exception as e:
        st.error(f"Execution Error: {e}")


# --- Review Stage (Human-in-the-Loop) ---
if st.session_state.current_step == "reviewing":
    st.subheader("📝 Draft Review")
    
    # Interactive Visualization
    if st.session_state.chart_data:
        import plotly.express as px
        import pandas as pd
        st.markdown("#### 📊 Extracted Trend Analysis")
        df = pd.DataFrame(st.session_state.chart_data)
        fig = px.bar(df, x="label", y="value", title="Data Visualization", color="label")
        st.plotly_chart(fig, use_container_width=True)

    # Tabs for different views
    tab1, tab2 = st.tabs(["📄 Newsletter Draft", "🔍 Raw Research Log"])
    
    with tab1:
        # Display HTML Preview
        with st.expander("View Rendered HTML", expanded=True):
            st.components.v1.html(st.session_state.draft_content, height=600, scrolling=True)
            
    with tab2:
        st.markdown("### 🕵️ Raw Research Findings")
        if st.session_state.research_data:
            for i, data in enumerate(st.session_state.research_data):
                st.info(f"**Finding {i+1}:**\n\n{data}")
        else:
            st.warning("No raw research data found. The agents might have relied on internal knowledge.")
    
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        feedback = st.text_input("Feedback (Leave empty to approve):", placeholder="e.g., 'Make the tone more formal'")
    
    with col_b:
        st.write("") 
        st.write("") 
        if st.button("Submit Decision"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            if feedback:
                agent_app.update_state(config, {"messages": [HumanMessage(content=feedback)]})
                for event in agent_app.stream(None, config):
                    pass
                state = agent_app.get_state(config)
                st.session_state.draft_content = state.values['messages'][-1].content
                st.session_state.chart_data = state.values.get('chart_data', [])
                st.rerun()
            else:
                st.session_state.current_step = "finished"
                mem_store = MemoryStore()
                topic_key = st.session_state.messages[0].content 
                mem_store.save_memory(topic_key, st.session_state.draft_content)
                st.rerun()

# --- Final Stage ---
if st.session_state.current_step == "finished":
    st.balloons()
    st.markdown('<div class="success-box">✅ Newsletter Approved & Archived!</div>', unsafe_allow_html=True)
    
    # Export Options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download HTML",
            data=st.session_state.draft_content,
            file_name=f"newsletter_{int(time.time())}.html",
            mime="text/html"
        )
    
    with col2:
        pdf_data = export_as_pdf(st.session_state.draft_content)
        if pdf_data:
            st.download_button(
                label="📁 Download PDF",
                data=pdf_data,
                file_name=f"newsletter_{int(time.time())}.pdf",
                mime="application/pdf"
            )
        else:
            st.error("PDF Export failed.")

    with col3:
        if st.button("🔄 New Research"):
            st.session_state.current_step = "idle"
            st.rerun()

    st.markdown("### Final Output Preview")
    st.components.v1.html(st.session_state.draft_content, height=800, scrolling=True)