import os
import streamlit as st

# Import the two mutually exclusive processing execution layers
from agent_orchestrator import legal_agent_loop
from upload_pipeline import process_and_analyze_upload

# --- PAGE CONFIG & AESTHETIC THEME ---
st.set_page_config(
    page_title="Legal Agent Workspace", 
    page_icon="⚖️", 
    layout="centered"
)

# --- CUSTOM CSS STYLING FOR INTERFACE LOOK ---


# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("⚖️ Workspace Controls")
    st.markdown("---")
    st.subheader("Data Input Mode")
    st.caption("Default Active Layer: Core Constitution Database (Permanent)")
    
    # File Uploader intercepts raw file stream into system RAM buffer
    uploaded_file = st.file_uploader(
        "Upload reference PDF for deep cross-examination (Transient RAM Only)", 
        type=["pdf"],
        key="pdf_uploader",
        help="This file is loaded strictly into volatile memory and is never persisted to disk."
    )
    
    st.markdown("---")
    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN INTERFACE RENDER ---
st.title("Legal Agent Workspace")
st.caption("Interact with the core constitutional archives or dynamically execute ReAct loops on uploaded documents on the fly.")

# Render contextual tracking badges based on user data status
if uploaded_file:
    st.markdown(
        f'<div class="context-badge upload-badge">⚡ Pipeline Routing: Active Document Target ➔ <b>{uploaded_file.name}</b> (Dynamic Memory Isolation Mode)</div>', 
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="context-badge">🏛️ Pipeline Routing: Core Database Target ➔ <b>Constitutional Vaults</b> (Standard Agent Loop)</div>', 
        unsafe_allow_html=True
    )

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture User Query Input Window
if user_query := st.chat_input("Enter your legal question or contract evaluation objective..."):
    
    # Echo User Query to UI State immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Initialize Assistant Response Container
    with st.chat_message("assistant"):
        # Create an expandable live status tracker container for the agent's thought traces
        status_placeholder = st.status("🤖 Agent Engine: Initializing execution track...", expanded=True)
        response_placeholder = st.empty()
        
        # Shared track list container to pipe terminal data directly into the web UI
        trace_logs = []
        
        try:
            # PATH B: Dynamic File Upload Analysis 
            if uploaded_file:
                status_placeholder.update(label=f"Analyzing Document: {uploaded_file.name}...", state="running")
                
                # Pass file stream pointer directly into the isolated upload pipeline setup
                agent_response = process_and_analyze_upload(
                    uploaded_file=uploaded_file, 
                    user_query=user_query, 
                    log_list=trace_logs
                )
                
            # PATH A: Permanent Core Database Retrieval Agent Loop
            else:
                status_placeholder.update(label="Querying Core Knowledge Bases...", state="running")
                
                # Capture custom logs using our shared tracking reference hook
                def ui_logger(msg):
                    trace_logs.append(msg)
                    print(msg)
                
                # Call standard core legal loop frame
                agent_response = legal_agent_loop(
                    user_query=user_query, 
                    log_list=trace_logs
                )
            
            # Print intermediate reasoning thoughts to the user inside the expander widget
            for log_entry in trace_logs:
                status_placeholder.write(f"📝 {log_entry}")
                
            status_placeholder.update(label="Analysis Complete", state="complete", expanded=False)
            
            # Render final final answer to screen
            response_placeholder.markdown(agent_response)
            st.session_state.messages.append({"role": "assistant", "content": agent_response})
            
        except Exception as e:
            status_placeholder.update(label="Pipeline Execution Collapse", state="error", expanded=True)
            response_placeholder.error(f"Execution Error: {str(e)}")